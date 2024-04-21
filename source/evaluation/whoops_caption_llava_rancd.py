import argparse
import torch
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from pathlib import Path
import h5py
# import kornia
from transformers import set_seed
from evaluation.eval_utils import get_timestamp, find_good_nns
from data.whoops.whoops_utils import load_data_for_whoops
from decoding.vcd_add_noise import add_diffusion_noise
from decoding.pensieve_sample import evolve_sampling
from decoding.pensieve_greedy_search import evolve_greedy_search

evolve_sampling()
evolve_greedy_search()

def eval_model(args):
    # Load Model
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device=device)
    
    whoops_data = load_data_for_whoops(args.whoops_path)
    
    answers_file = os.path.expanduser(os.path.join(args.result_path, args.answers_file_name))
    ans_file = open(answers_file, "w")
    
    if args.use_rancd or args.save_logits:
        q_nn_file_name = args.q_nn_file_path + \
            f'retrieved_{args.database}_imgs_clip_vit_l14_dino_vit_l14_32nns_whoops_images.json'
        q_nn_file = json.load(open(q_nn_file_name, 'r'))
    
    if args.save_logits:
        if args.logits_file_name is None:
            args.logits_file_name = args.answers_file_name.replace('.json', '.hdf5')
        h5py_file = os.path.expanduser(os.path.join(args.result_path, args.logits_file_name))
        logits_file = h5py.File(h5py_file, 'w')
        
    instruct = " please be precise and faithful to the image."
    
    for line in tqdm(whoops_data):
        idx = line['image_id']
        image_file = line['file_name']
        caption_gt = line['selected_caption']
        qs = "Provide a one-sentence caption for the provided image,"  # copy from llava1.5 paper
        
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + instruct)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, 
                                          return_tensors='pt').unsqueeze(0).cuda()  
        image = Image.open(image_file)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]  # [3, 336, 336]
        
        if args.use_rancd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.oracle_noise_step)
            image_id_q = line["image_id"]
            image_id_nns = q_nn_file[image_id_q]["nnimg_file_names"]
            image_id_nns_keep = find_good_nns(image_id_nns, image_id_q, args.kNN)
            image_tensor_nn_l = []
            for image_id_nn in image_id_nns_keep:
                if (not image_id_nn.endswith(".jpg")) and 'coco' in args.database:
                    image_id_nn += ".jpg"
                image_path_nn = os.path.join(args.coco_path, image_id_nn)
                image_nn = Image.open(image_path_nn)
                image_tensor_nn = image_processor.preprocess(image_nn, return_tensors='pt')['pixel_values'][0]
                image_tensor_nn_l.append(image_tensor_nn.unsqueeze(0).half().cuda())
        else:
            image_tensor_cd = None
            image_tensor_nn_l = None    
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            model_output = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() \
                    if image_tensor_cd is not None else None),
                images_racd=image_tensor_nn_l \
                        if image_tensor_nn_l is not None else None,
                alpha_base=args.alpha_base,
                alpha_noise=args.alpha_noise,
                alpha_nns=args.alpha_nns,
                racd_topk=args.racd_topk,
                jsd_thres=args.jsd_thres,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                max_new_tokens=1024,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=False,
                use_cache=True
                )
            scores_tuple = model_output.scores
            intact_scores = torch.cat(scores_tuple[:-2], dim=0)  # remove '.' and '</s>'
            output_ids = model_output.sequences
            
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        
        # observe score distribution change by replacing image tokens
        caption_ids = output_ids[:, input_token_len:-2].clone()  # remove '.' and '</s>'
        output_caption_len = caption_ids.shape[1]
        # decode per token
        outputs_per_token_list = [tokenizer.convert_ids_to_tokens(
            caption_ids[:, i:i+1])[0] for i in range(output_caption_len)]
        assert output_caption_len == intact_scores.shape[0]
        
        if args.save_logits:
            # for visual hallucination analysis
            # diffused image input
            noised_scores_dict = {}
            for noise_step in args.noise_steps:
                if noise_step not in noised_scores_dict.keys():
                    noised_scores_dict[noise_step] = ()
                image_tensor_noise = add_diffusion_noise(image_tensor, noise_step)
                for i in range(output_caption_len):
                    noise_input_ids = torch.cat([input_ids.clone(), caption_ids[:, :i].clone()], dim=1)
                    noise_output = model.generate(
                            noise_input_ids,
                            images=image_tensor_noise.unsqueeze(0).half().cuda(),
                            num_beams=1,
                            do_sample=False,
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            output_scores=True,
                            use_cache=True)
                    noise_scores_tuple = noise_output.scores
                    assert len(noise_scores_tuple) == 1
                    noised_scores_dict[noise_step] += noise_scores_tuple
                assert len(noised_scores_dict[noise_step]) == output_caption_len
                
            # retrieved image input
            nns_scores_dict = {}
            image_id_q = line["image_id"]
            image_id_nns = q_nn_file[image_id_q]["nnimg_file_names"]
            image_id_nns_keep = find_good_nns(image_id_nns, image_id_q, args.kNN)
            for k in range(args.kNN):
                nns_scores_dict[k] = ()
                image_id_nn = image_id_nns_keep[k]
                if not image_id_nn.endswith('.jpg') and 'coco' in args.database:
                    image_id_nn += '.jpg'
                image_path_nn = os.path.join(args.coco_path, image_id_nn)
                image_nn = Image.open(image_path_nn)
                image_tensor_nn = image_processor.preprocess(image_nn, return_tensors='pt')['pixel_values'][0]
                for i in range(output_caption_len):
                    nn_input_ids = torch.cat([input_ids.clone(), caption_ids[:, :i].clone()], dim=1)
                    nn_output = model.generate(
                            nn_input_ids,
                            images=image_tensor_nn.unsqueeze(0).half().cuda(),
                            num_beams=1,
                            do_sample=False,
                            temperature=args.temperature,
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            output_scores=True,
                            use_cache=True)
                    nn_scores_tuple = nn_output.scores
                    assert len(nn_scores_tuple) == 1
                    nns_scores_dict[k] += nn_scores_tuple
                assert len(nns_scores_dict[k]) == output_caption_len

        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        ans_file.write(json.dumps({"image_id": idx,
                                   "caption_gt": caption_gt,
                                   "caption_pred": outputs,
                                   "caption_tokens": outputs_per_token_list,
                                   "model_id": model_name,
                                   "metadata": {"noise steps": args.oracle_noise_step,
                                                "kNN": args.kNN}}) + "\n")
        ans_file.flush()
        
        if args.save_logits:
            logits_file.create_dataset(str(idx)+'_intact', 
                                    (intact_scores.shape[0], intact_scores.shape[1]), 
                                    data=intact_scores.cpu().numpy())
            for noise_step in args.noise_steps:
                noised_scores = torch.cat(noised_scores_dict[noise_step], dim=0)
                assert noised_scores.shape == intact_scores.shape
                logits_file.create_dataset(str(idx)+f'_noise{noise_step}', 
                                (noised_scores.shape[0], noised_scores.shape[1]), 
                                data=noised_scores.cpu().numpy())
            for k in range(args.kNN):
                nn_scores = torch.cat(nns_scores_dict[k], dim=0)
                assert nn_scores.shape == intact_scores.shape
                logits_file.create_dataset(str(idx)+f'_nn{k}', 
                                (nn_scores.shape[0], nn_scores.shape[1]), 
                                data=nn_scores.cpu().numpy())
            
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--whoops_path", type=str, default="")
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--answers_file_name", type=str, default=None)
    parser.add_argument("--save_logits", type=bool, default=False)
    parser.add_argument("--logits_file_name", type=str, default=None)
    
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--alpha_noise", type=float, default=0.1)
    parser.add_argument("--alpha_nns", type=float, default=0.1)
    parser.add_argument("--alpha_base", type=float, default=1.0)
    parser.add_argument("--jsd_thres", type=float, default=None)
    
    parser.add_argument("--use_rancd", action='store_true', default=True)
    parser.add_argument("--decode_method", type=str, default='',  choices=['greedy', 'sample', 'beamsearch'])
    parser.add_argument("--noise_steps", type=list, help="noise step list")
    parser.add_argument("--oracle_noise_step", type=int, default=900)
    parser.add_argument("--kNN", type=int)
    parser.add_argument("--racd_topk", type=int)
    args = parser.parse_args()
    
    #TODO set your path for llava model and whoops data
    args.model_path = '/DATA3/yangdingchen/checkpoint/llava-v1.5-7b'
    args.whoops_path = '/DATA3/yangdingchen/whoops/'
    args.result_path = args.whoops_path + 'results/' + get_timestamp() 
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    
    args.decode_method = 'greedy'
    args.use_rancd = True
    args.save_logits = False  # for visual hallucination evaluation
    
    if args.decode_method == 'greedy':
        args.num_beams = 1
        args.do_sample = False
    elif args.decode_method == 'sample':
        args.num_beams = 1
        args.do_sample = True
        args.top_p = 1
        args.top_k = None
        print(f"top-p {args.top_p}, top-k {args.top_k}")
    elif args.decode_method == 'beamsearch':
        args.num_beams = 3
        args.do_sample = False
    else:
        raise NotImplementedError
    
    if args.save_logits:
        assert not args.do_sample
        args.noise_steps = [999]
        args.kNN = 4
    
    decode_assist = 'wo-cd'
    if args.use_rancd:
        assert args.decode_method in ['greedy', 'sample']
        args.oracle_noise_step = 900
        args.racd_topk = 50
        args.kNN = 4
        decode_assist = 'w-rancd'
    
        args.alpha_noise = 0.1
        args.alpha_nns = 0.1
        args.alpha_base = 1.0
        
        args.jsd_thres = None
        
    answer_file_prefix = 'llava15_whoops_zeroshot_captions_image'
    args.answers_file_name = answer_file_prefix + f'_{args.decode_method}_{decode_assist}.json'
    
    args.database = 'coco'
    args.coco_path = '/DATA3/yangdingchen/coco/images/'
    args.q_nn_file_path = '/home/lufan/Projects/VCD/experiments/rag/q_nn_files/'
    set_seed(args.seed)
    eval_model(args)
