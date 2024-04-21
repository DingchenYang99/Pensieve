import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
import h5py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
# import kornia
from transformers import set_seed
from pathlib import Path
from data.mme.mme_utils import load_data_mme
from evaluation.eval_utils import get_timestamp, find_good_nns
from decoding.vcd_add_noise import add_diffusion_noise
from decoding.pensieve_sample import evolve_sampling
from decoding.pensieve_greedy_search import evolve_greedy_search

evolve_sampling()
evolve_greedy_search()
     
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    for interested_task_name in args.interested_task_names:
        answers_file_name = args.result_path + f"/llava15_mme_{interested_task_name}_answers_{args.decode_assist}.txt"
        questions = load_data_mme(args.image_folder, interested_task_name)
        answers_file = os.path.expanduser(answers_file_name)
        ans_file =  open(answers_file, "w")
        answers_file_json_name = answers_file_name.replace('.txt', '.jsonl')
        answers_file_json = os.path.expanduser(answers_file_json_name)
        ans_file_json =  open(answers_file_json, "w")
        
        if args.use_rancd:
            q_nn_file_name = args.q_nn_file_path + \
                f'retrieved_{args.database}_caps_clip_vit_l14_4nns_{interested_task_name}.json'
                # reranked using bleu score
            q_nn_file = json.load(open(q_nn_file_name, 'r'))
        
        for idx, line in enumerate(tqdm(questions)):
            image_file = line["image"]
            qs = line["text"]
            gt_ans = line["label"]
            
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image = Image.open(image_file)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            if args.use_rancd:
                image_tensor_cd = add_diffusion_noise(image_tensor, args.oracle_noise_step)
                image_id_q = line["image_id"]
                assert image_id_q == q_nn_file[str(idx)]["image_id"]
                image_id_nns = q_nn_file[str(idx)]["nnimg_file_names"]
                image_id_nns_keep = find_good_nns(image_id_nns, image_id_q, args.kNN)
                image_tensor_nn_l = []
                for image_id_nn in image_id_nns_keep:
                    if not image_id_nn.endswith('.jpg'):
                        image_id_nn += '.jpg'
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
                    alpha_noise=args.alpha_noise,
                    alpha_nns=args.alpha_nns,
                    alpha_base=args.alpha_base,
                    racd_topk=args.racd_topk,
                    jsd_thres=args.jsd_thres,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    num_beams=1,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=False,
                    use_cache=True
                    )
                # scores_tuple = model_output.scores
                # intact_scores = torch.cat(scores_tuple, dim=0)
                output_ids = model_output.sequences
                
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                
            caption_ids = output_ids[:, input_token_len:].clone()
            output_caption_len = caption_ids.shape[1]
            # assert output_caption_len == intact_scores.shape[0]
            # decode per token
            outputs_per_token_list = [tokenizer.convert_ids_to_tokens(
                caption_ids[:, i:i+1])[0] for i in range(output_caption_len)]
            
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            ans_file.write(image_file.split("/")[-1] + "\t" + cur_prompt + "\t" + gt_ans + "\t" + outputs + "\n")
            ans_file_json.write(json.dumps({"question_id": idx,
                                        "prompt": cur_prompt,
                                        "text": outputs,
                                        "label": gt_ans,
                                        "model_id": model_name,
                                        "image": image_file.split("/")[-1],
                                        "caption_tokens": outputs_per_token_list,
                                        "metadata": {"noise steps": args.oracle_noise_step,
                                                    "kNN": args.kNN}}) + "\n")
            ans_file_json.flush()
            ans_file.flush()
                
        ans_file.close()
        ans_file_json.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question_file", type=str, default="")
    parser.add_argument("--answers_file", type=str, default="")
    
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--alpha_noise", type=float)
    parser.add_argument("--alpha_nns", type=float)
    parser.add_argument("--alpha_base", type=float)
    parser.add_argument("--jsd_thres", type=float, default=None)

    parser.add_argument("--use_rancd", action='store_true', default=False)
    parser.add_argument("--oracle_noise_step", type=int)
    parser.add_argument("--kNN", type=int)
    parser.add_argument("--racd_topk", type=int)
    args = parser.parse_args()
    set_seed(args.seed)
    
    #TODO set your path for model and data
    args.mme_path = "/DATA3/yangdingchen/mme/"
    args.model_path = '/DATA3/yangdingchen/checkpoint/llava-v1.5-7b'
    args.result_path = args.mme_path + 'results/' + get_timestamp()
    args.image_folder = args.mme_path + 'MME_Benchmark_release_version/'
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    
    args.use_rancd = True
    args.do_sample = False
    
    args.interested_task_names = [
        "existence", 
        "count", 
        "position",
        "color",
        # "posters", 
        # "celebrity", 
        # "scene", 
        # "landmark", 
        # "artwork", 
        # "OCR"
        ]
    
    args.decode_assist = 'wo-cd'
    if args.use_rancd:
        args.decode_assist = 'w-rancd'
        args.oracle_noise_step = 700
        args.racd_topk = 2
        args.kNN = 1
        
        args.alpha_noise = 0.01
        args.alpha_nns = 0.1
        args.alpha_base = 1.0
        args.jsd_thres = None
    
    args.database = 'coco'
    args.coco_path = '/DATA3/yangdingchen/coco/images/'
    args.q_nn_file_path = '/home/lufan/Projects/VCD/experiments/rag/q_nn_files/'
    eval_model(args)
