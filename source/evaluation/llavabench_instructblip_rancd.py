import argparse
import torch
import json
from tqdm import tqdm
import shortuuid
import sys
import os
from transformers import set_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.utils import disable_torch_init
from lavis.models import load_model_and_preprocess

from PIL import Image
import math
from pathlib import Path
import h5py
# import kornia
from data.lbench.llavabench_utils import load_data_for_llavabench
from evaluation.eval_utils import get_timestamp, find_good_nns
from decoding.vcd_add_noise import add_diffusion_noise
from decoding.pensieve_sample import evolve_sampling
from decoding.pensieve_greedy_search import evolve_greedy_search

evolve_sampling()
evolve_greedy_search()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # Load Model
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads InstructBLIP model
    # For large_sized model,
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct",
                                                         model_type="vicuna7b", 
                                                         is_eval=True, 
                                                         device=device)
    
    lbench_data = load_data_for_llavabench(args.lbench_path)
    
    answers_file = os.path.expanduser(os.path.join(args.result_path, args.answers_file_name))
    ans_file = open(answers_file, "w")
    
    if args.use_rancd:
        q_nn_file_name = args.q_nn_file_path + \
            f'retrieved_{args.database}_imgs_clip_vit_l14_dino_vit_l14_32nns_llavabench_images.json'
        q_nn_file = json.load(open(q_nn_file_name, 'r'))
    
    for line in tqdm(lbench_data):
        idx = line['image_id']
        image_file = line['file_name']
        caption_gt = line['gt_ans']
        qs = line["text"]
        prompt = qs

        raw_image = Image.open(image_file).convert("RGB")
        # prepare the image
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        if args.use_rancd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.oracle_noise_step)
            image_id_q = line["image_id"]
            image_id_nns = q_nn_file[image_id_q]["nnimg_file_names"]
            image_id_nns_keep = find_good_nns(image_id_nns, image_id_q, args.kNN)
            image_tensor_nn_l = []
            for image_id_nn in image_id_nns_keep:
                if (not image_id_nn.endswith('.jpg')) and 'coco' in args.database:
                    image_id_nn += '.jpg'
                image_path_nn = os.path.join(args.database_path, image_id_nn)
                image_nn = Image.open(image_path_nn).convert("RGB")
                image_tensor_nn = vis_processors["eval"](image_nn).unsqueeze(0).to(device)
                image_tensor_nn_l.append(image_tensor_nn)
        else:
            image_tensor_cd = None
            image_tensor_nn_l = None  
                
        with torch.inference_mode():
            outputs, scores_tuple, caption_ids = model.generate(
                {"image": image_tensor, "prompt": prompt},
                images_cd=image_tensor_cd,
                images_racd=image_tensor_nn_l,
                alpha_noise=args.alpha_noise,
                alpha_nns=args.alpha_nns,
                alpha_base=args.alpha_base,
                racd_topk=args.racd_topk,
                jsd_thres=args.jsd_thres,
                use_nucleus_sampling=args.do_sample, 
                num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=1.,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=False,
                )
        outputs = outputs[0]
        caption_ids = caption_ids[:, 1:-2]
        
        outputs_per_token_list = [model.llm_tokenizer.convert_ids_to_tokens(
            caption_ids[:, i:i+1])[0] for i in range(caption_ids.shape[1])]
        output_caption_len = caption_ids.shape[1]
        assert len(outputs_per_token_list) == output_caption_len

        ans_file.write(json.dumps({"image_id": idx,
                                   "question": prompt,
                                   "caption_gt": caption_gt,
                                   "caption_pred": outputs,
                                   "caption_tokens": outputs_per_token_list,
                                   "model_id": "instructblip",
                                   "metadata": {"noise steps": args.oracle_noise_step,
                                                "kNN": args.kNN}}) + "\n")
        ans_file.flush()
            
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--lbench_path", type=str, default="")
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--answers_file_name", type=str, default=None)
    
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
    
    parser.add_argument("--use_rancd", action='store_true', default=True)
    parser.add_argument("--decode_method", type=str, default='', 
                        choices=['greedy', 'sample', 'beamsearch'])
    parser.add_argument("--oracle_noise_step", type=int)
    parser.add_argument("--kNN", type=int)
    parser.add_argument("--racd_topk", type=int)
    args = parser.parse_args()
    
    #TODO set your path for model and data
    args.lbench_path = '/path/to/your/llava-bench/'
    args.result_path = args.lbench_path + 'results/' + get_timestamp() 
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    
    args.decode_method = 'greedy'
    args.use_rancd = True
    
    if args.decode_method == 'greedy':
        args.num_beams = 1
        args.do_sample = False
    elif args.decode_method == 'sample':
        args.num_beams = 1
        args.do_sample = True
    elif args.decode_method == 'beamsearch':
        args.num_beams = 3
        args.do_sample = False
    else:
        raise NotImplementedError
        
    decode_assist = 'wo-cd'
        
    if args.use_rancd:
        assert args.decode_method in ['greedy', 'sample']
        args.oracle_noise_step = 900
        args.racd_topk = 50
        args.kNN = 2
        decode_assist = 'w-rancd'
    
        args.alpha_noise = 0.1
        args.alpha_nns = 0.1
        args.alpha_base = 1.0
        
        args.jsd_thres = None
        
    answer_file_prefix = 'instructblip_lbench_zeroshot_captions_image'
    args.answers_file_name = answer_file_prefix + f'_{args.decode_method}_{decode_assist}.json'
    
    args.database = 'coco'
    args.database_path = '/path/to/your/coco/images/'
    args.q_nn_file_path = '/path/to/your/Pensieve/source/rag/q_nn_files/'
    set_seed(args.seed)
    eval_model(args)
