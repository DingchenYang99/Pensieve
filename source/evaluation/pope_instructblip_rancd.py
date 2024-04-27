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
from PIL import Image
import math
import h5py
# import kornia
from lavis.models import load_model_and_preprocess
from pathlib import Path

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
    # Model
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads InstructBLIP model
    # For large_sized model,
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", 
                                                         model_type="vicuna7b", 
                                                         is_eval=True, device=device)

    for question_type in args.question_types:
        args.question_file = args.pope_path + \
            f"{args.dataset_name}/{args.dataset_name}_pope_{question_type}.json"
        answers_file_name = args.result_path + \
            f"/instructblip_{args.dataset_name}_pope_{question_type}_answers_{args.decode_assist}.jsonl"
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
        answers_file = os.path.expanduser(answers_file_name)
        ans_file = open(answers_file, "w")
        
        if args.use_rancd:
            q_nn_file_name = args.q_nn_file_path + \
                f'retrieved_{args.database}_caps_clip_vit_l14_4nns_{args.dataset_name}_{question_type}.json'
            q_nn_file = json.load(open(q_nn_file_name, 'r'))
        
        for line in tqdm(questions):
            idx = line["question_id"]
            image_file = line["image"]
            if args.dataset_name == "coco" or args.dataset_name == "aokvqa":
                image_file = image_file.split("_")[-1]
                
            qs = line["text"]
            gt_ans = line["label"]
            prompt = qs +  " Please answer this question with one word."

            raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
            image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            
            if args.use_rancd:
                image_tensor_cd = add_diffusion_noise(image_tensor, args.oracle_noise_step)
                image_id_q = image_file.rstrip('.jpg') if image_file.endswith(".jpg") else image_file
                
                assert image_id_q == q_nn_file[str(idx)]["image_id"]
                image_id_nns = q_nn_file[str(idx)]["nnimg_file_names"]
                image_id_nns_keep = find_good_nns(image_id_nns, image_id_q, args.kNN)
                image_tensor_nn_l = []
                for image_id_nn in image_id_nns_keep:
                    if not image_id_nn.endswith('.jpg'):
                        image_id_nn += '.jpg'
                    image_path_nn = os.path.join(args.coco_path, image_id_nn)
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
                    alpha_base=args.alpha_base,
                    alpha_noise=args.alpha_noise,
                    alpha_nns=args.alpha_nns,
                    racd_topk=args.racd_topk,
                    jsd_thres=args.jsd_thres,
                    use_nucleus_sampling=args.do_sample, 
                    num_beams=1,
                    repetition_penalty=1,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=False,
                    )
            # intact_scores = torch.cat(scores_tuple[:-2], dim=0)  # remove eos_token
            outputs = outputs[0]
            caption_ids = caption_ids[:, 1:-2]  # remove pad_token, eos_token

            outputs_per_token_list = [model.llm_tokenizer.convert_ids_to_tokens(
                caption_ids[:, i:i+1])[0] for i in range(caption_ids.shape[1])]
            output_caption_len = caption_ids.shape[1]
        
            ans_file.write(json.dumps({"question_id": idx,
                                        "prompt": prompt,
                                        "text": outputs,
                                        "label": gt_ans,
                                        "model_id": "instructblip",
                                        "image": image_file,
                                        "caption_tokens": outputs_per_token_list,
                                        "metadata": {"noise steps": args.oracle_noise_step,
                                                    "kNN": args.kNN}}) + "\n")
            ans_file.flush()
                
        ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
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
    
    parser.add_argument("--use_rancd", action='store_true', default=True)
    parser.add_argument("--oracle_noise_step", type=int)
    parser.add_argument("--kNN", type=int)
    parser.add_argument("--racd_topk", type=int)
    args = parser.parse_args()
    set_seed(args.seed)
    
    #TODO set your path for model and data
    args.pope_path = "/path/to/your/Pensieve/source/data/POPE/"
    args.result_path = '/path/to/your/pope/results/' + get_timestamp()
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    
    args.use_rancd = True  # enable pensieve
    args.do_sample = False  # use greedy decoding
        
    args.decode_assist = 'wo-cd'
    if args.use_rancd:
        args.decode_assist = 'w-rancd'
        args.oracle_noise_step = 900
        args.racd_topk = 2
        args.kNN = 2

        args.alpha_noise = 0.02
        args.alpha_nns = 0.02
        args.alpha_base = 1.5
        
        args.jsd_thres = 1e-2
    
    args.dataset_name = "coco"
    args.question_types = ["random", "popular", "adversarial"]
    
    if args.dataset_name == "coco" or args.dataset_name == "aokvqa":
        args.image_folder = "/path/to/your/coco/images"
    elif args.dataset_name == "gqa":
        args.image_folder = "/path/to/your/gqa/images"
    else:
        raise NotImplementedError
    
    args.database = 'coco'
    args.coco_path = '/path/to/your/coco/images/'
    args.q_nn_file_path = '/path/to/your/Pensieve/source/rag/q_nn_files/'
    eval_model(args)
