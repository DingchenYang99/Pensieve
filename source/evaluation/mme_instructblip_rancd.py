import argparse
import torch
import json
from tqdm import tqdm
import shortuuid
import sys
import os
import h5py
from transformers import set_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.utils import disable_torch_init
from PIL import Image
import math
# import kornia
from lavis.models import load_model_and_preprocess
from pathlib import Path
from data.mme.mme_utils import load_data_mme
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

    for interested_task_name in args.interested_task_names:
        answers_file_name = args.result_path + f"/instructblip_mme_{interested_task_name}_answers_{args.decode_assist}.txt"
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
            gt_ans = line["label"]
            qs = line["text"]
            cur_prompt = qs

            raw_image = Image.open(os.path.join(image_file)).convert("RGB")
            image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            
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
                    image_nn = Image.open(image_path_nn).convert("RGB")
                    image_tensor_nn = vis_processors["eval"](image_nn).unsqueeze(0).to(device)
                    image_tensor_nn_l.append(image_tensor_nn)
            else:
                image_tensor_cd = None
                image_tensor_nn_l = None   
                
            with torch.inference_mode():
                outputs, scores_tuple, caption_ids = model.generate(
                    {"image": image_tensor, "prompt": cur_prompt},
                    images_cd=image_tensor_cd,
                    images_racd=image_tensor_nn_l,
                    alpha_noise=args.alpha_noise,
                    alpha_nns=args.alpha_nns,
                    alpha_base=args.alpha_base,
                    racd_topk=args.racd_topk,
                    jsd_thres=args.jsd_thres,
                    use_nucleus_sampling=args.do_sample, 
                    num_beams=1,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=False,
                    )
            # intact_scores = torch.cat(scores_tuple[:-2], dim=0)  # remove eos_token
            outputs = outputs[0]
            caption_ids = caption_ids[:, 1:-2]  # remove pad_token-0, eos_token-2, bos_token-1
        
            outputs_per_token_list = [model.llm_tokenizer.convert_ids_to_tokens(
                caption_ids[:, i:i+1])[0] for i in range(caption_ids.shape[1])]
            output_caption_len = caption_ids.shape[1]
            # assert output_caption_len == intact_scores.shape[0]
            
            ans_file.write(image_file.split("/")[-1] + "\t" + cur_prompt + "\t" + gt_ans + "\t" + outputs + "\n")
            ans_file_json.write(json.dumps({"question_id": idx,
                                        "prompt": cur_prompt,
                                        "text": outputs,
                                        "label": gt_ans,
                                        "model_id": "instructblip",
                                        "image": image_file.split("/")[-1],
                                        "caption_tokens": outputs_per_token_list,
                                        "metadata": {"noise steps": args.oracle_noise_step,
                                                    "kNN": args.kNN}}) + "\n")
            ans_file.flush()
            ans_file_json.flush()
                
        ans_file.close()
        ans_file_json.close()

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
    
    parser.add_argument("--alpha_noise", type=float, default=1.0)
    parser.add_argument("--alpha_nns", type=float, default=1.0)
    parser.add_argument("--alpha_base", type=float, default=1.0)
    parser.add_argument("--jsd_thres", type=float, default=None)
    
    parser.add_argument("--use_rancd", action='store_true', default=False)
    parser.add_argument("--oracle_noise_step", type=int)
    parser.add_argument("--kNN", type=int)
    parser.add_argument("--racd_topk", type=int)
    args = parser.parse_args()
    set_seed(args.seed)
    
    #TODO set your path for model and data
    args.mme_path = "/DATA3/yangdingchen/mme/"
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
        args.oracle_noise_step = 900
        args.racd_topk = 2
        args.kNN = 2
        
        args.alpha_noise = 0.1
        args.alpha_nns = 0.5
        args.alpha_base = 1.0
        
        args.jsd_thres = None
    
    args.database = 'coco'
    args.coco_path = '/DATA3/yangdingchen/coco/images/'
    args.q_nn_file_path = '/home/lufan/Projects/VCD/experiments/rag/q_nn_files/'
    eval_model(args)
