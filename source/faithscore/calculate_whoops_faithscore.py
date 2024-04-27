import os
import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.faithscore.framework import FaithScore
from data.whoops.whoops_utils import *
import argparse
from transformers import set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--whoops_path',
                        type=str,)
    parser.add_argument('--answer_file',
                        type=str,)
    parser.add_argument('--model_name',
                        type=str,)
    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--vem_type',
                        type=str,
                        choices=["ofa", "ofa-ve", "llava"],
                        default="llava")
    parser.add_argument('--llava_path',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--llama_path',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--use_llama',
                        type=bool,
                        default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    
    args.vem_type = 'ofa_ve'
    #TODO set your openai-key
    args.openai_key = 'sk-'
    #TODO set your model, data and results path
    args.llava_path = "/path/to/your/llava-v1.5-13b"  # optional, set if vem_type==llava
    args.llama_path = "/path/to/your/llama-2-7b-hf"  # optional, set if use_llama==True
    args.whoops_path = '/path/to/your/whoops/'
    data_root = args.whoops_path + 'results/'
    time_dir = 'yymmdd-hhmmss'
    dataDir = data_root + time_dir
    
    image_suffix = 'png'
    args.use_llama = False
    
    args.model_name = 'llava15'
    # args.model_name = 'instructblip'
    
    decode_method = 'greedy'
    # decode_method = 'sample'
    
    # decode_assist = 'wo-cd'
    decode_assist = 'w-rancd'
    
    args.answer_file = dataDir + f'/{args.model_name}_whoops_zeroshot_captions_image_{decode_method}_{decode_assist}.json'
    result_file = args.answer_file.replace('.json', f'_{args.vem_type}_fs_results.jsonl')
    try:
        # if you have already extracted atomic facts
        atomic_facts_file = args.answer_file.replace('.json', f'_ofa_ve_fs_results.jsonl')
        atomic_facts_list = [json.loads(q) for q in open(os.path.expanduser(atomic_facts_file), "r")]
        pre_atomic_facts = [ii["atomic_facts"] for ii in atomic_facts_list]
        print("pre-extracted atomic facts found")
    except:
        pre_atomic_facts = None
        print("NO pre-extracted atomic facts found")
        pass

    images = []
    answers = []
    labeld_sub_sens = []
    ans_dict = {}
    image_ids = []
    cnt = 0
    for res in open(os.path.expanduser(args.answer_file), "r"):
        res_dict = json.loads(res)
        image_id = res_dict["image_id"]
        image_ids.append(image_id)
        file_path = args.whoops_path + f'whoops_images/{image_id}.{image_suffix}'
        images.append(file_path)
        answers.append(res_dict["caption_pred"])
        # in whoops benchmark,
        # there is only a single sentence image description,
        # i.e., there is no analytical content
        labeld_sub_sens.append(res_dict["caption_pred"] + ' [D]')
        cnt += 1
    assert len(answers) == len(images)
    print(f"data directory: {dataDir}")
    print(f"Faithscore is evaluated on {len(images)} samples.")

    score = FaithScore(vem_type=args.vem_type, api_key=args.openai_key,
                       llava_path=args.llava_path, use_llama=args.use_llama,
                       tokenzier_path=args.llama_path, llama_path=args.llama_path)
    
    f, atomic_facts, fact_scores = score.faithscore(answers, images, labeld_sub_sens, pre_atomic_facts)
    # print(f"Faithscore is {f}. Sentence-level faithscore is {sentence_f}.")
    print(f"data directory: {dataDir}")
    print(f"Faithscore is {f}.")

    fs_results = open(os.path.expanduser(result_file), "w")
    for image_id, caption_pred, atomic_fact, fs_list in zip(
        image_ids, answers, atomic_facts, fact_scores
    ):
        faith_score = sum(fs_list) / len(fs_list) if len(fs_list) > 0 else 0
        fs_results.write(json.dumps({
                "image_id": image_id,
                "faith_score": faith_score,
                "caption_pred": caption_pred,
                "atomic_facts": atomic_fact,
                "fs_list": fs_list,
                "vem_name":args.vem_type
                }) + "\n")
        fs_results.flush()
    fs_results.close()
    
    # print(atomic_facts)
    # print(fact_scores)