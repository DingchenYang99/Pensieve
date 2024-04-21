import torch
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import h5py
from transformers import AutoTokenizer, LlamaTokenizer
import torch.nn.functional as F
from decoding.jsd import calculate_jsd

# this script is used for top-1 confidence score and
# Jensen-Shannon Divergence analysis for each token in the sentence,
# used to extract confidence scores from saved logits hdf5 file
# to draw fig.2 and fig.10 ~ 15 in our paper

#TODO set your visual hallucination analysis directory

# model_name = 'llava15'
model_name = 'instructblip'

sample_method = 'greedy'
decode_assist = 'wo-cd'

noise_step = 999
topk = 50
# topd = 10

whoops_result_path = '/DATA3/yangdingchen/whoops/results/'
analysis_file_name = f'{model_name}_whoops_headvocab_analysis_image_{sample_method}_{decode_assist}.json'
results_file_name = f'{model_name}_whoops_zeroshot_captions_image_{sample_method}_{decode_assist}.json'
logits_file_name = results_file_name.replace('.json', '.hdf5')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("loading tokenizer")
if model_name == 'llava15':
    result_time_dir = ''
    model_path = '/DATA3/yangdingchen/checkpoint/llava-v1.5-7b'
    model_path = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
elif model_name == 'instructblip':
    result_time_dir = ''
    llm_model = "/DATA3/yangdingchen/checkpoint/vicuna-7b-v1.1"
    tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'bos_token': '</s>'})
    tokenizer.add_special_tokens({'eos_token': '</s>'})
    tokenizer.add_special_tokens({'unk_token': '</s>'})
    tokenizer.padding_side = "left"
else:
    raise NotImplementedError

# file names
results = [json.loads(q) for q in open(os.path.expanduser(
    os.path.join(whoops_result_path, result_time_dir, results_file_name)), "r")]
scores = h5py.File(os.path.expanduser(
    os.path.join(whoops_result_path, result_time_dir, logits_file_name)), 'r')
analysis = open(os.path.expanduser(
    os.path.join(whoops_result_path, result_time_dir, analysis_file_name)), "w")

#TODO set your interested test sample in whoops, 
# interested_hallucination_idx stands for the interested token's index in the sentence
query_sample_ids_dict = {
    # "1313395dfd6f998b1029f676ca336966c1583eb727f99af05a010c028632527f":{
    #     "caption_token_head_vocabs":{},
    # },
    # "42b85d70e840c34d44aee9cc55e216a8f147728e1fee181e6b4f76147ee93c67":{
    #     "caption_token_head_vocabs":{},
    # },
    # "f1ab0fe7d6a48fe63bf7ccaf849a9fdb186dc8c53b7a1e1dfbc1d3a9ba9d194a":{
    #     # "interested_hallucination_idx": 8,
    #     "caption_token_head_vocabs":{},
    # },
    # "512d74c66093d76a9bfcee93cacb80be74afa05386d86fad1bfd4115fadd3bc0":{
    #     "caption_token_head_vocabs":{},
    # },
    # "b06bef5a72e3149ecdeece8dd18c41cdb27c21c5e748180e5155e773654c369a":{
    #     # "interested_hallucination_idx": 10,
    #     "caption_token_head_vocabs":{},
    # }
    # "2c220934c912a40bfe64f6a650ae993dba2022f5babab4f8fe22e65d2882f0c7":{
    #     # "interested_hallucination_idx": 3,
    #     "caption_token_head_vocabs":{},
    # }
    # "ec651e98429fd77c7b0c379f7e7ad5220911ee35e74fb851632ec447447ca81b":{
    #     # "interested_hallucination_idx": 7,
    #     "caption_token_head_vocabs":{},
    # }
    "19948343aeef2c3c834bb2c623ada551b1c30383ad9d34482ef4cc478ed747dd":{
        "caption_token_head_vocabs":{},
    },
    # "583585d39f185ab38c3d25aad19f9a7c8c4d6286f363fd55520700695682404a":{
    #     "caption_token_head_vocabs":{},
    # },
    # "c66105596372fbca606367741cb84ffe4c741917ab9fed813b72bd727b857c12":{
    #     "caption_token_head_vocabs":{},
    # },
    # "fbe2d7c9f44da9a8fc0eb450ea2d0d83644685d50c86c8b1ec928d96cd83827b":{
    #     "caption_token_head_vocabs":{},
    # },
}

print("finding query sample information")
total_query_samples = len(query_sample_ids_dict.keys())
found = 0
for line in results:
    image_id = line["image_id"]
    if not image_id in query_sample_ids_dict.keys():
        continue
    query_sample_ids_dict[image_id]["caption_tokens"] = line["caption_tokens"]
    query_sample_ids_dict[image_id]["base_logits"] = torch.tensor(scores[image_id+'_intact'][()]).to(device)
    # [num_tokens, num_vocab]
    query_sample_ids_dict[image_id]["noise_logits"] = torch.tensor(scores[image_id+f'_noise{noise_step}'][()]).to(device)
    
    assert len(line["caption_tokens"]) == query_sample_ids_dict[image_id]["base_logits"].shape[0]
    assert len(line["caption_tokens"]) == query_sample_ids_dict[image_id]["noise_logits"].shape[0]
    
    found += 1
    if found == total_query_samples:
        break
    
for image_id, source_data in query_sample_ids_dict.items():
    
    greedy_base_logits_list = []
    greedy_noise_logits_list = []
    jsd_list = []
    jsd_topk_list = []
    for hallucination_idx, greedy_decoded_token in enumerate(source_data["caption_tokens"]):
        interested_hallucination_idx = source_data.get("interested_hallucination_idx", None)
        if interested_hallucination_idx is not None:
            if hallucination_idx != interested_hallucination_idx:
                continue
        base_logits = source_data["base_logits"][hallucination_idx, ...]
        noise_logits = source_data["noise_logits"][hallucination_idx, ...]
        
        jsd = calculate_jsd(base_logits[None, :], noise_logits[None, :])[0].item()
        jsd_list.append(jsd)
        
        # get head vocabulary with topk ranked candidates
        indices_to_keep = base_logits >= torch.topk(base_logits, topk)[0][..., -1]  
        base_head_idx = torch.nonzero(indices_to_keep.float()).t()  # [1, num_nonzero]
        base_logits_topk = base_logits[indices_to_keep].clone()
        noise_logits_topk = noise_logits[indices_to_keep].clone()
        delta_logits_topk = base_logits_topk - noise_logits_topk
        
        jsd_topk = calculate_jsd(base_logits_topk[None, :], noise_logits_topk[None, :])[0].item()
        jsd_topk_list.append(jsd_topk)
        base_logits_max = torch.max(base_logits_topk).item()
        
        head_vocab_list = []
        for i in range(base_head_idx.shape[1]):
            this_candidate_token = tokenizer.convert_ids_to_tokens(base_head_idx[:, i:i+1])[0]
            this_base_logit = base_logits[base_head_idx[0, i]].item()
            this_noise_logit = noise_logits[base_head_idx[0, i]].item()
            if this_candidate_token == greedy_decoded_token:
                assert this_base_logit == base_logits_max
                greedy_token_base_logit = this_base_logit
                greedy_base_logits_list.append(greedy_token_base_logit)
                greedy_token_noise_logit = this_noise_logit
                greedy_noise_logits_list.append(greedy_token_noise_logit)
            head_vocab_list.append({"token": this_candidate_token,
                                    "base_logit": this_base_logit,
                                    "noise_logit": this_noise_logit,
                                    "delta_logit%": (this_noise_logit - this_base_logit)/this_base_logit*100,
                                    "delta_logit": (this_base_logit - this_noise_logit)
                                    })
            
        sorted_head_vocab_list = sorted(head_vocab_list, reverse=True, key=lambda item: item["base_logit"])
        query_sample_ids_dict[image_id]["caption_token_head_vocabs"][greedy_decoded_token] = {
            "greedy_token_base_logit": greedy_token_base_logit,
            "greedy_token_noise_logit": greedy_token_noise_logit,
            "head_vocabs": sorted_head_vocab_list
        }
        
    print(query_sample_ids_dict[image_id]["caption_tokens"])  # tokens in sentence
    print(greedy_base_logits_list)  # top-1 base scores for each token in sentence
    print(greedy_noise_logits_list)  # top-1 txt scores for each token in sentence
    # print(jsd_list)
    print(jsd_topk_list)  # JSD within head vocabulary for each token in sentence
    
    del query_sample_ids_dict[image_id]["base_logits"], query_sample_ids_dict[image_id]["noise_logits"]
    analysis.write(json.dumps(
        {"image_id": image_id,
        "caption_tokens": query_sample_ids_dict[image_id]["caption_tokens"],
        "caption_token_head_vocabs": query_sample_ids_dict[image_id]["caption_token_head_vocabs"],
        }) + "\n"
    )
    analysis.flush()
    analysis.close()