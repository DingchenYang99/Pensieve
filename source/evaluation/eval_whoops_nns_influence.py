import torch
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import h5py
from transformers import AutoTokenizer, LlamaTokenizer

# this script is used for confidence scores analysis of retrieved images
# at a given token in the sentence,
# used to extract confidence scores from saved logits hdf5 file
# to draw fig.2 and fig.9 ~ 15 in our paper

model_name = 'llava15'
# model_name = 'instructblip'

sample_method = 'greedy'
decode_assist = 'wo-cd'

# noise_step = 999
topk = 50
topd = 10

#TODO set your data and results path

result_path = '/DATA3/yangdingchen/whoops/results/'
knn = 4

analysis_file_name = f'{model_name}_whoops_headvocab_nns_analysis_image_{sample_method}_{decode_assist}.json'
results_file_name = f'{model_name}_whoops_zeroshot_captions_image_{sample_method}_{decode_assist}.json'
logits_file_name = results_file_name.replace('.json', '.hdf5')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("loading tokenizer")
if model_name == 'llava15':
    result_time_dir = ''
    model_path = '/DATA3/yangdingchen/checkpoint/llava-v1.5-7b'
    model_path = os.path.expanduser(model_path)
    # disable_torch_init()
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
    os.path.join(result_path, result_time_dir, results_file_name)), "r")]
scores = h5py.File(os.path.expanduser(
    os.path.join(result_path, result_time_dir, logits_file_name)), 'r')
analysis = open(os.path.expanduser(
    os.path.join(result_path, result_time_dir, analysis_file_name)), "w")

#TODO set your interested test sample in whoops, 
query_sample_ids_dict = {
    # "512d74c66093d76a9bfcee93cacb80be74afa05386d86fad1bfd4115fadd3bc0":{
    #     "interested_hallucination_idx": 4,
    #     "caption_token_head_vocabs":{},
    # },
    # "a6768ca96f4c9a37e926a6a051437130b542a2d1506209da54f970a8f318ce5f":{
    #     "interested_hallucination_idx": 8,
    #     "caption_token_head_vocabs":{},
    # }
    # "98152160807b0d036e3c813e95eae1d0000f68a5fe8a2087eda391e68c53508f":{
    #     "interested_hallucination_idx": 1,
    #     "caption_token_head_vocabs":{},
    # }
    # "483452e205f21fd1830c4b8c8e90978c36928a5a02f5a90f9dbd75f5a9712b0e":{
    #     "interested_hallucination_idx": 1,
    #     "caption_token_head_vocabs":{},
    # }
    # "b06bef5a72e3149ecdeece8dd18c41cdb27c21c5e748180e5155e773654c369a":{
    #     "interested_hallucination_idx": 10,
    #     "caption_token_head_vocabs":{},
    # }
    "2c220934c912a40bfe64f6a650ae993dba2022f5babab4f8fe22e65d2882f0c7":{
        "interested_hallucination_idx": 10,
        "caption_token_head_vocabs":{},
    }
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
    for k in range(knn):
        nn_score = torch.tensor(scores[image_id+f'_nn{k}'][()]).to(device)
        query_sample_ids_dict[image_id][f"{k}nn_logits"] = nn_score
        assert len(line["caption_tokens"]) == nn_score.shape[0]
    assert len(line["caption_tokens"]) == query_sample_ids_dict[image_id]["base_logits"].shape[0]
    
    found += 1
    if found == total_query_samples:
        break
    
for image_id, source_data in query_sample_ids_dict.items():
    
    greedy_base_logits_list = []
    greedy_nns_logits_list = []
    for hallucination_idx, greedy_decoded_token in enumerate(source_data["caption_tokens"]):
        interested_hallucination_idx = source_data.get("interested_hallucination_idx", None)
        if interested_hallucination_idx is not None:
            if hallucination_idx != interested_hallucination_idx:
                continue
        base_logits = source_data["base_logits"][hallucination_idx, ...]
        # get head vocabulary with topk ranked candidates
        indices_to_keep = base_logits >= torch.topk(base_logits, topk)[0][..., -1]  
        base_head_idx = torch.nonzero(indices_to_keep.float()).t()  # [1, num_nonzero]
        base_logits_topk = base_logits[indices_to_keep].clone()
        base_logits_topk = base_logits_topk[None, :]
        
        nn_logits_l = []
        nn_topk_logits_l = []
        for k in range(knn):
            nn_logits = source_data[f"{k}nn_logits"][hallucination_idx, ...]
            nn_logits_topk = nn_logits[indices_to_keep].clone()
            nn_topk_logits_l.append(nn_logits_topk[None, :])
            nn_logits_l.append(nn_logits)

        base_logits_max = torch.max(base_logits_topk[0, :]).item()        
        head_vocab_list = []
        for i in range(base_head_idx.shape[1]):
            this_candidate_token = tokenizer.convert_ids_to_tokens(base_head_idx[:, i:i+1])[0]
            this_base_logit = base_logits[base_head_idx[0, i]].item()
            if this_candidate_token == greedy_decoded_token:
                assert this_base_logit == base_logits_max
                greedy_token_base_logit = this_base_logit
                greedy_base_logits_list.append(greedy_token_base_logit)
            output_dict = {"token": this_candidate_token,
                            "base_logit": this_base_logit}
            for k in range(knn):
                this_nn_logit = nn_logits_l[k][base_head_idx[0, i]].item()
                output_dict.update({f"{k}nn_logit": this_nn_logit})
            head_vocab_list.append(output_dict)
            
        sorted_head_vocab_list = sorted(head_vocab_list, reverse=True, key=lambda item: item["base_logit"])
        for j, head_voc in enumerate(sorted_head_vocab_list):
            print(f"{j}, {head_voc}")
        query_sample_ids_dict[image_id]["caption_token_head_vocabs"][greedy_decoded_token] = {
            "greedy_token_base_logit": greedy_token_base_logit,
            "head_vocabs": sorted_head_vocab_list}
        
    print(query_sample_ids_dict[image_id]["caption_tokens"])
    print(greedy_base_logits_list)

    del query_sample_ids_dict[image_id]["base_logits"]
    for k in range(knn):
        del query_sample_ids_dict[image_id][f"{k}nn_logits"]
    analysis.write(json.dumps(
        {"image_id": image_id,
        "caption_tokens": query_sample_ids_dict[image_id]["caption_tokens"],
        "caption_token_head_vocabs": query_sample_ids_dict[image_id]["caption_token_head_vocabs"],
        }) + "\n"
    )
    analysis.flush()
    analysis.close()