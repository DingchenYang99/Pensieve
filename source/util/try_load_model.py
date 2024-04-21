import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from lavis.models import load_model_and_preprocess

# model_name = "llava15"
model_name = "instructblip"

if model_name == "llava15":
    model_path = '/DATA3/yangdingchen/checkpoint/llava-v1.5-7b'
    YES_TOKEN = "\u2581Yes"
    NO_TOKEN = "\u2581No"
    # result_path = "/DATA3/yangdingchen/whoops/results/240110-152936"
elif model_name == "instructblip":
    model_path = "/DATA3/yangdingchen/checkpoint/vicuna-7b-v1.1"
    YES_TOKEN = "\u2581yes"
    NO_TOKEN = "\u2581no"
    # result_path = "/DATA3/yangdingchen/whoops/results/240118-130046"
else:
    raise NotImplementedError
            
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading {model_name} Model's lm_head weight.")

if model_name == "llava15":
    model_base = None
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _, _ = load_pretrained_model(model_path, model_base, model_name)
    lm_head_weight = model.lm_head.weight.data.clone()
elif model_name == "instructblip":
    model, _, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", 
                                                        is_eval=True, device=device)
    lm_head_weight = model.llm_model.lm_head.weight.data.clone()
    tokenizer = model.llm_tokenizer
    
print(f"lm_head weight of size {lm_head_weight.shape} loaded")

yes_ids = tokenizer.convert_tokens_to_ids(YES_TOKEN)
no_ids = tokenizer.convert_tokens_to_ids(NO_TOKEN)
print(yes_ids, no_ids)

yes_token_prototype = lm_head_weight[yes_ids, :].clone()
no_token_prototype = lm_head_weight[no_ids, :].clone()

print(yes_token_prototype.shape, no_token_prototype.shape)

yes_proto_norm = yes_token_prototype.norm(p=2, dim=-1).item()
no_proto_norm = no_token_prototype.norm(p=2, dim=-1).item()
print(yes_proto_norm, no_proto_norm)

all_proto_norm_mean = lm_head_weight.norm(p=2, dim=-1, keepdim=False).mean().item()
print(all_proto_norm_mean)