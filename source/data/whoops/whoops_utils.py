import os
import json
import pandas as pd

def load_data_for_whoops(whoops_path):
    
    json_keys = ['crowd_captions', 
                 'crowd_explanations', 
                 'crowd_underspecified_captions', 
                 'question_answering_pairs']
    image_suffix = 'png'
    examples_csv = whoops_path + 'whoops_dataset.csv'
    images_dir = whoops_path + 'whoops_images'
    
    df = pd.read_csv(examples_csv)
    for c in json_keys:
        df[c] = df[c].apply(json.loads)
    df.drop(columns=['image_url'],inplace=True)
    
    output = []
    for r_idx, r in df.iterrows():
        r_dict = r.to_dict()
        image_path = os.path.join(images_dir, f"{r_dict['image_id']}.{image_suffix}")
        r_dict['file_name'] = image_path
        output.append(r_dict)
        
    return output

def load_blip_results_for_whoops(whoops_path):
    
    json_keys = ['crowd_captions', 
                 'blip_vqa', 
                 'blip_matching', 
                 'blip_captioning']

    examples_csv = whoops_path + 'whoops_dataset_for_evaluation.csv'
    
    df = pd.read_csv(examples_csv)
    for c in json_keys:
        df[c] = df[c].apply(json.loads)
    
    output = []
    for r_idx, r in df.iterrows():
        r_dict = r.to_dict()
        output.append(r_dict)
        
    return output