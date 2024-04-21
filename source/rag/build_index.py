import faiss
import json
import os
import numpy as np
from PIL import Image
import torch
# import clip
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import CLIPFeatureExtractor, CLIPVisionModel, CLIPModel, CLIPTokenizer
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from tqdm import tqdm
from pathlib import Path
from data.mme.mme_utils import load_data_mme

def get_pope_images_names(pope_data_path):

    file_adversarial_name_coco = pope_data_path + '/coco/coco_pope_adversarial.json'
    file_adversarial_name_aokvqa = pope_data_path + '/aokvqa/aokvqa_pope_adversarial.json'
    file_adversarial_coco = os.path.expanduser(file_adversarial_name_coco)
    file_adversarial_aokvqa = os.path.expanduser(file_adversarial_name_aokvqa)

    file_popular_name_coco = pope_data_path + 'coco/coco_pope_popular.json'
    file_popular_name_aokvqa = pope_data_path + 'aokvqa/aokvqa_pope_popular.json'
    file_popular_coco = os.path.expanduser(file_popular_name_coco)
    file_popular_aokvqa = os.path.expanduser(file_popular_name_aokvqa)
    
    file_random_name_coco = pope_data_path + 'coco/coco_pope_random.json'
    file_random_name_aokvqa = pope_data_path + 'aokvqa/aokvqa_pope_random.json'
    file_random_coco = os.path.expanduser(file_random_name_coco)
    file_random_aokvqa = os.path.expanduser(file_random_name_aokvqa)

    adversarial_coco = [json.loads(q) for q in open(file_adversarial_coco, "r")]
    popular_coco = [json.loads(q) for q in open(file_popular_coco, "r")]
    random_coco = [json.loads(q) for q in open(file_random_coco, "r")]
    
    adversarial_aokvqa = [json.loads(q) for q in open(file_adversarial_aokvqa, "r")]
    popular_aokvqa = [json.loads(q) for q in open(file_popular_aokvqa, "r")]
    random_aokvqa = [json.loads(q) for q in open(file_random_aokvqa, "r")]

    all_file = adversarial_coco + popular_coco + random_coco + adversarial_aokvqa + popular_aokvqa + random_aokvqa
        
    image_id_list = []
    
    for line in all_file:
        image_file = line["image"]
        image_file = image_file.split("_")[-1]
        image_file = image_file.rstrip(".jpg")
        image_id_list.append(image_file)
        
    image_id_set = sorted(list(set(image_id_list)), reverse=True)
    print(f"{len(image_id_set)} images used for pope evaluation, they should be excluded from the index")
    return image_id_set

def get_mme_images_names(mme_data_path):

    # only the following four subtasks use image from COCO Caption
    # we do not include them in the reference database
    interested_task_names = [
            "existence", 
            "count", 
            "position",
            "color"
            ]

    image_id_list = []
    for task_name in interested_task_names:
        questions = load_data_mme(mme_data_path, task_name)
        
        for line in questions:
            mme_image_id = line['image_id']
            if mme_image_id.endswith(".jpg"):
                mme_image_id = mme_image_id.rstrip(".jpg")
            image_id_list.append(mme_image_id)
        
    image_id_set = sorted(list(set(image_id_list)), reverse=True)
    print(f"{len(image_id_set)} images used for mme evaluation, they should be excluded from the index")
    return image_id_set


def load_clip_vision_model(model_name, model_path):
    assert 'clip' in model_name
    image_processor = CLIPImageProcessor.from_pretrained(model_path + model_name)
    vision_tower = CLIPVisionModel.from_pretrained(model_path + model_name)
    return image_processor, vision_tower

def load_dino_vision_model(model_name, model_path):
    assert 'dinov2' in model_name
    # config = AutoConfig.from_pretrained(model_path + model_name)
    # image_processor = AutoImageProcessor.from_pretrained(model_path + model_name)
    # vision_tower = AutoModel.from_pretrained(model_path + model_name, config=config)
    vision_tower = torch.hub.load(repo_or_dir=model_path, model=model_name,
                                  source='local')
    return vision_tower

def load_clip_text_model(model_name, model_path):
    assert 'clip' in model_name
    tokenizer = CLIPTokenizer.from_pretrained(model_path + model_name)
    clip_model = CLIPModel.from_pretrained(model_path + model_name)
    
    return clip_model, tokenizer

def load_coco_train_restval_images(coco_data_path):
    annotations = json.load(open(coco_data_path))['images']
    images = []
    # captions = []
    for item in annotations:
        if item['split'] == 'restval':
            item['split'] = 'train'
        if item['split'] == 'train':
            images.append({'image_id': item['cocoid'], 'file_name': item['filename'].split('_')[-1]})
 
    return images

def load_coco_train_restval_caps(coco_data_path):
    annotations = json.load(open(coco_data_path))['images']
    captions = []
    for item in annotations:
        if item['split'] == 'restval':
            item['split'] = 'train'
        if item['split'] == 'train':
            # this_captions = []
            for sentence in item['sentences']:
                this_caption = ' '.join(sentence['tokens'])
                # this_captions.append(this_caption)
                captions.append({'image_id': item['cocoid'],  'caption': this_caption, 
                                 'file_name': item['filename'].split('_')[-1]})
 
    return captions

def filter_img_samples(images, idx_dataset_name):
    assert idx_dataset_name in ['coco']
    image_ids = [d['image_id'] for d in images]
    file_names = [d['file_name'] for d in images]
    # for whoops, we do not exclude samples in MME and POPE from the database
    
    filtered_image_ids, filtered_filenames = [], []
    assert len(image_ids) == len(file_names)
    for image_id, file_name in zip(image_ids, file_names):
        filtered_image_ids.append(image_id)
        filtered_filenames.append(file_name)

    return filtered_image_ids, filtered_filenames

def filter_cap_samples(captions, idx_dataset_name, 
                       pope_data_path, mme_data_path,):
    assert idx_dataset_name == 'coco'
    image_ids = [d['image_id'] for d in captions]
    caps = [d['caption'] for d in captions]
    file_names = [d['file_name'] for d in captions]

    test_image_ids_pope = get_pope_images_names(pope_data_path)
    test_image_ids_mme = get_mme_images_names(mme_data_path)
    test_image_ids = test_image_ids_pope + test_image_ids_mme
    filtered_image_ids, filtered_captions, filtered_file_names = [], [], []
    assert len(image_ids) == len(caps) == len(file_names)
    for image_id, cap, file_name in zip(image_ids, caps, file_names):
        if file_name.endswith(".jpg"):
            file_name = file_name.rstrip('.jpg')
        if file_name in test_image_ids:
            # manual filtering out test images in pope and mme
            continue  
        filtered_image_ids.append(image_id)
        filtered_captions.append(cap)
        filtered_file_names.append(file_name)

    return filtered_image_ids, filtered_captions, filtered_file_names

def encode_images(file_names, image_path, 
                  clip_model, clip_processor,
                  dino_model,
                  device):
    num_images = len(file_names)
    print(f'{num_images} images to be processed')
    bs = 128
    if bs > num_images:
        bs = num_images
    clip_image_feats_l, dino_image_feats_l = [], []
    for idx in tqdm(range(0, num_images, bs)):

        clip_image_input = [clip_processor.preprocess(Image.open(
            os.path.join(image_path, i)), return_tensors='pt')['pixel_values'][0] \
                for i in file_names[idx:min(idx+bs, num_images)]]
        
        with torch.no_grad():
            # get clip features
            concat_clip_images = torch.cat([image.unsqueeze(0).half().to(device) for image in clip_image_input], dim=0)
            clip_output = clip_model(concat_clip_images, output_hidden_states=True)
            clip_hidden_states = clip_output.hidden_states[-2]  # follow clip vision_tower
            clip_hidden_states = clip_hidden_states[:, :1]  # keep [cls] only
            clip_image_feats_l.append(clip_hidden_states.squeeze(1).cpu().numpy())
            
            # get dino features
            concat_dino_images = torch.cat([image.unsqueeze(0).half().to(device) for image in clip_image_input], dim=0)
            dino_output = dino_model.forward_features(concat_dino_images)
            dino_hidden_states = dino_output["x_prenorm"]  # follow dino vision_tower
            dino_hidden_states = dino_hidden_states[:, :1]  # keep [cls] only
            dino_image_feats_l.append(dino_hidden_states.squeeze(1).cpu().numpy())

    clip_image_feats = np.concatenate(clip_image_feats_l)
    dino_image_feats = np.concatenate(dino_image_feats_l)

    return clip_image_feats, dino_image_feats

def encode_captions(captions, clip_model, clip_tokenizer, device):
    num_caps = len(captions)
    print(f'{num_caps} captions to be processed')
    bs = 256
    if bs > num_caps:
        bs = num_caps
    encoded_captions = []
    for idx in tqdm(range(0, len(captions), bs)):
        with torch.no_grad():
            text_ids = clip_tokenizer(captions[idx:min(idx+bs, num_caps)], 
                                      padding=True, return_tensors='pt').to(device)
            text_encodings = clip_model.get_text_features(input_ids=text_ids['input_ids'])
            encoded_captions.append(text_encodings.cpu().numpy())
            
    encoded_captions = np.concatenate(encoded_captions, axis=0)

    return encoded_captions

def build_img_index(feats_clip, feats_dino):
    xb_clip = feats_clip.astype(np.float32)
    xb_dino = feats_dino.astype(np.float32)
    faiss.normalize_L2(xb_dino)
    faiss.normalize_L2(xb_clip)
    index = faiss.IndexFlatIP(xb_dino.shape[1] + xb_clip.shape[1])
    xb = np.concatenate([xb_clip, xb_dino], axis=-1).astype(np.float32)
    index.add(xb)
    return index

def build_txt_index(feats_clip):
    xb_clip = feats_clip.astype(np.float32)
    faiss.normalize_L2(xb_clip)
    index = faiss.IndexFlatIP(xb_clip.shape[1])
    index.add(xb_clip)
    return index

def build_coco_img_idx(clip_model_name, dino_model_name): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO set your data and save path
    coco_data_path = '/home/lufan/Projects/smallcap/caption/annotations/dataset_coco.json'
    coco_image_path = '/DATA3/yangdingchen/coco/images/'
    coco_index_path = '/DATA3/yangdingchen/datastore/coco_img/' + 'clip_vit_l14_dino_vit_l14/'
    clip_model_path = '/home/lufan/Projects/smallcap/pretrained/'
    dino_model_path = '/home/lufan/.cache/torch/hub/facebookresearch/dinov2/'
    Path(coco_index_path).mkdir(parents=True, exist_ok=True)
    
    print('Loading data')
    images = load_coco_train_restval_images(coco_data_path)
    
    print('Loading CLIP encoder')
    clip_image_processor, clip_vision_tower = load_clip_vision_model(clip_model_name, clip_model_path)
    clip_vision_tower.requires_grad_(False)
    clip_vision_tower = clip_vision_tower.to(device=device, dtype=torch.float16)
    
    print('Loading DINO encoder')
    dino_vision_tower = load_dino_vision_model(dino_model_name, dino_model_path)
    dino_vision_tower.requires_grad_(False)
    dino_vision_tower = dino_vision_tower.to(device=device, dtype=torch.float16)
    
    print('Filtering COCO samples')    
    xb_image_ids, xb_file_names = filter_img_samples(images, idx_dataset_name='coco')
    
    print('Encoding images')
    clip_image_feats, dino_image_feats = encode_images(xb_file_names, coco_image_path, 
                                                        clip_vision_tower, clip_image_processor,
                                                        dino_vision_tower,
                                                        device)
    
    print('Building coco image index')
    coco_index = build_img_index(clip_image_feats, dino_image_feats)
    
    print('Writing files')
    faiss.write_index(coco_index, coco_index_path + "coco_image_index_clip_dino")
    json.dump(xb_file_names, open(os.path.join(coco_index_path, 'coco_image_index_file_names.json'), 'w'))
    json.dump(xb_image_ids, open(os.path.join(coco_index_path, 'coco_image_index_imageids.json'), 'w'))
    
    print('finished')

def build_coco_cap_idx(clip_model_name, dino_model_name=None): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO set your data and save path
    pope_data_path = '/home/lufan/Projects/VCD/experiments/data/POPE/'
    mme_data_path = "/DATA3/yangdingchen/mme/"
    coco_data_path = '/home/lufan/Projects/smallcap/caption/annotations/dataset_coco.json'
    coco_index_path = '/DATA3/yangdingchen/datastore/coco_cap/' + 'clip_vit_l14/'
    clip_txt_model_path = '/home/lufan/Projects/smallcap/pretrained/'
    Path(coco_index_path).mkdir(parents=True, exist_ok=True)
    
    print('Loading data')
    caps = load_coco_train_restval_caps(coco_data_path)
    
    print('Loading CLIP encoder')
    clip_model, clip_tokenizer = load_clip_text_model(clip_model_name, clip_txt_model_path)
    clip_model.requires_grad_(False)
    clip_model = clip_model.to(device=device, dtype=torch.float16)
    
    print('Filtering COCO samples')    
    xb_image_ids, xb_caps, xb_file_names = filter_cap_samples(caps, idx_dataset_name='coco',
                                                              mme_data_path=mme_data_path, 
                                                              pope_data_path=pope_data_path)
    
    print('Encoding captions')
    clip_cap_feats = encode_captions(xb_caps, clip_model, clip_tokenizer, device)
    print('Building coco image index')
    coco_index = build_txt_index(clip_cap_feats)
    
    print('Writing files')
    faiss.write_index(coco_index, coco_index_path + "coco_caps_index_clip")
    json.dump(xb_caps, open(os.path.join(coco_index_path, 'coco_caps_index_captions.json'), 'w'))
    json.dump(xb_file_names, open(coco_index_path + 'coco_caps_index_file_names.json', 'w'))
    json.dump(xb_image_ids, open(os.path.join(coco_index_path, 'coco_caps_index_imageids.json'), 'w'))
    
    print('finished')
        
if __name__ == '__main__':
    
    # clip_model_name = 'openai/clip-vit-base-patch32'
    clip_model_name = 'openai/clip-vit-large-patch14'
    # dino_model_name = 'dinov2_vitb14'
    dino_model_name = 'dinov2_vitl14'
    
    dataset = 'coco'
    modality = 'img'
    if dataset == 'coco' and modality == 'img':
        build_coco_img_idx(clip_model_name, dino_model_name)
    elif dataset == 'coco' and modality == 'txt':
        build_coco_cap_idx(clip_model_name)
    else:
        raise NotImplementedError
    