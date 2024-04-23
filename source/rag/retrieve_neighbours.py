import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import faiss
import torch
import numpy as np
import json
import random
from rag.build_index import encode_images, encode_captions, \
    load_clip_vision_model, load_dino_vision_model, load_clip_text_model
from data.mme.mme_utils import load_data_mme
from data.whoops.whoops_utils import load_data_for_whoops
from data.lbench.llavabench_utils import load_data_for_llavabench
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

def retrieve_nns(index, query, k):
    if isinstance(query, list) or isinstance(query, tuple):
        assert len(query) == 2
        feats_clip = query[0]
        feats_dino = query[1]
        xq_clip = feats_clip.astype(np.float32)
        xq_dino = feats_dino.astype(np.float32)
        faiss.normalize_L2(xq_clip)
        faiss.normalize_L2(xq_dino)
        xq = np.concatenate([xq_clip, xq_dino], axis=-1).astype(np.float32)
    else:
        xq = query.astype(np.float32)
        faiss.normalize_L2(xq)
    D, I = index.search(xq, k) 
    
    return D, I

def retrieve_nn_imgs(
    dataset_q,
    query_file_path,
    sub_folder,
    index_dataset,
    read_path,
    save_path,
    retriever,
    k,
    ):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    read_file_path = read_path + f'{index_dataset}_img/{retriever}/'
    write_file = save_path + f'retrieved_{index_dataset}_imgs_{retriever}_{k}nns_{sub_folder}.json'
    
    if index_dataset == 'coco':
        index_name = 'coco'
    else:
        raise NotImplementedError
    file_name_path = read_file_path + f"{index_name}_image_index_file_names.json"
    
    if retriever == 'clip_vit_b32_dino_vit_b14':
        clip_model_name = 'openai/clip-vit-base-patch32'
        dino_model_name = 'dinov2_vitb14'
    elif retriever == 'clip_vit_l14_dino_vit_l14':
        clip_model_name = 'openai/clip-vit-large-patch14'
        dino_model_name = 'dinov2_vitl14'
    else:
        raise NotImplementedError
    
    print('Loading test data')
    data_dir = query_file_path + sub_folder

    if dataset_q == 'whoops':
        test_data = load_data_for_whoops(query_file_path)[:30]
        img_suffix = '.png'
    elif dataset_q == 'llavabench':
        test_data = load_data_for_llavabench(query_file_path)
        img_suffix = '.jpg'
    else:
        raise NotImplementedError

    print('Loading index')
    index = faiss.read_index(read_file_path + f"{index_name}_image_index_clip_dino")
    ## use faiss-gpu
    # print('moving index to gpu')
    # res = faiss.StandardGpuResources()
    # index = faiss.index_cpu_to_gpu(res, 0, index)
    
    print('Loading neighbour captions and file-names from coco caption dataset')
    xb_image_paths = json.load(open(file_name_path, 'r'))
    xb_image_paths_len = len(xb_image_paths)
    print(f"total {xb_image_paths_len} neighbor candidates")
    
    print('Loading CLIP encoder')
    #TODO set model path
    clip_model_path = '/home/lufan/Projects/smallcap/pretrained/'
    clip_image_processor, clip_vision_tower = load_clip_vision_model(clip_model_name, clip_model_path)
    clip_vision_tower.requires_grad_(False)
    clip_vision_tower = clip_vision_tower.to(device=device, dtype=torch.float16)
    
    print('Loading DINO encoder')
    #TODO set model path
    dino_model_path = '/home/lufan/.cache/torch/hub/facebookresearch/dinov2/'
    dino_vision_tower = load_dino_vision_model(dino_model_name, dino_model_path)
    dino_vision_tower.requires_grad_(False)
    dino_vision_tower = dino_vision_tower.to(device=device, dtype=torch.float16)
    
    xq_image_ids = list(set([d['image_id'] for d in test_data]))
    print(f'Encoding {len(xq_image_ids)} images')
    xq_file_names = [d+img_suffix for d in xq_image_ids]
    clip_image_feats, dino_image_feats = encode_images(xq_file_names, data_dir,
                                                        clip_vision_tower, clip_image_processor,
                                                        dino_vision_tower,
                                                        device)
    print('Retrieving neighbors')
    distances, nns = retrieve_nns(index, (clip_image_feats, dino_image_feats), k)
    retrieved_image_ids = {}
    for nns_list, dists_list, image_id, file_name in zip(nns, distances, xq_image_ids, xq_file_names):
        
        assert len(nns_list) == len(dists_list)
        good_nns = {"image_id": image_id,
                    "file_name": file_name,
                    "nnimg_file_names": [xb_image_paths[nn_id] for nn_id in nns_list], 
                    'nnimg_IPs': dists_list.tolist()
                    }
        assert len(good_nns["nnimg_file_names"]) == k
        retrieved_image_ids[image_id] = good_nns

    print('Writing files')
    json.dump(retrieved_image_ids, open(write_file, 'w'))
    
    print(f"saved to {write_file}")

def retrieve_nn_caps(
    dataset_q,
    query_file_path,
    sub_folder,
    index_dataset,
    read_path,
    save_path,
    retriever,
    k,
    bleu_rerank=False
    ):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    read_file_path = read_path + f'{index_dataset}_cap/{retriever}/'
    write_file = save_path + f'retrieved_{index_dataset}_caps_{retriever}_{k}nns_{sub_folder}.json'
    
    if index_dataset == 'coco':
        index_name = 'coco'
    else:
        raise NotImplementedError
    
    file_name_path = read_file_path + f"{index_name}_caps_index_file_names.json"
    captions_path = read_file_path + f"{index_name}_caps_index_captions.json"
    
    if retriever == 'clip_vit_b32':
        clip_model_name = 'openai/clip-vit-base-patch32'
    elif retriever == 'clip_vit_l14':
        clip_model_name = 'openai/clip-vit-large-patch14'
    else:
        raise NotImplementedError
    
    print('Loading test data')
    data_dir = query_file_path + sub_folder
    if dataset_q == 'mme':
        test_data = load_data_mme(query_file_path, sub_folder)
    elif 'pope' in dataset_q:
        dataset_name = sub_folder.split("_")[0]
        question_type = sub_folder.split("_")[1]
        question_file = query_file_path + f"{dataset_name}/{dataset_name}_pope_{question_type}.json"
        test_data = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    else:
        raise NotImplementedError
    
    if 'pope' in dataset_q:
        xq_image_ids = [d['image'].split("_")[-1].split(".")[0] for d in test_data]
    else:
        xq_image_ids = [d['image_id'] for d in test_data]
    
    # for mme, we transform question into narratives with predefined templates
    # color task
    xq_captions = [d['text'].split("in the image")[0].replace("Is there", 'A photo of') \
        for d in test_data]
    # existence, count task
    # xq_captions = ["A photo of "+d['text'].split(" in")[0].split('there ')[1] for d in test_data]
    # position task
    # xq_captions = ["A photo of"+d['text'].split(
        # " Please")[0].lstrip('Is').lstrip("Are").rstrip("\uff1f").rstrip("?") for d in test_data]
    # posters task
    # xq_captions = [d['text'].split("?")[0].replace("Is this", 'A poster of the') \
    #     for d in test_data]
    # celebrity task
    # xq_captions = ["A photo of a celebrity" + d['text'].split("?")[0].split("box")[1] \
    #     for d in test_data]
    # scene task
    # xq_captions = ["A" + d['text'].split("?")[0].split("this")[1].replace("a place of", "a") \
    #     for d in test_data]
    # landmark task
    # xq_captions = [d['text'].split("?")[0].lstrip("Is this ") \
        # for d in test_data]
    # artwork task
    # xq_captions = ["A" + d['text'].split("?")[0].split(" this")[1] \
    #     for d in test_data]
    # ocr task
    # xq_captions = ["A" + d['text'].split("?")[0].split("in the")[1].replace("\"", "") \
    #     for d in test_data]
    
    # for pope, we also transform question into narratives
    # xq_captions = [d['text'].split("in the image")[0].replace("Is there", "A photo of") for d in test_data]
    # print(test_data)
    
    print('Loading index')
    index = faiss.read_index(read_file_path + f"{index_name}_caps_index_clip")
    ## print('moving index to gpu')
    # res = faiss.StandardGpuResources()
    # index = faiss.index_cpu_to_gpu(res, 0, index)
    
    print('Loading neighbour captions and file names from coco caption dataset')
    xb_image_paths = json.load(open(file_name_path, 'r'))
    xb_captions = json.load(open(captions_path, 'r'))
    
    print('Loading CLIP encoder')
    clip_model, clip_tokenizer = load_clip_text_model(clip_model_name)
    clip_model.requires_grad_(False)
    clip_model = clip_model.to(device=device, dtype=torch.float16)
    
    print(f'Encoding {len(xq_captions)} captions')
    clip_cap_feats = encode_captions(xq_captions, clip_model, clip_tokenizer, device)
    print('Retrieving neighbors')
    distances, nns = retrieve_nns(index, clip_cap_feats, k*5)  # one image has 5 annoted caps
    retrieved_image_ids = {}
    
    if 'mme' in dataset_q:
        qs_id = 0
    elif 'pope' in dataset_q:
        qs_id = 1
    
    if bleu_rerank:
        print("reranking by BLEU score")
            
    for nns_list, dists_list, image_id, question in zip(nns, distances, xq_image_ids, xq_captions):
        assert len(nns_list) == len(dists_list)
        nnimg_file_names = [xb_image_paths[nn_id] for nn_id in nns_list]
        nnimg_captions = [xb_captions[nn_id] for nn_id in nns_list]
        if bleu_rerank:
            metric = "Bleu_1"
            _, score_list = calculate_bleu_score(qs_id, question, nnimg_captions, metric)
            score_tensor = torch.tensor(score_list)
            sorted_score_indices = torch.argsort(score_tensor, descending=True).tolist()
            nnimg_captions_reranked = [nnimg_captions[i] for i in sorted_score_indices]
            nnimg_file_names_reranked = [nnimg_file_names[i] for i in sorted_score_indices]
            dists_list_reranked = [float(dists_list[i]) for i in sorted_score_indices]
            score_list_reranked = [float(score_list[i]) for i in sorted_score_indices]
            
            good_nns = {"image_id": image_id,
                        "question": question,
                        f"{metric}_score_list": score_list_reranked,
                        "nnimg_file_names": nnimg_file_names_reranked, 
                        "nnimg_captions": nnimg_captions_reranked, 
                        'nnimg_IPs': dists_list_reranked
                        }
        else:
            good_nns = {"image_id": image_id,
                        "question": question,
                        "nnimg_file_names": nnimg_file_names, 
                        "nnimg_captions": nnimg_captions, 
                        'nnimg_IPs': dists_list.tolist()
                        }

        assert len(good_nns["nnimg_file_names"]) == k*5
        retrieved_image_ids[str(qs_id)] = good_nns
        qs_id += 1
    print(f"retrieval finished for {len(retrieved_image_ids.keys())} questions")
    print('Writing files')
    json.dump(retrieved_image_ids, open(write_file, 'w'))
    print(f"saved to {write_file}")
    
def calculate_bleu_score(qs_id, question, nnimg_captions, metric):
    assert isinstance(question, str)
    assert isinstance(nnimg_captions, list) or isinstance(nnimg_captions, tuple)
    method_list = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
    assert metric in method_list
    nnimg_captions_rerank_ids, score_list = [], []
    gts, res = {}, {}
    for i, nnimg_caption in enumerate(nnimg_captions):
        res[str(qs_id)+"_"+str(i)] = [{'caption':nnimg_caption}]
        gts[str(qs_id)+"_"+str(i)] = [{"caption":question}]
        
    bleu_tokenizer = PTBTokenizer()
    gts = bleu_tokenizer.tokenize(gts)
    res = bleu_tokenizer.tokenize(res)
    scorers = [
        (Bleu(4), method_list),
        ]
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res, verbose=0)
        if type(method) == list:
            for scs, m in zip(scores, method):
                if not m == metric:
                    continue
                for qs_nn_id, score in zip(sorted(gts.keys()), scs):
                    nnimg_captions_rerank_ids.append(int(qs_nn_id.split("_")[-1]))
                    score_list.append(score*100)
        else:
            raise NotImplementedError
        
    return nnimg_captions_rerank_ids, score_list


if __name__ == '__main__':
    
    dataset_q = 'whoops'
    bleu_rerank = False  # implemented for mme only
    # TODO set your test data path
    if dataset_q == 'mme':
        query_file_path = '/DATA3/yangdingchen/mme/MME_Benchmark_release_version/'
        interested_tasks = [
            "color", 
            # "count", 
            # "existence", 
            # "position", 
            # "posters",
            # "celebrity",
            # "scene",
            # "landmark",
            # "artwork",
            # "OCR",
        ]
    elif dataset_q == 'pope_coco':
        query_file_path = '/home/lufan/Projects/VCD/experiments/data/POPE/'
        interested_tasks = [
            'coco_adversarial',
            'coco_popular',
            'coco_random',
        ]
    elif dataset_q == 'pope_aokvqa':
        query_file_path = '/home/lufan/Projects/VCD/experiments/data/POPE/'
        interested_tasks = [
            'aokvqa_adversarial',
            'aokvqa_popular',
            'aokvqa_random',
        ]
    elif dataset_q == 'pope_gqa':
        query_file_path = '/home/lufan/Projects/VCD/experiments/data/POPE/'
        interested_tasks = [
            'gqa_adversarial',
            'gqa_popular',
            'gqa_random',
        ]
    elif dataset_q == 'whoops':
        query_file_path = '/DATA3/yangdingchen/whoops/'
        interested_tasks = [
            'whoops_images'
        ]
    elif dataset_q == 'llavabench':
        query_file_path = '/DATA3/yangdingchen/llava-bench/'
        interested_tasks = [
            'images'
        ]
    else:
        raise NotImplementedError
    
    #TODO set your read and write path
    read_path = '/DATA3/yangdingchen/datastore/'
    index_dataset = 'coco'
    save_path = '/home/lufan/Projects/Pensieve/source/rag/q_nn_files/'
    
    if dataset_q in ['whoops','llavabench']:
        retriever = 'clip_vit_l14_dino_vit_l14'
        for interested_task in interested_tasks:
            retrieve_nn_imgs(
                dataset_q,
                query_file_path,
                interested_task,
                index_dataset,
                read_path,
                save_path,
                retriever,
                k=32)
            
    elif dataset_q in ['pope_coco','mme']:
        retriever = 'clip_vit_l14' 
        for interested_task in interested_tasks:
            retrieve_nn_caps(
                dataset_q,
                query_file_path,
                interested_task,
                index_dataset,
                read_path,
                save_path,
                retriever,
                k=4,
                bleu_rerank=bleu_rerank)
