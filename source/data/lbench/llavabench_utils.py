import os
import json
from datetime import datetime

def load_data_for_llavabench(llavabench_path):
    question_file = os.path.join(llavabench_path, "questions.jsonl")
    caption_file = os.path.join(llavabench_path, "context.jsonl")
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    captions = [json.loads(q) for q in open(os.path.expanduser(caption_file), "r")]
    anno_dict = {}
    for anno in captions:
        if anno["id"] not in anno_dict.keys():
            anno_dict[anno["id"]] = anno
    out = []
    for qs in questions:
        image_id = qs["image"].split(".")[0]
        caption_gt = anno_dict[image_id]["caption"]
        file_name = llavabench_path + f'images/{qs["image"]}'
        out.append({
            "image_id": image_id,
            "question_id": qs["question_id"],
            "image": qs["image"],
            "text": qs["text"],
            "gt_ans": caption_gt,
            "file_name": file_name
        })
    return out
    