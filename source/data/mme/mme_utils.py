import os

def load_data_mme(mme_path, interested_task_name):
    sub_task_l = os.listdir(mme_path)
    assert interested_task_name in sub_task_l
    this_task_dir = mme_path + interested_task_name
    sample_names_ = os.listdir(this_task_dir)
    sample_names = []
    for sample_name in sample_names_:
        if os.path.isfile(os.path.join(this_task_dir, sample_name)):
            sample_names.append(sample_name)
    
    if len(sample_names) == 0:
        raise NotImplementedError  # read images and txts separately
    assert len(sample_names) % 2 == 0, 'missing sample'
    
    sample_ids = sorted(list(set([sample_name.split(".")[0] for sample_name in sample_names])))
    print(f"{len(sample_ids)} samples found in subtask {interested_task_name}")
    output_list = []
    for sample_id in sample_ids:
        image_file_name = os.path.join(this_task_dir, sample_id + '.jpg')
        anno_file_name = os.path.join(this_task_dir, sample_id + '.txt')
        lines = open(anno_file_name, 'r').readlines()
        assert len(lines) == 2
        for line in lines:
            if line.endswith("\n"):
                line = line.rstrip("\n")
            question, gt_ans = line.split("\t")
            output_list.append({
                "image_id":sample_id,
                "image":image_file_name,
                "text": question,
                "label":gt_ans
            })
        
    assert len(output_list) == len(sample_names)
    return output_list
