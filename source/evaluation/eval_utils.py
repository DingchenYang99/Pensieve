from datetime import datetime

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def find_good_nns(image_id_nns, image_id_q, knn):
    image_id_nns_keep = []
    for k in range(len(image_id_nns)):
        if image_id_nns[k] in image_id_nns_keep:
            continue
        if image_id_nns[k] == image_id_q:
            continue
        image_id_nns_keep.append(image_id_nns[k])
        if len(image_id_nns_keep) == knn:
            break
    return image_id_nns_keep