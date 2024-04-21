import torch.nn.functional as F

def calculate_jsd(base_logits, ref_logits):
    assert base_logits.shape == ref_logits.shape
    bs, d = base_logits.shape
    assert bs == 1, 'currently, jensen shannon divergence threshold support bs=1 only'
    base_logits_smx = F.softmax(base_logits, dim=-1)
    ref_logits_smx = F.softmax(ref_logits, dim=-1)
    M = 0.5 * (base_logits_smx + ref_logits_smx)
    base_logits_logsmx = F.log_softmax(base_logits, dim=-1)
    ref_logits_logsmx = F.log_softmax(ref_logits, dim=-1)
    kl1 = F.kl_div(base_logits_logsmx, M, reduction='none').mean(-1)
    kl2 = F.kl_div(ref_logits_logsmx, M, reduction='none').mean(-1)
    js_divs = 0.5 * (kl1 + kl2)
    return js_divs