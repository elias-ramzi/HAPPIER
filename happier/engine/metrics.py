from functools import partial

import torch
# import numpy as np

import happier.lib as lib


def ap(sorted_rel, at_R=False, reduce=True, **kwargs):
    normalizing_factor = sorted_rel.sum(-1)

    relevance_mask = torch.ones(sorted_rel.shape, device=sorted_rel.device)
    if at_R:
        for i, R in enumerate(normalizing_factor):
            relevance_mask[i, int(R.item()):] = 0

    ranks = torch.arange(1, 1 + sorted_rel.size(1), device=sorted_rel.device)
    pos_ranks = torch.cumsum(sorted_rel, -1) * sorted_rel * relevance_mask

    ap = (pos_ranks / ranks).sum(-1) / normalizing_factor

    if reduce:
        return ap.mean()
    return ap


def map_at_R(sorted_rel, reduce=True, **kwargs):
    """
    Truncated version of the AP:
    considers only the instances ranked among the R greatest
    (R being the number of relevant item for a query).
    The normalization factor is still the total number of relevant
    items.
    """
    return ap(sorted_rel, True, reduce)


def ap_for_hiearchy_level(sorted_target, hierarchy_level, reduce=True, **kwargs):
    """
    This version of AP considers only the positive with the requested level of relevance
    (hierarchy_level). The instances that have a higher / lower (being that the relevance
    is greater than min_treshold) relevance level are discarded (they play no role in the metric).
    """
    # kept positives
    pertinent = (sorted_target == hierarchy_level)
    # kept positives and negatives
    relevant = sorted_target <= hierarchy_level

    normalizing_factor = pertinent.sum(-1).float()
    keep = normalizing_factor > 0
    pertinent = pertinent[keep]
    relevant = relevant[keep]
    normalizing_factor = normalizing_factor[keep]

    ranks = (torch.cumsum(relevant, -1) * relevant).float()
    pos_ranks = (torch.cumsum(pertinent, -1) * pertinent).float()

    ap = pos_ranks / ranks
    ap[torch.isnan(ap)] = 0
    ap = ap.sum(-1) / normalizing_factor

    if reduce:
        return lib.safe_mean(ap)
    return ap


def precision_at_k(sorted_target, at_k, reduce=True, **kwargs):
    p_at_k = sorted_target[:, :at_k].mean(-1)

    if reduce:
        return p_at_k.mean()
    return p_at_k


def precision_at_1(sorted_target, reduce=True, **kwargs):
    return precision_at_k(sorted_target, 1, reduce)


def recall_rate_at_k(sorted_target, at_k, reduce=True, **kwargs):
    r_at_k = sorted_target[:, :at_k].any(1).float()

    if reduce:
        return r_at_k.mean()
    return r_at_k


def dcg(sorted_rel, reduce=True, **kwargs):
    ranks = torch.arange(1, sorted_rel.size(1) + 1, device=sorted_rel.device)

    dcg = (sorted_rel / torch.log2(1 + ranks)).sum(-1)

    if reduce:
        return dcg.mean()
    return dcg


def idcg(sorted_rel, reduce=True, **kwargs):
    ideal_ranks = 1 + sorted_rel.argsort(1, True).argsort(1).float()

    idcg = (sorted_rel / torch.log2(1 + ideal_ranks)).sum(-1)

    if reduce:
        return idcg.mean()
    return idcg


def ndcg(sorted_target, at_k=None, reduce=True, **kwargs):

    sorted_target = 2 ** sorted_target.float() - 1
    if at_k is not None:
        sorted_target = sorted_target[:, :at_k]

    ranks = torch.arange(1, sorted_target.size(1) + 1, device=sorted_target.device)
    ideal_ranks = 1 + sorted_target.argsort(1, True).argsort(1).float()

    dcg = (sorted_target / torch.log2(1 + ranks)).sum(-1)
    idcg = (sorted_target / torch.log2(1 + ideal_ranks)).sum(-1)

    ndcg = dcg / idcg

    if reduce:
        return ndcg.mean()
    return ndcg


def asi(sorted_target, at_R=True, reduce=True, **kwargs):
    real_sorted_target = sorted_target.sort(1, descending=True).values
    size = sorted_target.max()

    values = torch.arange(size+1, device=sorted_target.device).view(1, -1).unsqueeze(1).permute(0, 2, 1)
    eq = (sorted_target.unsqueeze(1) == values).cumsum(-1)
    real_eq = (real_sorted_target.unsqueeze(1) == values).cumsum(-1)

    at_k = 1 / torch.arange(1, sorted_target.size(-1) + 1, device=sorted_target.device).view(1, -1)
    norm_factor = sorted_target.size(-1)
    if at_R:
        at_k = at_k * real_sorted_target.bool()
        norm_factor = real_sorted_target.bool().sum(-1)
    # in the parantheses is SI for each depth
    asi = (torch.min(eq, real_eq).sum(1) * at_k).sum(-1) / norm_factor

    if reduce:
        return asi.mean()
    return asi


def hap(sorted_rel, relevances, reduce=True, **kwargs):
    device = sorted_rel.device
    relevances = relevances[:, 1:]

    eq = sorted_rel.unsqueeze(1) == relevances.unsqueeze(1).permute(0, 2, 1)
    cum = eq.cumsum(-1)

    weights = torch.minimum(
        sorted_rel.unsqueeze(1),
        relevances.unsqueeze(1).permute(0, 2, 1),
    )

    ranks = torch.arange(1, sorted_rel.size(1) + 1, device=device).view(1, -1)
    hranks = (weights * cum).sum(1)
    mhap = (hranks / ranks).sum(-1) / sorted_rel.sum(-1)

    if reduce:
        return mhap.mean()
    return mhap


def hap_at_R(sorted_rel, reduce=True, **kwargs):
    device = sorted_rel.device

    count_at_R = (sorted_rel == 1.).sum(-1)
    max_at_R = count_at_R.max()
    sorted_rel = sorted_rel[:, :max_at_R]

    relevance_mask = torch.ones(sorted_rel.shape, device=device)
    for i, R in enumerate(count_at_R):
        relevance_mask[i, int(R.item()):] = 0

    ranks = torch.arange(1, 1 + sorted_rel.size(1), device=device)
    min_rel = torch.minimum(
        sorted_rel.unsqueeze(1),
        sorted_rel.unsqueeze(1).permute(0, 2, 1),
    )
    # the mask indicates the instances that are before the considered instance
    mask = torch.ones_like(min_rel, device=device, dtype=torch.bool).tril()
    hranks = (min_rel * mask).sum(-1)

    hap_at_R = ((hranks / ranks) * relevance_mask).sum(-1) / count_at_R

    if reduce:
        return hap_at_R.mean().item()
    return hap_at_R


METRICS_DICT = {
    "binary": {
        "AP": ap,
        "P@1": precision_at_1,
        "mAP@R": map_at_R,
    },
    "multi_level": {
        "NDCG": ndcg,
        "H-AP": hap,
        "ASI": asi,
    },
    "exclude_level": {},
}


def get_metrics_dict(
    recall_rate=[],
    hard_ap_for_level=[],
    with_binary_asi=False,
):
    metrics_dict = METRICS_DICT.copy()

    for r in recall_rate:
        metrics_dict["binary"][f"R@{r}"] = partial(recall_rate_at_k, at_k=r)

    for level in hard_ap_for_level:
        metrics_dict["exclude_level"][f"AP#{level}"] = partial(ap_for_hiearchy_level, hierarchy_level=level)

    if with_binary_asi:
        metrics_dict["binary"]["ASI"] = asi

    return metrics_dict
