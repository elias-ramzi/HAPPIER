import torch


def mask_logsumexp(tens, mask):
    tens[~mask] = -float('inf')
    return torch.logsumexp(tens, dim=1)
