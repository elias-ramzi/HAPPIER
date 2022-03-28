import torch

from .avg_non_zero_reducer import avg_non_zero_reducer


__all__ = [
    'avg_non_zero_reducer',
]


REDUCE_DICT = {
    'none': torch.nn.Identity(),
    'mean': torch.mean,
    'sum': torch.sum,
    'avg_non_zero': avg_non_zero_reducer,
}


def reduce(tens, reduce_type='mean'):
    return REDUCE_DICT[reduce_type](tens)
