import torch


def avg_non_zero_reducer(losses):
    threshold_condition = losses > 0.
    num_past_filter = torch.sum(threshold_condition)
    if num_past_filter >= 1:
        loss = torch.mean(losses[threshold_condition])
    else:
        loss = torch.sum(losses * 0)
    return loss
