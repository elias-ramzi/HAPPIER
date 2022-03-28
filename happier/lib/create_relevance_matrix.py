import torch


def create_relevance_matrix(label_matrix, relevance):
    return torch.gather(relevance, 1, label_matrix)
