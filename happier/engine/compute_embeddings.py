import os

import torch
from tqdm import tqdm

import happier.lib as lib


def compute_embeddings(
    net,
    loader,
    convert_to_cuda=False,
    with_paths=False,
):
    features = []

    mode = net.training
    net.eval()
    lib.LOGGER.info("Computing embeddings")
    for i, batch in enumerate(tqdm(loader, disable=os.getenv("TQDM_DISABLE"))):
        with torch.no_grad():
            X = net(batch["image"].cuda())

        features.append(X)

    features = torch.cat(features)
    labels = torch.from_numpy(loader.dataset.labels).to('cuda' if convert_to_cuda else 'cpu')
    if loader.dataset.relevances is not None:
        relevances = loader.dataset.relevances.to('cuda' if convert_to_cuda else 'cpu')
    else:
        relevances = None

    net.train(mode)
    if with_paths:
        return features, labels, relevances, loader.dataset.paths
    else:
        return features, labels, relevances
