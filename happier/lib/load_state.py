import torch

from happier.lib.expand_path import expand_path


def load_state(path, key=None):
    state = torch.load(expand_path(path), map_location='cpu')

    if key is not None:
        return state[key]

    return state


def load_config(path):
    return load_state(path, 'config')
