import torch.nn as nn


def get_pooling(pool, cfg):
    name = cfg if isinstance(cfg, str) else cfg.name
    kwargs = {} if isinstance(cfg, str) else cfg.kwargs

    if name == 'default':
        return pool
    elif name == 'none':
        return nn.Identity()
    elif name == 'max':
        return nn.AdaptiveMaxPool2d(output_size=(1, 1))
    elif name == 'avg':
        return nn.AdaptiveAvgPool2d(output_size=(1, 1))
    else:
        return getattr(nn, name)(**kwargs)
