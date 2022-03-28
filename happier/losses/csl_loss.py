import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import happier.lib as lib


class CSLLoss(nn.Module):

    def __init__(
        self,
        num_proxies,
        margins=[0.25, 0.35, 0.45],
        scale=32.0,
        embedding_size=512,
        reduce_type='sum',
        proxies_seed=0,
        hierarchy_level="MULTI",
    ):
        super().__init__()
        self.num_proxies = num_proxies
        self.margins = margins
        self.scale = scale
        self.embedding_size = embedding_size
        self.reduce_type = reduce_type
        self.proxies_seed = proxies_seed
        self.hierarchy_level = hierarchy_level

        self.init_proxies()

        if self.hierarchy_level != 'MULTI':
            assert isinstance(hierarchy_level, int)
            lib.LOGGER.warning(f"Hierarchy_level (={self.hierarchy_level}) was passed for a HAP surrogate loss")

    @lib.get_set_random_state
    def init_proxies(self,):
        # Init as ProxyNCA++
        lib.random_seed(self.proxies_seed, backend=False)
        self.weight = nn.Parameter(torch.Tensor(self.num_proxies, self.embedding_size))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, embeddings, labels, **kwargs):
        P = F.normalize(self.weight)
        similarities = embeddings @ P.t()

        losses = []
        for lvl in range(labels.size(1)):
            index = labels[:, lvl].unsqueeze(1) == self.labels[:, lvl].unsqueeze(0)
            LSE_neg = lib.mask_logsumexp(similarities * self.scale, ~index)
            if lvl == 0:
                LSE_pos_high = lib.mask_logsumexp(- similarities * self.scale, index)
            lss = F.softplus(LSE_neg + LSE_pos_high + self.margins[lvl] * self.scale).mean()
            losses.append(lss)

        if self.reduce_type == 'sum':
            loss = sum(losses)
        elif self.reduce_type == 'mean':
            loss = sum(losses) / len(losses)
        elif self.reduce_type == 'none':
            loss = losses
        else:
            raise ValueError

        return loss

    def register_optimizers(self, opt, sch):
        self.opt = opt
        self.sch = sch
        lib.LOGGER.info(f"Optimizer registered for {self.__class__.__name__}")

    def register_labels(self, labels):
        assert len(self.margins) == labels.shape[1]
        assert self.num_proxies == len(labels[:, 0].unique())
        self.labels = nn.Parameter(labels.unique(dim=0), requires_grad=False)
        lib.LOGGER.info(f"Labels registered for {self.__class__.__name__}")

    def update(self, scaler=None):
        if scaler is None:
            self.opt.step()
        else:
            scaler.step(self.opt)

        if self.sch["on_step"]:
            self.sch["on_step"].step()

    def on_epoch(self,):
        if self.sch["on_epoch"]:
            self.sch["on_epoch"].step()

    def on_val(self, val):
        if self.sch["on_val"]:
            self.sch["on_val"].step(val)

    def state_dict(self, *args, **kwargs):
        state = {"super": super().state_dict(*args, **kwargs)}
        state["opt"] = self.opt.state_dict()
        state["sch_on_step"] = self.sch["on_step"].state_dict() if self.sch["on_step"] else None
        state["sch_on_epoch"] = self.sch["on_epoch"].state_dict() if self.sch["on_epoch"] else None
        state["sch_on_val"] = self.sch["on_val"].state_dict() if self.sch["on_val"] else None
        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        super().load_state_dict(state_dict["super"], *args, **kwargs)
        self.opt.load_state_dict(state_dict["opt"])
        if self.sch["on_step"]:
            self.sch["on_step"].load_state_dict(state_dict["sch_on_step"])
        if self.sch["on_epoch"]:
            self.sch["on_epoch"].load_state_dict(state_dict["sch_on_epoch"])
        if self.sch["on_val"]:
            self.sch["on_val"].load_state_dict(state_dict["sch_on_val"])

    def extra_repr(self,):
        repr = ''
        repr = repr + f"    num_proxies={self.num_proxies},\n"
        repr = repr + f"    embedding_size={self.embedding_size},\n"
        repr = repr + f"    margins={self.margins},\n"
        repr = repr + f"    scale={self.scale},\n"
        repr = repr + f"    reduce_type={self.reduce_type},\n"
        repr = repr + f"    proxies_seed={self.proxies_seed},\n"
        return repr
