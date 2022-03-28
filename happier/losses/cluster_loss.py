import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import happier.lib as lib


# adapted from :
# https://github.com/azgo14/classification_metric_learning/blob/master/metric_learning/modules/losses.py
class ClusterLoss(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    """
    def __init__(
        self,
        embedding_size,
        num_classes,
        num_centers=1,
        temperature=0.05,
        temperature_centers=0.1,
        hierarchy_level=None,
        multi_label=False,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.num_centers = num_centers
        self.temperature = temperature
        self.temperature_centers = temperature_centers
        self.hierarchy_level = hierarchy_level

        self.weight = nn.Parameter(torch.Tensor(num_classes * num_centers, embedding_size))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.loss_fn = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()

    def forward(self, embeddings, instance_targets, relevance_fn=None, **kwargs,):
        if self.hierarchy_level is not None:
            instance_targets = instance_targets[:, self.hierarchy_level]

        norm_weight = nn.functional.normalize(self.weight, dim=1)

        prediction_logits = nn.functional.linear(embeddings, norm_weight)

        if self.num_centers > 1:
            prediction_logits = prediction_logits.reshape(embeddings.size(0), self.num_classes, self.num_centers)
            prob = F.softmax(prediction_logits / self.temperature_centers, dim=2)
            prediction_logits = (prediction_logits * prob).sum(dim=2)

        loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)
        return loss

    def register_optimizers(self, opt, sch):
        self.opt = opt
        self.sch = sch
        lib.LOGGER.info(f"Optimizer registered for {self.__class__.__name__}")

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

    def load_state_dict(self, state_dict, override=False, *args, **kwargs):
        super().load_state_dict(state_dict["super"], *args, **kwargs)
        if not override:
            self.opt.load_state_dict(state_dict["opt"])
            if self.sch["on_step"]:
                self.sch["on_step"].load_state_dict(state_dict["sch_on_step"])
            if self.sch["on_epoch"]:
                self.sch["on_epoch"].load_state_dict(state_dict["sch_on_epoch"])
            if self.sch["on_val"]:
                self.sch["on_val"].load_state_dict(state_dict["sch_on_val"])

    def __repr__(self,):
        repr = f"{self.__class__.__name__}(\n"
        repr = repr + f"    temperature={self.temperature},\n"
        repr = repr + f"    num_classes={self.num_classes},\n"
        repr = repr + f"    embedding_size={self.embedding_size},\n"
        repr = repr + f"    opt={self.opt.__class__.__name__},\n"
        repr = repr + f"    hierarchy_level={self.hierarchy_level},\n"
        repr = repr + ")"
        return repr
