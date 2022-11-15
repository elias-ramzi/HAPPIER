from functools import partial

import torch
import torch.nn as nn
from tqdm import tqdm

import happier.lib as lib
from happier.losses.tools import reduce


def heaviside(tens, val=1., target=None, general=None):
    return torch.heaviside(tens, values=torch.tensor(val, device=tens.device, dtype=tens.dtype))


def tau_sigmoid(tensor, tau, target=None, general=False):
    exponent = -tensor / tau
    # clamp the input tensor for stability
    exponent = 1. + exponent.clamp(-50, 50).exp()
    return (1.0 / exponent).type(tensor.dtype)


def linear(tens, a, b):
    tens.mul_(a)
    tens.add_(b)


def step_rank(tens, tau, rho, offset, delta=None, start=0.5, target=None, general=False):
    device = tens.device
    dtype = tens.dtype

    if general:
        inferior_relevance = target.view(1, -1) < target[target.bool()].view(-1, 1)
    else:
        inferior_relevance = target.unsqueeze(target.ndim) > target.unsqueeze(target.ndim-1)

    if rho == -1:
        tens[inferior_relevance] = tau_sigmoid(tens[inferior_relevance], tau).type(dtype)
    else:
        pos_mask = (tens > 0).bool()
        if delta is None:
            tens[inferior_relevance & pos_mask] = rho * tens[inferior_relevance & pos_mask] + offset
        else:
            if offset is None:
                offset = tau_sigmoid(torch.tensor([delta], device=device), tau).type(dtype) + start

            margin_mask = tens > delta
            tens[inferior_relevance & pos_mask & ~margin_mask] = start + tau_sigmoid(tens[inferior_relevance & pos_mask & ~margin_mask], tau).type(dtype)
            tens[inferior_relevance & pos_mask & margin_mask] = rho * (tens[inferior_relevance & pos_mask & margin_mask] - delta) + offset

        tens[inferior_relevance & (~pos_mask)] = tau_sigmoid(tens[inferior_relevance & (~pos_mask)], tau).type(dtype)

    tens[~inferior_relevance] = heaviside(tens[~inferior_relevance])
    return tens


def step_hrank(tens, target, beta=None, leak=None, gamma=None, general=False):
    # dtype = tens.dtypes

    # inferior and equal relevance
    if general:
        inferior_relevance = target.view(1, -1) <= target[target.bool()].view(-1, 1)
    else:
        inferior_relevance = target.unsqueeze(target.ndim) >= target.unsqueeze(target.ndim-1)

    tens[inferior_relevance] = heaviside(tens[inferior_relevance])

    # superior relevance
    superior_relevance = ~inferior_relevance
    pos_mask = tens > 0

    tens[superior_relevance & ~pos_mask] *= leak
    if gamma > 1.0:
        tens[superior_relevance & pos_mask] = heaviside(tens[superior_relevance & pos_mask])
    else:
        tens[superior_relevance & pos_mask] = torch.clamp(beta * tens[superior_relevance & pos_mask] + gamma, max=1.)
    return tens


class SmoothRankHAP(nn.Module):

    def __init__(
        self,
        rank_approximation,
        hrank_approximation,
        hierarchy_level="MULTI",
        return_type='1-mHAP',
        reduce_type='mean',
        stop_grad=False,
    ):
        super().__init__()
        self.rank_approximation = rank_approximation
        self.hrank_approximation = hrank_approximation
        self.hierarchy_level = hierarchy_level
        self.return_type = return_type
        self.reduce_type = reduce_type
        self.stop_grad = stop_grad
        assert return_type in ["1-mHAP", "1-HAP", "HAP", 'mHAP']
        assert isinstance(self.stop_grad, bool)

        if self.hierarchy_level != 'MULTI':
            assert isinstance(hierarchy_level, int)
            lib.LOGGER.warning(f"Hierarchy_level (={self.hierarchy_level}) was passed for a HAP surrogate loss")

    # @profile
    def general_forward(self, embeddings, labels, ref_embeddings, ref_labels, relevance_fn, verbose=False):
        batch_size = embeddings.size(0)
        device = embeddings.device

        mhap_score = []

        iterator = range(batch_size)
        if verbose:
            iterator = tqdm(iterator, leave=None)

        for idx in iterator:
            _score = torch.mm(embeddings[idx].view(1, -1), ref_embeddings.t())[0]
            pos_mask = lib.create_label_matrix(
                labels[idx].view(1, -1), ref_labels,
                hierarchy_level=self.hierarchy_level,
                dtype=torch.long,
            )

            # shape M x M
            relevances = relevance_fn(pos_mask.view(1, -1)).type(_score.dtype)[0] / (labels.size(1) - 1)
            min_relevances = torch.where(
                relevances.view(1, -1) < relevances[pos_mask.bool()].view(-1, 1),
                relevances.view(1, -1),
                relevances[pos_mask.bool()].view(-1, 1),
            )

            query = _score.view(1, -1) - _score[pos_mask.bool()].view(-1, 1)
            rank = self.rank_approximation(torch.clone(query), target=pos_mask, general=True)
            hrank = self.hrank_approximation(torch.clone(query), target=pos_mask, general=True)
            mask = torch.ones_like(query, device=device, dtype=torch.bool)
            cond = torch.where(pos_mask)[0]
            mask[(torch.arange(len(cond)), cond)] = 0
            rank *= mask
            hrank *= mask

            # compute the approximated rank
            rank = 1. + torch.sum(rank, dim=-1)

            # compute the approximated Hrank
            hrank = relevances[pos_mask.bool()] + torch.sum(hrank * min_relevances, dim=-1)

            mhap = (hrank / rank).sum(-1) / relevances.sum(-1)
            mhap_score.append(mhap)
            del rank, hrank, mhap, query, cond, mask, relevances, min_relevances, _score, pos_mask
            torch.cuda.empty_cache()

        # shape N
        mhap_score = torch.stack(mhap_score)
        return mhap_score

    def quick_forward(self, scores, target, relevances):
        batch_size = target.size(0)
        device = scores.device

        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores difference of scores between an instance and itself
        mask = 1.0 - torch.eye(batch_size, device=device).unsqueeze(0)
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        # compute the difference matrix
        sim_diff = scores.unsqueeze(1) - scores.unsqueeze(1).permute(0, 2, 1)
        min_relevances = torch.where(
            relevances.unsqueeze(1) < relevances.unsqueeze(1).permute(0, 2, 1),
            relevances.unsqueeze(1),
            relevances.unsqueeze(1).permute(0, 2, 1),
        )

        # the relevance is bounded by the maximum (which is one)
        # clone is necessary as rank_approximation is modifying the tensor inplace
        rank = self.rank_approximation(torch.clone(sim_diff), target=target) * mask
        rank = 1. + torch.sum(rank, dim=-1)

        # compute the approximated Hrank
        hrank = self.hrank_approximation(torch.clone(sim_diff), target=target) * mask
        hrank = relevances + torch.sum(hrank * min_relevances, dim=-1)

        # approximation of Hap
        mhap = (hrank / rank).sum(-1) / relevances.sum(-1)
        return mhap

    def forward(
        self,
        embeddings,
        labels,
        relevance_fn,
        ref_embeddings=None,
        ref_labels=None,
        force_general=False,
        verbose=False,
        **kwargs,
    ):
        if self.hierarchy_level != "MULTI":
            labels = labels[:, self.hierarchy_level:self.hierarchy_level+1]

        if ref_labels is None:
            assert ref_embeddings is None
            ref_embeddings = embeddings
            ref_labels = labels
        else:
            assert embeddings.size(1) == ref_embeddings.size(1)
            assert labels.size(1) == ref_labels.size(1)
            if self.hierarchy_level != "MULTI":
                ref_labels = ref_labels[:, self.hierarchy_level:self.hierarchy_level+1]

        if (embeddings.size(0) == ref_embeddings.size(0)) and not force_general:
            if self.stop_grad:
                scores = torch.mm(embeddings, ref_embeddings.detach().t())
            else:
                scores = torch.mm(embeddings, ref_embeddings.t())
            target = lib.create_label_matrix(labels, ref_labels, dtype=torch.int64)
            relevances = relevance_fn(target).type(scores.dtype)  # / (labels.size(1) - 1)
            mhap = self.quick_forward(scores, target, relevances)
        else:
            mhap = self.general_forward(embeddings, labels, ref_embeddings, ref_labels, relevance_fn, verbose)

        if self.return_type == 'HAP':
            return mhap
        elif self.return_type == 'mHAP':
            return mhap.mean()
        elif self.return_type == '1-HAP':
            return 1 - mhap
        elif self.return_type == '1-mHAP':
            return reduce(1 - mhap, self.reduce_type)

    @property
    def my_repr(self,):
        repr = f"    return_type={self.return_type},\n"
        return repr


class HAPLoss(SmoothRankHAP):

    def __init__(self, tau=0.01, rho=100., offset=1.44, delta=0.05, start=0.5, beta=None, leak=None, gamma=None, with_hrank=False, **kwargs):
        self.tau = tau
        self.rho = rho
        self.offset = offset
        self.delta = delta
        self.start = start
        self.beta = beta
        self.leak = leak
        self.gamma = gamma
        self.with_hrank = with_hrank

        rank_approximation = partial(step_rank, tau=tau, rho=rho, offset=offset, delta=delta, start=start)
        hrank_approximation = partial(step_hrank, beta=beta, leak=leak, gamma=gamma) if with_hrank else heaviside

        super().__init__(
            rank_approximation=rank_approximation,
            hrank_approximation=hrank_approximation,
            **kwargs,
        )

    def extra_repr(self,):
        repr = (
            f"    tau={self.tau},\n"
            f"    rho={self.rho},\n"
            f"    offset={self.offset},\n"
            f"    delta={self.delta},\n"
            f"    start={self.start},\n"
            f"    with_hrank={self.with_hrank},\n"
            f"    beta={self.beta},\n"
            f"    gamma={self.gamma},\n"
            f"    leak={self.leak},\n"
        )
        repr = repr + self.my_repr
        return repr
