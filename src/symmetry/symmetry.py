# pylint: disable=line-too-long,too-many-arguments
from typing import Any, Tuple, Dict, Union, Callable, List
import torch
from torch import nn


def is_order_zero_rep(k):
    if isinstance(k, tuple):
        return all(is_order_zero_rep(k_) for k_ in k)
    return k == 0


class Symmetry():
    """Base class for symmetry operations."""
    group_name: Union[str, Tuple] = NotImplemented
    rep_dim: Union[int, Tuple] = NotImplemented
    rep_in: Union[Dict[int, int], Dict[Tuple, int]] = NotImplemented
    rep_out: Union[Dict[int, int], Dict[Tuple, int]] = NotImplemented
    metric_name: Union[str, List[str]] = NotImplemented
    interface: Callable = NotImplemented
    ignore_rep_in: List = []
    ignore_rep_out: List = []

    def __repr__(self):
        if isinstance(self.group_name, str):
            name = f"Group: {self.group_name}({self.rep_dim}), " \
                   f"Rep: {rep2str(self.rep_in)} -> {rep2str(self.rep_out)}"
        else:
            name = f"Group: {'*'.join([f'{g}({d})' for g, d in zip(self.group_name, self.rep_dim)])}, " \
                   f"Rep: {rep2str_prod(self.rep_in)} -> {rep2str_prod(self.rep_out)}"
        return name

    def __init__(self):
        pass

    def process_input(self, batch) -> Tuple:
        """Proprocess input batch to representation (tuple of (B, N ... N, C) tensors)"""
        raise NotImplementedError

    def process_output(self, xs: Tuple, batch) -> Any:
        """Postprocess output representation to target format"""
        raise NotImplementedError

    def samples_from_haar_distribution(self, xs: Tuple, batch) -> torch.Tensor:
        """Sample from Haar measure on the group"""
        raise NotImplementedError

    def samples_from_frame(self, xs: Tuple, batch, n_samples: int) -> torch.Tensor:
        """Sample from frame"""
        raise NotImplementedError

    def samples_from_prob(self, prob_interface_net: nn.Module, xs: Tuple, batch, n_samples: int, loss_dict: Dict) -> Tuple[torch.Tensor, Dict]:
        """Sample from equivariant distribution"""
        raise NotImplementedError

    def transform_input(self, xs: Tuple, gs: torch.Tensor) -> torch.Tensor:
        """Transform using base group representations gs"""
        raise NotImplementedError

    def transform_output(self, xs: Tuple, gs: torch.Tensor) -> torch.Tensor:
        """Transform using (inverse of) base group representations gs"""
        raise NotImplementedError

    def broadcast(self, xs: Tuple, n_samples: int) -> Tuple[Tuple, int]:
        """Broadcast to n_samples copies"""
        batch_size = xs[0].size(0)
        if n_samples == 1:
            return xs, batch_size
        xs = tuple(self._broadcast(x, n_samples) for x in xs)
        return xs, batch_size

    def reduce(self, xs: Tuple, batch_size: int, n_samples: int) -> Tuple:
        """Reduce from n_samples copies"""
        if n_samples == 1:
            return xs
        xs = tuple(self._reduce(x, batch_size, n_samples) for x in xs)
        return xs

    @staticmethod
    def _broadcast(x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Broadcast x to n_samples copies"""
        return x.repeat_interleave(n_samples, dim=0)

    @staticmethod
    def _reduce(x: torch.Tensor, batch_size: int, n_samples: int) -> torch.Tensor:
        """Reduce x from n_samples copies"""
        return x.reshape(batch_size, n_samples, *x.shape[1:]).mean(1)

    def criterion(self, y_hat: Any, y: Any) -> torch.Tensor:
        """Compute loss between prediction y_hat and target y"""
        raise NotImplementedError

    def evaluator(self, y_hat: Any, y: Any) -> torch.Tensor:
        """Compute metric between prediction y_hat and target y"""
        # caution: the metric should be higher the better
        # due to mode='max' in earlystopping and checkpoint callbacks
        raise NotImplementedError


def rep2str(rep: dict):
    def fn(p, c):
        if p == 0:
            return f"{c}"
        if p == 1:
            return f"{c}V"
        if p == 2:
            return f"{c}(V*V)"
        return f"{c}(V**{p})"
    return f'{"+".join([fn(k, v) for k, v in rep.items()])}'


def rep2str_prod(rep: dict):
    def fn(p, s):
        if p == 0:
            return ""
        if p == 1:
            return f"{s}"
        if p == 2:
            return f"({s}*{s})"
        return f"({s}**{p})"
    def fn_(p_tup, c):
        r = "*".join([fn(p_, f"V{i+1}") for i, p_ in enumerate(p_tup)]).strip("*")
        return f'{c}({r})' if len(r) > 0 else f'{c}'
    return f'{"+".join([fn_(k, v) for k, v in rep.items()])}'
