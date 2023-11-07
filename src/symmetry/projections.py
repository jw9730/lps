from typing import Tuple, List
import torch
from torch import nn

from .symmetry import Symmetry, is_order_zero_rep


def _sum(k):
    return k if isinstance(k, int) else sum(k)

class InputProj(nn.Module):
    proj: nn.ModuleList
    proj_keys: List

    def __init__(self, symmetry: Symmetry):
        super().__init__()
        self.symmetry = symmetry

    def forward(self, xs: Tuple) -> torch.Tensor:
        """Projection from input representations to (N, L, D) tensor"""
        assert [_sum(k) for k in self.proj_keys] == [x.ndim - 2 for x in xs]
        return torch.stack([proj(x) for proj, x in zip(self.proj, xs)]).sum(0)


class OutputProj(nn.Module):
    proj: nn.ModuleList
    proj_keys: List

    def __init__(self, symmetry: Symmetry):
        super().__init__()
        self.symmetry = symmetry

    def forward(self, x_unpooled: torch.Tensor, x_pooled: torch.Tensor) -> Tuple:
        """Projection from (N, L, D) tensor to output representations"""
        return tuple(proj(x_pooled if is_order_zero_rep(k) else x_unpooled) for proj, k in
                     zip(self.proj, self.proj_keys))
