# pylint: disable=line-too-long
from typing import Tuple, Any, Dict
import torch
from torch import nn

from .symmetry import Symmetry


class EquivariantCondDist(nn.Module):
    """Conditional equivariant distribution g ~ p(g|x)"""
    def __init__(self, symmetry: Symmetry):
        super().__init__()
        self.symmetry = symmetry

    def forward(self, xs: Tuple, batch: Any, n_samples: int, loss_dict: Dict) -> Tuple[torch.Tensor, Dict]:
        """Sample group elements g ~ p(g|x)"""
        raise NotImplementedError


class Uniform(EquivariantCondDist):
    """g ~ Unif(G)"""

    def forward(self, xs: Tuple, batch: Any, n_samples: int, loss_dict: Dict) -> Tuple[torch.Tensor, Dict]:
        xs, _ = self.symmetry.broadcast(xs, n_samples)
        gs = self.symmetry.samples_from_haar_distribution(xs, batch)
        return gs, loss_dict


class Frame(EquivariantCondDist):
    """g ~ Unif(F(x)) for predefined set-valued equivariant frame F"""

    def forward(self, xs: Tuple, batch: Any, n_samples: int, loss_dict: Dict) -> Tuple[torch.Tensor, Dict]:
        gs = self.symmetry.samples_from_frame(xs, batch, n_samples)
        return gs, loss_dict


class Probabilistic(EquivariantCondDist):
    """g ~ p(g|x) with trainable equivariant distribution p(g|x)"""
    def __init__(self, symmetry: Symmetry):
        super().__init__(symmetry)
        self.prob_interface_net = self.symmetry.interface()

    def forward(self, xs: Tuple, batch: Any, n_samples: int, loss_dict: Dict) -> Tuple[torch.Tensor, Dict]:
        gs, loss_dict = self.symmetry.samples_from_prob(self.prob_interface_net, xs, batch, n_samples, loss_dict)
        return gs, loss_dict
