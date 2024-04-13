# pylint: disable=too-many-arguments,too-many-instance-attributes
from typing import Tuple, Any, Dict
from torch import nn

from src.symmetry import Symmetry
from src.symmetry.interface import Uniform, Frame, Probabilistic
from src.symmetry.projections_conv import ConvInputProj, ConvOutputProj
from .backbone import Backbone

DIST = {'unif': Uniform, 'frame': Frame, 'prob': Probabilistic}


class InterfacedModel(nn.Module):
    def __init__(self, backbone: Backbone, symmetry: Symmetry, interface: str, centering: bool, pad_mode: str):
        super().__init__()
        # setup backbone
        self.backbone = backbone
        # setup group and reps
        self.symmetry = symmetry
        self.criterion = self.symmetry.criterion
        self.evaluator = self.symmetry.evaluator
        self.metric_name = self.symmetry.metric_name
        # setup equivariant distribution p(g|x)
        self.distribution = DIST[interface](self.symmetry)
        # setup input and output projections
        assert pad_mode in ['zero', 'reflect', 'replicate', 'circular']
        num_tokens = self.backbone.num_tokens
        hidden_size = self.backbone.hidden_size
        self.input_proj = ConvInputProj(symmetry, num_tokens, hidden_size, centering, pad_mode)
        self.output_proj = ConvOutputProj(symmetry, num_tokens, hidden_size, centering, pad_mode)

    def pretrained_parameters(self):
        return self.backbone.parameters()

    def scratch_parameters(self):
        for module in [self.input_proj, self.output_proj, self.distribution]:
            yield from module.parameters()

    def _forward(self, xs: tuple) -> tuple:
        x = self.input_proj(xs)
        x_unpooled, x_pooled = self.backbone(x)
        xs = self.output_proj(x_unpooled, x_pooled)
        return xs

    def forward(self, batch, n_samples: int=1, transform: bool=True) -> Tuple[Any, Dict]:
        """Forward pass with monte carlo sampling for symmetrization"""
        loss_dict = {}
        if not transform:
            xs = self.symmetry.process_input(batch)
            xs = self._forward(xs)
            x = self.symmetry.process_output(xs, batch)
            return x, loss_dict
        xs = self.symmetry.process_input(batch)
        gs, loss_dict = self.distribution(xs, batch, n_samples, loss_dict)
        xs, batch_size = self.symmetry.broadcast(xs, n_samples)
        xs = self.symmetry.transform_input(xs, gs)
        xs = self._forward(xs)
        xs = self.symmetry.transform_output(xs, gs)
        xs = self.symmetry.reduce(xs, batch_size, n_samples)
        x = self.symmetry.process_output(xs, batch)
        return x, loss_dict
