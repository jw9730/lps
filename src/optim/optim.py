# pylint: disable=too-few-public-methods
import torch
from torch import nn

from src.model import InterfacedModel


OPTIMIZERS = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}


class OptimizerConfig():
    def __init__(
            self,
            optimizer: str,
            weight_decay: float,
            lr_pretrained: float,
            lr: float
        ):
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.lr_pretrained = lr_pretrained
        self.lr = lr

    def setup(self, model: InterfacedModel):
        learnable_params = []
        learnable_params.append({'params': model.pretrained_parameters(),'lr': self.lr_pretrained})
        learnable_params.append({'params': model.scratch_parameters(), 'lr': self.lr})

        if self.optimizer == 'sgd':
            return torch.optim.SGD(learnable_params, weight_decay=self.weight_decay, momentum=0.9)
        if self.optimizer == 'adam':
            return torch.optim.Adam(learnable_params, weight_decay=self.weight_decay)
        if self.optimizer == 'adamw':
            return torch.optim.AdamW(learnable_params, weight_decay=self.weight_decay)

        raise ValueError(f'Optimizer {self.optimizer} not supported!')
