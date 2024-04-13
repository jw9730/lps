# pylint: disable=protected-access,too-many-locals,unused-argument,line-too-long,too-many-instance-attributes,too-many-arguments,not-callable
from typing import Tuple, Dict
from functools import partial
import torch
from torch import nn
from torch.nn.functional import one_hot
from torch_geometric.data import Data
from torchmetrics.functional import mean_absolute_error
from sklearn.metrics import r2_score

from src.symmetry import Symmetry
from src.symmetry.groups.S import samples_from_haar_distribution
from src.symmetry.frame.S import samples_from_frame
from src.symmetry.prob.S import EquivariantInterface


T = torch.Tensor

NODE_NUM_CLASSES = [17, 3, 7, 7, 5, 1, 6, 2, 2]
EDGE_NUM_CLASSES = [4, 1, 2]


class GraphRegressionPeptidesstruct(Symmetry):
    def __init__(self, config):
        super().__init__()
        self.group_name = 'S'
        self.rep_dim = 444
        self.rep_in = {1: sum(NODE_NUM_CLASSES), 2: 1 + sum(EDGE_NUM_CLASSES)}
        self.rep_out = {0: 11}
        self.metric_name = ['mae', 'r2']
        if config.interface == 'prob':
            self.entropy_loss_scale = config.entropy_loss_scale
            self.interface = partial(
                EquivariantInterface,
                noise_scale=config.noise_scale,
                tau=config.tau,
                hard=config.hard,
                rep_dim=self.rep_dim,
                node_rep_channels=self.rep_in[1],
                interface_num_layers=config.interface_num_layers,
                interface_hidden_dim=config.interface_hidden_dim,
                interface_dropout=config.interface_dropout
            )

    def process_input(self, batch: Data) -> Tuple[T, T]:
        device = batch.x.device
        batch_size = batch.num_graphs
        node_ptr = batch._slice_dict['x'].to(device)
        edge_ptr = batch._slice_dict['edge_index'].to(device)
        num_nodes = node_ptr[1:] - node_ptr[:-1]
        num_edges = edge_ptr[1:] - edge_ptr[:-1]
        n = self.rep_dim
        # convert node and edge classes to one-hot vectors
        node_attr_one_hot = torch.cat([one_hot(x, num_classes=NODE_NUM_CLASSES[idx]) for idx, x in enumerate(batch.x.unbind(-1))], dim=-1).float()
        edge_attr_one_hot = torch.cat([one_hot(x, num_classes=EDGE_NUM_CLASSES[idx]) for idx, x in enumerate(batch.edge_attr.unbind(-1))], dim=-1).float()
        # parse node features (B, N, C)
        tri = torch.tril(torch.ones(n, n, device=device, dtype=torch.bool))
        node_mask = tri[num_nodes - 1]
        node_features = torch.zeros(batch_size, n, self.rep_in[1], device=device, dtype=torch.float)
        node_features[node_mask] = node_attr_one_hot
        # parse edge features (B, N, N, C)
        edge_features = torch.zeros(batch_size, n, n, self.rep_in[2], device=device, dtype=torch.float)
        edge_batch_index = torch.arange(batch_size, device=device).repeat_interleave(num_edges)
        edge_index_offset = node_ptr[:-1].repeat_interleave(num_edges)
        edge_index = batch.edge_index - edge_index_offset[None, :]
        edge_features[edge_batch_index, edge_index[0], edge_index[1], 0] = 1.
        edge_features[edge_batch_index, edge_index[0], edge_index[1], 1:] = edge_attr_one_hot
        # return features
        xs = (node_features, edge_features)
        return xs

    def process_output(self, xs: Tuple[T], batch: Data) -> torch.Tensor:
        graph_output = xs[0]
        return graph_output

    def samples_from_haar_distribution(self, xs: Tuple[T, T], batch: Data) -> torch.Tensor:
        n, bsize, device, dtype = self.rep_dim, xs[0].size(0), xs[0].device, xs[0].dtype
        node_ptr = batch._slice_dict['x'].to(device)
        num_nodes = node_ptr[1:] - node_ptr[:-1]
        return samples_from_haar_distribution(num_nodes, n, bsize, device, dtype)

    def samples_from_frame(self, xs: Tuple[T, T], batch: Data, n_samples: int) -> torch.Tensor:
        n = self.rep_dim
        gs = samples_from_frame(n, xs, batch, n_samples)
        return gs

    def samples_from_prob(self, prob_interface_net: nn.Module, xs: Tuple[T, T], batch: Data, n_samples: int, loss_dict: Dict) -> Tuple[torch.Tensor, Dict]:
        gs, entropy_loss = prob_interface_net(xs, batch, n_samples)
        loss_dict['entropy'] = {'weight': self.entropy_loss_scale, 'value': entropy_loss}
        return gs, loss_dict

    def calculate_entropy_prob(self, prob_entropy_net: nn.Module, prob_interface_net: nn.Module, xs: Tuple[T, T], batch):
        entropy_loss = prob_entropy_net(prob_interface_net, xs, batch)
        return entropy_loss

    def transform_input(self, xs: Tuple[T, T], gs: torch.Tensor) -> Tuple[T, T]:
        node_features, edge_features = xs
        # leverage orthogonality of permutation matrices
        # gs_inv = torch.linalg.inv(gs)
        gs_inv = gs.transpose(1, 2)
        node_features = gs_inv @ node_features
        edge_features = torch.einsum('bij,bjkd,blk->bild', gs_inv, edge_features, gs_inv)
        xs = (node_features, edge_features)
        return xs

    def transform_output(self, xs: Tuple[T], gs: torch.Tensor) -> Tuple[T]:
        graph_output = xs[0]
        xs = (graph_output,)
        return xs

    def criterion(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Implementation from LRGB:
        https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/loss/l1.py
        """
        return nn.L1Loss()(y_hat, y)

    def evaluator(self, y_hat: list, y: list) -> dict:
        """Implementation from LRGB:
        https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/logger.py
        """
        y_hat, y = torch.cat(y_hat, dim=0), torch.cat(y, dim=0)
        batch_size = y.size(0)
        mae_val = mean_absolute_error(y, y_hat)
        # caution: torchmetric and sklearn r2_score give different values
        # official LRGB implementation uses sklearn
        r2_score_val = r2_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy(), multioutput='uniform_average')
        return {
            'mae': {
                'metric_sum': mae_val * batch_size,
                'metric_count': batch_size
            },
            'r2': {
                'metric_sum': r2_score_val * batch_size,
                'metric_count': batch_size
            }
        }
