# pylint: disable=protected-access,too-many-locals,unused-argument,line-too-long,too-many-instance-attributes,too-many-arguments
from typing import Tuple, Dict
from functools import partial
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
from torch_geometric.data import Data

from src.symmetry import Symmetry
from src.symmetry.groups.S import samples_from_haar_distribution
from src.symmetry.frame.S import samples_from_frame
from src.symmetry.prob.S import EquivariantInterface


T = torch.Tensor

NODE_NUM_FEAT = 3


class NodeClassificationPATTERN(Symmetry):
    def __init__(self, config):
        super().__init__()
        self.group_name = 'S'
        self.rep_dim = 188
        self.rep_in = {1: NODE_NUM_FEAT, 2: 1 + NODE_NUM_FEAT}
        self.rep_out = {2: 2}
        self.metric_name = ['accuracy', 'trivial_accuracy']
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
        device, dtype = batch.x.device, batch.x.dtype
        batch_size = batch.num_graphs
        node_ptr = batch._slice_dict['x'].to(device)
        edge_ptr = batch._slice_dict['edge_index'].to(device)
        num_nodes = node_ptr[1:] - node_ptr[:-1]
        num_edges = edge_ptr[1:] - edge_ptr[:-1]
        n = self.rep_dim
        # parse node features (B, N, C)
        tri = torch.tril(torch.ones(n, n, device=device, dtype=torch.bool))
        node_mask = tri[num_nodes - 1]
        node_features = torch.zeros(batch_size, n, self.rep_in[1], device=device, dtype=dtype)
        node_features[node_mask] = batch.x
        # parse edge features (B, N, N, C)
        edge_features = torch.zeros(batch_size, n, n, self.rep_in[2], device=device, dtype=dtype)
        edge_batch_index = torch.arange(batch_size, device=device).repeat_interleave(num_edges)
        edge_index_offset = node_ptr[:-1].repeat_interleave(num_edges)
        edge_index = batch.edge_index - edge_index_offset[None, :]
        edge_features[edge_batch_index, edge_index[0], edge_index[1], 0] = 1.
        # place node features on the diagonals of the edge features
        node_indices = torch.arange(n, device=device)
        edge_features[:, node_indices, node_indices, -NODE_NUM_FEAT:] = node_features
        # return features
        # only edge features are used in the backbone
        # node features are only used for the interface
        xs = (node_features, edge_features)
        return xs

    def process_output(self, xs: Tuple[T], batch: Data) -> torch.Tensor:
        edge_output = xs[0]
        # (B, N, N, C) -> (B, N, C) -> (sum(Ni), C)
        device = batch.x.device
        node_ptr = batch._slice_dict['x'].to(device)
        num_nodes = node_ptr[1:] - node_ptr[:-1]
        n = self.rep_dim
        tri = torch.tril(torch.ones(n, n, device=device, dtype=torch.bool))
        node_mask = tri[num_nodes - 1]
        node_output = edge_output.diagonal(dim1=1, dim2=2).transpose(1, 2)
        node_output = node_output[node_mask]
        return node_output

    def samples_from_haar_distribution(self, xs: Tuple[T, T], batch: Data) -> torch.Tensor:
        n, bsize, device, dtype = self.rep_dim, xs[0].size(0), xs[0].device, xs[0].dtype
        node_ptr = batch._slice_dict['x'].to(device)
        num_nodes = node_ptr[1:] - node_ptr[:-1]
        return samples_from_haar_distribution(num_nodes, n, bsize, device, dtype)

    def samples_from_frame(self, xs: Tuple[T, T], batch: Data, n_samples: int):
        n = self.rep_dim
        gs = samples_from_frame(n, xs, batch, n_samples)
        return gs

    def samples_from_prob(self, prob_interface_net: nn.Module, xs: Tuple[T, T], batch: Data, n_samples: int, loss_dict: Dict) -> Tuple[torch.Tensor, Dict]:
        gs, entropy_loss = prob_interface_net(xs, batch, n_samples)
        loss_dict['entropy'] = {'weight': self.entropy_loss_scale, 'value': entropy_loss}
        return gs, loss_dict

    def transform_input(self, xs: Tuple[T, T], gs: torch.Tensor) -> Tuple[T, T]:
        node_features, edge_features = xs
        # leverage orthogonality of permutation matrices
        # gs_inv = torch.linalg.inv(gs)
        gs_inv = gs.transpose(1, 2)
        node_features = gs_inv @ node_features
        edge_features = torch.einsum('bij,bjkd,blk->bild', gs_inv, edge_features, gs_inv)
        # remove node features as they are already encoded in the edge features
        node_features = node_features.fill_(0)
        xs = (node_features, edge_features)
        return xs

    def transform_output(self, xs: Tuple[T], gs: torch.Tensor) -> Tuple[T]:
        edge_output = xs[0]
        edge_output = torch.einsum('bij,bjkd,blk->bild', gs, edge_output, gs)
        xs = (edge_output,)
        return xs

    # def criterion(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    #     return nn.functional.cross_entropy(y_hat, y)

    def criterion(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Multiclass weighted cross-entropy for imbalanced classification.
        Implementation from LRGB:
        https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/loss/weighted_cross_entropy.py
        """
        y = y.long()
        # calculating label weights for weighted loss computation
        V = y.size(0)
        n_classes = y_hat.size(-1)
        torch.use_deterministic_algorithms(False)
        label_count = torch.bincount(y)
        torch.use_deterministic_algorithms(True)
        label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
        cluster_sizes = torch.zeros(n_classes, device=y_hat.device, dtype=torch.long)
        cluster_sizes[torch.unique(y)] = label_count
        weight = (V - cluster_sizes).float() / max(V, 1e-5)
        weight *= (cluster_sizes > 0).float()
        # weighted cross-entropy
        return nn.CrossEntropyLoss(weight=weight)(y_hat, y)

    def evaluator(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Implementation from Benchmarking GNNs:
        https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/train/metrics.py"""
        batch_size = y.size(0)
        scores, targets = y_hat, y
        S = targets.cpu().numpy()
        C = np.argmax(torch.nn.Softmax(dim=1)(scores.float()).cpu().detach().numpy(), axis=1)
        CM = confusion_matrix(S, C).astype(np.float32)
        nb_classes = CM.shape[0]
        targets = targets.cpu().detach().numpy()
        nb_non_empty_classes = 0
        pr_classes = np.zeros(nb_classes)
        for r in range(nb_classes):
            cluster = np.where(targets==r)[0]
            if cluster.shape[0] != 0:
                pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
                if CM[r,r]>0:
                    nb_non_empty_classes += 1
            else:
                pr_classes[r] = 0.0
        acc = 100.* np.sum(pr_classes)/ float(nb_classes)
        return {
            'accuracy':{
                'metric_sum': acc * batch_size,
                'metric_count': batch_size,
            },
            'trivial_accuracy': {
                'metric_sum': (y_hat.argmax(dim=-1) == y).float().sum(),
                'metric_count': batch_size,
            }
        }
