# pylint: disable=protected-access,too-many-locals,unused-argument,line-too-long,too-many-instance-attributes,too-many-arguments,not-callable
from typing import Tuple, Dict
from functools import partial
import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot
from torch_geometric.data import Data

from src.symmetry import Symmetry
from src.symmetry.groups.S import samples_from_haar_distribution
from src.symmetry.frame.S import samples_from_frame
from src.symmetry.prob.S import EquivariantInterface


T = torch.Tensor

NODE_NUM_CLASSES = [35,  3,  7,  7,  2,  4,  6,  2,  2]
EDGE_NUM_CLASSES = [4, 1, 1]


class LinkPredictionPCQMContact(Symmetry):
    def __init__(self, config):
        super().__init__()
        self.group_name = 'S'
        self.rep_dim = 53
        self.rep_in = {1: sum(NODE_NUM_CLASSES), 2: 1 + sum(EDGE_NUM_CLASSES)}
        self.rep_out = {2: 1}
        self.metric_name = ['hits@1', 'hits@3', 'hits@10', 'mrr', 'mrr_filtered', 'mrr_filtered_noself']
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
        edge_output = xs[0]
        # (B, N, N, 1) -> (B, N, N)
        return edge_output[..., 0]

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
        edge_output = xs[0]
        edge_output = torch.einsum('bij,bjkd,blk->bild', gs, edge_output, gs)
        xs = (edge_output,)
        return xs

    def criterion(self, y_hat: torch.Tensor, y: Data) -> torch.Tensor:
        # parse inputs
        batch = y
        y = batch.edge_label.float()
        device = batch.x.device
        batch_size = batch.num_graphs
        node_ptr = batch._slice_dict['x'].to(device)
        edge_ptr = batch._slice_dict['edge_label_index'].to(device)
        num_edges = edge_ptr[1:] - edge_ptr[:-1]
        # (B, N, N) -> (sum(Ei),)
        edge_batch_index = torch.arange(batch_size, device=device).repeat_interleave(num_edges)
        edge_index_offset = node_ptr[:-1].repeat_interleave(num_edges)
        edge_index = batch.edge_label_index - edge_index_offset[None, :]
        y_hat = y_hat[edge_batch_index, edge_index[0], edge_index[1]]
        return nn.BCEWithLogitsLoss()(y_hat, y)

    @torch.no_grad()
    def _eval_mrr(self, y_pred_pos, y_pred_neg, suffix='') -> dict:
        """ Compute Hits@k and Mean Reciprocal Rank (MRR).
        Implementation from OGB:
        https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py
        Args:
            y_pred_neg: array with shape (batch size, num_entities_neg).
            y_pred_pos: array with shape (batch size, )
        """
        assert y_pred_pos.ndim == 1 and y_pred_neg.ndim == 2
        assert y_pred_pos.shape[0] == y_pred_neg.shape[0]
        y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
        argsort = torch.argsort(y_pred, dim=1, descending=True)
        ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
        ranking_list = ranking_list[:, 1] + 1
        hits1_list = (ranking_list <= 1).to(torch.float)
        hits3_list = (ranking_list <= 3).to(torch.float)
        hits10_list = (ranking_list <= 10).to(torch.float)
        mrr_list = 1. / ranking_list.to(torch.float)
        return {
            f'hits@1{suffix}': hits1_list,
            f'hits@3{suffix}': hits3_list,
            f'hits@10{suffix}': hits10_list,
            f'mrr{suffix}': mrr_list
        }

    @torch.no_grad()
    def _eval(self, pred, data) -> dict:
        pred = pred[:data.num_nodes, :data.num_nodes]
        pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
        num_pos_edges = pos_edge_index.shape[1]
        pred_pos = pred[pos_edge_index[0], pos_edge_index[1]]
        if num_pos_edges > 0:
            # raw MRR
            neg_mask = torch.ones([num_pos_edges, data.num_nodes], dtype=torch.bool)
            neg_mask[torch.arange(num_pos_edges), pos_edge_index[1]] = False
            pred_neg = pred[pos_edge_index[0]][neg_mask].view(num_pos_edges, -1)
            mrr_list = self._eval_mrr(pred_pos, pred_neg, suffix='')

            # filtered MRR
            pred_masked = pred.clone()
            pred_masked[pos_edge_index[0], pos_edge_index[1]] -= float("inf")
            pred_neg = pred_masked[pos_edge_index[0]]
            mrr_list.update(self._eval_mrr(pred_pos, pred_neg, suffix='_filtered'))

            # extended filter without self-loops
            pred_masked.fill_diagonal_(-float("inf"))
            pred_neg = pred_masked[pos_edge_index[0]]
            mrr_list.update(self._eval_mrr(pred_pos, pred_neg, suffix='_filtered_noself'))
        else:
            mrr_list = self._eval_mrr(pred_pos, pred_pos)
        return mrr_list

    @torch.no_grad()
    def evaluator(self, y_hat: list, y: list) -> dict:
        """ Compute Hits@k and Mean Reciprocal Rank (MRR).
        Implementation from LRGB:
        https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/head/inductive_edge.py
        """
        y_hat = torch.cat(y_hat, dim=0)
        data_list = []
        for batch in y:
            data_list.extend(batch.to_data_list())

        stats = {}
        for pred, data in zip(y_hat, data_list):
            mrr_list = self._eval(pred, data)
            for key, val in mrr_list.items():
                val = float(val.mean().item())
                if np.isnan(val):
                    val = 0.
                if key not in stats:
                    stats[key] = [val]
                else:
                    stats[key].append(val)

        batch_stats = {}
        for key, val in stats.items():
            batch_stats[key] = {
                'metric_sum': sum(val),
                'metric_count': len(val)
            }
        return batch_stats
