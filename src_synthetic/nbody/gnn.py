# https://github.com/omri1348/Frame-Averaging/blob/master/nbody/models/gcl.py
import torch
from torch import nn


class GNNLayer(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, input_ef=2, act_fn=nn.SiLU(), bias=True):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + input_ef, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, output_nf, bias=bias)
        )

    def edge_model(self, source, target, edge_attr):
        edge_in = torch.cat([source, target, edge_attr], dim=-1)
        out = self.edge_mlp(edge_in)
        return out

    @staticmethod
    def _unsorted_segment_sum(data, segment_ids, num_segments):
        """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
        assert data.ndim == 2 and segment_ids.ndim == 1
        assert data.size(0) == segment_ids.size(0)
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        return result

    def node_model(self, h, edge_index, edge_attr):
        row, _ = edge_index[0], edge_index[1]
        agg = self._unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=-1)
        out = self.node_mlp(out)
        return out

    def forward(self, x, edge_features, edge_idx):
        edge_feat = self.edge_model(x[edge_idx[0]], x[edge_idx[1]], edge_features)
        x = self.node_model(x, edge_idx, edge_feat)
        return x


class GNN(nn.Module):
    def __init__(self, input_dim=6, hidden_nf=60, n_layers=4, act_fn=nn.SiLU()):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Linear(input_dim, hidden_nf)
        for i in range(n_layers):
            self.add_module(f"layer_{i}", GNNLayer(hidden_nf, hidden_nf, hidden_nf))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 3)
        )

    def _incr_edge_idx(self, edge_idx, step: int):
        # batch increment edge_idx by step
        assert step == 5
        b, _, n_edges = edge_idx.shape
        assert edge_idx.shape == (b, 2, n_edges)
        incr = step * torch.arange(b, device=edge_idx.device)
        incr_edge_idx = edge_idx + incr[:, None, None]
        assert incr_edge_idx.shape == (b, 2, n_edges)
        return incr_edge_idx

    def pyg_format(self, node_features, edge_features, edge_idx):
        b, n, _, d_node = node_features.shape
        _, _, _, d_edge = edge_features.shape
        _, _, n_edges = edge_idx.shape
        assert node_features.shape == (b, n, 3, d_node)
        assert edge_features.shape == (b, n, n, d_edge)
        assert edge_idx.shape == (b, 2, n_edges)
        pyg_node_features = node_features.transpose(-1, -2).reshape(b*n, 3*d_node)
        batch_idx = torch.arange(b, device=node_features.device).repeat_interleave(n_edges)
        row_idx = edge_idx[:, 0, :].flatten()
        col_idx = edge_idx[:, 1, :].flatten()
        pyg_edge_features = edge_features[batch_idx, row_idx, col_idx, :]
        assert pyg_edge_features.shape == (b*n_edges, d_edge)
        incr_edge_idx = self._incr_edge_idx(edge_idx, n)
        assert incr_edge_idx.shape == (b, 2, n_edges)
        pyg_edge_idx = incr_edge_idx.transpose(0, 1).reshape(2, b*n_edges)
        return pyg_node_features, pyg_edge_features, pyg_edge_idx

    def forward(self, node_features, edge_features, edge_idx):
        b, n, _, d_node = node_features.shape
        _, _, _, d_edge = edge_features.shape
        _, _, n_edges = edge_idx.shape
        assert node_features.shape == (b, n, 3, d_node)
        assert edge_features.shape == (b, n, n, d_edge)
        assert edge_idx.shape == (b, 2, n_edges)
        assert n_edges == n * (n - 1)
        # standize input format
        x, edge_features, edge_idx = self.pyg_format(node_features, edge_features, edge_idx)
        assert x.shape == (b*n, 3*d_node)
        assert edge_features.shape == (b*n_edges, d_edge)
        assert edge_idx.shape == (2, b*n_edges)
        # forward
        x = self.embedding(x)
        for i in range(self.n_layers):
            x = self._modules[f"layer_{i}"](x, edge_features, edge_idx)
        x = self.decoder(x)
        x = x.reshape(b, n, 3)
        return x
