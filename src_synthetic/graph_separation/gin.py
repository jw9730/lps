# pylint: disable=not-callable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        assert num_layers > 1, "number of layers must be greater than 1"
        self.num_layers = num_layers
        self.linear_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.linear_layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers.append(nn.Linear(hidden_dim, output_dim))
        for _ in range(num_layers - 1):
            self.norms.append(nn.BatchNorm1d((hidden_dim)))
        # initialize batchnorm statistics and bias
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.linear_layers[i](x)
            x = self.norms[i](x)
            x = F.relu(x)
        x = self.linear_layers[self.num_layers - 1](x)
        return x


class GIN(nn.Module):
    def __init__(
        self,
        num_layers,
        num_mlp_layers,
        input_dim,
        hidden_dim,
        output_dim,
        learn_eps,
        initial_eps
    ):
        """
        num_layers: including input layer, excluding output layer
        num_mlp_layers: number of nn.Linear layers in each MLP block
        """
        super().__init__()
        self.num_layers = num_layers
        if initial_eps is None:
            eps = torch.tensor([np.random.uniform() for _ in range(num_layers)])
        else:
            eps = torch.tensor(initial_eps).float().repeat(num_layers)
        if learn_eps:
            self.eps = nn.Parameter(eps)
        else:
            self.register_buffer("eps", eps)
        self.input = nn.Linear(input_dim, hidden_dim)
        self.mlps = nn.ModuleList()
        self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
        for _ in range(self.num_layers - 1):
            self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.norms.append(nn.BatchNorm1d((hidden_dim)))
        self.head = nn.Linear(hidden_dim, output_dim)
        # initialize batchnorm statistics and bias
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def layer_forward(self, x, sparse_adj, layer_idx):
        x = torch_sparse.matmul(sparse_adj.t(), x) + (1 + self.eps[layer_idx]) * x
        x = self.mlps[layer_idx](x)
        x = self.norms[layer_idx](x)
        x = F.relu(x)
        return x

    def forward(self, x, edge_index):
        # x: [sum(n), input_dim]
        # edge_index: [2, sum(e)]
        assert x.ndim == edge_index.ndim == 2
        n, _ = x.shape
        # create sparse adj
        row, col = edge_index
        sparse_adj = torch_sparse.SparseTensor(row=row, col=col, sparse_sizes=(n, n))
        # forward
        for i in range(self.num_layers):
            x = self.layer_forward(x, sparse_adj, i)
        x = self.head(x)
        # x: [sum(n), output_dim]
        return x
