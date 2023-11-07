import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class MLPNetGraph8C(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(64, 128)
        self.conv2 = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        assert x.ndim == 3  # [b, k, d]
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.fc1(x))
        # mean over samples
        x = x.mean(1)
        return x


class MLPNetEXPiso(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(4224, 2048)
        self.conv2 = nn.Linear(2048, 4096)
        self.conv3 = nn.Linear(4096, 2048)
        self.fc1 = nn.Linear(2048, 10)

    def forward(self, x):
        assert x.ndim == 3  # [b, k, d]
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = torch.tanh(self.fc1(x))
        # mean over samples
        x = x.mean(1)
        return x


class MLPNetEXPclassify(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(4224, 2048)
        self.conv2 = nn.Linear(2048, 4096)
        self.conv3 = nn.Linear(4096, 2048)
        self.fc1 = nn.Linear(2048, 10)
        self.fc2 = nn.Linear(10, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        assert x.ndim == 3  # [b, k, d]
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        # mean over samples
        x = x.mean(1)
        return x


class MLPNetAutomorphism(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(25*25, 2048)
        self.conv2 = nn.Linear(2048, 4096)
        self.conv3 = nn.Linear(4096, 2048)
        self.fc1 = nn.Linear(2048, 25)
        self.activation = nn.ReLU()

    def forward(self, x):
        assert x.ndim == 3  # [b, k, d]
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.fc1(x)
        return x


class GINNetGRAPH8C(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.n = n
        self.d = d
        neuron = 64
        r1 = np.random.uniform()
        r2 = np.random.uniform()
        r3 = np.random.uniform()
        nn1 = nn.Sequential(nn.Linear(self.n + self.d, neuron))
        nn2 = nn.Sequential(nn.Linear(neuron, neuron))
        nn3 = nn.Sequential(nn.Linear(neuron, neuron))
        self.conv1 = GINConv(nn1, eps=r1, train_eps=True)
        self.conv2 = GINConv(nn2, eps=r2, train_eps=True)
        self.conv3 = GINConv(nn3, eps=r3, train_eps=True)
        self.fc1 = nn.Linear(neuron, 10)

    def forward(self, x, edge_index, node_mask, k):
        x = torch.tanh(self.conv1(x, edge_index))
        x = torch.tanh(self.conv2(x, edge_index))
        x = torch.tanh(self.conv3(x, edge_index))
        # sum pooling
        b, n = node_mask.shape
        _, d = x.shape
        assert n == self.n
        assert x.shape == (b * k * n, d)
        x = x.reshape(b, k, n, d)
        node_mask = node_mask[:, None, :, None].expand(b, k, n, d)
        x[~node_mask] = 0
        # sum over nodes
        x = x.sum(2)
        # fc
        x = torch.tanh(self.fc1(x))
        # mean over samples
        x = x.mean(1)
        return x


class GINNetEXPiso(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.n = n
        self.d = d
        neuron = 64
        r1 = np.random.uniform()
        r2 = np.random.uniform()
        r3 = np.random.uniform()
        nn1 = nn.Sequential(nn.Linear(self.n + self.d, neuron))
        nn2 = nn.Sequential(nn.Linear(neuron, neuron))
        nn3 = nn.Sequential(nn.Linear(neuron, neuron))
        self.conv1 = GINConv(nn1, eps=r1, train_eps=True)
        self.conv2 = GINConv(nn2, eps=r2, train_eps=True)
        self.conv3 = GINConv(nn3, eps=r3, train_eps=True)
        self.fc1 = nn.Linear(neuron, 10)

    def forward(self, x, edge_index, node_mask, k):
        x = torch.tanh(self.conv1(x, edge_index))
        x = torch.tanh(self.conv2(x, edge_index))
        x = torch.tanh(self.conv3(x, edge_index))
        # sum pooling
        b, n = node_mask.shape
        _, d = x.shape
        assert n == self.n
        assert x.shape == (b * k * n, d)
        x = x.reshape(b, k, n, d)
        node_mask = node_mask[:, None, :, None].expand(b, k, n, d)
        x[~node_mask] = 0
        # sum over nodes
        x = x.sum(2)
        # fc
        x = torch.tanh(self.fc1(x))
        # mean over samples
        x = x.mean(1)
        return x


class GINNetEXPclassify(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.n = n
        self.d = d
        neuron = 64
        r1 = np.random.uniform()
        r2 = np.random.uniform()
        r3 = np.random.uniform()
        nn1 = nn.Sequential(nn.Linear(self.n + self.d, neuron))
        nn2 = nn.Sequential(nn.Linear(neuron, neuron))
        nn3 = nn.Sequential(nn.Linear(neuron, neuron))
        self.conv1 = GINConv(nn1, eps=r1, train_eps=True)
        self.conv2 = GINConv(nn2, eps=r2, train_eps=True)
        self.conv3 = GINConv(nn3, eps=r3, train_eps=True)
        self.fc1 = nn.Linear(neuron, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x, edge_index, node_mask, k):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index)) # [6400, 64]
        # sum pooling
        b, n = node_mask.shape
        _, d = x.shape
        assert n == self.n
        assert x.shape == (b * k * n, d)
        x = x.reshape(b, k, n, d)
        node_mask = node_mask[:, None, :, None].expand(b, k, n, d) 
        x = torch.where(node_mask, x, torch.zeros_like(x)) # [b, k, n, d]

        # sum over nodes
        x = x.sum(2) # [b, k, d]
        # fc
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # mean over samples
        x = x.mean(1)
        return x


def setup_backbone(backbone_type, task, n, d, backbone_seed=None):
    assert backbone_type in ('mlp', 'gin')
    if backbone_seed:
        print(f'initializing backbone with random seed {backbone_seed}')
        # get current random states
        torch_state = torch.get_rng_state()
        np_state = np.random.get_state()
        # seed random states
        torch.manual_seed(backbone_seed)
        np.random.seed(backbone_seed)
    if backbone_type == 'mlp':
        backbone = {
            'GRAPH8c': MLPNetGraph8C,
            'EXPiso': MLPNetEXPiso,
            'EXPclassify': MLPNetEXPclassify,
            'automorphism': MLPNetAutomorphism,
        }[task]()
    else:
        backbone = {
            'GRAPH8c': GINNetGRAPH8C,
            'EXPiso': GINNetEXPiso,
            'EXPclassify': GINNetEXPclassify
        }[task](n=n, d=d)
    if backbone_seed:
        # restore random states
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
    return backbone
