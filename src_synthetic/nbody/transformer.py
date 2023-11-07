# pylint: disable=missing-module-docstring,missing-class-docstring,line-too-long
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 25):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 25):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(max_len, 1, d_model))

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
            self,
            d_model=64,
            dim_feedforward=64,
            nhead=4,
            num_encoder_layers=4,
            dropout=0.5,
            activation=F.gelu
        ):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(6+2, d_model), nn.Dropout(dropout))
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, norm_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

    def forward(self, node_features, edge_features):
        # get shapes
        b, n, c, d_node = node_features.shape
        _, _, _, d_edge = edge_features.shape
        assert n == 5 and c == 3
        assert edge_features.shape == (b, n, n, d_edge)
        node_channels = 3*d_node
        edge_channels = d_edge
        # embed input into a (b, n, n, c) tensor
        x = torch.zeros(b, n, n, node_channels + edge_channels, device=node_features.device)
        node_features = node_features.reshape(b, n, node_channels)
        x[:, torch.arange(n), torch.arange(n), :node_channels] = node_features
        x[:, :, :, node_channels:] = edge_features
        # reshape to (b, l=n*n, c)
        x = x.reshape(b, n*n, node_channels + edge_channels)
        # transpose to (l, b, c)
        x = x.transpose(0, 1)
        # push through transformer
        x = self.input(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.output(x)
        # transpose to (b, l, 3)
        x = x.transpose(0, 1)
        # reshape into (b, n, n, 3)
        x = x.reshape(b, n, n, 3)
        # take diagonal, (b, n, 3)
        x = x[:, torch.arange(n), torch.arange(n)]
        return x
