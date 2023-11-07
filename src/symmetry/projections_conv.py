# pylint: disable=too-many-arguments,unused-variable,too-many-instance-attributes,line-too-long,unused-argument
from typing import Optional, Callable, Tuple
import warnings
import torch
from torch import nn
from torch.nn import functional as F

from .symmetry import Symmetry
from .projections import InputProj, OutputProj


class ConvInputProj(InputProj):
    def __init__(self, symmetry: Symmetry, num_tokens: int, embed_dim: int, bias: bool=True):
        super().__init__(symmetry)
        proj_keys = []
        proj = []
        if 0 in symmetry.rep_in and 0 not in symmetry.ignore_rep_in:
            proj_keys.append(0)
            proj.append(Order0ConvInputProj(in_chans=symmetry.rep_in[0], num_tokens=num_tokens, embed_dim=embed_dim, bias=bias))
        if 1 in symmetry.rep_in and 1 not in symmetry.ignore_rep_in:
            proj_keys.append(1)
            proj.append(Order1ConvInputProj(seq_size=symmetry.rep_dim, in_chans=symmetry.rep_in[1], num_tokens=num_tokens, embed_dim=embed_dim, bias=bias))
        if 2 in symmetry.rep_in and 2 not in symmetry.ignore_rep_in:
            proj_keys.append(2)
            proj.append(Order2ConvInputProj(img_size=symmetry.rep_dim, in_chans=symmetry.rep_in[2], num_tokens=num_tokens, embed_dim=embed_dim, bias=bias))
        if (0, 0) in symmetry.rep_in and (0, 0) not in symmetry.ignore_rep_in:
            proj_keys.append((0, 0))
            proj.append(Order0ConvInputProj(in_chans=symmetry.rep_in[(0, 0)], num_tokens=num_tokens, embed_dim=embed_dim, bias=bias))
        if (1, 1) in symmetry.rep_in and (1, 1) not in symmetry.ignore_rep_in:
            proj_keys.append((1, 1))
            if symmetry.rep_dim[1] == 3:
                print(f"Point cloud inputs detected, using Order1ConvInputProj({symmetry.rep_dim[0]},) instead of Order1x1ConvInputProj({symmetry.rep_dim})")
                proj.append(Order1ConvInputProj(seq_size=symmetry.rep_dim[0], in_chans=symmetry.rep_dim[1] * symmetry.rep_in[(1, 1)], num_tokens=num_tokens, embed_dim=embed_dim, bias=bias))
            else:
                proj.append(Order1x1ConvInputProj(img_size=symmetry.rep_dim, in_chans=symmetry.rep_in[(1, 1)], num_tokens=num_tokens, embed_dim=embed_dim, bias=bias))
        for k in symmetry.rep_in:
            assert k in proj_keys and k not in symmetry.ignore_rep_in, f"Missing projection for {k}-order representation"
        self.proj_keys = proj_keys
        self.proj = nn.ModuleList(proj)


class ConvOutputProj(OutputProj):
    def __init__(self, symmetry: Symmetry, num_tokens: int, embed_dim: int, bias: bool=True):
        super().__init__(symmetry)
        proj_keys = []
        proj = []
        if 0 in symmetry.rep_out and 0 not in symmetry.ignore_rep_out:
            proj_keys.append(0)
            proj.append(nn.Linear(embed_dim, symmetry.rep_out[0], bias=bias))
        if 1 in symmetry.rep_out and 1 not in symmetry.ignore_rep_out:
            proj_keys.append(1)
            proj.append(Order1ConvOutputProj(seq_size=symmetry.rep_dim, out_chans=symmetry.rep_out[1], num_tokens=num_tokens, embed_dim=embed_dim, bias=bias))
        if 2 in symmetry.rep_out and 2 not in symmetry.ignore_rep_out:
            proj_keys.append(2)
            proj.append(Order2ConvOutputProj(img_size=symmetry.rep_dim, out_chans=symmetry.rep_out[2], num_tokens=num_tokens, embed_dim=embed_dim, bias=bias))
        if (0, 0) in symmetry.rep_out and (0, 0) not in symmetry.ignore_rep_out:
            proj_keys.append((0, 0))
            proj.append(nn.Linear(embed_dim, symmetry.rep_out[(0, 0)], bias=bias))
        if (1, 0) in symmetry.rep_out and (1, 0) not in symmetry.ignore_rep_out:
            proj_keys.append((1, 0))
            proj.append(Order1ConvOutputProj(seq_size=symmetry.rep_dim[0], out_chans=symmetry.rep_out[(1, 0)], num_tokens=num_tokens, embed_dim=embed_dim, bias=bias))
        for k in symmetry.rep_out:
            assert k in proj_keys and k not in symmetry.ignore_rep_out, f"Missing projection for {k}-order representation"
        self.proj_keys = proj_keys
        self.proj = nn.ModuleList(proj)


def setup_1d(seq_size: int, num_tokens: int):
    if num_tokens <= seq_size:
        patch_size = (seq_size - 1) // num_tokens + 1
        pad_size = (patch_size * num_tokens) - seq_size
    else:
        patch_size = 1
        pad_size = num_tokens - seq_size
    return patch_size, pad_size


class Order0ConvInputProj(nn.Module):
    """0D to Tokens"""
    def __init__(self, in_chans: int, num_tokens: int, embed_dim: int, norm_layer: Optional[Callable]=None, bias: bool=True):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        x = self.proj(x)
        x = x.unsqueeze(1).repeat(1, self.num_tokens, 1)
        x = self.norm(x)
        return x


class Order1ConvInputProj(nn.Module):
    """1D to Tokens"""
    def __init__(self, seq_size: int, in_chans: int, num_tokens: int, embed_dim: int, norm_layer: Optional[Callable]=None, bias: bool=True):
        super().__init__()
        patch_size, pad_size = setup_1d(seq_size, num_tokens)
        self.seq_size = seq_size
        self.pad_size = pad_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            # point cloud inputs, x is (B, N, 3, C)
            assert x.size(1) == self.seq_size
            x = x.flatten(2, 3)
        B, L, C = x.shape
        assert L == self.seq_size, f"Input sequence length ({L}) doesn't match model ({self.seq_size})."
        x = x.transpose(1, 2)
        x = F.pad(x, (0, self.pad_size))
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class Order1ConvOutputProj(nn.Module):
    """Tokens to 1D"""
    def __init__(self, seq_size: int, out_chans: int, num_tokens: int, embed_dim: int, norm_layer: Optional[Callable]=None, bias: bool=True):
        super().__init__()
        patch_size, pad_size = setup_1d(seq_size, num_tokens)
        self.num_tokens = num_tokens
        self.pad_size = pad_size
        self.proj = nn.ConvTranspose1d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(out_chans) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        assert L == self.num_tokens, f"Input sequence length ({L}) doesn't match model ({self.num_tokens})."
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x[:, :, :-self.pad_size]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


def setup_2d(img_size: int, num_tokens: int):
    assert int(num_tokens ** 0.5) ** 2 == num_tokens, "num_patches must be a perfect square"
    img_size = (img_size, img_size)
    grid_size = (
        int(num_tokens ** 0.5),
        int(num_tokens ** 0.5)
    )
    if grid_size[0] <= img_size[0]:
        assert grid_size[1] <= img_size[1]
        patch_size = (
            (img_size[0] - 1) // grid_size[0] + 1,
            (img_size[1] - 1) // grid_size[1] + 1
        )
        pad_size = (
            (patch_size[0] * grid_size[0]) - img_size[0],
            (patch_size[1] * grid_size[1]) - img_size[1]
        )
    else:
        patch_size = (1, 1)
        pad_size = (
            grid_size[0] - img_size[0],
            grid_size[1] - img_size[1]
        )
    return img_size, grid_size, patch_size, pad_size


class Order2ConvInputProj(nn.Module):
    """2D to Tokens"""
    def __init__(self, img_size: int, in_chans: int, num_tokens: int, embed_dim: int, norm_layer: Optional[Callable]=None, bias: bool=True):
        super().__init__()
        img_size, _, patch_size, pad_size = setup_2d(img_size, num_tokens)
        self.img_size = img_size
        self.pad_size = pad_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, (0, self.pad_size[1], 0, self.pad_size[0]))
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x


class Order2ConvOutputProj(nn.Module):
    """Tokens to 2D"""
    def __init__(self, img_size: int, out_chans: int, num_tokens: int, embed_dim: int, norm_layer: Optional[Callable]=None, bias: bool=True):
        super().__init__()
        _, grid_size, patch_size, pad_size = setup_2d(img_size, num_tokens)
        self.grid_size = grid_size
        self.num_tokens = num_tokens
        self.pad_size = pad_size
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        assert L == self.num_tokens, f"Input sequence length ({L}) doesn't match model ({self.num_tokens})."
        x = x.transpose(1, 2)
        x = x.reshape(B, C, self.grid_size[0], self.grid_size[1])
        x = self.proj(x)
        x = x[:, :, :-self.pad_size[0], :-self.pad_size[1]]
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        return x


def setup_1x1d(img_size: Tuple[int, int], num_tokens: int):
    assert int(num_tokens ** 0.5) ** 2 == num_tokens, "num_patches must be a perfect square"
    grid_size = (
        int(num_tokens ** 0.5),
        int(num_tokens ** 0.5)
    )
    patch_size = [None, None]
    pad_size = [None, None]
    if grid_size[0] <= img_size[0]:
        patch_size[0] = (img_size[0] - 1) // grid_size[0] + 1
        pad_size[0] = (patch_size[0] * grid_size[0]) - img_size[0]
    else:
        patch_size[0] = 1
        pad_size[0] = grid_size[0] - img_size[0]
    if grid_size[1] <= img_size[1]:
        patch_size[1] = (img_size[1] - 1) // grid_size[1] + 1
        pad_size[1] = (patch_size[1] * grid_size[1]) - img_size[1]
    else:
        patch_size[1] = 1
        pad_size[1] = grid_size[1] - img_size[1]
    patch_size = tuple(patch_size)
    pad_size = tuple(pad_size)
    return img_size, grid_size, patch_size, pad_size


class Order1x1ConvInputProj(nn.Module):
    """2D (1Dx1D) to Tokens"""
    def __init__(self, img_size: Tuple[int, int], in_chans: int, num_tokens: int, embed_dim: int, norm_layer: Optional[Callable]=None, bias: bool=True):
        super().__init__()
        img_size, _, patch_size, pad_size = setup_1x1d(img_size, num_tokens)
        self.img_size = img_size
        self.pad_size = pad_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, (0, self.pad_size[1], 0, self.pad_size[0]))
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x
