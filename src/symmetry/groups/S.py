# pylint: disable=line-too-long
import torch


def samples_from_haar_distribution_constant_n(n, bsize, device, dtype) -> torch.Tensor:
    """Random permutations within given numbers of nodes"""
    # sample permutations
    scores = torch.rand(bsize, n, device=device, dtype=dtype)
    perms = scores.argsort(dim=-1)
    gs = torch.eye(n, device=device, dtype=dtype)[None].expand(bsize, -1, -1)
    gs = gs.gather(1, perms[..., None].expand(-1, -1, n))
    return gs


def samples_from_haar_distribution(num_nodes: torch.Tensor, n, bsize, device, dtype) -> torch.Tensor:
    """Random permutations within given numbers of nodes"""
    # handle broadcasted case (n_samples > 1)
    n_samples = bsize // num_nodes.size(0)
    if n_samples > 1:
        num_nodes = num_nodes[None, ...].expand(n_samples, *num_nodes.shape).reshape(-1, *num_nodes.shape[1:])
    # get padding mask
    tri = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
    padding_mask = tri[num_nodes - 1]
    # sample permutations
    scores = torch.rand(bsize, n, device=device, dtype=dtype)
    padding_scores = (torch.arange(n, device=device, dtype=dtype) + 10)[None, :].expand(bsize, -1)
    scores[padding_mask] = padding_scores[padding_mask]
    perms = scores.argsort(dim=-1)
    gs = torch.eye(n, device=device, dtype=dtype)[None].expand(bsize, -1, -1)
    gs = gs.gather(1, perms[..., None].expand(-1, -1, n))
    return gs
