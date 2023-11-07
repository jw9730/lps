# pylint:disable=unused-variable,line-too-long
import torch


def compute_node_mask(num_nodes, n, device):
    tri = torch.tril(torch.ones(n, n, device=device, dtype=torch.bool))
    node_mask = tri[num_nodes - 1]
    b, n = node_mask.shape
    arr0 = node_mask.unsqueeze(1).expand(b, n, n)
    arr1 = node_mask.unsqueeze(2).expand(b, n, n)
    node_mask2d = torch.logical_and(arr0, arr1)
    return node_mask, node_mask2d


def perm1d_to_2d(perm1d, device):
    b, n = perm1d.shape
    perm2d = torch.eye(n, device=device)[None, :, :].expand(b, n, n)
    perm2d = perm2d.gather(1, perm1d[:, :, None].expand(b, n, n))
    return perm2d


def sample_perm(perm_idx_pad, mask_perm_pad, sort_idx_pad, n_samples):
    device = perm_idx_pad.device
    b, n = perm_idx_pad.shape
    assert mask_perm_pad.shape == sort_idx_pad.shape == (b, n)

    sort_idx_perm = perm1d_to_2d(sort_idx_pad, device)
    sort_idx_perm = sort_idx_perm[:, None, :, :].expand(b, n_samples, n, n)
    perm_idx_pad = perm_idx_pad[:, None, :].expand(b, n_samples, n)
    mask_perm_pad = mask_perm_pad[:, None, :].expand(b, n_samples, n)

    sort_idx_perm = sort_idx_perm.reshape(b * n_samples, n, n)
    perm_idx_pad = perm_idx_pad.reshape(b * n_samples, n)
    mask_perm_pad = mask_perm_pad.reshape(b * n_samples, n)

    z = torch.zeros(b * n_samples, n, device=device).uniform_(0.1, 0.5)
    perm_idx_pad = perm_idx_pad.to(z.dtype)
    perm_idx_pad[mask_perm_pad] += z[mask_perm_pad]
    perms = perm_idx_pad.argsort(dim=-1)
    assert perms.shape == (b * n_samples, n)

    gs = perm1d_to_2d(perms, device)
    assert gs.shape == (b * n_samples, n, n)

    gs = torch.matmul(gs, sort_idx_perm)  # gs[0, -1, -1] = 1, not 0
    return gs


@torch.no_grad()
def samples_from_frame(n, xs, batch, n_samples):
    b, n, _ = xs[0].shape
    assert b == batch.num_graphs
    device = xs[0].device

    n_nodes = batch.ptr[1:] - batch.ptr[:-1]
    node_mask, _ = compute_node_mask(n_nodes, n, device)

    perm_idx_pad = torch.arange(n, device=device)[None, :].expand(b, n).clone()
    mask_perm_pad = torch.ones(b, n, device=device) == 0
    sort_idx_pad = torch.arange(n, device=device)[None, :].expand(b, n).clone()

    perm_idx_pad[node_mask] = batch.perm_idx.to(device)
    mask_perm_pad[node_mask] = batch.mask_perm.to(device)
    sort_idx_pad[node_mask] = batch.sort_idx.to(device)

    gs = sample_perm(perm_idx_pad, mask_perm_pad, sort_idx_pad, n_samples)
    gs = gs.transpose(1,2)
    return gs
