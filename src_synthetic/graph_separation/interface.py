# pylint: disable=no-member
import numpy as np
import torch
from torch import nn, Tensor as T

from gin import GIN
from backbone import setup_backbone


@torch.no_grad()
def compute_edge_mask(n_edges: T, n: int):
    b = n_edges.shape[0]
    idxs = torch.arange(0, n*n, device=n_edges.device)[None, :].expand(b, n*n)
    edge_mask = idxs < n_edges[:, None]
    return edge_mask


@torch.no_grad()
def increment_edge_idx(edge_idx: T, mask: T, h: int):
    b, l = mask.shape
    device = edge_idx.device
    increment = torch.arange(start=0, end=h*(b-1)+1, step=h, device=device)
    increment = increment[:, None, None].expand(b, 2, l)
    new_edge_idx = edge_idx + increment
    mask = mask.unsqueeze(1).expand(b, 2, l)
    new_edge_idx[~mask] = -1
    return new_edge_idx


@torch.no_grad()
def compute_k_edge_idx(edge_idx: T, mask: T, h: int, k: int):
    b, _, l = edge_idx.shape
    k_edge_idx = edge_idx.unsqueeze(1).expand(b, k, 2, l)
    k_edge_idx = k_edge_idx.reshape(b*k, 2, l)
    k_mask = mask.unsqueeze(1).expand(b, k, l)
    k_mask = k_mask.reshape(b*k, l)
    k_edge_idx = increment_edge_idx(k_edge_idx, k_mask, h=h)
    return k_edge_idx, k_mask


def flatten_edge_idx(edge_idx: T, mask: T):
    device = edge_idx.device
    result = torch.zeros(2, mask.sum(), dtype=edge_idx.dtype, device=device)
    result[0] = edge_idx[:, 0, :][mask]
    result[1] = edge_idx[:, 1, :][mask]
    return result


@torch.no_grad()
def compute_node_mask(n_nodes: T, n: int):
    b = n_nodes.shape[0]
    idxs = torch.arange(0, n, device=n_nodes.device)[None, :].expand(b, n)
    node_mask = idxs < n_nodes[:, None]
    return node_mask


@torch.no_grad()
def compute_node_pair_mask(node_mask: T):
    b, n = node_mask.shape
    mask1 = node_mask[:, None, :].expand(b, n, n)
    mask2 = node_mask[:, :, None].expand(b, n, n)
    node_pair_mask = torch.logical_and(mask1, mask2)
    return node_pair_mask


class InterfacedModel(nn.Module):
    def __init__(
        self,
        n=64,
        d=2,
        interface='prob',
        num_interface_layers=1,
        backbone='mlp',
        fixed_noise=False,
        noise_scale=1,
        tau=0.1,
        hard=True,
        task='EXPclassify',
        backbone_seed=None,
        interface_seed=None
    ):
        super().__init__()
        assert task in ('EXPclassify', 'EXPiso', 'GRAPH8c', 'automorphism')
        assert interface in ('prob', 'unif')
        assert backbone in ('mlp', 'gin')
        self.task = task
        self.n = n
        self.d = d
        self.mlp_backbone = backbone == 'mlp'
        self.skip_transform = False
        self.backbone = setup_backbone(
            backbone_type=backbone,
            task=task,
            n=n,
            d=d,
            backbone_seed=backbone_seed
        )
        self.interface = EquivariantInterface(
            n=n,
            d=d,
            interface=interface,
            num_interface_layers=num_interface_layers,
            backbone=backbone,
            fixed_noise=fixed_noise,
            noise_scale=noise_scale,
            tau=tau,
            hard=hard,
            task=task,
            interface_seed=interface_seed
        )

    def transform_input(self, perm, x, adj, n_nodes):
        b, k, n, _ = perm.shape
        d = self.d
        assert perm.shape == (b, k, n, n)
        assert x.shape == (b, n, d)
        assert adj.shape == (b, n, n)
        assert n_nodes.shape == (b,)
        x = x[:, None, :, :].expand(b, k, n, d)
        adj = adj[:, None, :, :].expand(b, k, n, n)
        node_mask = compute_node_mask(n_nodes, n)
        node_pair_mask = compute_node_pair_mask(node_mask)
        node_pair_mask = node_pair_mask[:, None, :, :].expand(b, k, n, n)

        if self.mlp_backbone:
            # apply permutation to input data
            if not self.skip_transform:
                # left-equivariance, so use inverse
                inv_perm = perm.transpose(2, 3)
                x = inv_perm @ x
                adj = inv_perm @ adj @ inv_perm.transpose(2, 3)
            if self.task in ('EXPclassify', 'EXPiso'):
                x = x.reshape(b, k, n*d)
                adj = adj.reshape(b, k, n*n)
                x = torch.cat((x, adj), -1)
                return x
            if self.task in ('GRAPH8c', 'automorphism'):
                x = adj.reshape(b, k, n*n)
                return x
            raise NotImplementedError
        # gin-id
        # add node identifiers
        if not self.skip_transform:
            # left-equivariance
            x_id = torch.where(node_pair_mask, perm, torch.zeros_like(perm))
        else:
            x_id = torch.eye(n, device=perm.device)[None, None, :, :].expand(b, k, n, n)
            x_id = torch.where(node_pair_mask, x_id, torch.zeros_like(perm))
        x = torch.cat((x, x_id), -1).reshape(b*k*n, d+n)
        return x

    def transform_output(self, perm, x, n_nodes):
        if self.task in ('EXPclassify', 'EXPiso', 'GRAPH8c'):
            return x
        if self.task == 'automorphism':
            b, k, n, _ = perm.shape
            assert perm.shape == (b, k, n, n)
            assert x.shape == (b, k, n)
            x = x[:, :, :, None]
            node_mask = compute_node_mask(n_nodes, n)
            # apply permutation to output data
            # left-equivariance, so use inverse of inverse
            x = perm @ x
            # mean over samples
            x = x.mean(1)
            # mask invalid nodes
            x[~node_mask] = 0
            return x
        raise NotImplementedError

    def forward(self, data, n_samples: int = 1, return_perm=False):
        if self.mlp_backbone:
            perm, entropy_loss = self.interface(data, k=n_samples)
            x = self.transform_input(perm, data.pad_x, data.pad_adj, data.n)
            x = self.backbone(x)
            x = self.transform_output(perm, x, data.n)
        else:
            # gin-id
            perm, entropy_loss = self.interface(data, k=n_samples)
            _, k, n, _ = perm.shape
            x = self.transform_input(perm, data.pad_x, data.pad_adj, data.n)
            edge_idx = data.pad_edge_idx
            edge_mask = compute_edge_mask(data.e, n)
            node_mask = compute_node_mask(data.n, n)
            k_edge_idx, k_edge_mask = compute_k_edge_idx(edge_idx, edge_mask, h=n, k=k)
            flat_edge_idx = flatten_edge_idx(k_edge_idx, k_edge_mask)
            x = self.backbone(x, flat_edge_idx, node_mask, k)
            x = self.transform_output(perm, x, data.n)
        if return_perm:
            return x, entropy_loss, perm
        return x, entropy_loss


class PermutaionMatrixPenalty(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    @staticmethod
    def _normalize(scores, axis, eps=1e-12):
        normalizer = torch.sum(scores, axis).clamp(min=eps)
        normalizer = normalizer.unsqueeze(axis)
        prob = torch.div(scores, normalizer)
        return prob

    @staticmethod
    def _entropy(prob, axis):
        return -torch.sum(prob * prob.log().clamp(min=-100), axis)

    def entropy(self, scores, node_mask, eps=1e-12):
        b, k, n, _ = scores.shape
        # clamp min to avoid zero logarithm
        scores = scores.clone()
        node_pair_mask = compute_node_pair_mask(node_mask)
        node_pair_mask = node_pair_mask[:, None, :, :].expand(b, k, n, n)
        scores[node_pair_mask] = scores[node_pair_mask].clamp(min=eps)
        # compute columnwise entropy
        col_prob = self._normalize(scores, axis=2)
        col_prob = torch.where(node_pair_mask, col_prob, torch.ones_like(scores))
        entropy_col = self._entropy(col_prob, axis=2)
        # compute rowwise entropy
        row_prob = self._normalize(scores, axis=3)
        row_prob = torch.where(node_pair_mask, row_prob, torch.ones_like(scores))
        entropy_row = self._entropy(row_prob, axis=3)
        # return entropy
        assert entropy_col.shape == entropy_row.shape == (b, k, n)
        return entropy_col, entropy_row

    def forward(self, perm_soft: T, n_nodes: T):
        b, k, n, _ = perm_soft.shape
        assert n == self.n
        assert perm_soft.shape == (b, k, n, n)
        # compute node mask
        node_mask = compute_node_mask(n_nodes, n)
        # compute entropy
        entropy_col, entropy_row = self.entropy(perm_soft, node_mask)
        # compute mean over samples
        loss = entropy_col.mean(1) + entropy_row.mean(1)
        # compute mean over nodes
        loss[~node_mask] = 0
        loss = loss.sum(1) / n_nodes
        # compute mean over batch
        loss = loss.mean()
        return loss


class EquivariantInterface(nn.Module):
    def __init__(
        self,
        n=64,
        d=2,
        interface='prob',
        num_interface_layers=1,
        backbone='mlp',
        fixed_noise=False,
        noise_scale=0.1,
        tau=0.1,
        hard=True,
        task='EXPclassify',
        interface_seed=None
    ):
        super().__init__()
        assert interface in ('prob', 'unif')
        self.task = task
        self.interface = interface
        self.mlp_backbone = backbone == 'mlp'
        self.n = n
        self.d = d
        self.use_virtual_node = True
        self.fixed_noise = fixed_noise
        self.noise_scale = noise_scale
        self.tau = tau
        self.hard = hard
        if interface_seed:
            print(f'initializing interface with random seed {interface_seed}')
            # get current random states
            torch_state = torch.get_rng_state()
            np_state = np.random.get_state()
            # seed random states
            torch.manual_seed(interface_seed)
            np.random.seed(interface_seed)
        # build graph isomorphism network
        self.d_h = 64
        self.gin_interface = GIN(
            num_layers=num_interface_layers,
            num_mlp_layers=2,
            input_dim=self.d,
            hidden_dim=self.d_h,
            output_dim=1,
            learn_eps=True,
            initial_eps=None
        )
        self.virtual_node = nn.Parameter(torch.randn(self.d))
        if interface_seed:
            # restore random states
            torch.set_rng_state(torch_state)
            np.random.set_state(np_state)
        # entropy loss for soft permutation matrix
        self.compute_entropy_loss = PermutaionMatrixPenalty(n)

    def _add_vnode_x(self, x: T):
        b, n, d = x.shape
        new_x = torch.zeros(b, 1+n, d, dtype=x.dtype, device=x.device)
        new_x[:, 1:, :] = x
        new_x[:, 0, :] = self.virtual_node[None, :].expand(b, d)
        return new_x

    @torch.no_grad()
    def _add_vnode_node_mask(self, n_nodes: T):
        b, n = n_nodes.shape[0], self.n
        new_n_nodes = 1 + n_nodes
        idx = torch.arange(0, 1+n, device=n_nodes.device)
        idx = idx[None, :].expand(b, 1+n)
        new_node_mask = idx < new_n_nodes[:, None]
        return new_node_mask

    @torch.no_grad()
    def _add_vnode_edge_idx(self, edge_idx: T, n_nodes: T, n_edges: T):
        b, n = n_nodes.shape[0], self.n
        device, dtype = edge_idx.device, edge_idx.dtype
        # initialize edge_idx with virtual node
        new_pad_edge_idx = torch.zeros(b, 2, (1+n)*(1+n), device=device, dtype=dtype)
        # setup intermediate tensors
        edge_mask = compute_edge_mask(n_edges, n)
        new_idx = torch.arange(0, (1+n)*(1+n), device=device, dtype=dtype)
        new_idx = new_idx[None, None, :].expand(b, 2, (1+n)*(1+n))
        # first entry is self-loop on virtual node
        # from second entry are edges connecting virtual node and original nodes
        ve = torch.zeros(b, 2, n, device=device, dtype=dtype)
        ve[:, 1, :] = torch.arange(n, device=device, dtype=dtype) + 1
        # ev = ve[:, [1, 0], :]
        idx = torch.arange(0, n, device=device, dtype=dtype)
        idx = idx[None, :].expand(b, n)
        idx_mask = idx < n_nodes[:, None]
        idx_mask = idx_mask[:, None, :].expand(b, 2, n)
        n_nodes = n_nodes[:, None, None].expand(b, 2, (1+n)*(1+n))
        new_idx_mask = (new_idx >= 1) & (new_idx < 1 + n_nodes)
        new_pad_edge_idx[new_idx_mask] = ve[idx_mask]
        new_idx_mask = (new_idx >= 1 + n_nodes) & (new_idx < 1 + 2 * n_nodes)
        new_pad_edge_idx[new_idx_mask] = ve[:, [1, 0], :][idx_mask]
        # the remaining are edges connecting original nodes
        # ee = pad_edge_idx + 1
        idx_mask = edge_mask[:, None, :].expand(b, 2, n*n)
        n_edges = n_edges[:, None, None].expand(b, 2, (1+n)*(1+n))
        new_idx_mask = (new_idx >= 1 + 2 * n_nodes) & (new_idx < 1 + 2 * n_nodes + n_edges)
        new_pad_edge_idx[new_idx_mask] = (edge_idx + 1)[idx_mask]
        return new_pad_edge_idx

    @torch.no_grad()
    def _add_vnode_edge_mask(self, n_nodes: T, n_edges: T, dtype):
        b, n = n_nodes.shape[0], self.n
        n_total = (1 + 2 * n_nodes + n_edges)[:, None]
        idx = torch.arange(0, (1+n)*(1+n), device=n_nodes.device, dtype=dtype)
        idx = idx[None, :].expand(b, (1+n)*(1+n))
        new_edge_mask = idx < n_total
        return new_edge_mask

    def add_vnode(self, x, edge_idx, n_nodes, n_edges):
        b, n, _ = x.shape
        assert n == self.n
        assert edge_idx.shape == (b, 2, n*n)
        assert n_nodes.shape == n_edges.shape == (b,)
        new_x = self._add_vnode_x(x)
        new_node_mask = self._add_vnode_node_mask(n_nodes)
        new_pad_edge_idx = self._add_vnode_edge_idx(edge_idx, n_nodes, n_edges)
        new_edge_mask = self._add_vnode_edge_mask(n_nodes, n_edges, edge_idx.dtype)
        return new_x, new_pad_edge_idx, new_node_mask, new_edge_mask

    def repeat(self, x: T, edge_idx: T, node_mask: T, edge_mask: T, k: int):
        b, n, d = x.shape
        _, _, l = edge_idx.shape
        assert edge_idx.shape == (b, 2, l)
        assert node_mask.shape == (b, n)
        assert edge_mask.shape == (b, l)
        x = x[:, None, :, :].expand(b, k, n, d).reshape(b * k, n, d)
        node_mask = node_mask[:, None, :].expand(b, k, n).reshape(b * k, n)
        edge_idx = edge_idx[:, None, :, :].expand(b, k, 2, l).reshape(b * k, 2, l)
        edge_mask = edge_mask[:, None, :].expand(b, k, l).reshape(b * k, l)
        return x, edge_idx, node_mask, edge_mask

    @torch.no_grad()
    def _incr_edge_idx(self, edge_idx: T, edge_mask: T, step: int):
        # batch increment edge_idx by step
        b, _, l = edge_idx.shape
        assert edge_idx.shape == (b, 2, l)
        assert edge_mask.shape == (b, l)
        edge_mask = edge_mask[:, None, :].expand(b, 2, l)
        incr = torch.arange(0, step*(b-1)+1, step, device=edge_idx.device)
        assert incr.shape == (b,)
        inc_edge_idx = edge_idx + incr[:, None, None]
        inc_edge_idx[~edge_mask] = -1
        assert inc_edge_idx.shape == (b, 2, l)
        return inc_edge_idx

    def pyg_format(self, x: T, edge_idx: T, node_mask: T, edge_mask: T):
        b, n, d = x.shape
        _, _, l = edge_idx.shape
        assert edge_idx.shape == (b, 2, l)
        assert node_mask.shape == (b, n)
        assert edge_mask.shape == (b, l)
        pyg_x = x.view(b*n, d)
        n_total_edges = int(edge_mask.sum())
        incr_edge_idx = self._incr_edge_idx(edge_idx, edge_mask, n)
        pyg_edge_idx = torch.zeros(2, n_total_edges,
                                   dtype=edge_idx.dtype,
                                   device=edge_idx.device)
        pyg_edge_idx[0] = incr_edge_idx[:, 0, :][edge_mask]
        pyg_edge_idx[1] = incr_edge_idx[:, 1, :][edge_mask]
        return pyg_x, pyg_edge_idx

    def normalize_scores(self, scores: T, n_nodes: T, eps=1e-6):
        # normalize scores considering masked nodes
        # this does not affect hard permutation, but affects
        # straight-through gradient from soft permutation matrix
        b, k, n = scores.shape
        assert n == self.n, f'{n} != {self.n}'
        assert n_nodes.shape == (b,)
        # compute node mask
        node_mask = compute_node_mask(n_nodes, n)
        node_mask = node_mask[:, None, :].expand(b, k, n)
        n_nodes = n_nodes[:, None, None].expand(b, k, n)
        # normalize
        scores[~node_mask] = 0.
        l2_norm = scores.pow(2).sum(2, keepdim=True).sqrt()
        normalized_scores = scores / l2_norm.clamp(min=eps)
        # mask
        normalized_scores[~node_mask] = -float('inf')
        return normalized_scores

    def argsort(self, scores: T, n_nodes: T, sinkhorn_iter=20):
        b, k, n = scores.shape
        assert n == self.n
        assert n_nodes.shape == (b,)
        assert sinkhorn_iter >= 0
        # compute node pair mask
        node_mask = compute_node_mask(n_nodes, n)
        node_pair_mask = compute_node_pair_mask(node_mask)
        node_pair_mask = node_pair_mask[:, None, :, :].expand(b, k, n, n)
        # sort scores, result is unique up to permutations of tied scores
        scores = scores[:, :, :, None]
        scores_sorted, indices = scores.sort(descending=True, dim=2)
        scores = scores.expand(b, k, n, n)
        scores_sorted = scores_sorted.transpose(2, 3).expand(b, k, n, n)
        # softsort + log sinkhorn operator for computing soft permutation matrix
        log_perm_soft = (scores - scores_sorted).abs().neg() / self.tau
        log_zero = torch.zeros_like(log_perm_soft).fill_(-float('inf'))
        for _ in range(sinkhorn_iter):
            log_perm_soft = torch.where(node_pair_mask, log_perm_soft, log_zero)
            log_perm_soft = log_perm_soft - torch.logsumexp(log_perm_soft, dim=-1, keepdim=True)
            log_perm_soft[~torch.isfinite(log_perm_soft)] = -float('inf')
            log_perm_soft = torch.where(node_pair_mask, log_perm_soft, log_zero)
            log_perm_soft = log_perm_soft - torch.logsumexp(log_perm_soft, dim=-2, keepdim=True)
            log_perm_soft[~torch.isfinite(log_perm_soft)] = -float('inf')
        perm_soft = log_perm_soft.exp()
        if self.hard:
            # argsort for hard permutation matrix
            with torch.no_grad():
                perm_hard = torch.zeros_like(perm_soft).scatter(dim=-1, index=indices, value=1)
                perm_hard = perm_hard.transpose(2, 3)
                perm_hard[~node_pair_mask] = 0
                # (optional) test if perm_hard is a permutation matrix
                assert torch.allclose(perm_hard.sum(-1).mean(1), node_mask.float())
                assert torch.allclose(perm_hard.sum(-2).mean(1), node_mask.float())
            # differentiability with straight-through gradient
            # the estimated gradient is accurate if perm_soft is close to perm_hard
            # for this, entropy regularization is necessary
            perm_hard = (perm_hard - perm_soft).detach() + perm_soft
            return perm_hard, perm_soft
        return perm_soft, perm_soft

    def sample_invariant_noise(self, x, idx, k):
        b, n, d = x.shape
        if self.fixed_noise:
            zs = []
            for i in idx:
                seed = torch.seed()
                torch.manual_seed(i)
                z = torch.zeros(k, n, d, device=x.device, dtype=x.dtype)
                z = z.uniform_(0, self.noise_scale)
                zs.append(z)
                torch.manual_seed(seed)
            z = torch.stack(zs, dim=0).reshape(b, n, self.d)
        else:
            z = torch.zeros_like(x).uniform_(0, self.noise_scale)
        if self.use_virtual_node:
            z[:, 0, :] = 0
        return z

    def _forward_prob(self, data, k):
        # k is the number of interface samples
        x = data.pad_x
        edge_idx = data.pad_edge_idx
        b, n, d = x.shape
        assert n == self.n and d == self.d
        assert edge_idx.shape == (b, 2, n*n)
        # add virtual node and compute masks
        if self.use_virtual_node:
            x, edge_idx, node_mask, edge_mask = self.add_vnode(x, edge_idx, data.n, data.e)
            n = n + 1
        else:
            node_mask = compute_node_mask(data.n, n)
            edge_mask = compute_edge_mask(data.e, n)
        # replicate input k times
        x, edge_idx, node_mask, edge_mask = self.repeat(x, edge_idx, node_mask, edge_mask, k)
        # add noise
        x = x + self.sample_invariant_noise(x, data.idx, k)
        # remove masked nodes
        x[~node_mask] = 0
        # compute scores
        scores = self.gin_interface(*self.pyg_format(x, edge_idx, node_mask, edge_mask))
        assert scores.shape == (b*k*n, 1) and node_mask.shape == (b*k, n)
        scores = scores.view(b, k, n)
        node_mask = node_mask.view(b, k, n)
        # add small noise to break ties
        # without this, models can exploit the ordering of tied scores
        scores = scores + torch.zeros_like(scores).uniform_(0, 1e-6)
        # mask invalid nodes
        scores[~node_mask] = -float('inf')
        # remove virtual node from scores
        if self.use_virtual_node:
            scores = scores[:, :, 1:]
        # normalize
        # this is optional, for improving training stability
        scores = self.normalize_scores(scores, data.n)
        # compute argsort and permutation without virtual node
        perm, perm_soft = self.argsort(scores, data.n)
        # compute entropy loss
        entropy_loss = self.compute_entropy_loss(perm_soft, data.n)
        return perm, entropy_loss

    def _forward_unif(self, data, k):
        # k is the number of interface samples
        x = data.pad_x
        edge_idx = data.pad_edge_idx
        b, n, d = x.shape
        assert n == self.n and d == self.d
        assert edge_idx.shape == (b, 2, n*n)
        node_mask = compute_node_mask(data.n, n)
        edge_mask = compute_edge_mask(data.e, n)
        # replicate input k times
        x, edge_idx, node_mask, edge_mask = self.repeat(x, edge_idx, node_mask, edge_mask, k)
        # random noise
        assert self.hard
        if self.fixed_noise:
            raise NotImplementedError
        scores = torch.zeros(b*k, n, device=x.device).uniform_(0, 1)
        assert scores.shape == (b*k, n) and node_mask.shape == (b*k, n)
        scores = scores.view(b, k, n)
        node_mask = node_mask.view(b, k, n)
        # mask invalid nodes
        scores[~node_mask] = -float('inf')
        # compute argsort and permutation
        # for group averaging, soft permutation can be discarded (sinkhorn_iter=1)
        perm, _ = self.argsort(scores, data.n, sinkhorn_iter=1)
        return perm

    def forward(self, data, k):
        if self.interface == 'prob':
            perm, entropy_loss = self._forward_prob(data, k)
            return perm, entropy_loss
        if self.interface == 'unif':
            perm = self._forward_unif(data, k)
            return perm, torch.tensor(0, device=perm.device)
        raise NotImplementedError
