import itertools
import numpy as np
import torch
from torch_geometric.data.data import Data
from torch_geometric.utils import get_laplacian


class PrecomputeSpectral():
    def __init__(
        self,
        nmax=0,
        recfield=1,
        dv=5,
        nfreq=5,
        adddegree=False,
        laplacian=True,
        addadj=False,
        vmax=None,
    ):
        # receptive field. 0: adj, 1; adj+I, n: n-hop area
        self.recfield = recfield
        # b parameter
        self.dv = dv
        # number of sampled point of spectrum
        self.nfreq = nfreq
        # if degree is added to node feature
        self.adddegree = adddegree
        # use laplacian or adjacency for spectrum
        self.laplacian = laplacian
        # add adjacecny as edge feature
        self.addadj = addadj
        # use given max eigenvalue
        self.vmax = vmax
        # max node for PPGN algorithm, set 0 if you do not use PPGN
        self.nmax = nmax

    def __call__(self, data):
        n = data.x.shape[0]
        nf = data.x.shape[1]

        data.x = data.x.type(torch.float32)

        nsup = self.nfreq + 1
        if self.addadj:
            nsup += 1

        A = np.zeros((n, n), dtype=np.float32)
        SP = np.zeros((nsup, n, n), dtype=np.float32)
        A[data.edge_index[0], data.edge_index[1]] = 1

        if self.adddegree:
            data.x = torch.cat(
                [data.x, torch.tensor(A.sum(0)).unsqueeze(-1)], 1)

        # calculate receptive field. 0: adj, 1; adj+I, n: n-hop area
        if self.recfield == 0:
            M = A
        else:
            M = A + np.eye(n)
            for i in range(1, self.recfield):
                M = M.dot(M)

        M = M > 0

        d = A.sum(axis=0)
        # normalized Laplacian matrix.
        dis = 1 / np.sqrt(d)
        dis[np.isinf(dis)] = 0
        dis[np.isnan(dis)] = 0
        D = np.diag(dis)
        nL = np.eye(D.shape[0]) - (A.dot(D)).T.dot(D)
        V, U = np.linalg.eigh(nL)
        V[V < 0] = 0
        # keep maximum eigenvalue for Chebnet if it is needed
        data.lmax = V.max().astype(np.float32)

        if not self.laplacian:
            V, U = np.linalg.eigh(A)

        # design convolution supports
        vmax = self.vmax
        if vmax is None:
            vmax = V.max()

        freqcenter = np.linspace(V.min(), vmax, self.nfreq)

        # design convolution supports (aka edge features)
        for i, fc in enumerate(freqcenter):
            SP[i, :, :] = M * \
                U.dot(np.diag(np.exp(-(self.dv * (V - fc) ** 2))).dot(U.T))
        # add identity
        SP[len(freqcenter), :, :] = np.eye(n)
        # add adjacency if it is desired
        if self.addadj:
            SP[len(freqcenter) + 1, :, :] = A

        # set convolution support weigths as an edge feature
        E = np.where(M > 0)
        data.edge_index2 = torch.Tensor(
            np.vstack((E[0], E[1]))).type(torch.int64)
        data.edge_attr2 = torch.Tensor(SP[:, E[0], E[1]].T).type(torch.float32)

        # set tensor for PPGN
        if self.nmax > 0:
            H = torch.zeros(1, nf + 2, self.nmax, self.nmax)
            H[0, 0, data.edge_index[0], data.edge_index[1]] = 1
            H[0, 1, 0:n, 0:n] = torch.diag(torch.ones(data.x.shape[0]))
            for j in range(0, nf):
                H[0, j + 2, 0:n, 0:n] = torch.diag(data.x[:, j])
            data.X2 = H
            M = torch.zeros(1, 2, self.nmax, self.nmax)
            for i in range(0, n):
                M[0, 0, i, i] = 1
            M[0, 1, 0:n, 0:n] = 1 - M[0, 0, 0:n, 0:n]
            data.M = M

        return data


def generate_perm(perm_idx):
    perm = [np.random.permutation(perm) for perm in perm_idx]
    return perm


def generate_A(edge_index, n):
    A = np.zeros((n, n), dtype=np.float32)
    A[edge_index[0], edge_index[1]] = 1
    return torch.from_numpy(A.flatten())


def edge_index_to_adjacency(edge_index, n):
    A = np.zeros((n, n), dtype=np.float32)
    A[edge_index[0], edge_index[1]] = 1
    return torch.from_numpy(A)


def adjacency_to_edge_index(A):
    n = A.shape[0]
    mask_entries = A == 1
    e = mask_entries.sum().item()

    arr0 = torch.arange(0, n).unsqueeze(-1).expand(-1, n)  # [n, n]
    arr1 = torch.arange(0, n).unsqueeze(0).expand(n, -1)  # [n, n]
    arr2 = torch.cat((arr0.unsqueeze(0), arr1.unsqueeze(0)), dim=0)  # [2, n, n]
    result = torch.zeros(2, e, dtype=A.dtype, device=A.device)  # [2, e]
    result[0] = arr2[0][mask_entries]
    result[1] = arr2[1][mask_entries]
    return result


def arr2perm(arr, device=None):
    n = arr.shape[0]
    I = torch.eye(n, dtype=torch.float32, device=device)
    result = I[arr]
    return result


def sort_fn_laplacian(x, edge_index):
    # construct laplacian
    L_e, L_w = get_laplacian(edge_index)
    L = np.zeros((x.shape[0], x.shape[0]), dtype=np.float32)
    L[L_e[0], L_e[1]] = L_w

    # compute eigen decomposition of Laplacian, evals are returned in ascending order
    evals, evecs = np.linalg.eigh(L)

    # ----- create sorting criterion -----
    unique_vals, evals_idx, evals_mult = np.unique(
        evals, return_counts=True, return_index=True
    )  # get eigenvals multiplicity

    chosen_evecs = []
    for eigen_space_start_idx, eigen_space_size in zip(evals_idx, evals_mult):
        if eigen_space_size == 1:
            chosen_evecs.append(np.abs(evecs[:, eigen_space_start_idx]))
        else:
            eig_space_basis = evecs[:, eigen_space_start_idx: (
                eigen_space_start_idx + eigen_space_size)]
            chosen_evecs.append(np.sqrt((eig_space_basis**2).sum(1)))

    chosen_evecs = np.stack(chosen_evecs, axis=1).round(decimals=2)
    sort_idx = np.lexsort(
        [col for col in chosen_evecs.transpose()[::-1]])  # consider regular sort
    return sort_idx, chosen_evecs


class PrecomputeSortFrame():
    def __init__(self, pre_transform, device, sort_fn=sort_fn_laplacian):
        self.pre_transform = pre_transform
        self.sort_fn = sort_fn
        self.device = device

    def __call__(self, data):
        data = self.pre_transform(data)
        sort_idx, to_sort = self.sort_fn(data.x, data.edge_index)
        sorted_x = to_sort[sort_idx, :]
        unique_rows, dup_rows_idx, dup_rows_mult = np.unique(
            sorted_x, axis=0, return_index=True, return_counts=True)

        perm_start_idx = dup_rows_idx[dup_rows_mult != 1]
        perm_size = dup_rows_mult[dup_rows_mult != 1]
        perm_idx = [np.arange(start, start + size)
                    for start, size in zip(perm_start_idx, perm_size)]
        data.perm_idx = perm_idx
        data.sort_idx = sort_idx
        data.size = data.x.shape[0]
        return data


class SampleFrame():
    def __init__(self, size=64, sample_size=10, GA=False, MLP=False):
        self.sample_size = sample_size
        self.size = size
        self.GA = GA
        self.id = not MLP
        self.MLP = MLP

    def apply_permutation(self, perm_idx, edge_index, x):
        inv_perm_idx = np.zeros_like(perm_idx)
        inv_perm_idx[perm_idx] = np.arange(perm_idx.shape[0])
        sorted_edge_index = torch.tensor(inv_perm_idx[edge_index])
        sorted_x = x[perm_idx, :]
        return sorted_edge_index, sorted_x

    def permute_with_perm(self, perm_idx, perm, sort_idx, edge_index, x):
        inv_sort_idx = np.zeros_like(sort_idx)
        inv_sort_idx[sort_idx] = np.arange(sort_idx.shape[0])
        inv_sort_idx[sort_idx[list(itertools.chain(*perm_idx))]] = list(
            itertools.chain(*perm))
        sorted_edge_index = inv_sort_idx[edge_index]
        cur_sort_idx = np.zeros_like(sort_idx)
        cur_sort_idx[inv_sort_idx] = np.arange(sort_idx.shape[0])
        sorted_x_feat = x[cur_sort_idx, :]
        return sorted_edge_index, sorted_x_feat

    def permute_with_perm2(self, perm_idx, perm, sort_idx, edge_index, x):
        inv_sort_idx = np.zeros_like(sort_idx)
        inv_sort_idx[sort_idx] = np.arange(sort_idx.shape[0])
        inv_sort_idx[sort_idx[list(itertools.chain(*perm_idx))]] = list(
            itertools.chain(*perm))
        cur_sort_idx = np.zeros_like(sort_idx)
        cur_sort_idx[inv_sort_idx] = np.arange(sort_idx.shape[0])

        P = arr2perm(cur_sort_idx, device=x.device)
        sorted_x = torch.matmul(P, x)
        A = edge_index_to_adjacency(edge_index, x.size(0))
        sorted_A = torch.matmul(torch.matmul(P, A), P.transpose(0, 1))
        sorted_edge_index = adjacency_to_edge_index(sorted_A)
        return sorted_edge_index, sorted_x

    def __call__(self, data):
        n, d = data.x.shape[0], data.x.shape[1]
        m = self.size
        x = data.x
        edge_index = data.edge_index  # [2, e]

        # edge_index is directed, that means (a,b) and (b,a) are both included.
        sort_idx = data.sort_idx
        perm_idx = data.perm_idx
        x_arr = []
        e_arr = []
        if not perm_idx:
            if self.GA:
                sorted_edge_index, sorted_x = self.apply_permutation(
                    np.random.permutation(n), edge_index.clone(), x.clone())
            else:
                sorted_edge_index, sorted_x = self.apply_permutation(
                    sort_idx, edge_index.clone(), x.clone())
                data.edge_index = sorted_edge_index.detach()

            if self.MLP:
                new_x = torch.cat(
                    [sorted_x.flatten(), torch.zeros((m - n, d), dtype=x.dtype),
                     generate_A(sorted_edge_index, m)])
                data = new_x, data.y
            else:
                if self.id:
                    data.x = torch.cat(
                        [sorted_x.detach(), torch.eye(n, dtype=x.dtype),
                         torch.zeros((n, m - n), dtype=x.dtype)], 1).clone()
                else:
                    data.x = sorted_x.detach()

                data.edge_index = sorted_edge_index

        else:
            for i in range(self.sample_size):
                if self.GA:
                    sorted_edge_index, sorted_x = self.apply_permutation(
                        np.random.permutation(n), edge_index.clone(), x.clone())
                else:
                    perm = generate_perm(perm_idx)
                    sorted_edge_index, sorted_x = self.permute_with_perm(
                        perm_idx, perm, sort_idx, edge_index.clone(), x.clone())
                    sorted_edge_index = torch.from_numpy(sorted_edge_index)
                    # sort_x for different i are identical

                if self.MLP:
                    x_arr.append(
                        torch.cat(
                            [sorted_x.flatten(),
                             torch.zeros((m - n, d), dtype=x.dtype).flatten(),
                             generate_A(sorted_edge_index, m)]))
                    # x_arr[-1].shape = m*(m+d)
                    # sorted_x: n*d, torch.zeros: (m-n)*d, generate_A: m*m
                else:
                    if self.id:
                        x_arr.append(
                            torch.cat(
                                [sorted_x.detach(), torch.eye(n, dtype=x.dtype),
                                 torch.zeros((n, m - n), dtype=x.dtype)], 1).clone())
                    else:
                        x_arr.append(sorted_x.detach())

                    e_arr.append(torch.tensor(
                        sorted_edge_index.clone()) + (i * n))

            if self.MLP:
                data = torch.stack(x_arr, dim=0), data.y
            else:
                data.x = torch.cat(x_arr, 0).detach()
                data.edge_index = torch.cat(e_arr, 1).detach()

        return data


class SampleFrame8C():
    def __init__(self, size=64, sample_size=10, GA=False, MLP=False):
        self.sample_size = sample_size
        self.size = size
        self.GA = GA
        self.id = not MLP
        self.MLP = MLP

    def apply_permutation(self, perm_idx, edge_index, x):
        inv_perm_idx = np.zeros_like(perm_idx)
        inv_perm_idx[perm_idx] = np.arange(perm_idx.shape[0])
        sorted_edge_index = torch.tensor(inv_perm_idx[edge_index])
        sorted_x = x[perm_idx, :]
        return sorted_edge_index, sorted_x

    def permute_with_perm(self, perm_idx, perm, sort_idx, edge_index, x):
        inv_sort_idx = np.zeros_like(sort_idx)
        inv_sort_idx[sort_idx] = np.arange(sort_idx.shape[0])
        inv_sort_idx[sort_idx[list(itertools.chain(*perm_idx))]] = list(
            itertools.chain(*perm)
        )
        sorted_edge_index = inv_sort_idx[edge_index]
        cur_sort_idx = np.zeros_like(sort_idx)
        cur_sort_idx[inv_sort_idx] = np.arange(sort_idx.shape[0])

        sorted_x_feat = x[cur_sort_idx, :]

        return sorted_edge_index, sorted_x_feat

    def __call__(self, data):
        n = data.x.shape[0]
        m = self.size
        x = data.x
        edge_index = data.edge_index
        sort_idx = data.sort_idx
        perm_idx = data.perm_idx
        x_arr = []
        e_arr = []

        if not perm_idx:
            if self.GA:
                sorted_edge_index, sorted_x = self.apply_permutation(
                    np.random.permutation(n), edge_index.clone(), x.clone())
            else:
                sorted_edge_index, sorted_x = self.apply_permutation(
                    sort_idx, edge_index.clone(), x.clone())
                data.edge_index = sorted_edge_index.detach()

            if self.MLP:
                new_x = generate_A(sorted_edge_index, m).unsqueeze(0)
                data = new_x, data.y
            else:
                if self.id:
                    data.x = torch.cat(
                        [sorted_x.detach(), torch.eye(n, dtype=x.dtype),
                         torch.zeros((n, m - n), dtype=x.dtype)], 1).clone()
                else:
                    data.x = sorted_x.detach()
                data.edge_index = sorted_edge_index

        else:
            for i in range(self.sample_size):
                if self.GA:
                    sorted_edge_index, sorted_x = self.apply_permutation(
                        np.random.permutation(n), edge_index.clone(), x.clone())
                else:
                    perm = generate_perm(perm_idx)
                    sorted_edge_index, sorted_x = self.permute_with_perm(
                        perm_idx, perm, sort_idx, edge_index.clone(), x.clone())
                    sorted_edge_index = torch.from_numpy(sorted_edge_index)

                if self.MLP:
                    x_arr.append(generate_A(sorted_edge_index, m))
                else:
                    if self.id:
                        x_arr.append(
                            torch.cat(
                                [sorted_x.detach(), torch.eye(n, dtype=x.dtype),
                                 torch.zeros((n, m - n), dtype=x.dtype)], 1).clone())
                    else:
                        x_arr.append(sorted_x.detach())

                    e_arr.append(torch.tensor(
                        sorted_edge_index.clone()) + (i * n))

            if self.MLP:
                data = torch.stack(x_arr, dim=0), data.y
            else:
                data.x = torch.cat(x_arr, 0).detach()
                data.edge_index = torch.cat(e_arr, 1).detach()

        return data


class PrecomputePad():
    def __init__(self, nmax=64):
        self.nmax = nmax

    def __call__(self, data):
        x = data.x
        y = data.y
        idx = data.idx
        edge_index = data.edge_index  # [2, e]
        n, d = x.shape
        e = edge_index.size(-1)
        # pad data to nmax
        pad_x = torch.zeros((1, self.nmax, d), dtype=x.dtype, device=x.device)
        pad_x[0, :n] = x
        pad_adj = edge_index_to_adjacency(edge_index, self.nmax)[None, :, :]
        pad_edge_idx = torch.full((1, 2, self.nmax * self.nmax), -1,
                                  dtype=edge_index.dtype, device=edge_index.device)
        pad_edge_idx[0, :, :e] = edge_index
        # return padded data
        new_data = Data(
            x=x,
            idx=idx,
            edge_index=edge_index,
            pad_x=pad_x,
            pad_edge_idx=pad_edge_idx,
            pad_adj=pad_adj,
            n=n,
            e=e,
            y=y
        )
        return new_data
