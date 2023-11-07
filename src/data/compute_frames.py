from tqdm import tqdm
import numpy as np
import torch
import torch_geometric


def load_graphs(filepath): 
    data_obj, dict_slice = torch.load(filepath)

    if 'pos' in dict_slice:
        slice_node = dict_slice['pos']
    else:
        slice_node = dict_slice['x']
    slice_edge_index = dict_slice['edge_index']
    num_node = (slice_node[1:] - slice_node[:-1]).tolist()
    num_edge_index = (slice_edge_index[1:] - slice_edge_index[:-1]).tolist()

    assert sum(num_edge_index) == data_obj.edge_index.shape[1]

    list_edge_index = torch.split(data_obj.edge_index, num_edge_index, dim=1)
    assert len(num_node) == len(list_edge_index)

    dict_graphs = {"Ns": num_node, "edge_indexs": list_edge_index, "slice_node": slice_node}
    return dict_graphs


def sort_fn_laplacian(N, edge_index):
    # N: number of nodes
    # construct laplacian
    L_e, L_w = torch_geometric.utils.get_laplacian(edge_index)
    L = np.zeros((N,N),dtype=np.float32)
    L[L_e[0],L_e[1]]=L_w

    # compute eigen decomposition of Laplacian, evals are returned in ascending order
    evals, evecs = np.linalg.eigh(L)

    # ----- create sorting criterion -----
    unique_vals, evals_idx, evals_mult = np.unique(evals, return_counts=True, return_index=True) # get eigenvals multiplicity

    chosen_evecs = []
    len_evals_idx = len(evals_idx)
    for ii in range(len_evals_idx):
        if evals_mult[ii] == 1:
            chosen_evecs.append(np.abs(evecs[:,evals_idx[ii]]))
        else:
            eigen_space_start_idx = evals_idx[ii]
            eigen_space_size = evals_mult[ii]
            eig_space_basis = evecs[:, eigen_space_start_idx:(eigen_space_start_idx+eigen_space_size)] # (52, 2)
            chosen_evecs.append(np.sqrt((eig_space_basis ** 2).sum(1))) # (52,)
    chosen_evecs = np.stack(chosen_evecs, axis=1).round(decimals=2)  # (52, 37), it's the matrix S(X) in section 3.2
    sort_idx = np.lexsort([col for col in chosen_evecs.transpose()[::-1]]) # consider regular sort, there are 37 elements in this list
    return sort_idx, chosen_evecs

def compute_frame(N, edge_index):
    """
    N: number of nodes
    edge_index: [2, 4884]
    edge_index = tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, ...],
                         [ 6,  9, 11, 12, 13, 16, 27, 35, 37, 43, ...]])"""
    e = edge_index.shape[1]
    sort_idx, to_sort = sort_fn_laplacian(N, edge_index) # to_sort is S(X) in section 3.2 
    # sort_idx = [ 96  40  65  55   2 ... 73   0  11]
    sorted_x = to_sort[sort_idx,:]
    unique_rows, dup_rows_idx, dup_rows_mult = np.unique(sorted_x, axis=0, return_index=True, return_counts=True) # return unique elements in sorted order
    # dup_rows_idx =  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 18 19 20 22 23 25 27 28 34 46]
    # dup_rows_mult =  [ 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  2  1  1  2  1  2  2  1 6 12  6]
    
    perm_idx = torch.repeat_interleave(torch.tensor(dup_rows_idx), torch.tensor(dup_rows_mult)) 
    # perm_idx =  tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 16, 18, 19, 20, 20, 22, 23, 23, 25, 25, 27, 28, 28, 28, 28, 28, 28, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 46, 46, 46, 46, 46, 46])
    # create binary mask perm 
    mask_perm = dup_rows_mult != 1 
    mask_perm = torch.from_numpy(mask_perm)
    mask_perm = torch.repeat_interleave(mask_perm, torch.tensor(dup_rows_mult))
    sort_idx = torch.tensor(sort_idx)
    
    return sort_idx, perm_idx, mask_perm


def compute_all_frames(data_filepath):
    dict_graphs = load_graphs(data_filepath)
    num_graphs = len(dict_graphs["Ns"])
    list_sort_idx = []
    list_perm_idx = []
    list_mask_perm = []
    for i in tqdm(range(num_graphs)): 
        N = dict_graphs["Ns"][i]
        edge_index = dict_graphs["edge_indexs"][i]
        sort_idx, perm_idx, mask_perm = compute_frame(N, edge_index)
        list_sort_idx.append(sort_idx)
        list_perm_idx.append(perm_idx)
        list_mask_perm.append(mask_perm)

    sort_idxs = torch.cat(list_sort_idx, dim=0)
    perm_idxs = torch.cat(list_perm_idx, dim=0)
    mask_perms = torch.cat(list_mask_perm, dim=0)

    slice_sort_idxs = dict_graphs["slice_node"].clone()
    slice_perm_idxs = dict_graphs["slice_node"].clone()
    slice_mask_perms = dict_graphs["slice_node"].clone()

    frame_data = {
        "sort_idxs": sort_idxs,
        "perm_idxs": perm_idxs,
        "mask_perms": mask_perms,
        "slice": {
            "sort_idxs": slice_sort_idxs,
            "perm_idxs": slice_perm_idxs,
            "mask_perms": slice_mask_perms
        }
    }
    return frame_data
