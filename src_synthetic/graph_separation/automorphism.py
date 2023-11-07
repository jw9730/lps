# pylint: disable=not-callable,line-too-long,no-value-for-parameter
import os
from pathlib import Path
import random
import numpy as np
from matplotlib import pyplot as plt, colors, cm
import networkx as nx
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

from args import get_args
from preprocess import PrecomputePad
from interface import InterfacedModel


MAX_NODES = 25


def to_edge_index(edge_list):
    return to_undirected(torch.tensor(edge_list, dtype=torch.long).t())


def data_trivial():
    edge_index = to_edge_index([[0, 1], [1, 2], [2, 0], [2, 3], [1, 4], [4, 5], [2, 5], [5, 6]])
    node_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6])[:, None]
    return Data(idx=0, x=torch.zeros(7, 1), edge_index=edge_index, y=node_labels)


def data_s6():
    edge_index = to_edge_index([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]])
    node_labels = torch.tensor([0, 1, 1, 1, 1, 1, 1])[:, None]
    return Data(idx=1, x=torch.zeros(7, 1), edge_index=edge_index, y=node_labels)


def data_s3xs2():
    edge_index = to_edge_index([[0, 1], [0, 2], [0, 3], [4, 5], [4, 6], [0, 4]])
    node_labels = torch.tensor([0, 1, 1, 1, 2, 3, 3])[:, None]
    return Data(idx=2, x=torch.zeros(7, 1), edge_index=edge_index, y=node_labels)


def data_d7():
    edge_index = to_edge_index([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0]])
    node_labels = torch.tensor([0, 0, 0, 0, 0, 0, 0])[:, None]
    return Data(idx=3, x=torch.zeros(7, 1), edge_index=edge_index, y=node_labels)


def data_d4():
    G = nx.grid_2d_graph(5, 5)
    nodes = list(G.nodes())
    edges = list(G.edges())
    edge_index = to_edge_index([[nodes.index(u), nodes.index(v)] for u, v in edges])
    pos = torch.tensor(nodes)
    node_labels = (pos - 2).pow(2).sum(-1, keepdim=True)
    return Data(idx=4, x=torch.zeros(25, 1), edge_index=edge_index, y=node_labels)


def draw_graph(data, fname, label, vmin, vmax):
    result_dir = Path('experiments/automorphism_results')
    result_dir.mkdir(exist_ok=True, parents=True)
    G = nx.Graph()
    G.add_nodes_from(list(range(data.num_nodes)))
    G.add_edges_from(data.edge_index.t().tolist())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap='rainbow')
    node_color = [mapper.to_rgba(val) for val in label]
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(4, 4))
    node_size = 1000 if data.num_nodes <= 10 else 500
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color=node_color, edgecolors='black')
    plt.colorbar(mapper, ax=plt.gca(), shrink=0.5)
    plt.axis('off')
    plt.savefig(result_dir / f'{fname}.pdf', bbox_inches='tight')
    plt.close()


def main(args):
    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    # configure device
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

    # configure data
    transform = PrecomputePad(nmax=MAX_NODES)
    data_list = [data_trivial(), data_s6(), data_s3xs2(), data_d7(), data_d4()]
    labels_list = [data.y[:, 0].tolist() for data in data_list]
    for data, label in zip(data_list, labels_list):
        vmin, vmax = min(label), max(label)
        draw_graph(data, f'data_{data.idx}_label', label, vmin, vmax)
    batched_data = Batch.from_data_list([transform(data) for data in data_list]).to(device)

    # configure model
    model = InterfacedModel(
        n=MAX_NODES,
        d=1,
        num_interface_layers=args.num_interface_layers,
        backbone=args.backbone,
        noise_scale=args.noise_scale,
        tau=args.tau,
        hard=args.hard,
        task='automorphism'
    ).to(device)
    model.eval()

    # run test
    emb_1, _ = model(batched_data, n_samples=args.eval_sample_size)
    labels_list_1 = emb_1[:, :, 0].cpu().tolist()
    model.interface.noise_scale = 0
    emb_2, _ = model(batched_data, n_samples=1)
    labels_list_2 = emb_2[:, :, 0].cpu().tolist()
    for data, label_1, label_2 in zip(data_list, labels_list_1, labels_list_2):
        label_1 = label_1[:data.num_nodes]
        label_2 = label_2[:data.num_nodes]
        vmin, vmax = min(label_1), max(label_1)
        draw_graph(data, f'data_{data.idx}_embed_ps', label_1, vmin, vmax)
        vmin, vmax = min(label_2), max(label_2)
        draw_graph(data, f'data_{data.idx}_embed_canonical', label_2, vmin, vmax)
    print("Done, saved results to automorphism_results/")


if __name__ == '__main__':
    args_ = get_args()
    main(args_)
