import pickle
from pathlib import Path
import networkx as nx
import torch
import torch_geometric.data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_undirected


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['GRAPHSAT.pkl']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    @staticmethod
    def parse_old_version_data(data):
        return torch_geometric.data.Data(
            x=data.__dict__['x'],
            edge_index=data.__dict__['edge_index'],
            y=data.__dict__['y'],
        )

    def process(self):
        # Read data into huge `Data` list.
        with open(Path(self.root) / 'raw/GRAPHSAT.pkl', 'rb') as f:
            data_list = pickle.load(f)

        # Parse old PyG version data
        keys = data_list[0].__dict__.keys()
        for k in keys:
            if k not in ('x', 'edge_index', 'y'):
                assert all(d.__dict__[k] is None for d in data_list)
        data_list = [self.parse_old_version_data(data) for data in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # add idx for fixing noise
        for idx, data in enumerate(data_list):
            data.idx = idx

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GRAPH8cDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['graph8c.g6']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for datum in dataset:
            x = torch.ones(datum.number_of_nodes(), 1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1, 0))
            data_list.append(Data(edge_index=edge_index, x=x, y=0))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # add idx for fixing noise
        for idx, data in enumerate(data_list):
            data.idx = idx

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
