# pylint: disable=too-many-return-statements,too-many-branches,line-too-long
from functools import partial
from pathlib import Path
import pickle
from torch.utils.data import Dataset
import torch_geometric.datasets

from .node_classification_pattern import NodeClassificationPATTERN
from .graph_classification_peptides_func import GraphClassificationPeptidesfunc
from .graph_regression_peptides_struct import GraphRegressionPeptidesstruct
from .link_prediction_pcqm_contact import LinkPredictionPCQMContact

from .compute_frames import compute_all_frames


def setup_symmetry(dataset: str, config):
    if dataset == "gnn_benchmark/pattern":
        return NodeClassificationPATTERN(config)
    if dataset == "lrgb/pcqm_contact":
        return LinkPredictionPCQMContact(config)
    if dataset == "lrgb/peptides_func":
        return GraphClassificationPeptidesfunc(config)
    if dataset == "lrgb/peptides_struct":
        return GraphRegressionPeptidesstruct(config)
    raise NotImplementedError(f"Dataset ({dataset}) not supported!")


class DatasetBuilder():
    """Dataset configuration class"""
    def __init__(
            self,
            dataset: str,
            root_dir: str,
            compute_frames: bool
        ):
        self.root_dir = root_dir
        self.compute_frames = compute_frames
        # pyg datasets
        self.is_pyg_dataset = True
        if dataset == "gnn_benchmark/pattern":
            self.ds_builder = partial(torch_geometric.datasets.GNNBenchmarkDataset, name="PATTERN")
        elif dataset == "lrgb/pcqm_contact":
            self.ds_builder = partial(torch_geometric.datasets.LRGBDataset, name="PCQM-Contact")
        elif dataset == "lrgb/peptides_func":
            self.ds_builder = partial(torch_geometric.datasets.LRGBDataset, name="Peptides-func")
        elif dataset == "lrgb/peptides_struct":
            self.ds_builder = partial(torch_geometric.datasets.LRGBDataset, name="Peptides-struct")
        else:
            # non-pyg datasets
            self.is_pyg_dataset = False
            raise NotImplementedError(f"Dataset ({dataset}) not supported!")

    def setup_frames(self, data_name, split):
        assert split in ['train', 'val', 'test']
        frame_filepath = Path(self.root_dir) / data_name / f"processed/{split}_frame.pickle"
        if not frame_filepath.exists():
            print(f"Computing frames for the {split} dataset at {frame_filepath}...")
            data_filepath = Path(self.root_dir) / data_name / f"processed/{split}_data.pt"
            if not data_filepath.exists():
                data_filepath = Path(self.root_dir) / data_name / f"processed/{split}.pt"
            assert data_filepath.exists()
            frame_data = compute_all_frames(data_filepath)
            with open(frame_filepath, "wb") as f:
                pickle.dump(frame_data, f)

    def prepare_data(self):
        if self.is_pyg_dataset:
            ds_builder = self.ds_builder(self.root_dir)
            if self.compute_frames:
                data_name = getattr(ds_builder, 'name')
                self.setup_frames(data_name, 'train')
                self.setup_frames(data_name, 'val')
                self.setup_frames(data_name, 'test')
                print("Done!")
        else:
            raise NotImplementedError

    def load_frames(self, ds_builder, data_name, split):
        frame_filepath = Path(self.root_dir) / data_name / f"processed/{split}_frame.pickle"
        with open(frame_filepath, "rb") as f:
            frame_dict = pickle.load(f)
        sort_idxs = frame_dict["sort_idxs"]
        perm_idxs = frame_dict["perm_idxs"]
        mask_perms = frame_dict["mask_perms"]
        slice_sort_idxs = frame_dict["slice"]["sort_idxs"]
        slice_perm_idxs = frame_dict["slice"]["perm_idxs"]
        slice_mask_perms = frame_dict["slice"]["mask_perms"]
        _data = getattr(ds_builder, "_data")
        slices = getattr(ds_builder, "slices")
        _data.sort_idx = sort_idxs
        _data.perm_idx = perm_idxs
        _data.mask_perm = mask_perms
        slices["sort_idx"] = slice_sort_idxs
        slices["perm_idx"] = slice_perm_idxs
        slices["mask_perm"] = slice_mask_perms
        return ds_builder

    def train_dataset(self) -> Dataset:
        if self.is_pyg_dataset:
            ds_builder = self.ds_builder(self.root_dir, split='train')
            data_name = getattr(ds_builder, "name")
            if self.compute_frames:
                ds_builder = self.load_frames(ds_builder, data_name, 'train')
            return ds_builder
        raise NotImplementedError

    def val_dataset(self) -> Dataset:
        if self.is_pyg_dataset:
            ds_builder = self.ds_builder(self.root_dir, split='val')
            data_name = getattr(ds_builder, "name")
            if self.compute_frames:
                ds_builder = self.load_frames(ds_builder, data_name, 'val')
            return ds_builder
        raise NotImplementedError

    def test_dataset(self)  -> Dataset:
        if self.is_pyg_dataset:
            ds_builder = self.ds_builder(self.root_dir, split='test')
            data_name = getattr(ds_builder, "name")
            if self.compute_frames:
                ds_builder = self.load_frames(ds_builder, data_name, 'test')
            return ds_builder
        raise NotImplementedError

    def predict_dataset(self) -> Dataset:
        if self.is_pyg_dataset:
            return NotImplemented
        raise NotImplementedError
