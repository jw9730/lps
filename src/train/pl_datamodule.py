# pylint: disable=too-many-instance-attributes,disable=too-many-arguments,unused-argument
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch_geometric.data.lightning

from src.data import DatasetBuilder


class LitDataModule(pl.LightningDataModule):
    def __init__(
            self,
            ds_builder: DatasetBuilder,
            global_batch_size: int,
            devices: int,
            num_workers: int,
            verbose: bool=True
        ):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.ds_builder = ds_builder
        self.batch_size = global_batch_size // devices
        self.num_workers = num_workers

        self.verbose = verbose

    def prepare_data(self):
        """Download data, split, etc. Only called on 1 GPU/TPU in distributed."""
        self.ds_builder.prepare_data()

    def setup(self, stage: str):
        """Make assignments here (val/train/test split). Called on every GPU/TPU in DDP."""

        # assign train/val split(s) for use in Dataloaders
        if stage in ('fit', 'validate'):
            self.train_dataset = self.ds_builder.train_dataset()
            self.val_dataset = self.ds_builder.val_dataset()

        # assign test split(s) for use in Dataloaders
        if stage == 'test':
            self.test_dataset = self.ds_builder.test_dataset()

        # assign predict split(s) for use in Dataloaders
        if stage == 'predict':
            self.predict_dataset = self.ds_builder.predict_dataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=1
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=1
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=1
        )


@pl.utilities.rank_zero_only
def prepare_pyg_data(ds_builder: DatasetBuilder):
    """Prepare PyTorch Geometric data"""
    ds_builder.prepare_data()


def setup_pyg_datamodule(
        ds_builder: DatasetBuilder,
        global_batch_size: int,
        devices: int,
        num_workers: int,
        verbose: bool=True
    ):
    """Setup PyTorch Geometric datamodule"""
    prepare_pyg_data(ds_builder=ds_builder)
    train_dataset = ds_builder.train_dataset()
    val_dataset = ds_builder.val_dataset()
    test_dataset = ds_builder.test_dataset()
    pred_dataset = ds_builder.predict_dataset()
    batch_size = global_batch_size // devices
    datamodule = torch_geometric.data.lightning.LightningDataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        pred_dataset=pred_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return datamodule
