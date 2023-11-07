# pylint:disable=line-too-long
from pathlib import Path
import torch

from src.data import DatasetBuilder, setup_symmetry
from .pl_datamodule import LitDataModule, setup_pyg_datamodule


def configure_data(config, verbose=True):
    # setup data directory
    data_dir = setup_data_directory(
        root_dir=config.root_dir,
        data_dir=config.data_dir
    )
    # setup dataset
    compute_frames = config.interface == 'frame'
    ds_builder = DatasetBuilder(
        root_dir=data_dir,
        dataset=config.dataset,
        compute_frames=compute_frames
    )
    # setup lightning datamodule
    global_batch_size = config.global_batch_size
    devices = torch.cuda.device_count() if config.accelerator == 'gpu' else 1
    if config.test_mode:
        global_batch_size = config.global_batch_size // devices
        devices = 1
    datamodule = setup_datamodule(
        ds_builder=ds_builder,
        global_batch_size=global_batch_size,
        devices=devices,
        num_workers=config.num_workers,
        verbose=verbose
    )
    # setup data symmetry
    symmetry = setup_symmetry(
        dataset=config.dataset,
        config=config,
    )
    if verbose:
        print(symmetry)
    return datamodule, symmetry


def setup_data_directory(root_dir='experiments', data_dir='data'):
    data_dir = Path(root_dir) / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir.as_posix()


def setup_datamodule(ds_builder: DatasetBuilder, global_batch_size, devices, num_workers, verbose=True):
    dm_builder = setup_pyg_datamodule if ds_builder.is_pyg_dataset else LitDataModule
    datamodule = dm_builder(
        ds_builder=ds_builder,
        global_batch_size=global_batch_size,
        devices=devices,
        num_workers=num_workers,
        verbose=verbose
    )
    return datamodule
