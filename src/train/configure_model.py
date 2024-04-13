# pylint:disable=too-many-arguments,line-too-long
from pathlib import Path
import torch

from src.model import Backbone, InterfacedModel
from src.optim import OptimizerConfig, LRSchedulerConfig
from .pl_module import LitModule


def configure_model(config, symmetry, verbose=True):
    # setup model
    backbone = Backbone(
        backbone=config.backbone,
        pretrained=getattr(config, 'pretrained', True),
        patch_dropout=getattr(config, 'patch_dropout', None),
        max_patch_dropout=getattr(config, 'max_patch_dropout', None),
    )
    model = InterfacedModel(
        backbone=backbone,
        symmetry=symmetry,
        interface=config.interface,
        centering=getattr(config, 'centering', False),
        pad_mode=getattr(config, 'pad_mode', 'zero'),
    )
    # setup optimizer and lr scheduler
    optimizer_config = OptimizerConfig(
        optimizer=config.optimizer,
        weight_decay=config.weight_decay,
        lr_pretrained=config.lr_pretrained,
        lr=config.lr
    )
    lr_scheduler_config = LRSchedulerConfig(
        lr_schedule=config.lr_schedule,
        n_steps=config.n_steps,
        lr_pretrained=config.lr_pretrained,
        lr=config.lr,
        lr_warmup=config.lr_warmup,
        lr_warmup_scale=config.lr_warmup_scale,
        lr_decay_degree=config.lr_decay_degree
    )
    # setup lightning model
    model = LitModule(
        model=model,
        sample_size=config.sample_size,
        eval_sample_size=config.eval_sample_size if not config.test_mode else config.test_sample_size,
        optimizer_config=optimizer_config,
        lr_scheduler_config=lr_scheduler_config,
        verbose=verbose
    )
    if verbose:
        print(model)
    # setup and load trained ckeckpoint
    ckpt_path = setup_ckpt_path(
        load_dir=config.load_dir,
        dataset=config.dataset,
        exp_name=config.exp_name,
        resume_mode=config.resume_mode,
        test_mode=config.test_mode,
        test_ckpt_path=config.test_ckpt_path
    )
    model = load_ckpt(
        model=model,
        ckpt_path=ckpt_path,
        resume_mode=config.resume_mode,
        test_mode=config.test_mode
    )
    if verbose:
        print(f'trained checkpoint loaded from {ckpt_path}' if ckpt_path is not None \
              else 'no trained checkpoint loaded')
    return model, ckpt_path


def setup_ckpt_path(load_dir, dataset, exp_name, resume_mode=False, test_mode=False, test_ckpt_path=None):
    dataset = dataset.replace('/', '_')
    if resume_mode or test_mode:
        if resume_mode:
            assert test_ckpt_path is None, "test_ckpt_path must be None if resume_mode is True!"
            ckpt_name = 'last.ckpt'
        else:
            assert test_mode, "test_mode must be True if resume_mode is False!"
            if test_ckpt_path is not None:
                return test_ckpt_path
            # check the directory Path('experiments') / load_dir / exp_name
            # the best checkpoints have format 'best.ckpt', 'bestv1.ckpt', ...
            # select the latest one
            ckpt_dir = Path('experiments') / load_dir / dataset / exp_name
            ckpt_names = [ckpt.name for ckpt in ckpt_dir.iterdir() if ckpt.name.startswith('best')]
            if len(ckpt_names) > 1:
                ckpt_names.sort()
                version_postfix = ckpt_names[-1].split('best')[-1].split('.')[0]
            else:
                version_postfix = ''
            ckpt_name = f'best{version_postfix}.ckpt'
        ckpt_path = Path('experiments') / load_dir / dataset / exp_name / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint ({ckpt_path}) does not exist!")
        return ckpt_path.as_posix()
    return None


def load_ckpt(model, ckpt_path, resume_mode, test_mode):
    if resume_mode or test_mode:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
    return model
