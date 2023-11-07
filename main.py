# pylint: disable=line-too-long,no-member,protected-access
import os
import sys
import logging
import argparse
import yaml
from easydict import EasyDict as edict
import torch
import torch._dynamo.config
import pytorch_lightning as pl

from src.train import configure_data, configure_model, configure_experiment

torch._dynamo.config.log_level = logging.ERROR


def str2bool(v):
    if v in ('True', 'true'):
        return True
    if v in ('False', 'false'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # necessary arguments
    parser.add_argument('--config', '-cfg', type=str, default=None)
    parser.add_argument('--debug_mode', '-debug', default=False, action='store_true')
    parser.add_argument('--resume_mode', '-resume', default=False, action='store_true')
    parser.add_argument('--test_mode', '-test', default=False, action='store_true')
    parser.add_argument('--skip_mode', '-skip', default=False, action='store_true')
    parser.add_argument('--reset_mode', '-reset', default=False, action='store_true')
    parser.add_argument('--no_eval', '-ne', default=False, action='store_true')
    parser.add_argument('--no_save', '-ns', default=False, action='store_true')
    parser.add_argument('--test_ckpt_path', '-ckpt', type=str, default=None)

    # experiment arguments
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--test_seed', type=int, default=None)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--name_postfix', '-pf', type=str, default=None)
    parser.add_argument('--exp_subname', type=str, default='')

    # data arguments
    parser.add_argument('--dataset', '-ds', type=str, default=None)
    parser.add_argument('--strategy', '-str', type=str, default=None)
    parser.add_argument('--accelerator', '-acc', type=str, default=None)
    parser.add_argument('--num_workers', '-nw', type=int, default=None)
    parser.add_argument('--global_batch_size', '-gbs', type=int, default=None)
    parser.add_argument('--accumulate_grad_batches', '-agb', type=int, default=None)

    # model arguments
    parser.add_argument('--backbone', '-bb', type=str, default=None)
    parser.add_argument('--pretrained', '-pre', type=str2bool, default=None)

    # symmetry arguments
    parser.add_argument('--interface', '-io', type=str, default=None, choices=['unif', 'frame', 'prob'])
    parser.add_argument('--sample_size', '-sz', type=int, default=None)
    parser.add_argument('--eval_sample_size', '-esz', type=int, default=None)
    parser.add_argument('--test_sample_size', '-tsz', type=int, default=None)

    # probabilistic symmetrization arguments
    parser.add_argument('--hard', '-hrd', type=str2bool, default=None)

    # training arguments
    parser.add_argument('--n_steps', '-nst', type=int, default=None)
    parser.add_argument('--optimizer', '-opt', type=str, default=None, choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--gradient_clip_val', '-clip', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lr_pretrained', '-lrp', type=float, default=None)
    parser.add_argument('--lr_schedule', '-lrs', type=str, default=None, choices=['const', 'sqrt', 'cos', 'poly'])
    parser.add_argument('--early_stopping_monitor', '-esm', type=str, default=None)
    parser.add_argument('--early_stopping_mode', '-esd', type=str, default=None, choices=['min', 'max'])
    parser.add_argument('--early_stopping_patience', '-esp', type=int, default=None)

    # logging arguments
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--val_iter', '-viter', type=int, default=None)
    parser.add_argument('--save_iter', '-siter', type=int, default=None)

    return parser


def get_config() -> edict:
    # parse arguments
    parser = argparse.ArgumentParser(description='Probabilistic Symmetrization')
    parser = add_args(parser)
    args = parser.parse_args()

    # load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        config = edict(config)

    # update config with parsed arguments
    for k, v in vars(args).items():
        if v is not None:
            setattr(config, k, v)

    # create experiment name
    if config.exp_name == '':
        postfix = config.name_postfix if hasattr(config, 'name_postfix') else ''
        config.exp_name = f"pt_{config.pretrained}," \
            + f"k_{config.sample_size}," \
            + (f'eval_k_{config.eval_sample_size},' if config.sample_size != config.eval_sample_size else '')
        if config.interface == 'unif':
            config.exp_name += 'ga,'
        elif config.interface == 'frame':
            config.exp_name += 'fa,'
        elif config.interface == 'prob':
            config.exp_name += f"z_{config.noise_scale}," \
                + f"tau_{config.tau}," \
                + ('hard,' if config.hard else '') \
                + (f'l_{config.interface_num_layers},' if hasattr(config, 'interface_num_layers') else '') \
                + (f'd_{config.interface_hidden_dim},' if hasattr(config, 'interface_hidden_dim') else '') \
                + (f'drop_{config.interface_dropout},' if hasattr(config, 'interface_dropout') else '')
        else:
            raise NotImplementedError
        config.exp_name += f"b_{config.global_batch_size}{(f'x{config.accumulate_grad_batches}' if hasattr(config, 'accumulate_grad_batches') else '')}," \
            + f"es_{config.early_stopping_monitor.replace('/', '_')}_{config.early_stopping_mode}_{config.early_stopping_patience}," \
            + f"lr_{config.lr}_{config.lr_pretrained}," \
            + f"steps_{config.n_steps}," \
            + f"wu_{config.lr_warmup}," \
            + f"wd_{config.weight_decay}," \
            + (f'clip_{config.gradient_clip_val},' if hasattr(config, 'gradient_clip_val') else '') \
            + f"seed_{config.seed},"
        config.exp_name += postfix

    # create seed for testing
    if not hasattr(config, 'test_seed'):
        config.test_seed = config.seed

    # create checkpoint for testing
    if not hasattr(config, 'test_ckpt_path'):
        config.test_ckpt_path = None

    # create team name for wandb logging
    config.team_name = 'vl-kaist'

    # # this is a hack for debugging with visual studio code
    # config.debug_mode = True

    # setup debugging
    if config.debug_mode:
        config.accelerator = 'cpu'
        config.num_workers = 0
        config.global_batch_size = 2
        config.n_samples = 2
        config.n_samples_eval = 2
        config.n_steps = 10
        config.log_iter = 1
        config.val_iter = 5
        config.save_iter = 5
        config.log_dir += '_debug'
        config.save_dir += '_debug'
        config.load_dir += '_debug'

    return config


def main(config):
    # reproducibility (this and deterministic=True in trainer)
    pl.seed_everything(config.seed, workers=True)

    # utilize Tensor Cores (RTX 3090)
    torch.set_float32_matmul_precision('medium')

    # configure data and task
    datamodule, symmetry = configure_data(config, verbose=IS_RANK_ZERO)

    # configure model
    model, ckpt_path = configure_model(config, symmetry, verbose=IS_RANK_ZERO)

    # configure experiment
    logger, log_dir, callbacks, precision, strategy, plugins = configure_experiment(config, model)

    # compile the model and *step (training/validation/test/prediction)
    # note: can lead to nondeterministic behavior
    # note: temporarily disabled due to an issue with python 3.8
    # model = torch.compile(model)

    if config.test_mode:
        # test routine reproducibility (this and deterministic=True in trainer)
        pl.seed_everything(config.test_seed, workers=True)

        # setup trainer
        # during evaluation, it is recommended to use `Trainer(devices=1, num_nodes=1)`
        # to ensure each sample/batch gets evaluated exactly once. Otherwise,
        # multi-device settings use `DistributedSampler` that replicates some
        # samples to make sure all devices have same batch size in case of uneven inputs.
        # https://github.com/Lightning-AI/lightning/issues/12862
        trainer = pl.Trainer(
            logger=logger,
            default_root_dir=log_dir,
            accelerator=config.accelerator,
            num_sanity_val_steps=0,
            callbacks=callbacks,
            deterministic=True,
            devices=1,
            num_nodes=1,
            strategy=strategy,
            precision=precision,
            plugins=plugins,
            sync_batchnorm=True
        )

        # start evaluation
        trainer.test(model, datamodule=datamodule, verbose=IS_RANK_ZERO)

        # terminate
        sys.exit()

    # setup trainer
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir=log_dir,
        accelerator=config.accelerator,
        max_steps=config.n_steps,
        log_every_n_steps=-1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        deterministic=True,
        devices=torch.cuda.device_count() if config.accelerator == 'gpu' else 1,
        strategy=strategy,
        precision=precision,
        plugins=plugins,
        sync_batchnorm=True,
        gradient_clip_val=0.0 if not hasattr(config, 'gradient_clip_val') else config.gradient_clip_val,
        accumulate_grad_batches=1 if not hasattr(config, 'accumulate_grad_batches') else config.accumulate_grad_batches
    )

    if not config.resume_mode:
        # validation at start
        trainer.validate(model, datamodule=datamodule, verbose=IS_RANK_ZERO)

    # start training
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # start evaluation
    # this uses the last checkpoint for testing, and replicates some test samples.
    # for exact evaluation using the best checkpoint, it is recommended to run a
    # separate process with command `python3 main,py ... --test_mode` after training.
    trainer.test(model, datamodule=datamodule, verbose=IS_RANK_ZERO)

    # terminate
    sys.exit()


if __name__ == '__main__':
    IS_RANK_ZERO = int(os.environ.get('LOCAL_RANK', 0)) == 0
    config_ = get_config()
    main(config_)
