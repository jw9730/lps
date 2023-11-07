# pylint: disable=not-callable,line-too-long
import os
import sys
import argparse
from pathlib import Path
import json
import random
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch_geometric.loader import DataLoader

from args import add_args
from exp_classify import configure_device, configure_experiment_name
from dataset import PlanarSATPairsDataset
from preprocess import PrecomputeSpectral, PrecomputeSortFrame, PrecomputePad
from interface import InterfacedModel
from analysis import analyze_interface, analyze_forward, analyze_backward


def get_args() -> argparse.Namespace:
    # parse arguments
    parser = argparse.ArgumentParser('Probabilistic Symmetrization')
    # experiment arguments
    parser = add_args(parser)
    # analysis arguments
    parser.add_argument('--all_epochs', action='store_true')
    parser.add_argument('--compile_results', action='store_true')
    args = parser.parse_args()
    return args


def configure_data(args):
    device = configure_device(args)
    # setup dataset
    pre_transform = PrecomputeSpectral(nmax=64, recfield=1, dv=2, nfreq=5, adddegree=True)
    pre_transform = PrecomputeSortFrame(pre_transform, device)
    transform = PrecomputePad(nmax=64)
    dataset = PlanarSATPairsDataset(root='dataset/EXP/', transform=transform, pre_transform=pre_transform)
    # setup loaders
    val_loader = DataLoader(dataset[0:200], args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(dataset[200:400], args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    train_loader = DataLoader(dataset[400:1200], args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    return train_loader, val_loader, test_loader


def configure_model(args):
    # setup save directory
    exp_name = configure_experiment_name(args)
    ckpt_dir = Path(args.save_dir) / exp_name
    assert ckpt_dir.exists(), f'checkpoint directory {ckpt_dir} does not exist'
    # setup model
    device = configure_device(args)
    model = InterfacedModel(
        n=64,
        d=2,
        interface=args.interface,
        num_interface_layers=args.num_interface_layers,
        backbone=args.backbone,
        fixed_noise=args.fixed_noise,
        noise_scale=args.noise_scale,
        tau=args.tau,
        hard=args.hard,
        task='EXPclassify',
        backbone_seed=args.backbone_seed,
        interface_seed=args.interface_seed
    ).to(device)
    return model, ckpt_dir.as_posix()


def configure_experiment(args):
    # setup log directory
    exp_name = configure_experiment_name(args)
    log_dir = Path(args.log_dir) / exp_name
    assert log_dir.exists(), f'log directory {log_dir} does not exist'
    return log_dir.as_posix()


def analyze(model, log_dir, epoch, train_loader, val_loader, device, args):
    # analyze
    val_entropy_list = analyze_interface(model, log_dir, epoch, val_loader, device, args)
    val_pred_std_dict, val_loss_mean_dict, val_loss_std_dict = analyze_forward(model, val_loader, device)
    backbone_grad_norm_list, backbone_grad_direction_norm = analyze_backward(model, train_loader, device, args)
    # organize
    val_entropy = np.mean(val_entropy_list)
    backbone_grad_norm = np.mean(backbone_grad_norm_list)
    # save results as json
    results_dict = {
        'val_entropy': val_entropy,
        'backbone_grad_norm': backbone_grad_norm,
        'backbone_grad_direction_norm': backbone_grad_direction_norm,
        'val_pred_std_dict': val_pred_std_dict,
        'val_loss_mean_dict': val_loss_mean_dict,
        'val_loss_std_dict': val_loss_std_dict
    }
    results_path = Path(log_dir) / f'epoch_{epoch}.json'
    with open(results_path.as_posix(), 'w', encoding='utf-8') as f:
        json.dump(results_dict, f)
    # return
    results = (
        val_entropy,
        backbone_grad_norm,
        backbone_grad_direction_norm,
        val_pred_std_dict,
        val_loss_mean_dict,
        val_loss_std_dict
    )
    return results


def save_results(results_list, epochs_list, log_dir, args):
    # extract
    (
        val_entropy_list,
        backbone_grad_norm_list,
        backbone_grad_direction_norm_list,
        val_pred_std_dict_list,
        val_loss_mean_dict_list,
        val_loss_std_dict_list,
    ) = zip(*results_list)
    eval_sample_size_list = list(val_pred_std_dict_list[0].keys())
    colors = plt.colormaps.get_cmap('viridis').resampled(len(eval_sample_size_list)).colors
    # plot val entropy
    plt.figure()
    plt.plot(epochs_list, val_entropy_list)
    plt.xlabel('epoch')
    plt.ylabel('entropy')
    plt.savefig(Path(log_dir) / 'entropy_val.pdf')
    plt.close()
    # plot grad norm
    plt.figure()
    plt.plot(epochs_list, backbone_grad_norm_list)
    plt.xlabel('epoch')
    plt.ylabel('grad norm')
    plt.savefig(Path(log_dir) / 'grad_norm.pdf')
    # plot grad direction norm
    plt.figure()
    plt.plot(epochs_list, backbone_grad_direction_norm_list)
    plt.xlabel('epoch')
    plt.ylabel('grad direction norm')
    plt.savefig(Path(log_dir) / 'grad_direction_norm.pdf')
    plt.close()
    # plot output (logit) std
    plt.figure()
    for eval_sample_size, color in zip(eval_sample_size_list, colors):
        val_pred_std_list = [val_pred_std_dict[eval_sample_size] for val_pred_std_dict in val_pred_std_dict_list]
        plt.plot(epochs_list, val_pred_std_list, label=f'{eval_sample_size} samples', color=color)
    plt.xlabel('epoch')
    plt.ylabel('output std')
    plt.legend()
    plt.savefig(Path(log_dir) / 'output_std.pdf')
    plt.close()
    # plot loss mean
    plt.figure()
    for eval_sample_size, color in zip(eval_sample_size_list, colors):
        val_loss_mean_list = [val_loss_mean_dict[eval_sample_size] for val_loss_mean_dict in val_loss_mean_dict_list]
        plt.plot(epochs_list, val_loss_mean_list, label=f'{eval_sample_size} samples', color=color)
    plt.xlabel('epoch')
    plt.ylabel('loss mean')
    plt.legend()
    plt.savefig(Path(log_dir) / 'loss_mean.pdf')
    plt.close()
    # plot loss std
    plt.figure()
    for eval_sample_size, color in zip(eval_sample_size_list, colors):
        val_loss_std_list = [val_loss_std_dict[eval_sample_size] for val_loss_std_dict in val_loss_std_dict_list]
        plt.plot(epochs_list, val_loss_std_list, label=f'{eval_sample_size} samples', color=color)
    plt.xlabel('epoch')
    plt.ylabel('loss std')
    plt.legend()
    plt.savefig(Path(log_dir) / 'loss_std.pdf')
    plt.close()
    if args.all_epochs:
        # plot output (logit) std at initialization
        plt.figure()
        plt.plot(*zip(*sorted(val_pred_std_dict_list[0].items())))
        plt.xlabel('sample size')
        plt.ylabel('output std')
        plt.savefig(Path(log_dir) / 'output_std_init.pdf')
        plt.close()
        # plot loss mean at initialization
        plt.figure()
        plt.plot(*zip(*sorted(val_loss_mean_dict_list[0].items())))
        plt.xlabel('sample size')
        plt.ylabel('loss mean')
        plt.savefig(Path(log_dir) / 'loss_mean_init.pdf')
        plt.close()
        # plot loss std at initialization
        plt.figure()
        plt.plot(*zip(*sorted(val_loss_std_dict_list[0].items())))
        plt.xlabel('sample size')
        plt.ylabel('loss std')
        plt.savefig(Path(log_dir) / 'loss_std_init.pdf')
        plt.close()


def read_results(log_dir, epoch):
    # read log_dir / epoch_{epoch}.json
    results_path = Path(log_dir) / f'epoch_{epoch}.json'
    with open(results_path.as_posix(), 'r', encoding='utf-8') as f:
        results_dict = json.load(f)
    return results_dict


def compile_results(args):
    # configure experiments
    args.hard = True
    args.interface = 'unif'
    log_dir_ga = configure_experiment(args)
    args.interface = 'prob'
    args.sample_size = 1
    log_dir_ps_1 = configure_experiment(args)
    args.sample_size = 2
    log_dir_ps_2 = configure_experiment(args)
    args.sample_size = 5
    log_dir_ps_5 = configure_experiment(args)
    args.sample_size = 10
    log_dir_ps = configure_experiment(args)
    log_dir_ps_10 = configure_experiment(args)
    args.sample_size = 20
    log_dir_ps_20 = configure_experiment(args)
    args.sample_size = 50
    log_dir_ps_50 = configure_experiment(args)
    # analyze ps in comparison to ga
    epochs_list = [100, 500, 1000, 1500, 2000]
    entropy_ga = [read_results(log_dir_ga, epoch)['val_entropy'] for epoch in epochs_list]
    entropy_ps = [read_results(log_dir_ps, epoch)['val_entropy'] for epoch in epochs_list]
    grad_norm_ga = [read_results(log_dir_ga, epoch)['backbone_grad_direction_norm'] for epoch in epochs_list]
    grad_norm_ps = [read_results(log_dir_ps, epoch)['backbone_grad_direction_norm'] for epoch in epochs_list]
    pred_std_ga = read_results(log_dir_ga, '_init')['val_pred_std_dict']
    pred_std_ps = read_results(log_dir_ps, '_init')['val_pred_std_dict']
    loss_std_ga = read_results(log_dir_ga, '_init')['val_loss_std_dict']
    loss_std_ps = read_results(log_dir_ps, '_init')['val_loss_std_dict']
    pred_var_ga = {int(k): v ** 2 for k, v in pred_std_ga.items()}
    pred_var_ps = {int(k): v ** 2 for k, v in pred_std_ps.items()}
    loss_var_ga = {int(k): v ** 2 for k, v in loss_std_ga.items()}
    loss_var_ps = {int(k): v ** 2 for k, v in loss_std_ps.items()}
    print('entropy of permutation matrices averaged over validation samples')
    print('ga: epoch, entropy')
    print('\n'.join([f'{k}, {v}' for k, v in zip(epochs_list, entropy_ga)]))
    print('ps: epoch, entropy')
    print('\n'.join([f'{k}, {v}' for k, v in zip(epochs_list, entropy_ps)]))
    print('\ngradient norm of backbone averaged over training samples')
    print('ga: epoch, grad norm')
    print('\n'.join([f'{k}, {v}' for k, v in zip(epochs_list, grad_norm_ga)]))
    print('ps: epoch, grad norm')
    print('\n'.join([f'{k}, {v}' for k, v in zip(epochs_list, grad_norm_ps)]))
    print('\nvariance of estimated output')
    print('ga: inference sample size, output variance')
    print('\n'.join([f'{k}, {v}' for k, v in sorted(pred_var_ga.items())]))
    print('ps: inference sample size, output variance')
    print('\n'.join([f'{k}, {v}' for k, v in sorted(pred_var_ps.items())]))
    print('\nvariance of estimated loss')
    print('ga: inference sample size, loss variance')
    print('\n'.join([f'{k}, {v}' for k, v in sorted(loss_var_ga.items())]))
    print('ps: inference sample size, loss variance')
    print('\n'.join([f'{k}, {v}' for k, v in sorted(loss_var_ps.items())]))
    # analyze ps on the effect of training sample size
    entropy_ps = {
        1: read_results(log_dir_ps_1, 'best')['val_entropy'],
        2: read_results(log_dir_ps_2, 'best')['val_entropy'],
        5: read_results(log_dir_ps_5, 'best')['val_entropy'],
        10: read_results(log_dir_ps_10, 'best')['val_entropy'],
        20: read_results(log_dir_ps_20, 'best')['val_entropy'],
        50: read_results(log_dir_ps_50, 'best')['val_entropy']
    }
    pred_std_ps = {
        1: read_results(log_dir_ps_1, 'best')['val_pred_std_dict'],
        2: read_results(log_dir_ps_2, 'best')['val_pred_std_dict'],
        5: read_results(log_dir_ps_5, 'best')['val_pred_std_dict'],
        10: read_results(log_dir_ps_10, 'best')['val_pred_std_dict'],
        20: read_results(log_dir_ps_20, 'best')['val_pred_std_dict'],
        50: read_results(log_dir_ps_50, 'best')['val_pred_std_dict']
    }
    loss_std_ps = {
        1: read_results(log_dir_ps_1, 'best')['val_loss_std_dict'],
        2: read_results(log_dir_ps_2, 'best')['val_loss_std_dict'],
        5: read_results(log_dir_ps_5, 'best')['val_loss_std_dict'],
        10: read_results(log_dir_ps_10, 'best')['val_loss_std_dict'],
        20: read_results(log_dir_ps_20, 'best')['val_loss_std_dict'],
        50: read_results(log_dir_ps_50, 'best')['val_loss_std_dict']
    }
    loss_mean_ps = {
        1: read_results(log_dir_ps_1, 'best')['val_loss_mean_dict'],
        2: read_results(log_dir_ps_2, 'best')['val_loss_mean_dict'],
        5: read_results(log_dir_ps_5, 'best')['val_loss_mean_dict'],
        10: read_results(log_dir_ps_10, 'best')['val_loss_mean_dict'],
        20: read_results(log_dir_ps_20, 'best')['val_loss_mean_dict'],
        50: read_results(log_dir_ps_50, 'best')['val_loss_mean_dict']
    }
    pred_var_ps = {k: {int(kk): vv ** 2 for kk, vv in v.items()} for k, v in pred_std_ps.items()}
    loss_var_ps = {k: {int(kk): vv ** 2 for kk, vv in v.items()} for k, v in loss_std_ps.items()}
    loss_mean_ps = {k: {int(kk): vv for kk, vv in v.items()} for k, v in loss_mean_ps.items()}
    print('\nentropy of permutation matrices averaged over validation samples')
    print('ps: train sample size, entropy')
    print('\n'.join([f'{k}, {v}' for k, v in entropy_ps.items()]))
    print('\nvariance of estimated output')
    print('ps: train sample size, output variance for each inference sample size')
    print('\n'.join([f'{k}, {v}' for k, v in pred_var_ps.items()]))
    print('\nvariance of estimated loss')
    print('ps: train sample size, loss variance for each inference sample size')
    print('\n'.join([f'{k}, {v}' for k, v in loss_var_ps.items()]))
    print('\nmean of estimated loss')
    print('ps: train sample size, loss mean for each inference sample size')
    print('\n'.join([f'{k}, {v}' for k, v in loss_mean_ps.items()]))
    print('done')


def main(args):
    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    # compile results if specified
    if args.compile_results:
        compile_results(args)
        sys.exit()

    # configure device
    device = configure_device(args)

    # configure data
    train_loader, val_loader, _ = configure_data(args)

    # configure model
    model, ckpt_dir = configure_model(args)

    # configure experiment
    log_dir = configure_experiment(args)

    # analyze
    print(f'checkpoint directory found at src_synthetic/graph_separation/{ckpt_dir}')
    epochs_list = []
    results_list = []
    if args.all_epochs:
        # initialization
        # setup
        epoch = '_init'
        # analyze
        print('analyzing initialized model')
        results = analyze(model, log_dir, epoch, train_loader, val_loader, device, args)
        # record
        epochs_list.append(-100)
        results_list.append(results)
        # main loop
        for epoch in range(args.num_epochs + 1):
            # setup
            ckpt_path = Path(ckpt_dir) / f'epoch_{epoch}.ckpt'
            if not ckpt_path.exists():
                continue
            # load
            model.load_state_dict(torch.load(ckpt_path.as_posix(), map_location=device))
            # analyze
            print(f'analyzing epoch_{epoch}.ckpt')
            results = analyze(model, log_dir, epoch, train_loader, val_loader, device, args)
            # record
            epochs_list.append(epoch)
            results_list.append(results)
    # best checkpoint
    # setup
    epoch = 'best'
    ckpt_path = Path(ckpt_dir) / 'best.ckpt'
    # load
    model.load_state_dict(torch.load(ckpt_path.as_posix(), map_location=device))
    # analyze
    print('analyzing best.ckpt')
    results = analyze(model, log_dir, epoch, train_loader, val_loader, device, args)
    # record
    epochs_list.append(args.num_epochs + 100)
    results_list.append(results)
    # save
    save_results(results_list, epochs_list, log_dir, args)
    # close
    print(f'done, results saved to src_synthetic/graph_separation/{log_dir}')


if __name__ == '__main__':
    args_ = get_args()
    main(args_)
