# pylint: disable=not-callable,line-too-long
# https://github.com/omri1348/Frame-Averaging/blob/master/nbody/nbody.py
import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from args import get_args
from dataset import NBodyDataset
from interface import InterfacedModel


def configure_device(args):
    return torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')


def configure_data(args):
    # setup dataset
    train_dataset = NBodyDataset('train', dataset_name=args.dataset, max_samples=args.max_train_samples)
    val_dataset = NBodyDataset('val', dataset_name='nbody_small')
    test_dataset = NBodyDataset('test', dataset_name='nbody_small')
    # setup loaders
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)
    return train_loader, val_loader, test_loader


def configure_experiment_name(args):
    exp_name = f'seed_{args.seed},' \
        + f'backbone_{args.backbone},'
    if args.backbone == 'transformer':
        exp_name += f'l_{args.transformer_num_layers},' \
            + f'd_{args.transformer_hidden_dim},' \
            + f'h_{args.transformer_n_head},' \
            + f'drop_{args.transformer_dropout},'
    elif args.backbone == 'gnn':
        exp_name += f'l_{args.gnn_num_layers},' \
            + f'd_{args.gnn_hidden_dim},'
    else:
        raise NotImplementedError
    exp_name += f'g_{args.symmetry},'
    if args.interface == 'unif':
        exp_name += 'ga,hard,'
    elif args.interface == 'prob':
        exp_name += 'prob,' \
            + f'z_scale_{args.noise_scale},' \
            + f'tau_{args.tau},' \
            + ('hard,' if args.hard else '') \
            + ('fix,' if args.fixed_noise else '') \
            + f'vnn_d{args.vnn_hidden_dim}_knn{args.vnn_k_nearest_neighbors}_drop{args.vnn_dropout},'
    else:
        raise NotImplementedError
    exp_name += f'epo_{args.num_epochs},' \
        + f'b_{args.batch_size},' \
        + f'lr_{args.lr},' \
        + f'k_{args.sample_size},' \
        + (f'eval_k_{args.eval_sample_size},' if args.sample_size != args.eval_sample_size else '') \
        + f'decay_{args.weight_decay},' \
        + (f'clip_{args.gradient_clip},' if args.gradient_clip else '')
    if args.interface == 'prob':
        exp_name += (f'entropy_{args.entropy_loss_scale},' if args.entropy_loss_scale > 0.0 else '')
    exp_name += args.postfix
    return exp_name


def configure_model(args):
    # setup save directory
    exp_name = configure_experiment_name(args)
    ckpt_dir = Path(args.save_dir) / exp_name
    if args.test:
        assert ckpt_dir.exists(), f'ckpt_dir {ckpt_dir} does not exist'
    elif ckpt_dir.exists():
        while True:
            print(f'ckpt_dir {ckpt_dir} already exists. overwrite? [y/n]')
            inp = input()
            if inp == 'y':
                break
            if inp == 'n':
                sys.exit()
            print('invalid input')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # setup model
    device = configure_device(args)
    model = InterfacedModel(
        symmetry=args.symmetry,
        interface=args.interface,
        backbone=args.backbone,
        transformer_num_layers=args.transformer_num_layers,
        transformer_hidden_dim=args.transformer_hidden_dim,
        transformer_n_head=args.transformer_n_head,
        transformer_dropout=args.transformer_dropout,
        gnn_num_layers=args.gnn_num_layers,
        gnn_hidden_dim=args.gnn_hidden_dim,
        fixed_noise=args.fixed_noise,
        noise_scale=args.noise_scale,
        tau=args.tau,
        hard=args.hard,
        vnn_hidden_dim=args.vnn_hidden_dim,
        vnn_k_nearest_neighbors=args.vnn_k_nearest_neighbors,
        vnn_dropout=args.vnn_dropout
    ).to(device)
    # print number of parameters
    backbone_n_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    interface_n_params = sum(p.numel() for p in model.interface.parameters() if p.requires_grad)
    print(f'number of parameters: backbone {backbone_n_params}, interface {interface_n_params}')
    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer, ckpt_dir.as_posix()


def configure_experiment(args):
    # setup log directory
    exp_name = configure_experiment_name(args)
    log_dir = Path(args.log_dir) / exp_name
    if args.test:
        if not log_dir.exists():
            print(f'log_dir {log_dir} does not exist')
        args.eval_sample_size = args.test_sample_size
    elif log_dir.exists():
        while True:
            print(f'log_dir {log_dir} already exists. overwrite? [y/n]')
            inp = input()
            if inp == 'y':
                break
            if inp == 'n':
                sys.exit()
            print('invalid input')
    log_dir.mkdir(parents=True, exist_ok=True)
    # setup logger
    logger = SummaryWriter(log_dir)
    return logger


@torch.no_grad()
def parse_batch(data, loader, device):
    data = [d.to(device) for d in data]
    loc, vel, edge_attr, _, loc_end, idx = data
    b, n, _ = loc.shape
    edges = loader.dataset.get_edges_no_incr(b).to(device)
    edges0 = edges[:, 0, :, None].expand(-1, -1, 3)
    edges1 = edges[:, 1, :, None].expand(-1, -1, 3)
    loc0 = torch.gather(loc, dim=1, index=edges0)
    loc1 = torch.gather(loc, dim=1, index=edges1)
    loc_dist = torch.sum((loc0 - loc1).pow(2), dim=-1)[:, :, None]
    edge_attr = torch.cat([edge_attr, loc_dist], dim=-1)
    node_features = torch.cat([loc, vel], dim=-1)
    node_features = node_features.reshape(b, n, 2, 3)
    node_features = node_features.transpose(-1, -2)
    assert (loc - node_features[:, :, :, 0]).abs().sum().item() == 0
    edge_features = torch.zeros(b, n, n, edge_attr.size(-1), device=device)
    batch_idxs = torch.arange(b, device=device).repeat_interleave(edges.shape[-1])
    edge_features[batch_idxs, edges[:, 0, :].flatten(), edges[:, 1, :].flatten(), :] \
        = edge_attr.reshape(-1, edge_attr.size(-1))
    return node_features, edge_features, edges, loc_end, idx


def train_epoch(model, optimizer, train_loader, device, args):
    model.train()
    epoch_loss_sum = 0
    epoch_entropy_loss_sum = 0
    total = 0
    for data in train_loader:
        node_features, edge_features, edge_idx, target, idx = parse_batch(data, train_loader, device)
        optimizer.zero_grad()
        pred, entropy_loss = model(node_features, edge_features, edge_idx, idx, n_samples=args.sample_size)
        loss = F.mse_loss(pred, target)
        if args.entropy_loss_scale > 0.0:
            total_loss = loss + args.entropy_loss_scale * entropy_loss
        total_loss.backward()
        if args.gradient_clip:
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        epoch_loss_sum += loss.item() * len(target)
        epoch_entropy_loss_sum += entropy_loss.item() * len(target)
        total += len(target)
    epoch_loss_mean = epoch_loss_sum / total
    epoch_entropy_loss_mean = epoch_entropy_loss_sum / total
    return epoch_loss_mean, epoch_entropy_loss_mean


@torch.no_grad()
def eval_epoch(model, loader, device, args):
    model.eval()
    epoch_loss_sum = 0
    epoch_entropy_loss_sum = 0
    total = 0
    for data in loader:
        node_features, edge_features, edge_idx, target, idx = parse_batch(data, loader, device)
        pred, entropy_loss = model(node_features, edge_features, edge_idx, idx, n_samples=args.eval_sample_size)
        loss = F.mse_loss(pred, target)
        epoch_loss_sum += loss.item() * len(target)
        epoch_entropy_loss_sum += entropy_loss.item() * len(target)
        total += len(target)
    epoch_loss_mean = epoch_loss_sum / total
    epoch_entropy_loss_mean = epoch_entropy_loss_sum / total
    return epoch_loss_mean, epoch_entropy_loss_mean


@torch.no_grad()
def test_epoch(model, loader, device, args):
    model.eval()
    epoch_loss_trials = []
    for _ in range(args.test_n_trials):
        epoch_loss_sum = 0
        total = 0
        for data in loader:
            node_features, edge_features, edge_idx, target, idx = parse_batch(data, loader, device)
            pred, _ = model(node_features, edge_features, edge_idx, idx, n_samples=args.eval_sample_size)
            loss = F.mse_loss(pred, target)
            epoch_loss_sum += loss.item() * len(target)
            total += len(target)
        epoch_loss_mean = epoch_loss_sum / total
        epoch_loss_trials.append(epoch_loss_mean)
    epoch_loss_trials_mean = np.mean(epoch_loss_trials)
    epoch_loss_trials_std = np.std(epoch_loss_trials)
    return epoch_loss_trials_mean, epoch_loss_trials_std


def main(args):
    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    # configure device
    device = configure_device(args)

    # configure data
    train_loader, val_loader, test_loader = configure_data(args)

    # configure model
    model, optimizer, ckpt_dir = configure_model(args)

    # configure experiment
    logger = configure_experiment(args)

    # load and evaluate a trained checkpoint if specified
    if args.test:
        model.load_state_dict(torch.load(Path(ckpt_dir) / 'best.ckpt'))
        print(f'loaded checkpoint {ckpt_dir}/best.ckpt')
        best_test_loss_mean, best_test_loss_std = test_epoch(model, test_loader, device, args)
        print(f'test loss {best_test_loss_mean:.8f} ± {best_test_loss_std:.8f}')
        sys.exit()

    # main loop
    best_val_loss = float('inf')
    best_test_loss_mean = float('inf')
    best_test_loss_std = float('inf')
    for epoch in range(0, args.num_epochs + 1):
        # train
        train_loss, train_entropy_loss = train_epoch(model, optimizer, train_loader, device, args)
        # eval
        val_loss, val_entropy_loss = eval_epoch(model, val_loader, device, args)
        # test
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss_mean, best_test_loss_std = test_epoch(model, test_loader, device, args)
            torch.save(model.state_dict(), Path(ckpt_dir) / 'best.ckpt')
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), Path(ckpt_dir) / f'epoch_{epoch}.ckpt')
        # log
        logger.add_scalar('train/loss', train_loss, epoch)
        logger.add_scalar('train/entropy loss', train_entropy_loss, epoch)
        logger.add_scalar('val/loss', val_loss, epoch)
        logger.add_scalar('val/entropy loss', val_entropy_loss, epoch)
        logger.add_scalar('val/best loss', best_val_loss, epoch)
        logger.add_scalar('test/best loss', best_test_loss_mean, epoch)
        logger.add_scalar('test/best loss std', best_test_loss_std, epoch)
        print(f'epoch {epoch:04d} | ' \
              f'loss {train_loss:.4f} | ' \
              f'val loss {val_loss:.4f} | ' \
              f'best val loss {best_val_loss:.4f} | ' \
              f'best test loss {best_test_loss_mean:.8f} ± {best_test_loss_std:.8f}')
    # close
    logger.flush()
    logger.close()


if __name__ == '__main__':
    args_ = get_args()
    main(args_)
