# pylint: disable=not-callable,line-too-long
# https://github.com/omri1348/Frame-Averaging/blob/master/graph_separation/exp_classify.py
import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.loader import DataLoader

from args import get_args
from dataset import PlanarSATPairsDataset
from preprocess import PrecomputeSpectral, PrecomputeSortFrame, PrecomputePad
from interface import InterfacedModel


def configure_device(args):
    return torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')


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
    train_loader = DataLoader(dataset[400:1200], args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    return train_loader, val_loader, test_loader


def configure_experiment_name(args):
    exp_name = f'seed_{args.seed},' \
        + (f'backbone_seed_{args.backbone_seed},' if args.backbone_seed else '') \
        + (f'interface_seed_{args.interface_seed},' if args.interface_seed else '') \
        + f'backbone_{args.backbone},'
    if args.interface == 'unif':
        exp_name += 'ga,hard,'
    elif args.interface == 'prob':
        exp_name += f'z_scale_{args.noise_scale},' \
            + f'tau_{args.tau},' \
            + ('hard,' if args.hard else '') \
            + ('fix,' if args.fixed_noise else '')
    else:
        raise NotImplementedError
    exp_name += f'epo_{args.num_epochs},' \
        + f'b_{args.batch_size},' \
        + f'lr_{args.lr},' \
        + f'k_{args.sample_size},' \
        + (f'eval_k_{args.eval_sample_size},' if args.sample_size != args.eval_sample_size else '') \
        + (f'clip_{args.gradient_clip},' if args.gradient_clip else '') \
        + (f'wu_{args.lr_warmup_epochs},' if args.lr_warmup_epochs > 0 else '')
    if args.interface == 'prob':
        exp_name += (f'entropy_{args.entropy_loss_scale},' if args.entropy_loss_scale > 0.0 else '')
    exp_name += args.postfix
    return exp_name


def configure_model(args, step_per_epoch):
    # setup save directory
    exp_name = configure_experiment_name(args)
    ckpt_dir = Path(args.save_dir) / exp_name
    if ckpt_dir.exists():
        if args.skip_if_run_exists:
            print(f'ckpt_dir {ckpt_dir} already exists. terminating.')
            sys.exit()
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
    # setup optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = step_per_epoch * args.num_epochs
    warmup_steps = int(args.lr_warmup_epochs / args.num_epochs * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if args.lr_decay_after_warmup:
            return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
        return 1.0
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return model, optimizer, lr_scheduler, ckpt_dir.as_posix()


def configure_experiment(args):
    # setup log directory
    exp_name = configure_experiment_name(args)
    log_dir = Path(args.log_dir) / exp_name
    if log_dir.exists():
        if args.skip_if_run_exists:
            print(f'log_dir {log_dir} already exists. terminating.')
            sys.exit()
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
    logger =  SummaryWriter(log_dir)
    return logger


def train_epoch(model, optimizer, lr_scheduler, train_loader, device, args):
    model.train()
    epoch_loss_sum = 0
    epoch_entropy_loss_sum = 0
    total = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred, entropy_loss = model(data, n_samples=args.sample_size)
        # pred: [b, 1]
        loss = F.binary_cross_entropy_with_logits(pred, data.y[:, None].float())
        total_loss = loss
        if args.entropy_loss_scale > 0.0:
            total_loss = loss + args.entropy_loss_scale * entropy_loss
        total_loss.backward()
        if args.gradient_clip:
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        lr_scheduler.step()
        epoch_loss_sum += loss.item() * len(data.y)
        epoch_entropy_loss_sum += entropy_loss.item() * len(data.y)
        total += len(data.y)
    epoch_loss_mean = epoch_loss_sum / total
    epoch_entropy_loss_mean = epoch_entropy_loss_sum / total
    return epoch_loss_mean, epoch_entropy_loss_mean


@torch.no_grad()
def eval_epoch(model, loader, device, args):
    model.eval()
    epoch_loss_sum = 0
    epoch_entropy_loss_sum = 0
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        pred, entropy_loss = model(data, n_samples=args.eval_sample_size)
        correct += torch.round(F.sigmoid(pred)).eq(data.y[:, None].float()).sum().item()
        loss = F.binary_cross_entropy_with_logits(pred, data.y[:, None].float())
        epoch_loss_sum += loss.item() * len(data.y)
        epoch_entropy_loss_sum += entropy_loss.item() * len(data.y)
        total += len(data.y)
    epoch_acc = correct / total
    epoch_loss_mean = epoch_loss_sum / total
    epoch_entropy_loss_mean = epoch_entropy_loss_sum / total
    return epoch_acc, epoch_loss_mean, epoch_entropy_loss_mean


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
    model, optimizer, lr_scheduler, ckpt_dir = configure_model(args, len(train_loader))

    # configure experiment
    logger = configure_experiment(args)

    # main loop
    best_val_loss = float('inf')
    best_val_acc = 0
    best_test_acc = 0
    torch.save(model.state_dict(), Path(ckpt_dir) / 'epoch_init.ckpt')
    for epoch in range(args.num_epochs + 1):
        # train
        train_loss, train_entropy_loss = train_epoch(model, optimizer, lr_scheduler, train_loader, device, args)
        # eval
        val_acc, val_loss, val_entropy_loss = eval_epoch(model, val_loader, device, args)
        test_acc, _, _ = eval_epoch(model, test_loader, device, args)
        # save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            torch.save(model.state_dict(), Path(ckpt_dir) / 'best.ckpt')
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), Path(ckpt_dir) / f'epoch_{epoch}.ckpt')
        # log
        logger.add_scalar('train/loss', train_loss, epoch)
        logger.add_scalar('train/entropy loss', train_entropy_loss, epoch)
        logger.add_scalar('val/acc', val_acc, epoch)
        logger.add_scalar('val/loss', val_loss, epoch)
        logger.add_scalar('val/entropy loss', val_entropy_loss, epoch)
        logger.add_scalar('test/acc', test_acc, epoch)
        logger.add_scalar('val/best acc', best_val_acc, epoch)
        logger.add_scalar('val/best loss', best_val_loss, epoch)
        logger.add_scalar('test/best acc', best_test_acc, epoch)
        print(f'epoch {epoch:04d} | ' \
              f'loss {train_loss:.4f} | ' \
              f'val loss {val_loss:.4f} | ' \
              f'best val loss {best_val_loss:.4f} | ' \
              f'best test acc {best_test_acc * 100:.1f}%')
    # close
    logger.flush()
    logger.close()


if __name__ == '__main__':
    args_ = get_args()
    main(args_)
