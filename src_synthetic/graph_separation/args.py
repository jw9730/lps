import argparse


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # experiment arguments
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--backbone_seed', type=int, default=None)
    parser.add_argument('--interface_seed', type=int, default=None)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('--skip_if_run_exists', action='store_true')

    # data arguments
    parser.add_argument('--batch_size', type=int, default=100)

    # model arguments
    parser.add_argument('--backbone', type=str, default='mlp', choices=['mlp', 'gin'])

    # probabilistic symmetrization arguments
    parser.add_argument('--interface', type=str, default='prob', choices=['prob', 'unif'])
    parser.add_argument('--sample_size', type=int, default=10)
    parser.add_argument('--eval_sample_size', type=int, default=10)
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--noise_scale', type=float, default=1)
    parser.add_argument('--fixed_noise', action='store_true')
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--entropy_loss_scale', type=float, default=0.1)
    parser.add_argument('--num_interface_layers', type=int, default=3)

    # training arguments
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gradient_clip', type=float, default=0.1)
    parser.add_argument('--lr_warmup_epochs', type=int, default=200)
    parser.add_argument('--lr_decay_after_warmup', action='store_true')

    # logging arguments
    parser.add_argument('--log_dir', type=str, default='experiments/logs')
    parser.add_argument('--save_dir', type=str, default='experiments/checkpoints')
    parser.add_argument('--save_interval', type=int, default=100)

    return parser


def get_args() -> argparse.Namespace:
    # parse arguments
    parser = argparse.ArgumentParser('Probabilistic Symmetrization')
    parser = add_args(parser)
    args = parser.parse_args()
    return args
