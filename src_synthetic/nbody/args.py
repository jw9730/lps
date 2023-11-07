# pylint: disable=line-too-long
import argparse


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # experiment arguments
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('--test', action='store_true')

    # data arguments
    parser.add_argument('--dataset', type=str, default='nbody_small')
    parser.add_argument('--max_train_samples', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=100)

    # model arguments
    parser.add_argument('--backbone', type=str, default='transformer', choices=['transformer', 'gnn'])
    # transformer arguments
    parser.add_argument('--transformer_num_layers', type=int, default=8)
    parser.add_argument('--transformer_hidden_dim', type=int, default=64)
    parser.add_argument('--transformer_n_head', type=int, default=4)
    parser.add_argument('--transformer_dropout', type=float, default=0)
    # gnn arguments
    parser.add_argument('--gnn_num_layers', type=int, default=4)
    parser.add_argument('--gnn_hidden_dim', type=int, default=60)

    # symmetry arguments
    parser.add_argument('--symmetry', type=str, default='SnxO3', choices=['SnxSO3', 'SnxO3', 'SO3', 'O3', 'Sn'])
    parser.add_argument('--interface', type=str, default='prob', choices=['prob', 'unif'])
    parser.add_argument('--sample_size', type=int, default=20)
    parser.add_argument('--eval_sample_size', type=int, default=20)
    parser.add_argument('--test_sample_size', type=int, default=20)

    # probabilistic symmetrization arguments
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--noise_scale', type=float, default=1)
    parser.add_argument('--fixed_noise', action='store_true')
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--entropy_loss_scale', type=float, default=0.1)
    # vector neurons arguments
    parser.add_argument('--vnn_hidden_dim', type=int, default=64)
    parser.add_argument('--vnn_k_nearest_neighbors', type=int, default=4)
    parser.add_argument('--vnn_dropout', type=float, default=0.08)

    # training arguments
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-12)
    parser.add_argument('--gradient_clip', type=float, default=0)

    # logging arguments
    parser.add_argument('--log_dir', type=str, default='experiments/logs')
    parser.add_argument('--save_dir', type=str, default='experiments/checkpoints')
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--test_n_trials', type=int, default=5)

    return parser


def get_args() -> argparse.Namespace:
    # parse arguments
    parser = argparse.ArgumentParser('Probabilistic Symmetrization')
    parser = add_args(parser)
    args = parser.parse_args()
    return args
