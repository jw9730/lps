# pylint: disable=not-callable,line-too-long
# https://github.com/omri1348/Frame-Averaging/blob/master/graph_separation/graph8c.py
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from args import get_args
from dataset import GRAPH8cDataset
from preprocess import PrecomputeSpectral, PrecomputeSortFrame, PrecomputePad
from interface import InterfacedModel


def main(args):
    # configure device
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

    # configure data
    pre_transform = PrecomputeSpectral(nmax=8, recfield=1, dv=2, nfreq=5, adddegree=True)
    pre_transform = PrecomputeSortFrame(pre_transform, device)
    transform = PrecomputePad(nmax=8)
    dataset = GRAPH8cDataset(root='dataset/graph8c/', transform=transform, pre_transform=pre_transform)
    train_loader = DataLoader(dataset, args.batch_size, shuffle=False)

    # main loop
    M = 0
    for seed in range(100):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # configure model
        model = InterfacedModel(
            n=8,
            d=2,
            interface=args.interface,
            num_interface_layers=args.num_interface_layers,
            backbone=args.backbone,
            noise_scale=args.noise_scale,
            tau=args.tau,
            hard=args.hard,
            task='GRAPH8c'
        ).to(device)

        # run test
        embeddings = []
        model.eval()
        for data in train_loader:
            data = data.to(device)
            emb, _ = model(data, n_samples=args.eval_sample_size)
            embeddings.append(emb)

        E = torch.cat(embeddings).cpu().detach().numpy()
        M = M+1*((np.abs(np.expand_dims(E, 1)-np.expand_dims(E, 0))).sum(2) > 0.001)
        sm = ((M == 0).sum()-M.shape[0])/2
        print('similar:', sm)


if __name__ == '__main__':
    args_ = get_args()
    main(args_)
