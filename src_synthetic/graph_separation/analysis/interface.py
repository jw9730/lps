# pylint: disable=too-many-arguments,line-too-long
from pathlib import Path
from matplotlib import pyplot as plt
import torch


def visualize(perm, name, log_dir):
    perm = perm.cpu().numpy()
    plt.figure(figsize=(5, 5))
    img = plt.imshow(perm, norm=plt.Normalize(0, 0.25), interpolation='nearest')
    plt.axis('off')
    img.axes.get_xaxis().set_visible(False)
    img.axes.get_yaxis().set_visible(False)
    plt.savefig(Path(log_dir) / f'{name}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


class EntropyMetric():
    def __init__(self, n):
        super().__init__()
        self.n = n

    @torch.no_grad()
    def compute_node_mask(self, n_nodes: torch.Tensor):
        b = n_nodes.shape[0]
        idxs = torch.arange(0, self.n, device=n_nodes.device)[None, :].expand(b, self.n)
        node_mask = idxs < n_nodes[:, None]
        return node_mask

    @torch.no_grad()
    def compute_node_pair_mask(self, node_mask: torch.Tensor):
        b = node_mask.shape[0]
        mask1 = node_mask[:, None, :].expand(b, self.n, self.n)
        mask2 = node_mask[:, :, None].expand(b, self.n, self.n)
        node_pair_mask = torch.logical_and(mask1, mask2)
        return node_pair_mask

    @staticmethod
    def _normalize(scores, axis, eps=1e-12):
        normalizer = torch.sum(scores, axis).clamp(min=eps)
        normalizer = normalizer.unsqueeze(axis)
        prob = torch.div(scores, normalizer)
        return prob

    @staticmethod
    def _entropy(prob, axis):
        return -torch.sum(prob * prob.log().clamp(min=-100), axis)

    def entropy(self, scores, node_mask, eps=1e-12):
        b, n, _ = scores.shape
        # clamp min to avoid zero logarithm
        scores = scores.clone()
        node_pair_mask = self.compute_node_pair_mask(node_mask)
        scores[node_pair_mask] = scores[node_pair_mask].clamp(min=eps)
        scores[~node_pair_mask] = 0
        # compute columnwise entropy
        col_prob = self._normalize(scores, axis=1)
        col_prob = torch.where(node_pair_mask, col_prob, torch.ones_like(scores))
        entropy_col = self._entropy(col_prob, axis=1)
        # compute rowwise entropy
        row_prob = self._normalize(scores, axis=2)
        row_prob = torch.where(node_pair_mask, row_prob, torch.ones_like(scores))
        entropy_row = self._entropy(row_prob, axis=2)
        # return entropy
        assert entropy_col.shape == entropy_row.shape == (b, n)
        return entropy_col, entropy_row

    def get_entropy(self, perm: torch.Tensor, n_nodes: torch.Tensor):
        b, n, _ = perm.shape
        assert n == self.n
        assert perm.shape == (b, n, n)
        # compute node mask
        node_mask = self.compute_node_mask(n_nodes)
        # compute entropy
        entropy_col, entropy_row = self.entropy(perm, node_mask)
        entropy = entropy_col + entropy_row
        # compute mean over nodes
        entropy[~node_mask] = 0
        entropy = entropy.sum(1) / n_nodes
        # unbind to list
        return entropy.tolist()


@torch.no_grad()
def analyze_interface(model, log_dir, epoch, val_loader, device, args):
    # analyze trained distribution p(g|x)
    entropy_metric = EntropyMetric(64)
    num_visualize = 1
    eval_sample_size = 100
    # val data
    model.eval()
    entropy_list = []
    for batch_idx, data in enumerate(val_loader):
        data = data.to(device)
        # entropy of averaged permutation samples
        _, _, perm = model(data, n_samples=eval_sample_size, return_perm=True)
        average_perm = perm.mean(1)
        entropy_list += entropy_metric.get_entropy(average_perm, data.n)
        # visualization of averaged permutation samples
        if batch_idx == 0:
            for data_idx, (p, n) in enumerate(zip(average_perm, data.n)):
                if data_idx < num_visualize:
                    visualize(p[:n, :n], f'perm_val_data_{data_idx}_epoch_{epoch}_k_{eval_sample_size}', log_dir)
    # return results
    return entropy_list
