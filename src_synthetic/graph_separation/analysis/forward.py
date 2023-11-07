# pylint: disable=line-too-long
import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def analyze_forward(model, val_loader, device):
    # analyze trained backbone gf(g-1x)
    # variance of prediction and loss of gf(g-1x) over sample sizes
    eval_sample_size_list = [1, 2, 5, 10, 20, 50, 100, 200]
    n_trials = 100
    # val data
    model.eval()
    pred_mean_dict = {eval_sample_size: [] for eval_sample_size in eval_sample_size_list}
    pred_std_dict = {eval_sample_size: [] for eval_sample_size in eval_sample_size_list}
    loss_mean_dict = {eval_sample_size: [] for eval_sample_size in eval_sample_size_list}
    loss_std_dict = {eval_sample_size: [] for eval_sample_size in eval_sample_size_list}
    for data in val_loader:
        data = data.to(device)
        for eval_sample_size in eval_sample_size_list:
            pred_list = []
            loss_list = []
            for _ in range(n_trials):
                # pred, loss: [b, 1]
                # gf(g-1x)
                pred, _ = model(data, n_samples=eval_sample_size)
                loss = F.binary_cross_entropy_with_logits(pred, data.y[:, None].float(), reduction='none')
                pred_list.append(F.sigmoid(pred))
                loss_list.append(loss)
            # pred, loss: [b, n_trials]
            pred_std, pred_mean = torch.std_mean(torch.cat(pred_list, dim=1), dim=1)
            loss_std, loss_mean = torch.std_mean(torch.cat(loss_list, dim=1), dim=1)
            pred_mean_dict[eval_sample_size] += pred_mean.tolist()
            pred_std_dict[eval_sample_size] += pred_std.tolist()
            loss_mean_dict[eval_sample_size] += loss_mean.tolist()
            loss_std_dict[eval_sample_size] += loss_std.tolist()
    # average over val data
    pred_mean_dict = {eval_sample_size: np.mean(pred_mean_dict[eval_sample_size]) for eval_sample_size in eval_sample_size_list}
    pred_std_dict = {eval_sample_size: np.mean(pred_std_dict[eval_sample_size]) for eval_sample_size in eval_sample_size_list}
    loss_mean_dict = {eval_sample_size: np.mean(loss_mean_dict[eval_sample_size]) for eval_sample_size in eval_sample_size_list}
    loss_std_dict = {eval_sample_size: np.mean(loss_std_dict[eval_sample_size]) for eval_sample_size in eval_sample_size_list}
    # return results
    return pred_std_dict, loss_mean_dict, loss_std_dict
