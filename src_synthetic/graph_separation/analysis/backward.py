# pylint: disable=line-too-long
import torch
from torch.nn import functional as F


def get_backbone_grad(model, data, args):
    model.zero_grad()
    pred, _ = model(data, n_samples=args.sample_size)
    # pred: [b, 1]
    loss = F.binary_cross_entropy_with_logits(pred, data.y[:, None].float())
    # only task loss affects gradient of backbone, entropy loss does not
    backbone_grad = torch.autograd.grad(loss, model.backbone.parameters())
    backbone_grad = torch.cat([g.view(-1) for g in backbone_grad])
    backbone_grad_norm = torch.norm(backbone_grad, p=2)
    return backbone_grad.detach(), backbone_grad_norm.item()


def analyze_backward(model, train_loader, device, args):
    # analyze trained backbone gf(g-1x)
    # gradient norm and direction of gf(g-1x)
    model.train()
    grad_direction = 0
    grad_norm_list = []
    for data in train_loader:
        data = data.to(device)
        for i in range(args.batch_size):
            data_instance = data[i]
            # gradient of gf(g-1x)
            grad, grad_norm = get_backbone_grad(model, data_instance, args)
            grad_direction += grad
            grad_norm_list.append(grad_norm)
    # gradient direction of gf(g-1x)
    count = len(grad_norm_list)
    grad_direction_norm = torch.norm(grad_direction / count, p=2).item()
    # return results
    return grad_norm_list, grad_direction_norm
