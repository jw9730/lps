import torch


def knn(x, k):
    # (B, C*3, N) -> (B, N, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x) # [B, N, N]
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx.to(x.device)


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    # x: (B, C, 3, N) = [1000, 2, 3, 5]
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)  # (B, C*3, N)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)
        else: # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()  # (B, N, C*3)
    feature = x.view(batch_size*num_points, -1)[idx, :]  # (B*N*k, C*3)
    feature = feature.view(batch_size, num_points, k, num_dims, 3)  # (B, N, k, C, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)  # (B, N, k, C, 3)

    # (B, N, k, C, 3) -> (B, N, k, 2C, 3) -> (B, 2C, 3, N, k)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    return feature


def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)

    feature = torch.cat((feature-x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature
