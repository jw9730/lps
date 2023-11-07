import torch


def batched_gram_schmidt_3d(bvv):
    assert bvv.ndim == 3
    assert bvv.shape[1] == bvv.shape[2] == 3

    def projection(bu, bv):
        return (bv * bu).sum(-1, keepdim=True) / (bu * bu).sum(-1, keepdim=True) * bu

    buu = torch.zeros_like(bvv)
    buu[:, :, 0] = bvv[:, :, 0].clone()

    # k = 1 start
    bv1 = bvv[:, :, 1].clone()
    bu1 = torch.zeros_like(bv1)
    # j = 0
    bu0 = buu[:, :, 0].clone()
    bu1 = bu1 + projection(bu0, bv1)
    # k = 1 end
    buu[:, :, 1] = bv1 - bu1

    # k = 2 start
    bv2 = bvv[:, :, 2].clone()
    bu2 = torch.zeros_like(bv2)
    # j = 0
    bu0 = buu[:, :, 0].clone()
    bu2 = bu2 + projection(bu0, bv2)
    # j = 1
    bu1 = buu[:, :, 1].clone()
    bu2 = bu2 + projection(bu1, bv2)
    # k = 2 end
    buu[:, :, 2] = bv2 - bu2

    # normalize
    buu = torch.nn.functional.normalize(buu, dim=1)
    return buu
