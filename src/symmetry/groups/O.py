import torch


def samples_from_haar_distribution(d, bsize, device, dtype) -> torch.Tensor:
    """Random orthogonal matrices drawn from O(d) Haar distribution
    Adopted from scipy.stats.ortho_group, which implements the algorithm described in
    Mezzadri, How to generate random matrices from the classical compact groups (2006)
    """
    z = torch.randn(bsize, d, d, device=device, dtype=dtype)
    q, r = torch.linalg.qr(z)
    # The last two dimensions are the rows and columns of R matrices.
    # Extract the diagonals. Note that this eliminates a dimension.
    d = r.diagonal(offset=0, dim1=-2, dim2=-1)
    # Add back a dimension for proper broadcasting: we're dividing
    # each row of each R matrix by the diagonal of the R matrix.
    q *= (d/abs(d))[..., None, :]  # to broadcast properly
    return q
