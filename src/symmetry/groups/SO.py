import torch


def samples_from_haar_distribution(d, bsize, device, dtype) -> torch.Tensor:
    """Random orthogonal matrices with determinant 1 drawn from SO(d) Haar distribution
    Adopted from scipy.stats.special_ortho_group, which implements the algorithm described in
    Mezzadri, How to generate random matrices from the classical compact groups (2006)
    """
    # H represents a (dim, dim) matrix, while D represents the diagonal of
    # a (dim, dim) diagonal matrix. The algorithm that follows is
    # broadcasted on the leading shape in `size` to vectorize along
    # samples.
    H = torch.empty(bsize, d, d, device=device, dtype=dtype)
    H[..., :, :] = torch.eye(d, device=device, dtype=dtype)
    D = torch.empty(bsize, d, device=device, dtype=dtype).fill_(float('inf'))
    for n in range(d-1):
        # x is a vector with length dim-n, xrow and xcol are views of it as
        # a row vector and column vector respectively. It's important they
        # are views and not copies because we are going to modify x
        # in-place.
        x = torch.randn(bsize, d-n, device=device, dtype=dtype)
        xrow = x[..., None, :]
        xcol = x[..., :, None]

        # This is the squared norm of x, without vectorization it would be
        # dot(x, x), to have proper broadcasting we use matmul and squeeze
        # out (convert to scalar) the resulting 1x1 matrix
        norm2 = torch.matmul(xrow, xcol).squeeze((-2, -1))

        x0 = x[..., 0].clone()
        D[..., n] = torch.where(x0 != 0, torch.sign(x0), 1)
        x[..., 0] += D[..., n] * torch.sqrt(norm2)

        # In renormalizing x we have to append an additional axis with
        # [..., None] to broadcast the scalar against the vector x
        x /= torch.sqrt((norm2 - x0**2 + x[..., 0]**2) / 2.)[..., None]

        # Householder transformation, without vectorization the RHS can be
        # written as outer(H @ x, x) (apart from the slicing)
        H[..., :, n:] -= torch.matmul(H[..., :, n:], xcol) * xrow

    D[..., -1] = (-1)**(d-1) * D[..., :-1].prod(dim=-1)

    # Without vectorization this could be written as H = diag(D) @ H,
    # left-multiplication by a diagonal matrix amounts to multiplying each
    # row of H by an element of the diagonal, so we add a dummy axis for
    # the column index
    H *= D[..., :, None]
    return H
