"""Distance functions used by few-shot methods."""
import torch


def euclidean_distance(x, y):
    """Compute pairwise Euclidean distances between x and y.

    x: [n_x, d], y: [n_y, d] -> returns [n_x, n_y]
    """
    n_x = x.size(0)
    n_y = y.size(0)
    xx = (x ** 2).sum(dim=1, keepdim=True)  # [n_x, 1]
    yy = (y ** 2).sum(dim=1, keepdim=True).t()  # [1, n_y]
    xy = x @ y.t()  # [n_x, n_y]
    dists = xx + yy - 2 * xy
    return dists
