"""
Distance functions used by few-shot methods.
"""

def euclidean_distance(x, y):
    """
    Compute pairwise Euclidean distances between x and y.

    x: [n_x, d], y: [n_y, d] -> returns [n_x, n_y]
    """
    xx = (x ** 2).sum(dim=1, keepdim=True)
    yy = (y ** 2).sum(dim=1, keepdim=True).t()

    xy = x @ y.t()
    dists = xx + yy - 2 * xy
    return dists