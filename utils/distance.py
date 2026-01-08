"""
Distance functions and enum used by few-shot methods.
"""

from enum import Enum

import torch


class Distance(Enum):
    """Distance metric enum for prototypical networks."""

    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances between x and y.

    Args:
        x: Tensor of shape [n_x, d]
        y: Tensor of shape [n_y, d]

    Returns:
        Tensor of shape [n_x, n_y] containing pairwise distances.
    """
    xx = (x**2).sum(dim=1, keepdim=True)
    yy = (y**2).sum(dim=1, keepdim=True).t()

    xy = x @ y.t()
    dists = xx + yy - 2 * xy
    return dists
