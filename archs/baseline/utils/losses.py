# utils/losses.py

from __future__ import annotations

import torch
import torch.nn.functional as F


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Squared Euclidean distance between rows of x and y.

    x: (Nq, D)
    y: (Nc, D)
    returns: (Nq, Nc)
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise ValueError("Embedding dimension mismatch")

    x_exp = x.unsqueeze(1).expand(n, m, d)
    y_exp = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x_exp - y_exp, 2).sum(dim=2)


def prototypical_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, n_support: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Plain prototypical loss (similar to original baseline). :contentReference[oaicite:2]{index=2}

    embeddings: (N, D)
    labels:     (N,)
    n_support:  number of support examples per class

    Returns:
        loss: scalar
        acc:  scalar accuracy over queries
    """

    def supp_idxs(c: torch.Tensor) -> torch.Tensor:
        return labels.eq(c).nonzero()[:n_support].squeeze(1)

    def query_idxs(c: torch.Tensor) -> torch.Tensor:
        return labels.eq(c).nonzero()[n_support:]

    classes = torch.unique(labels)
    n_classes = len(classes)
    n_query = labels.eq(classes[0].item()).sum().item() - n_support

    support_indices = list(map(supp_idxs, classes))
    prototypes = torch.stack(
        [embeddings[idxs].mean(0) for idxs in support_indices]
    )  # (Nc, D)

    query_indices = torch.stack(list(map(query_idxs, classes))).view(-1)
    query_samples = embeddings[query_indices]  # (Nq, D)

    dists = euclidean_dist(query_samples, prototypes)  # (Nq, Nc)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes, device=embeddings.device)
    target_inds = target_inds.view(n_classes, 1, 1).expand(n_classes, n_query, 1).long()

    loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss, acc_val
