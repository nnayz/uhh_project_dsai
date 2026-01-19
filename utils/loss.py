import torch
from torch.nn import functional as F


def euclidean_dist(x, y):
    """Compute euclidean distance between two tensors."""
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


def cosine_dist(x, y):
    """Compute cosine distance (1 - cosine similarity) between two tensors."""
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    similarity = torch.mm(x_norm, y_norm.t())
    return 1 - similarity


def get_distance_fn(distance: str = "euclidean"):
    """Return the distance function based on the distance type."""
    if distance == "cosine":
        return cosine_dist
    return euclidean_dist


def prototypical_loss(_input, target, n_support, distance: str = "euclidean"):
    def supp_idxs(c):
        return target.eq(c).nonzero()[:n_support].squeeze(1)

    def query_idxs(c):
        return target.eq(c).nonzero()[n_support:]

    classes = torch.unique(target)
    n_classes = len(classes)
    n_query = target.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([_input[idx_list].mean(0) for idx_list in support_idxs])

    query_idxs = torch.stack(list(map(query_idxs, classes))).view(-1)
    query_samples = _input[query_idxs]
    dist_fn = get_distance_fn(distance)
    dists = dist_fn(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    dist_loss = torch.tensor([0.0]).to(log_p_y.device)

    target_inds = torch.arange(0, n_classes).view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    target_inds = target_inds.to(log_p_y.device)
    loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss, acc_val, dist_loss


def prototypical_loss_filter_negative(_input, target, n_support, distance: str = "euclidean"):
    def supp_idxs(c):
        return target.eq(c).nonzero()[:n_support].squeeze(1)

    def query_idxs(c):
        return target.eq(c).nonzero()[n_support:]

    classes = torch.unique(target)
    pos_classes = classes[~(classes % 2 == 1)]

    n_classes = len(classes)
    n_query = target.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([_input[idx_list].mean(0) for idx_list in support_idxs])

    dist_loss = torch.tensor([0.0]).to(_input.device)

    query_idxs = torch.stack(list(map(query_idxs, pos_classes))).view(-1)
    query_samples = _input[query_idxs]
    dist_fn = get_distance_fn(distance)
    dists = dist_fn(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes // 2, n_query, -1)

    target_inds = torch.arange(0, n_classes // 2) * 2
    target_inds = target_inds.view(n_classes // 2, 1, 1)
    target_inds = target_inds.expand(n_classes // 2, n_query, 1).long()
    target_inds = target_inds.to(log_p_y.device)
    loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss, acc_val, dist_loss
