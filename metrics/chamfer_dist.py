import torch


def chamfer_dist(x, y):
    """Chamfer distance between two point clouds

    Args:
        x (torch.Tensor): point cloud of shape (bs, n, 3)
        y (torch.Tensor): point cloud of shape (bs, n, 3)

    Returns:
        torch.Tensor: chamfer distance between x and y
    """
    x = x.permute(0, 2, 1)
    y = y.permute(0, 2, 1)
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)
    dist = torch.sum((x - y) ** 2, dim=1)
    dist1 = torch.min(dist, dim=2)[0]
    dist2 = torch.min(dist, dim=1)[0]
    loss = torch.mean(dist1) + torch.mean(dist2)
    return loss
