import torch
from chamferdist import ChamferDistance


def _pointnet_loss(outputs, labels, m3x3, m64x64, criterion, alpha=0.0001):
    """PointNet loss function with transformation regularization.

    Args:
        outputs: (Variable) size batchsize x num_classes
        labels: (Variable) size batchsize
        m3x3: (Variable) size batchsize x 3 x 3
        m64x64: (Variable) size batchsize x 64 x 64
        criterion: (function) loss function
        alpha: (float) weight for transformation regularization

    Returns:
        (Variable) loss
    """
    criterion = criterion()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).expand(bs, -1, -1)
    id64x64 = torch.eye(64, requires_grad=True).expand(bs, -1, -1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (
        torch.norm(diff3x3) + torch.norm(diff64x64)
    ) / float(bs)


def pointnet_classification_loss(outputs, labels, m3x3, m64x64, alpha=0.0001):
    """PointNet classification loss function with transformation regularization.

    Args:
        outputs: (Variable) size batchsize x num_classes
        labels: (Variable) size batchsize
        m3x3: (Variable) size batchsize x 3 x 3
        m64x64: (Variable) size batchsize x 64 x 64
        alpha: (float) weight for transformation regularization

    Returns:
        (Variable) loss
    """
    return _pointnet_loss(
        outputs, labels, m3x3, m64x64, torch.nn.CrossEntropyLoss, alpha
    )


def pointnet_generative_loss(outputs, labels, m3x3, m64x64, alpha=0.0001):
    """PointNet generative loss function with transformation regularization.

    Args:
        outputs: (Variable) size batchsize x num_classes
        labels: (Variable) size batchsize
        m3x3: (Variable) size batchsize x 3 x 3
        m64x64: (Variable) size batchsize x 64 x 64
        alpha: (float) weight for transformation regularization

    Returns:
        (Variable) loss
    """
    return (
        ChamferDistance()(outputs, labels)
    )
