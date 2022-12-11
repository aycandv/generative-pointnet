# A generic script to calculate accuracy of a model depending on the model type

import argparse
import torch
import torch.nn as nn

from metrics.chamfer_dist import chamfer_dist


def calculate_acc(args, input_data, labels, predictions):
    """Calculate accuracy of model

    Args:
        args (argparse.Namespace): Arguments
        input_data (torch.Tensor): Input data
        labels (torch.Tensor): Labels
        predictions (torch.Tensor): Predictions

    Returns:
        acc (float): Accuracy of model
    """
    if args.model == "pointnet-ae":
        acc = _calculate_acc_pointnet_ae(args, input_data, labels, predictions)
    elif args.model == "pointnet":
        acc = _calculate_acc_pointnet_cls(args, input_data, labels, predictions)
    else:
        raise ValueError("Model not supported")
    return acc


def _calculate_acc_pointnet_ae(args, input_data, labels, predictions):
    """Calculate accuracy of PointNet AE

    Args:
        args (argparse.Namespace): Arguments
        input_data (torch.Tensor): Input data
        labels (torch.Tensor): Labels
        predictions (torch.Tensor): Predictions

    Returns:
        acc (float): Accuracy of model
    """
    # Calculate accuracy
    dist = chamfer_dist(input_data, predictions)
    acc = torch.mean(dist)
    return acc


def _calculate_acc_pointnet_cls(args, input_data, labels, predictions):
    """Calculate accuracy of PointNet classifier

    Args:
        args (argparse.Namespace): Arguments
        input_data (torch.Tensor): Input data
        labels (torch.Tensor): Labels
        predictions (torch.Tensor): Predictions

    Returns:
        acc (float): Accuracy of model
    """
    # Calculate accuracy
    _, predicted = torch.max(predictions.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    acc = correct / total
    return acc
