from .loss import pointnet_classification_loss, pointnet_generative_loss


def generate_loss(type: str):
    """Generate loss function based on model type

    Args:
        type (str): model type

    Returns:
        loss: loss function

    Raises:
        ValueError: Invalid model type

    Examples:

    >>> loss = generate_loss('pointnet')
    >>> loss = generate_loss('pointnet-ae')
    >>> loss = generate_loss('pvcnn')
    >>> loss = generate_loss('pvcnn-ae')
    >>> loss = generate_loss('pvcnn-res-ae')
    """

    if type == "pointnet":
        return pointnet_classification_loss
    elif type == "pointnet-ae":
        return pointnet_generative_loss
    elif type == "pvcnn":
        return pointnet_classification_loss
    elif type == "pvcnn-ae":
        return pointnet_generative_loss
    elif type == "pvcnn-res-ae":
        return pointnet_generative_loss
    else:
        raise ValueError("Invalid model type")
