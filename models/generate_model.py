from models.pointnet import PointNet
from models.pointnet_ae import PointNetAE

def generate_model(type: str, device: str = 'cpu'): 
    """Generate model based on type

    Args:
        type (str): model type
        device (str, optional): device to load model. Defaults to 'cpu'.

    Returns:
        model: model

    Raises:
        ValueError: Invalid model type

    Examples:

    >>> model = generate_model('pointnet')
    >>> model = generate_model('pointnet-ae')
    >>> model = generate_model('pvcnn')
    >>> model = generate_model('pvcnn-ae')
    >>> model = generate_model('pvcnn-res-ae')
    """
    
    if type == 'pointnet':
        return PointNet().to(device)
    elif type == 'pointnet-ae':
        return PointNetAE().to(device)
        raise NotImplementedError
    elif type == 'pvcnn':
        # return PVCNN().to(device)
        raise NotImplementedError
    elif type == 'pvcnn-ae':
        # return PVCNNAE().to(device)
        raise NotImplementedError
    elif type == 'pvcnn-res-ae':
        # return PVCNNResAE().to(device)
        raise NotImplementedError