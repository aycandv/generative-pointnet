# Create Autoencoder using PointNet as encoder and a Shared MLP as decoder
import torch
import torch.nn as nn
from models.pointnet import PointNet


class PointNetAE(nn.Module):
    """PointNet Autoencoder"""
    def __init__(self, device='cpu'):
        """Initialize PointNet Autoencoder model with encoder and decoder modules

        Args:
            device (str, optional): device to load model. Defaults to 'cpu'.

        """
        super().__init__()
        self.encoder = PointNet().transform
        self.decoder = []
    
    def forward(self, x):
        """Forward pass of PointNet Autoencoder
        
        Args:
            x (torch.Tensor): input tensor of shape (bs, n, 3)
            
        Returns:
            x (torch.Tensor): output tensor of shape (bs, n, 3)
    
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
