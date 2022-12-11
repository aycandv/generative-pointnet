# Create Autoencoder using PointNet as encoder and a Shared MLP as decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pointnet import PointNet

class InvTransform(nn.Module):
   def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
       
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
   def forward(self, input):
        xb = nn.MaxPool1d(input.size(-1))(input)
        xb = self.bn3(self.conv3(xb))
        xb = F.relu(self.bn2(self.conv2(xb)))
        output = F.relu(self.bn1(self.conv1(xb))) 
        return output

class PointNetAE(nn.Module):
    """PointNet Autoencoder"""
    def __init__(self, device='cpu'):
        """Initialize PointNet Autoencoder model with encoder and decoder modules

        Args:
            device (str, optional): device to load model. Defaults to 'cpu'.

        """
        super().__init__()
        self.encoder = PointNet().transform
        self.decoder = InvTransform()
    
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
