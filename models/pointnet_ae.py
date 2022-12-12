# Create Autoencoder using PointNet as encoder and a Shared MLP as decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pointnet import PointNet


class InvTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.ConvTranspose1d(1024, 1024, 1)
        self.fc2 = nn.ConvTranspose1d(1024, 1024, 1)
        self.fc3 = nn.ConvTranspose1d(1024, 1024, 2)
        self.fc4 = nn.ConvTranspose1d(1024, 1024, 2)

        # batch norm
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(1024)


    def forward(self, x):
        # x.shape == (bs,1024)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        x = x.view(-1, 1024, 3)
        return x


class PointNetAE(nn.Module):
    """PointNet Autoencoder"""

    def __init__(self, device="cpu"):
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
        x, matrix3x3, matrix64x64 = self.encoder(x)
        x = self.decoder(x.unsqueeze(-1))
        return x, matrix3x3, matrix64x64
