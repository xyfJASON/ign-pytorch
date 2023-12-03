from typing import List

import torch.nn as nn
from torch import Tensor

from models.init_weights import init_weights


class AutoEncoder(nn.Module):
    def __init__(
            self,
            img_channels: int = 3,
            dim: int = 64,
            dim_mults: List[int] = (1, 2, 4, 8),
            with_bn: bool = True,
            with_tanh: bool = True,
            init_type: str = 'normal',
    ):
        """ A simple CNN autoencoder. """
        super().__init__()
        self.encoder = Encoder(
            in_dim=img_channels,
            dim=dim,
            dim_mults=dim_mults,
            with_bn=with_bn,
            init_type=init_type,
        )
        self.decoder = Decoder(
            z_dim=dim * dim_mults[-1],
            dim=dim,
            dim_mults=dim_mults[::-1],
            out_dim=img_channels,
            with_bn=with_bn,
            with_tanh=with_tanh,
            init_type=init_type,
        )

    def forward(self, X: Tensor):
        X = self.encoder(X)
        X = self.decoder(X)
        return X


class Encoder(nn.Module):
    def __init__(
            self,
            in_dim: int = 3,
            dim: int = 64,
            dim_mults: List[int] = (1, 2, 4, 8),
            with_bn: bool = True,
            init_type: str = 'normal',
    ):
        """ A simple CNN encoder.

        The network is composed of a stack of convolutional layers.
        Each layer except the last layer reduces the resolution by half.
        The last layer reduces the resolution from 4x4 to 1x1.

        Args:
            in_dim: Input dimension.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            with_bn: Use batch normalization.
            init_type: Type of weight initialization.

        """
        super().__init__()
        self.first_conv = nn.Conv2d(in_dim, dim * dim_mults[0], (4, 4), stride=2, padding=1)
        self.layers = nn.ModuleList([])
        for i in range(len(dim_mults) - 1):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(dim * dim_mults[i]) if i > 0 and with_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim * dim_mults[i], dim * dim_mults[i+1], (4, 4), stride=2, padding=1)
            ))
        self.last_conv = nn.Sequential(
            nn.BatchNorm2d(dim * dim_mults[-1]) if with_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim * dim_mults[-1], dim * dim_mults[-1], (4, 4), stride=1, padding=0),
        )

        self.apply(init_weights(init_type))

    def forward(self, X: Tensor):
        X = self.first_conv(X)
        for layer in self.layers:
            X = layer(X)
        X = self.last_conv(X)
        return X


class Decoder(nn.Module):
    def __init__(
            self,
            z_dim: int = 512,
            dim: int = 64,
            dim_mults: List[int] = (8, 4, 2, 1),
            out_dim: int = 3,
            with_bn: bool = True,
            with_tanh: bool = True,
            init_type: str = 'normal',
    ):
        """ A simple CNN decoder.

        The network is composed of a stack of convolutional layers.
        The first layer outputs a Tensor with a resolution of 4x4.
        Each following layer doubles the resolution using transposed convolution.

        Args:
            z_dim: Dimension of the bottleneck.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            out_dim: Output dimension.
            with_bn: Use batch normalization.
            with_tanh: Use nn.Tanh() at last.
            init_type: Type of weight initialization.

        """
        super().__init__()
        self.first_conv = nn.ConvTranspose2d(z_dim, dim * dim_mults[0], (4, 4), stride=1, padding=0)
        self.layers = nn.ModuleList([])
        for i in range(len(dim_mults) - 1):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(dim * dim_mults[i]) if with_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(dim * dim_mults[i], dim * dim_mults[i+1], (4, 4), stride=2, padding=1),
            ))
        self.last_conv = nn.Sequential(
            nn.BatchNorm2d(dim * dim_mults[-1]) if with_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim * dim_mults[-1], out_dim, (4, 4), stride=2, padding=1),
        )
        self.act = nn.Tanh() if with_tanh else nn.Identity()

        self.apply(init_weights(init_type))

    def forward(self, X: Tensor):
        X = self.first_conv(X)
        for layer in self.layers:
            X = layer(X)
        X = self.last_conv(X)
        X = self.act(X)
        return X


def _test():
    import torch
    autoencoder = AutoEncoder()
    z = torch.randn(10, 3, 64, 64)
    recz = autoencoder(z)
    print(z.shape)
    print(recz.shape)


if __name__ == '__main__':
    _test()
