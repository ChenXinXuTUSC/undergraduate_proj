import torch.nn as nn
from . import block

class Mapper(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: list
    ) -> None:
        super().__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels
        
        layer_channels = [in_channels, *mid_channels]
        conv_layers = []
        for i in range(1, len(layer_channels)):
            conv_layers.append(
                nn.Sequential(
                    block.ConvBlock1d(layer_channels[i - 1], layer_channels[i], 1),
                    block.ResidualBlock1d(layer_channels[i], layer_channels[i])
                )
            )
            # conv_layers.append(nn.Conv1d(layer_channels[i - 1], layer_channels[i], 1, 1, bias=True))
            # conv_layers.append(nn.BatchNorm1d(layer_channels[i]))
            # conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv1d(layer_channels[-1], out_channels, 1, 1, bias=True))
        conv_layers.append(nn.BatchNorm1d(out_channels))
        # no relu at the last layer
        
        self.mapper = nn.Sequential(*conv_layers)
    
    def forward(self, x):
        x = self.mapper(x)
        return x

