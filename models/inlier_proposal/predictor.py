import torch
import torch.nn as nn
import torch.nn.functional as F

from . import block

class predictor(nn.Module):
    '''
    ResUNet structure from FCGF, and model idea
    from DGR. DGR use  the  sigmoid  activation
    to transform logits  to  probabilities  and
    use BCELoss to update the model's parameter
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        unet_channels: list
    ) -> None:
        super().__init__()
        
        unet_channels = [in_channels, *unet_channels]
        
        self.unet_layers = []
        for i in range(1, len(unet_channels)):
            self.unet_layers.append(
                nn.Sequential(
                    block.ConvBlock1d(unet_channels[i - 1], unet_channels[i], 1, 1),
                    block.ResidualBlock1d(unet_channels[i])
                )
            )
        
        self.convo = nn.Conv1d(unet_channels[-1], out_channels, 1, 1, bias=True)
        self.normo = nn.BatchNorm1d(out_channels)
        
    
    def forward(self, x: torch.Tensor):
        y = self.unet(x)
        y = self.normo(self.convo(y))
        return y # no sigmoid to transform into probabilities

    def unet(self, x: torch.Tensor):
        half_layer = len(self.unet_layers) // 2
        unet_output = []
        for i, layer in enumerate(self.unet_layers):
            if i < half_layer:
                x = layer(x)
                unet_output.append(x)
            else:
                x = torch.concat([x, unet_output[half_layer - i]], dim=1)
                x = layer(x)
        
        y = x # this just makes it look nicer...
        return y
