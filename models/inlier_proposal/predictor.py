import torch
import torch.nn as nn
import torch.nn.functional as F

from . import block

class Predictor(nn.Module):
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
        
        self.unet = block.UnetBlock1d(in_channels, unet_channels)
        self.convo = nn.Conv1d(unet_channels[-1], out_channels, 1, 1, bias=True)
        self.normo = nn.BatchNorm1d(out_channels)
        
    
    def forward(self, x: torch.Tensor):
        y = self.unet(x)
        y = self.normo(self.convo(y))
        return y # no sigmoid to transform into probabilities

