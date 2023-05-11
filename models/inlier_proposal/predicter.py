import os
import torch
import torch.nn as nn

from . import block

import utils

class Predicter(nn.Module):
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
        mid_channels: list,
        weight: str=None
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.unet = block.UnetBlock1d(in_channels, mid_channels)
        self.convo = nn.Conv1d(mid_channels[-1], out_channels, 1, 1, bias=True)
        self.normo = nn.BatchNorm1d(out_channels)

        if os.path.exists(weight):
            try:
                self.load_state_dict(torch.load(weight))
                utils.log_info("successfully load state dict for predictor")
            except Exception as e:
                utils.log_warn("fail to load weight for predictor:", e)
                utils.log_warn("run without pretrained weight")
        else:
            utils.log_warn("weights not found, load with empty weights")
    
    @classmethod
    def conf_init(cls, conf_file: str):
        with open(conf_file, 'r') as f:
            try:
                import yaml
                pred_conf = yaml.safe_load(f)
                in_channels = pred_conf["in_channels"]
                out_channels = pred_conf["out_channels"]
                mid_channels = pred_conf["mid_channels"]
                weight = pred_conf["weight"]
            except Exception as e:
                raise Exception("conf load error:", e)
        return cls(in_channels, out_channels, mid_channels, weight)
    
    def forward(self, x: torch.Tensor):
        y = self.unet(x)
        y = self.normo(self.convo(y))
        return y # no sigmoid to transform into probabilities
