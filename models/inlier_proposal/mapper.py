import os

import torch
import torch.nn as nn
from . import block

import utils

class Mapper(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: list,
        weight: str=None
    ) -> None:
        '''
        will use conf_file if it is provided
        '''
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
        
        if os.path.exists(weight):
            try:
                self.load_state_dict(torch.load(weight))
            except Exception as e:
                utils.log_warn("fail to load weight for mapper:", e)
            utils.log_info("successfully load state dict for mapper")
    
    @classmethod
    def conf_init(cls, conf_file: str):
        with open(conf_file, 'r') as f:
            try:
                import yaml
                mapper_conf = yaml.safe_load(f)
                in_channels = mapper_conf["in_channels"]
                out_channels = mapper_conf["out_channels"]
                mid_channels = mapper_conf["mid_channels"]
                weight = mapper_conf["weight"]
            except Exception as e:
                raise Exception("conf load error:", e)
        return cls(in_channels, out_channels, mid_channels, weight)
    
    def num_feats(self):
        return self.out_channels
    
    def forward(self, x):
        x = self.mapper(x)
        return x

