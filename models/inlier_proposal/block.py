import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int=1,
        padding: int=0,
        bias: bool=True
    ) -> None:
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.BatchNorm1d(out_channels)
    
    def forward(self, x: torch.Tensor):
        y = F.relu(self.norm(self.conv(x)))
        return y

class ResidualBlock1d(nn.Module):
    '''
    copy from d2lzh
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        change_input: bool=False
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, 1, bias=True)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1, 1, bias=True)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.convi = None
        if change_input:
            self.convi =nn.Conv1d(in_channels, out_channels, 1, 1, bias=True)
    
    def forward(self, x: torch.Tensor):
        import torch.nn.functional as F
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(x))
        
        if self.convi:
            x = self.convi(x)
        
        y += x
        return F.relu(y)

class UnetBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        unet_channels: int
    ) -> None:
        temp_modules_list = []
        
        import math
        self.half1 = int(math.ceil(len(unet_channels) / 2) - 1)
        self.half2 = len(unet_channels) // 2
        
        num_channels = [in_channels, *unet_channels]
        super().__init__()
        for i in range(1, len(num_channels)):
            temp_modules_list.append(
                nn.Sequential(
                    ConvBlock1d(num_channels[i - 1] if i - 1 <= self.half2 else num_channels[i - 1] + unet_channels[self.half1 - ((i - 1) - self.half2)], num_channels[i], 1, 1),
                    ResidualBlock1d(num_channels[i], num_channels[i])
                )
            )
        
        self.Umodules = nn.ModuleList(temp_modules_list)
    
    def forward(self, x: torch.Tensor):
        y = self.u_propagate(x)
        return y

    def u_propagate(self, x: torch.Tensor):
        unet_output = []
        for i, module in enumerate(self.Umodules):
            if i <= self.half2:
                x = module[0](x)
                unet_output.append(x)
                x = module[1](x)
            else:
                x = torch.concat([x, unet_output[self.half1 - (i - self.half2)]], dim=1)
                x = module(x)
        y = x # this just makes it look nicer...
        return y
