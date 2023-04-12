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
