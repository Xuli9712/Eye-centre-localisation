from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class Double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1):
        super().__init__()
        self.double_conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.double_conv2d(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            Double_conv(in_channels, out_channels, 3, 1, 1))
    def forward(self, x):
        x = self.downsample(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = Double_conv(in_channels, out_channels, 3, 1, 1)
        
    
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        #diffH = torch.tensor(x2.size()[2]-x1.size()[2])
        #diffW = torch.tensor(x2.size()[3]-x1.size()[3])
        #对x1填充至和x2相同大小
        #x1 = F.pad(x1, (diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2))
        #在通道维度上进行拼接，B，H，W值大小应一致
        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x


#1*1卷积，不改变图像高宽，改变通道数
class Out_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0):
        super().__init__()
        self.outconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):
        x = self.outconv(x)
        return x
    

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out

    

