from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

#(conv2d+bn+relu)*2 两次conv均为3*3，步长和padding为1的卷积,不改变图片高宽。只在第一次卷积改变通道数。
#1个double_conv模块不改变图片宽高，改变一次图片通道数
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

#使用maxpooling进行下采样/2 downscaling then a double conv
class Down(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            Double_conv(in_channels, out_channels, 3, 1, 1))
    def forward(self, x):
        x = self.downsample(x)
        return x


#上采样 *2  bilinear upsampling + skip_connection + double_conv 
#此处加入了U-Net的skip-connection跳级连接 如果进行concat拼接的两个图片HW不同，需要调整。x1为上采样的特征图，x2为来拼接的特征图
#得到skip connection之后的feature map
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
    

