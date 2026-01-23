import torch.nn as nn

class DEMAttention(nn.Module):
    def __init__(self, in_channels, dem_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dem_channels, in_channels//2, 3, padding=1), # 通过卷积将地形数据转换为特征
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels, 1), # 通过卷积将特征转换为与输入相同通道数的特征--生成注意力图
            nn.Sigmoid()
        )
    def forward(self, x, dem):
        return x * self.conv(dem)


class LUAttention(nn.Module):
    def __init__(self, in_channels, lu_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(lu_channels, in_channels//2, 3, padding=1), # 通过卷积将土地利用数据转换为特征
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels, 1),  # 通过卷积将特征转换为与输入相同通道数的特征--生成注意力图
            nn.Sigmoid()
        )
    def forward(self, x, lu):
        return x * self.conv(lu)
