import torch
import torch.nn as nn
import torch.nn.functional as F

from .convlstm import ConvLSTMCell
from .attention import DEMAttention, LUAttention
from .coordconv import add_coord_channels


class UpsampleBlock(nn.Module):
    """逐级上采样块，使用PixelShuffle实现2x超分辨率"""
    def __init__(self, in_channels, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(
            in_channels, 
            in_channels * (upscale_factor ** 2), 
            kernel_size=3, 
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=1, dem_channels=1, lu_channels=0, 
                 hidden_dims=[32,64], target_grid_size=None, scale_factor=None):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.lu_channels = lu_channels
        
        # 计算上采样配置
        if target_grid_size is not None:
            # 情况1：使用目标网格尺寸
            self.target_grid_size = target_grid_size
            self.scale_factor = None
            self.target_size = None
        elif scale_factor is not None:
            # 情况2：使用缩放因子
            self.scale_factor = scale_factor
            self.target_grid_size = None
            self.target_size = None

        self.init_conv = nn.Conv2d(
            in_channels + 2,  # 输入通道数：降水 + 坐标
            hidden_dims[0],   # 隐藏层通道数
            kernel_size=3,    # 卷积核大小
            padding=1         # 填充
        )

        self.cell1 = ConvLSTMCell(hidden_dims[0], hidden_dims[0])
        self.cell2 = ConvLSTMCell(hidden_dims[0], hidden_dims[1])

        self.dem_attn = DEMAttention(hidden_dims[1], dem_channels)
        self.lu_attn = LUAttention(hidden_dims[1], lu_channels)
        
        # 动态上采样模块列表（运行时初始化）
        self.upsample_blocks = None
        
        # 后处理层，适应超分辨率后的特征
        self.post_process = nn.Sequential(
            nn.Conv2d(hidden_dims[1], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1)
        )
        
    def _setup_upsample_blocks(self, scale_factor, device):
        """根据缩放因子设置上采样模块"""
        if scale_factor <= 1:
            self.upsample_blocks = nn.ModuleList()
            return 1
            
        # 分解scale_factor为2的幂次组合
        blocks = []
        current_factor = scale_factor
        
        while current_factor > 1:
            if current_factor >= 2:
                blocks.append(UpsampleBlock(self.hidden_dims[1], 2))
                current_factor //= 2
            else:
                # 对于非整数倍，使用插值
                break
                
        self.upsample_blocks = nn.ModuleList(blocks).to(device)
        return scale_factor / (2 ** len(blocks))

    def forward(self, rain_lr, dem, lu, input_grid_size=None):
        """
        Args:
            rain_lr: [B, T, 1, H, W] 低分辨率降水
            dem: [B, 1, H_dem, W_dem] 地形数据（可以是高分辨率，如 1km）
            lu: [B, C, H_lu, W_lu] 土地利用数据（可以是高分辨率，如 1km）
            input_grid_size: (grid_size_x, grid_size_y) 输入数据的网格尺寸（米），用于计算目标尺寸
        """
        B, T, C, H, W = rain_lr.shape
        device = rain_lr.device
        
        # 计算目标分辨率
        if self.target_grid_size is not None and input_grid_size is not None:
            # 根据输入和目标网格尺寸计算缩放因子
            input_grid_x, input_grid_y = input_grid_size
            target_grid_x, target_grid_y = self.target_grid_size
            
            # 计算缩放因子（输入网格尺寸 / 目标网格尺寸）
            scale_factor_w = input_grid_x / target_grid_x
            scale_factor_h = input_grid_y / target_grid_y
            
            # 计算目标网格数量
            target_W = int(W * scale_factor_w)
            target_H = int(H * scale_factor_h)
            
            self.target_size = (target_H, target_W)
            scale_factor = max(scale_factor_h, scale_factor_w)
        elif self.scale_factor is not None:
            scale_factor = self.scale_factor
            self.target_size = None
        else:
            scale_factor = 1
            self.target_size = None
            
        # 设置上采样模块（如果还未设置）
        if self.upsample_blocks is None:
            remaining_factor = self._setup_upsample_blocks(int(scale_factor), device)
        else:
            remaining_factor = scale_factor / (2 ** len(self.upsample_blocks))
        
        # 计算降水特征上采样后的最终尺寸
        if self.target_size is not None:
            final_H, final_W = self.target_size
        else:
            # 根据 scale_factor 计算最终尺寸
            final_H = int(H * scale_factor)
            final_W = int(W * scale_factor)
        
        # 将 DEM 和 LUCC 上采样到与降水特征相同的最终尺寸
        dem_hr = F.interpolate(
            dem, 
            size=(final_H, final_W), 
            mode='bilinear', 
            align_corners=False
        )
        lu_hr = F.interpolate(
            lu, 
            size=(final_H, final_W), 
            mode='nearest'
        )

        # 初始化隐藏状态（在原始分辨率下）
        h1 = torch.zeros(B, self.hidden_dims[0], H, W, device=rain_lr.device)
        c1 = torch.zeros_like(h1)

        h2 = torch.zeros(B, self.hidden_dims[1], H, W, device=rain_lr.device)
        c2 = torch.zeros_like(h2)

        outputs = []

        for t in range(T):
            # 只处理降水数据，不拼接 DEM/LUCC
            x_t = rain_lr[:, t]  # (B, 1, H, W)
            x_t = add_coord_channels(x_t)  # (B, 3, H, W)
            x_t = F.relu(self.init_conv(x_t))  # (B, hidden_dims[0], H, W)

            h1, c1 = self.cell1(x_t, h1, c1)
            h2, c2 = self.cell2(h1, h2, c2)
            
            # 上采样到目标分辨率
            feat = h2
            for upsample_block in self.upsample_blocks:
                feat = upsample_block(feat)
            
            # 处理剩余的非整数倍缩放
            if remaining_factor > 1:
                feat = F.interpolate(
                    feat, 
                    scale_factor=remaining_factor, 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 如果指定了目标尺寸，精确调整
            if self.target_size is not None:
                target_H, target_W = self.target_size
                feat = F.interpolate(
                    feat, 
                    size=(target_H, target_W), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 在最终分辨率下融合 DEM 和 LUCC
            feat = self.dem_attn(feat, dem_hr)
            feat = self.lu_attn(feat, lu_hr)
            
            # 后处理输出
            out_t = self.post_process(feat)
            outputs.append(out_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)
