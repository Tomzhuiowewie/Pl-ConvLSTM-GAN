import torch
import torch.nn as nn
import torch.nn.functional as F


class TerrainPrecipitationLoss(nn.Module):
    """
    地形-降水关系约束损失
    
    物理原理:
    1. 海拔越高，降水越多（地形抬升效应）
    2. 降水梯度与地形梯度正相关
    """
    
    def __init__(self, elevation_bins=10):
        super().__init__()
        self.elevation_bins = elevation_bins
    
    def forward(self, rain_hr, dem):
        """
        Args:
            rain_hr: [B, T, 1, H, W] 生成的高分辨率降水
            dem: [B, 1, H_dem, W_dem] 地形高程
        
        Returns:
            loss: 标量损失值
        """
        B, T, C, H, W = rain_hr.shape
        device = rain_hr.device
        
        # 调整 DEM 到相同分辨率
        dem_resized = F.interpolate(dem, size=(H, W), mode='bilinear', align_corners=False)
        
        # 时间平均降水
        rain_mean = rain_hr.mean(dim=1)  # [B, 1, H, W]
        
        # 损失1: 降水与海拔的正相关性
        dem_flat = dem_resized.flatten()
        rain_flat = rain_mean.flatten()
        
        # 计算每个高程带的平均降水
        dem_min, dem_max = dem_flat.min(), dem_flat.max()
        if dem_max - dem_min < 1e-6:
            elevation_loss = torch.tensor(0.0, device=device)
        else:
            bin_edges = torch.linspace(dem_min, dem_max, self.elevation_bins + 1, device=device)
            
            elevation_rain = []
            for i in range(self.elevation_bins):
                mask = (dem_flat >= bin_edges[i]) & (dem_flat < bin_edges[i+1])
                if mask.sum() > 0:
                    elevation_rain.append(rain_flat[mask].mean())
                else:
                    elevation_rain.append(torch.tensor(0.0, device=device))
            
            elevation_rain = torch.stack(elevation_rain)
            
            # 期望: 高海拔降水 >= 低海拔降水（使用 ReLU 惩罚违反的情况）
            elevation_loss = F.relu(elevation_rain[:-1] - elevation_rain[1:]).mean()
        
        # 损失2: 降水梯度与地形梯度的相关性
        if H > 1 and W > 1:
            dem_grad_x = dem_resized[:, :, :, 1:] - dem_resized[:, :, :, :-1]
            dem_grad_y = dem_resized[:, :, 1:, :] - dem_resized[:, :, :-1, :]
            
            rain_grad_x = rain_mean[:, :, :, 1:] - rain_mean[:, :, :, :-1]
            rain_grad_y = rain_mean[:, :, 1:, :] - rain_mean[:, :, :-1, :]
            
            # 使用余弦相似度
            correlation_x = F.cosine_similarity(
                rain_grad_x.flatten(1),
                dem_grad_x.flatten(1),
                dim=1
            )
            correlation_y = F.cosine_similarity(
                rain_grad_y.flatten(1),
                dem_grad_y.flatten(1),
                dim=1
            )
            
            # 期望: 正相关（相关系数接近1）
            correlation_loss = (1 - correlation_x).mean() + (1 - correlation_y).mean()
        else:
            correlation_loss = torch.tensor(0.0, device=device)
        
        total_loss = elevation_loss + 0.5 * correlation_loss
        
        return total_loss


class SpatialVariabilityLoss(nn.Module):
    """
    空间变异性约束损失
    
    物理原理:
    真实降水有合理的空间变化尺度，不应过度平滑或过度噪声
    """
    
    def __init__(self, target_variance=0.15, kernel_size=5):
        super().__init__()
        self.target_var = target_variance
        self.kernel_size = kernel_size
    
    def compute_local_variance(self, x):
        """计算局部方差"""
        # 局部均值
        local_mean = F.avg_pool2d(
            x, self.kernel_size, stride=1, padding=self.kernel_size//2
        )
        
        # 局部方差
        local_var = F.avg_pool2d(
            (x - local_mean) ** 2,
            self.kernel_size, stride=1, padding=self.kernel_size//2
        )
        
        return local_var
    
    def forward(self, rain_hr):
        """
        Args:
            rain_hr: [B, T, 1, H, W]
        
        Returns:
            loss: 标量损失值
        """
        # 时间平均
        rain_mean = rain_hr.mean(dim=1)  # [B, 1, H, W]
        
        # 计算局部方差
        local_var = self.compute_local_variance(rain_mean)
        
        # 期望: 局部方差在合理范围
        var_loss = F.mse_loss(
            local_var.mean(),
            torch.tensor(self.target_var, device=rain_hr.device)
        )
        
        return var_loss


class ExtremeValueLoss(nn.Module):
    """
    极端值分布约束损失
    
    物理原理:
    极端降水（暴雨）应该符合真实观测的统计分布
    """
    
    def __init__(self, percentile=95):
        super().__init__()
        self.percentile = percentile
    
    def forward(self, rain_hr, s_values):
        """
        Args:
            rain_hr: [B, T, 1, H, W] 生成的降水
            s_values: [B, T, num_stations] 站点观测
        
        Returns:
            loss: 标量损失值
        """
        device = rain_hr.device
        
        # 提取生成降水的极端值
        gen_flat = rain_hr.flatten()
        gen_flat = gen_flat[gen_flat > 0]  # 只考虑有降水的网格
        
        if len(gen_flat) == 0:
            return torch.tensor(0.0, device=device)
        
        threshold_idx = max(1, int(len(gen_flat) * self.percentile / 100))
        gen_extremes = torch.topk(gen_flat, k=threshold_idx).values
        
        # 提取观测的极端值
        obs_flat = s_values.flatten()
        obs_flat = obs_flat[~torch.isnan(obs_flat)]
        obs_flat = obs_flat[obs_flat > 0]
        
        if len(obs_flat) == 0:
            return torch.tensor(0.0, device=device)
        
        obs_threshold_idx = max(1, int(len(obs_flat) * self.percentile / 100))
        obs_extremes = torch.topk(obs_flat, k=obs_threshold_idx).values
        
        # 损失1: 极端值的均值应该接近
        mean_loss = F.mse_loss(
            gen_extremes.mean(),
            obs_extremes.mean()
        )
        
        # 损失2: 极端值的标准差应该接近
        std_loss = F.mse_loss(
            gen_extremes.std(),
            obs_extremes.std()
        )
        
        total_loss = mean_loss + 0.5 * std_loss
        
        return total_loss
