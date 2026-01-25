import torch
import torch.nn as nn

from .combined_loss import CombinedLoss
from .physics_loss import TerrainPrecipitationLoss, SpatialVariabilityLoss, ExtremeValueLoss
from .statistical_loss import DistributionMatchingLoss, SpatialCorrelationLoss


class EnhancedCombinedLoss(nn.Module):
    """
    增强版组合损失函数
    
    整合:
    - 原有损失: 站点监督 + 物理守恒 + 空间平滑 + 时间一致性
    - 方案1: 地形约束 + 空间变异性 + 极端值分布
    - 方案3: 概率分布匹配 + 空间相关性
    """
    
    def __init__(self, 
                 # 原有损失权重
                 lambda_point=1.0,
                 lambda_conserve=1.0,
                 lambda_smooth=0.1,
                 lambda_temporal=0.05,
                 # 方案1: 物理约束权重
                 lambda_terrain=0.5,
                 lambda_variability=0.2,
                 lambda_extreme=0.3,
                 # 方案3: 统计匹配权重
                 lambda_distribution=0.4,
                 lambda_correlation=0.2,
                 # 可选: 频谱匹配
                 lambda_spectrum=0.0,
                 # 其他参数
                 use_physics_loss=True,
                 use_statistical_loss=True):
        super().__init__()
        
        # 原有损失函数
        self.base_loss = CombinedLoss(
            lambda_point=lambda_point,
            lambda_conserve=lambda_conserve,
            lambda_smooth=lambda_smooth,
            lambda_temporal=lambda_temporal
        )
        
        # 方案1: 物理约束损失
        self.use_physics_loss = use_physics_loss
        if use_physics_loss:
            self.terrain_loss = TerrainPrecipitationLoss()
            self.variability_loss = SpatialVariabilityLoss()
            self.extreme_loss = ExtremeValueLoss()
        
        # 方案3: 统计匹配损失
        self.use_statistical_loss = use_statistical_loss
        if use_statistical_loss:
            self.distribution_loss = DistributionMatchingLoss()
            self.correlation_loss = SpatialCorrelationLoss()
            
            # 可选: 频谱匹配
            if lambda_spectrum > 0:
                from .statistical_loss import SpectralLoss
                self.spectrum_loss = SpectralLoss()
            else:
                self.spectrum_loss = None
        
        # 保存权重
        self.lambda_terrain = lambda_terrain
        self.lambda_variability = lambda_variability
        self.lambda_extreme = lambda_extreme
        self.lambda_distribution = lambda_distribution
        self.lambda_correlation = lambda_correlation
        self.lambda_spectrum = lambda_spectrum
    
    def forward(self, pred, lr_input, s_coords, s_values, scale_factor, dem=None, lucc=None):
        """
        Args:
            pred: [B, T, 1, H, W] 生成的高分辨率降水
            lr_input: [B, T, 1, H_lr, W_lr] 低分辨率输入
            s_coords: [num_stations, 2] 或 [B, num_stations, 2] 站点坐标
            s_values: [B, T, num_stations] 站点观测值
            scale_factor: float 缩放因子
            dem: [B, 1, H_dem, W_dem] 地形数据 (可选)
            lucc: [B, C, H_lu, W_lu] 土地利用数据 (可选)
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        device = pred.device
        
        # 1. 计算原有损失
        base_total_loss, base_loss_dict = self.base_loss(
            pred, lr_input, s_coords, s_values, scale_factor
        )
        
        # 初始化损失字典
        loss_dict = base_loss_dict.copy()
        total_loss = base_total_loss
        
        # 2. 计算方案1: 物理约束损失
        if self.use_physics_loss:
            # 地形-降水关系
            if dem is not None and self.lambda_terrain > 0:
                loss_terrain = self.terrain_loss(pred, dem)
                total_loss = total_loss + self.lambda_terrain * loss_terrain
                loss_dict['terrain'] = loss_terrain.item()
            
            # 空间变异性
            if self.lambda_variability > 0:
                loss_variability = self.variability_loss(pred)
                total_loss = total_loss + self.lambda_variability * loss_variability
                loss_dict['variability'] = loss_variability.item()
            
            # 极端值分布
            if self.lambda_extreme > 0:
                loss_extreme = self.extreme_loss(pred, s_values)
                total_loss = total_loss + self.lambda_extreme * loss_extreme
                loss_dict['extreme'] = loss_extreme.item()
        
        # 3. 计算方案3: 统计匹配损失
        if self.use_statistical_loss:
            # 概率分布匹配
            if self.lambda_distribution > 0:
                loss_distribution = self.distribution_loss(pred, s_values)
                total_loss = total_loss + self.lambda_distribution * loss_distribution
                loss_dict['distribution'] = loss_distribution.item()
            
            # 空间相关性
            if self.lambda_correlation > 0:
                loss_correlation = self.correlation_loss(pred)
                total_loss = total_loss + self.lambda_correlation * loss_correlation
                loss_dict['correlation'] = loss_correlation.item()
            
            # 频谱匹配 (可选)
            if self.spectrum_loss is not None and self.lambda_spectrum > 0:
                loss_spectrum = self.spectrum_loss(pred)
                total_loss = total_loss + self.lambda_spectrum * loss_spectrum
                loss_dict['spectrum'] = loss_spectrum.item()
        
        return total_loss, loss_dict
