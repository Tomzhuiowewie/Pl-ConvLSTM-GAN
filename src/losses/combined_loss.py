import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, lambda_point=1.0, lambda_conserve=1.0, lambda_smooth=0.1, lambda_temporal=0.05, 
                 use_weighted_loss=True, weight_strategy='log'):
        super().__init__()
        self.lambda_point = lambda_point    # 站点监督损失权重
        self.lambda_conserve = lambda_conserve  # 物理守恒损失权重
        self.lambda_smooth = lambda_smooth    # 空间平滑损失权重
        self.lambda_temporal = lambda_temporal  # 时间一致性损失权重
        self.l1 = nn.L1Loss()
        
        # 样本不平衡处理参数
        self.use_weighted_loss = use_weighted_loss  # 是否使用加权损失
        self.weight_strategy = weight_strategy      # 权重策略: 'log', 'stratified', 'sqrt'

    # ---------------------------------------------
    # 计算样本权重(处理样本不平衡)
    # ---------------------------------------------
    def compute_sample_weights(self, rain_values):
        """
        根据降水强度计算样本权重
        
        Args:
            rain_values: 降水观测值 (任意shape的tensor)
        
        Returns:
            weights: 与rain_values相同shape的权重tensor
        """
        if not self.use_weighted_loss:
            return torch.ones_like(rain_values)
        
        if self.weight_strategy == 'log':
            # 对数加权: weight = 1 + log(1 + rain)
            # 小雨(0-10mm): 权重 1.0-2.4
            # 中雨(10-25mm): 权重 2.4-3.3
            # 大雨(25-50mm): 权重 3.3-3.9
            # 暴雨(>50mm): 权重 >3.9
            weights = 1.0 + torch.log1p(rain_values)
            
        elif self.weight_strategy == 'stratified':
            # 分层加权: 根据降水等级设置固定权重
            weights = torch.ones_like(rain_values)
            weights = torch.where(rain_values >= 10, torch.tensor(2.0, device=rain_values.device), weights)  # 中雨: 2倍
            weights = torch.where(rain_values >= 25, torch.tensor(3.0, device=rain_values.device), weights)  # 大雨: 3倍
            weights = torch.where(rain_values >= 50, torch.tensor(5.0, device=rain_values.device), weights)  # 暴雨: 5倍
            
        elif self.weight_strategy == 'sqrt':
            # 平方根加权: weight = 1 + sqrt(rain)
            # 比log加权更温和
            weights = 1.0 + torch.sqrt(rain_values)
            
        else:
            weights = torch.ones_like(rain_values)
        
        return weights
    
    # ---------------------------------------------
    # 物理守恒损失
    # ---------------------------------------------
    def conservation_loss(self, pred, lr_input):
        B, T, C, H, W = pred.shape
        H_lr, W_lr = lr_input.shape[-2:]

        pred_lr = F.interpolate(
            pred.view(B * T, C, H, W),
            size=(H_lr, W_lr),
            mode="area"
        ).view(B, T, C, H_lr, W_lr)

        return self.l1(pred_lr, lr_input)

    # ---------------------------------------------
    # 站点监督损失--向量化版本
    # ---------------------------------------------
    def point_supervision_loss(self, pred, s_coords, s_values, scale_factor=1.0):
        """
        向量化的站点监督损失
        """
        if s_values is None or s_coords.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        B, T, _, H, W = pred.shape
        pred_vals = pred[:, :, 0, :, :]  # (B, T, H, W)
        
        # 获取站点坐标
        if len(s_coords.shape) == 3:  # (B, num_stations, 2)
            coords = s_coords[0]
        else:  # (num_stations, 2)
            coords = s_coords
        
        # 缩放站点坐标到高分辨率（考虑网格中心对齐）
        scaled_coords = ((coords.float() + 0.5) * scale_factor - 0.5).long()
        rows = scaled_coords[:, 0]
        cols = scaled_coords[:, 1]
        
        # 边界检查
        valid_station_mask = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        valid_rows = rows[valid_station_mask]
        valid_cols = cols[valid_station_mask]
        valid_num_stations = valid_rows.shape[0]
        
        if valid_num_stations == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # 向量化索引：创建索引张量
        batch_idx = torch.arange(B, device=pred.device).view(B, 1, 1).expand(B, T, valid_num_stations)
        time_idx = torch.arange(T, device=pred.device).view(1, T, 1).expand(B, T, valid_num_stations)
        rows_expand = valid_rows.view(1, 1, -1).expand(B, T, valid_num_stations)
        cols_expand = valid_cols.view(1, 1, -1).expand(B, T, valid_num_stations)
        
        # 提取站点位置的预测值 (B, T, valid_num_stations)
        pred_at_stations = pred_vals[batch_idx, time_idx, rows_expand, cols_expand]
        
        # 获取对应的观测值
        if len(s_values.shape) == 3:  # (B, T, num_stations)
            obs_at_stations = s_values[:, :, valid_station_mask]  # (B, T, valid_num_stations)
        else:  # (T, num_stations)
            obs_at_stations = s_values[:, valid_station_mask].unsqueeze(0).expand(B, -1, -1)
        
        # 过滤 NaN 值
        valid_mask = ~torch.isnan(obs_at_stations)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # 计算样本权重(根据降水强度)
        weights = self.compute_sample_weights(obs_at_stations[valid_mask])
        
        # 使用加权 L1 损失
        loss_per_sample = F.l1_loss(
            pred_at_stations[valid_mask], 
            obs_at_stations[valid_mask],
            reduction='none'
        )
        weighted_loss = (loss_per_sample * weights).mean()
        
        return weighted_loss

    # ---------------------------------------------
    # 空间梯度损失（平滑性约束）
    # ---------------------------------------------
    def gradient_loss(self, pred):
        """
        空间梯度损失，鼓励预测结果平滑
        """
        # 水平梯度
        grad_x = torch.abs(pred[:, :, :, :, :-1] - pred[:, :, :, :, 1:])
        # 垂直梯度
        grad_y = torch.abs(pred[:, :, :, :-1, :] - pred[:, :, :, 1:, :])
        
        return grad_x.mean() + grad_y.mean()
    
    # ---------------------------------------------
    # 时间一致性损失
    # ---------------------------------------------
    def temporal_consistency_loss(self, pred):
        """
        时间一致性损失，鼓励相邻时间步的预测平滑
        """
        # pred: [B, T, 1, H, W]
        # 计算相邻时间步的差异
        temporal_diff = torch.abs(pred[:, :-1] - pred[:, 1:])
        
        return temporal_diff.mean()
    
    # ---------------------------------------------
    # 总损失
    # ---------------------------------------------
    def forward(self, pred, lr_input, s_coords, s_values, scale_factor=1.0):
        loss_point = self.point_supervision_loss(pred, s_coords, s_values, scale_factor)
        loss_conserve = self.conservation_loss(pred, lr_input)
        loss_smooth = self.gradient_loss(pred)
        loss_temporal = self.temporal_consistency_loss(pred)
        
        total_loss = (
            self.lambda_point * loss_point + 
            self.lambda_conserve * loss_conserve +
            self.lambda_smooth * loss_smooth +
            self.lambda_temporal * loss_temporal
        )

        return total_loss, {
            "point": loss_point,
            "conserve": loss_conserve,
            "smooth": loss_smooth,
            "temporal": loss_temporal
        }