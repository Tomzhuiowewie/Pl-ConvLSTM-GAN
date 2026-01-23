import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, lambda_point=1.0, lambda_conserve=1.0, lambda_smooth=0.1, lambda_temporal=0.05):
        super().__init__()
        self.lambda_point = lambda_point    # 站点监督损失权重
        self.lambda_conserve = lambda_conserve  # 物理守恒损失权重
        self.lambda_smooth = lambda_smooth    # 空间平滑损失权重
        self.lambda_temporal = lambda_temporal  # 时间一致性损失权重
        self.l1 = nn.L1Loss()

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
        
        # 使用 L1 损失（与守恒损失一致）
        loss = F.l1_loss(pred_at_stations[valid_mask], obs_at_stations[valid_mask])
        
        return loss

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