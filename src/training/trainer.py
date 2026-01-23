import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os

from src.datasets.fenhe_dataset import FenheDataset
from src.losses.combined_loss import CombinedLoss
from src.models.generator import Generator
from src.utils.visualization import plot_stations_vs_pred, plot_rmse_per_time_step
from src.config import Config, load_config


class Trainer:
    def __init__(self, config=None, config_name="default"):
        if config is None:
            self.config = load_config(config_name)
        else:
            self.config = config
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建输出目录
        self.output_dir = self.config.output.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def setup_data(self):
        """设置数据集和数据加载器"""
        dataset = FenheDataset(
            rain_lr_path=self.config.data.rain_lr_path,
            dem_path=self.config.data.dem_path,
            lucc_path=self.config.data.lucc_path,
            rain_meta_path=self.config.data.meta_path,
            rain_station_path=self.config.data.rain_excel_path,
            shp_path=self.config.data.shp_path,
            T=self.config.model.T
        )
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.config.training.batch_size, 
            shuffle=True
        )
        return dataset
        
    def setup_model(self, dataset):
        """设置模型、优化器和损失函数"""
        num_lu_classes = dataset.lucc_onehot.shape[0]
        
        # 准备模型参数
        model_kwargs = {
            'hidden_dims': self.config.model.hidden_dims,
            'lu_channels': num_lu_classes
        }
        
        # 添加上采样配置
        if hasattr(self.config.model, 'target_grid_size') and self.config.model.target_grid_size:
            # 情况1：使用目标网格尺寸
            model_kwargs['target_grid_size'] = tuple(self.config.model.target_grid_size)
        elif hasattr(self.config.model, 'scale_factor') and self.config.model.scale_factor:
            # 情况2：使用缩放因子
            model_kwargs['scale_factor'] = self.config.model.scale_factor
        
        self.model = Generator(**model_kwargs).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.training.learning_rate
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=self.config.training.scheduler_factor, 
            patience=self.config.training.scheduler_patience
        )
        
        self.loss_module = CombinedLoss(
            lambda_point=self.config.training.lambda_point, 
            lambda_conserve=self.config.training.lambda_conserve,
            lambda_smooth=self.config.training.lambda_smooth,
            lambda_temporal=self.config.training.lambda_temporal
        )
        
    def compute_station_rmse(self, fake_hr, s_coords, s_values, scale_factor=1.0):
        """计算站点RMSE"""
        B, T, C, H, W = fake_hr.shape
        pred_vals = fake_hr[:,:,0,:,:]  # (B,T,H,W)
        
        # 获取站点坐标
        if len(s_coords.shape) == 3:  # (B, num_stations, 2)
            coords = s_coords[0]
        else:  # (num_stations, 2)
            coords = s_coords
        
        # 缩放站点坐标到高分辨率（考虑网格中心对齐）
        scaled_coords = ((coords.float() + 0.5) * scale_factor - 0.5).long()
        rows = scaled_coords[:, 0]
        cols = scaled_coords[:, 1]
            
        num_stations = rows.shape[0]
        
        # 边界检查
        valid_station_mask = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        valid_rows = rows[valid_station_mask]
        valid_cols = cols[valid_station_mask]
        valid_num_stations = valid_rows.shape[0]
        
        if valid_num_stations > 0:
            batch_idx = torch.arange(B, device=self.device).view(B,1,1).expand(B,T,valid_num_stations)
            time_idx = torch.arange(T, device=self.device).view(1,T,1).expand(B,T,valid_num_stations)
            rows_expand = valid_rows.view(1,1,-1).expand(B,T,valid_num_stations)
            cols_expand = valid_cols.view(1,1,-1).expand(B,T,valid_num_stations)
            
            pred_at_stations = pred_vals[batch_idx, time_idx, rows_expand, cols_expand]
            
            # 正确索引 s_values
            if len(s_coords.shape) == 3:
                true_vals = s_values[:, :, valid_station_mask]
            else:
                true_vals = s_values[:, :, valid_station_mask]
                
            mask = ~torch.isnan(true_vals)
            
            if mask.sum() > 0:
                batch_rmse = torch.sqrt(F.mse_loss(pred_at_stations[mask], true_vals[mask]))
            else:
                batch_rmse = torch.tensor(0.0, device=self.device)
        else:
            batch_rmse = torch.tensor(0.0, device=self.device)
            
        return batch_rmse, pred_at_stations, true_vals, valid_station_mask, mask
        
    def train_epoch(self, epoch, T):
        """训练一个epoch"""
        all_batch_rmse = []
        rmse_time_history = [[] for _ in range(T)]
        
        for i, (lr, dem, lu, s_coords, s_values) in enumerate(self.dataloader):
            lr, dem, lu = lr.to(self.device), dem.to(self.device), lu.to(self.device)
            s_values, s_coords = s_values.to(self.device), s_coords.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 准备 forward 参数
            forward_kwargs = {}
            if hasattr(self.config.model, 'input_grid_size') and self.config.model.input_grid_size:
                forward_kwargs['input_grid_size'] = tuple(self.config.model.input_grid_size)
            
            fake_hr = self.model(lr, dem, lu, **forward_kwargs)
            
            # 计算缩放因子
            _, _, _, H_lr, W_lr = lr.shape
            _, _, _, H_hr, W_hr = fake_hr.shape
            scale_factor = H_hr / H_lr
            
            loss, loss_dict = self.loss_module(fake_hr, lr, s_coords, s_values, scale_factor)
            
            if torch.isnan(loss):
                print("NaN detected, skip this batch")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.training.grad_clip_norm
            )
            self.optimizer.step()
            
            # 计算站点RMSE
            with torch.no_grad():
                batch_rmse, pred_at_stations, true_vals, valid_station_mask, mask = self.compute_station_rmse(
                    fake_hr, s_coords, s_values, scale_factor
                )
                all_batch_rmse.append(batch_rmse.item())
                
                # 分时间步RMSE
                if valid_station_mask.sum() > 0:
                    for t in range(T):
                        mask_t = mask[:,t,:]
                        if mask_t.sum() > 0:
                            rmse_t = torch.sqrt(F.mse_loss(pred_at_stations[:,t,:][mask_t], true_vals[:,t,:][mask_t]))
                        else:
                            rmse_t = torch.tensor(0.0, device=self.device)
                        rmse_time_history[t].append(rmse_t.item())
            
            # 日志输出
            if i % self.config.output.log_interval == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f} | Point: {loss_dict['point']:.4f} | "
                      f"Conserve: {loss_dict['conserve']:.4f} | Smooth: {loss_dict['smooth']:.4f} | "
                      f"Temporal: {loss_dict['temporal']:.4f} | Batch RMSE: {batch_rmse:.4f}")
                      
        return all_batch_rmse, rmse_time_history, pred_at_stations, true_vals, s_coords, valid_station_mask
        
    def train(self):
        """完整的训练流程"""
        # 设置数据和模型
        dataset = self.setup_data()
        self.setup_model(dataset)
        
        T = self.config.model.T
        
        for epoch in range(self.config.training.epochs):
            all_batch_rmse, rmse_time_history, pred_at_stations, true_vals, s_coords, valid_station_mask = self.train_epoch(epoch, T)
            
            # 保存站点对比图
            if valid_station_mask.sum() > 0:
                pred_station_mean = pred_at_stations[0].mean(dim=0).cpu().numpy()
                true_station_mean = true_vals[0].mean(dim=0).cpu().numpy()
                plot_stations_vs_pred(
                    s_coords[0, valid_station_mask].cpu().numpy(),
                    true_station_mean,
                    pred_station_mean,
                    save_path=os.path.join(self.output_dir, f"station_comparison_epoch_{epoch}.png")
                )
            
            print(f"Epoch {epoch} finished. Avg Batch RMSE: {np.mean(all_batch_rmse):.4f}")
            self.scheduler.step(np.mean(all_batch_rmse))
            
            # 绘制时间步RMSE图
            plot_rmse_per_time_step(rmse_time_history, epoch, self.output_dir)
            
            # 保存模型
            if (epoch + 1) % self.config.output.save_model_interval == 0:
                model_path = os.path.join(self.output_dir, f"model_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'rmse': np.mean(all_batch_rmse)
                }, model_path)
                print(f"Model saved to {model_path}")
