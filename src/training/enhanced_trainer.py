import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os

from src.datasets.fenhe_dataset import FenheDataset
from src.datasets.fenhe_dataset_split import split_dataset_by_year, split_dataset_random
from src.losses.enhanced_loss import EnhancedCombinedLoss
from src.models.generator import Generator
from src.utils.visualization import plot_stations_vs_pred, plot_training_curves
from src.config_enhanced import EnhancedConfig, load_enhanced_config


class EnhancedTrainer:
    """
    增强版训练器
    
    支持方案1+3的物理约束和统计匹配损失
    完全兼容原有训练器的接口
    """
    
    def __init__(self, config_name="enhanced"):
        # 加载配置、设备
        self.config = load_enhanced_config(config_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建输出目录
        self.output_dir = self.config.output.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 训练历史记录
        self.history = {
            'epoch': [],
            'total_loss': [],
            'point_loss': [],
            'conserve_loss': [],
            'smooth_loss': [],
            'temporal_loss': [],
            'terrain_loss': [],
            'variability_loss': [],
            'extreme_loss': [],
            'distribution_loss': [],
            'correlation_loss': [],
            'rmse': [],
            'learning_rate': []
        }
        
        # 追踪最佳模型
        self.best_rmse = float('inf')
        self.best_epoch = -1
        
    def setup_data(self):
        """设置数据集和数据加载器"""
        # 创建完整数据集
        full_dataset = FenheDataset(
            rain_lr_path=self.config.data.rain_lr_path,
            dem_path=self.config.data.dem_path,
            lucc_path=self.config.data.lucc_path,
            rain_meta_path=self.config.data.meta_path,
            rain_station_path=self.config.data.rain_excel_path,
            shp_path=self.config.data.shp_path,
            T=self.config.model.T,
            start_year=self.config.data.start_year,
            end_year=self.config.data.end_year
        )
        
        # 判断是否需要划分数据集
        if self.config.training.use_split:
            print(f"\n使用 {self.config.training.split_method} 方法划分数据集...")
            
            if self.config.training.split_method == "year":
                # 按年份划分
                train_dataset, val_dataset, test_dataset = split_dataset_by_year(
                    full_dataset,
                    train_years=tuple(self.config.training.train_years),
                    val_years=tuple(self.config.training.val_years),
                    test_years=tuple(self.config.training.test_years)
                )
            else:
                # 随机划分
                train_dataset, val_dataset, test_dataset = split_dataset_random(
                    full_dataset,
                    train_ratio=0.7,
                    val_ratio=0.15,
                    test_ratio=0.15
                )
            
            # 创建数据加载器
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.training.batch_size, 
                shuffle=True
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False
            )
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False
            )
            
            # 保持向后兼容
            self.dataloader = self.train_loader
            
            print(f"训练集: {len(train_dataset)} 样本")
            print(f"验证集: {len(val_dataset)} 样本")
            print(f"测试集: {len(test_dataset)} 样本\n")
            
        else:
            # 不划分，使用全部数据训练
            print("\n使用全部数据进行训练（未划分数据集）\n")
            self.dataloader = DataLoader(
                full_dataset, 
                batch_size=self.config.training.batch_size, 
                shuffle=True
            )
            self.train_loader = self.dataloader
            self.val_loader = None
            self.test_loader = None
        
        return full_dataset
        
    def setup_model(self, dataset):
        """设置模型、优化器和损失函数"""
        # 获取LUCC通道数（兼容单年和多年数据）
        if hasattr(dataset, 'is_multiyear_lucc') and dataset.is_multiyear_lucc:
            num_lu_classes = dataset.lucc_onehot_list[0].shape[0]
        else:
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
        
        # 使用增强版损失函数
        loss_kwargs = {
            'lambda_point': self.config.training.lambda_point,
            'lambda_conserve': self.config.training.lambda_conserve,
            'lambda_smooth': self.config.training.lambda_smooth,
            'lambda_temporal': self.config.training.lambda_temporal,
        }
        
        # 添加方案1和方案3的权重（如果配置中存在）
        if hasattr(self.config.training, 'lambda_terrain'):
            loss_kwargs['lambda_terrain'] = self.config.training.lambda_terrain
        if hasattr(self.config.training, 'lambda_variability'):
            loss_kwargs['lambda_variability'] = self.config.training.lambda_variability
        if hasattr(self.config.training, 'lambda_extreme'):
            loss_kwargs['lambda_extreme'] = self.config.training.lambda_extreme
        if hasattr(self.config.training, 'lambda_distribution'):
            loss_kwargs['lambda_distribution'] = self.config.training.lambda_distribution
        if hasattr(self.config.training, 'lambda_correlation'):
            loss_kwargs['lambda_correlation'] = self.config.training.lambda_correlation
        if hasattr(self.config.training, 'lambda_spectrum'):
            loss_kwargs['lambda_spectrum'] = self.config.training.lambda_spectrum
        
        self.loss_module = EnhancedCombinedLoss(**loss_kwargs)
        
    def validate(self):
        """在验证集上评估模型"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        val_losses = []
        val_rmses = []
        
        with torch.no_grad():
            for lr, dem, lu, s_coords, s_values in self.val_loader:
                lr, dem, lu = lr.to(self.device), dem.to(self.device), lu.to(self.device)
                s_values, s_coords = s_values.to(self.device), s_coords.to(self.device)
                
                # 前向传播
                forward_kwargs = {}
                if hasattr(self.config.model, 'input_grid_size') and self.config.model.input_grid_size:
                    forward_kwargs['input_grid_size'] = tuple(self.config.model.input_grid_size)
                
                fake_hr = self.model(lr, dem, lu, **forward_kwargs)
                
                # 计算缩放因子
                _, _, _, H_lr, W_lr = lr.shape
                _, _, _, H_hr, W_hr = fake_hr.shape
                scale_factor = H_hr / H_lr
                
                # 计算损失
                loss, _ = self.loss_module(fake_hr, lr, s_coords, s_values, scale_factor, dem, lu)
                val_losses.append(loss.item())
                
                # 计算RMSE
                batch_rmse, _, _, _, _ = self.compute_station_rmse(
                    fake_hr, s_coords, s_values, scale_factor
                )
                val_rmses.append(batch_rmse.item())
        
        self.model.train()
        
        return {
            'loss': np.mean(val_losses),
            'rmse': np.mean(val_rmses)
        }
    
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
            pred_at_stations = None
            true_vals = None
            
        return batch_rmse, pred_at_stations, true_vals, valid_station_mask, mask if valid_num_stations > 0 else None
        
    def train_epoch(self, epoch, T):
        """训练一个epoch"""
        self.model.train()
        all_batch_rmse = []
        epoch_losses = {
            'total': [],
            'point': [],
            'conserve': [],
            'smooth': [],
            'temporal': [],
            'terrain': [],
            'variability': [],
            'extreme': [],
            'distribution': [],
            'correlation': []
        }
        
        for i, (lr, dem, lu, s_coords, s_values) in enumerate(self.train_loader):
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
            
            # 使用增强版损失函数（传入 dem 和 lu）
            loss, loss_dict = self.loss_module(fake_hr, lr, s_coords, s_values, scale_factor, dem, lu)
            
            if torch.isnan(loss):
                print("NaN detected, skip this batch")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.training.grad_clip_norm
            )
            self.optimizer.step()
            
            # 记录损失
            epoch_losses['total'].append(loss.item())
            for key in ['point', 'conserve', 'smooth', 'temporal', 'terrain', 'variability', 'extreme', 'distribution', 'correlation']:
                if key in loss_dict:
                    epoch_losses[key].append(float(loss_dict[key]))
            
            # 计算站点RMSE
            with torch.no_grad():
                batch_rmse, pred_at_stations, true_vals, valid_station_mask, mask = self.compute_station_rmse(
                    fake_hr, s_coords, s_values, scale_factor
                )
                all_batch_rmse.append(batch_rmse.item())
            
            # 日志输出
            if i % self.config.output.log_interval == 0:
                log_msg = f"Epoch {epoch} | Loss: {loss:.4f}"
                for key in ['point', 'conserve', 'terrain', 'distribution']:
                    if key in loss_dict:
                        log_msg += f" | {key.capitalize()}: {loss_dict[key]:.4f}"
                log_msg += f" | RMSE: {batch_rmse:.4f}"
                print(log_msg)
        
        # 计算平均损失
        avg_losses = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in epoch_losses.items()}
        return all_batch_rmse, pred_at_stations, true_vals, s_coords, valid_station_mask, avg_losses
        
    def train(self):
        """完整的训练流程"""
        # 设置数据和模型
        dataset = self.setup_data()
        self.setup_model(dataset)
        
        T = self.config.model.T
        
        for epoch in range(self.config.training.epochs):
            all_batch_rmse, pred_at_stations, true_vals, s_coords, valid_station_mask, avg_losses = self.train_epoch(epoch, T)
            
            # 记录训练历史
            self.history['epoch'].append(epoch)
            self.history['total_loss'].append(avg_losses['total'])
            self.history['point_loss'].append(avg_losses.get('point', 0.0))
            self.history['conserve_loss'].append(avg_losses.get('conserve', 0.0))
            self.history['smooth_loss'].append(avg_losses.get('smooth', 0.0))
            self.history['temporal_loss'].append(avg_losses.get('temporal', 0.0))
            self.history['terrain_loss'].append(avg_losses.get('terrain', 0.0))
            self.history['variability_loss'].append(avg_losses.get('variability', 0.0))
            self.history['extreme_loss'].append(avg_losses.get('extreme', 0.0))
            self.history['distribution_loss'].append(avg_losses.get('distribution', 0.0))
            self.history['correlation_loss'].append(avg_losses.get('correlation', 0.0))
            self.history['rmse'].append(np.mean(all_batch_rmse))
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # 验证
            val_metrics = self.validate()
            if val_metrics:
                print(f"Epoch {epoch} | Train RMSE: {np.mean(all_batch_rmse):.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | Val RMSE: {val_metrics['rmse']:.4f}")
                self.scheduler.step(val_metrics['rmse'])
            else:
                print(f"Epoch {epoch} finished. Avg Batch RMSE: {np.mean(all_batch_rmse):.4f}")
                self.scheduler.step(np.mean(all_batch_rmse))
            
            # 每10个epoch绘制一次收敛曲线
            if (epoch + 1) % 10 == 0:
                plot_training_curves(
                    self.history,
                    save_path=os.path.join(self.output_dir, "training_curves.png")
                )
            
            # 只保存最佳模型
            if val_metrics:
                current_rmse = val_metrics['rmse']
            else:
                current_rmse = np.mean(all_batch_rmse)
            
            if current_rmse < self.best_rmse:
                self.best_rmse = current_rmse
                self.best_epoch = epoch
                
                # 删除旧的最佳模型
                old_best_path = os.path.join(self.output_dir, "best_model.pth")
                if os.path.exists(old_best_path):
                    os.remove(old_best_path)
                
                # 保存新的最佳模型
                best_model_path = os.path.join(self.output_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'rmse': current_rmse,
                    'history': self.history
                }, best_model_path)
                print(f"✓ New best model saved! Epoch {epoch+1}, RMSE: {current_rmse:.4f}")
        
        # 训练结束后绘制最终收敛图和站点对比图
        plot_training_curves(
            self.history,
            save_path=os.path.join(self.output_dir, "final_training_curves.png")
        )
        
        # 保存最终站点对比图
        if pred_at_stations is not None and valid_station_mask.sum() > 0:
            pred_station_mean = pred_at_stations[0].mean(dim=0).cpu().numpy()
            true_station_mean = true_vals[0].mean(dim=0).cpu().numpy()
            plot_stations_vs_pred(
                s_coords[0, valid_station_mask].cpu().numpy(),
                true_station_mean,
                pred_station_mean,
                save_path=os.path.join(self.output_dir, "final_station_comparison.png")
            )
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best model: Epoch {self.best_epoch+1}, RMSE: {self.best_rmse:.4f}")
        print(f"Results saved to {self.output_dir}/")
        print(f"{'='*60}")
