import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionMatchingLoss(nn.Module):
    """
    概率分布匹配损失
    
    匹配生成降水与真实降水的概率分布
    使用 KL 散度或 Wasserstein 距离
    """
    
    def __init__(self, num_bins=50):
        super().__init__()
        self.num_bins = num_bins
    
    def compute_histogram(self, x, bins):
        """计算归一化直方图"""
        hist = torch.histc(x, bins=self.num_bins, min=bins[0], max=bins[-1])
        hist = hist / (hist.sum() + 1e-8)  # 归一化，避免除零
        return hist
    
    def forward(self, rain_hr, s_values):
        """
        Args:
            rain_hr: [B, T, 1, H, W] 生成的降水
            s_values: [B, T, num_stations] 站点观测（真实分布）
        
        Returns:
            loss: 标量损失值
        """
        device = rain_hr.device
        
        # 提取非零降水（只关注有降水的情况）
        gen_rain = rain_hr[rain_hr > 0.1]
        obs_rain = s_values[s_values > 0.1]
        
        if len(gen_rain) < 10 or len(obs_rain) < 10:
            return torch.tensor(0.0, device=device)
        
        # 确定分位数范围
        min_val = min(gen_rain.min(), obs_rain.min())
        max_val = max(gen_rain.max(), obs_rain.max())
        
        if max_val - min_val < 1e-6:
            return torch.tensor(0.0, device=device)
        
        bins = torch.linspace(min_val, max_val, self.num_bins + 1, device=device)
        
        # 计算直方图
        hist_gen = self.compute_histogram(gen_rain, bins)
        hist_obs = self.compute_histogram(obs_rain, bins)
        
        # 使用 KL 散度（添加小常数避免 log(0)）
        hist_gen_safe = hist_gen + 1e-8
        hist_obs_safe = hist_obs + 1e-8
        
        loss = F.kl_div(
            hist_gen_safe.log(),
            hist_obs_safe,
            reduction='batchmean'
        )
        
        return loss


class SpatialCorrelationLoss(nn.Module):
    """
    空间相关性匹配损失
    
    匹配空间自相关函数（简化版变异函数）
    """
    
    def __init__(self, max_lag=10, correlation_length=5.0):
        super().__init__()
        self.max_lag = max_lag
        self.correlation_length = correlation_length
    
    def compute_autocorrelation(self, x):
        """
        计算空间自相关函数
        
        Args:
            x: [B, C, H, W]
        
        Returns:
            correlations: [max_lag] 不同距离的相关系数
        """
        B, C, H, W = x.shape
        correlations = []
        
        for lag in range(1, self.max_lag + 1):
            if lag >= min(H, W):
                break
            
            # 水平方向自相关
            if W > lag:
                corr_h = F.cosine_similarity(
                    x[:, :, :, :-lag].flatten(1),
                    x[:, :, :, lag:].flatten(1),
                    dim=1
                )
            else:
                corr_h = torch.tensor(0.0, device=x.device)
            
            # 垂直方向自相关
            if H > lag:
                corr_v = F.cosine_similarity(
                    x[:, :, :-lag, :].flatten(1),
                    x[:, :, lag:, :].flatten(1),
                    dim=1
                )
            else:
                corr_v = torch.tensor(0.0, device=x.device)
            
            # 平均
            avg_corr = (corr_h.mean() + corr_v.mean()) / 2
            correlations.append(avg_corr)
        
        if len(correlations) == 0:
            return torch.tensor([0.0], device=x.device)
        
        return torch.stack(correlations)
    
    def forward(self, rain_hr):
        """
        Args:
            rain_hr: [B, T, 1, H, W]
        
        Returns:
            loss: 标量损失值
        """
        device = rain_hr.device
        
        # 时间平均
        rain_mean = rain_hr.mean(dim=1)  # [B, 1, H, W]
        
        # 计算生成降水的自相关
        gen_corr = self.compute_autocorrelation(rain_mean)
        
        # 理论模型: 指数衰减 corr(d) = exp(-d/L)
        lags = torch.arange(1, len(gen_corr) + 1, device=device).float()
        expected_corr = torch.exp(-lags / self.correlation_length)
        
        # 匹配自相关函数
        loss = F.mse_loss(gen_corr, expected_corr)
        
        return loss


class SpectralLoss(nn.Module):
    """
    频谱匹配损失（可选）
    
    匹配功率谱密度，确保生成降水的频域特性合理
    """
    
    def __init__(self, use_radial_average=True):
        super().__init__()
        self.use_radial_average = use_radial_average
    
    def compute_power_spectrum(self, x):
        """
        计算2D功率谱
        
        Args:
            x: [B, C, H, W]
        
        Returns:
            spectrum: 功率谱
        """
        # FFT
        fft = torch.fft.fft2(x)
        
        # 功率谱
        power = torch.abs(fft) ** 2
        
        if self.use_radial_average:
            # 径向平均（转换为1D谱）
            B, C, H, W = x.shape
            center_h, center_w = H // 2, W // 2
            
            # 创建频率网格
            y, x_grid = torch.meshgrid(
                torch.arange(H, device=x.device) - center_h,
                torch.arange(W, device=x.device) - center_w,
                indexing='ij'
            )
            radius = torch.sqrt(x_grid.float()**2 + y.float()**2).long()
            
            # 径向平均
            max_radius = min(H, W) // 2
            radial_spectrum = []
            for r in range(1, max_radius):
                mask = (radius == r)
                if mask.sum() > 0:
                    radial_spectrum.append(power[..., mask].mean())
                else:
                    radial_spectrum.append(torch.tensor(0.0, device=x.device))
            
            if len(radial_spectrum) == 0:
                return torch.tensor([0.0], device=x.device)
            
            return torch.stack(radial_spectrum)
        else:
            return power
    
    def forward(self, rain_hr):
        """
        Args:
            rain_hr: [B, T, 1, H, W]
        
        Returns:
            loss: 标量损失值
        """
        device = rain_hr.device
        
        # 时间平均
        rain_mean = rain_hr.mean(dim=1)  # [B, 1, H, W]
        
        # 计算功率谱
        gen_spectrum = self.compute_power_spectrum(rain_mean)
        
        # 理论谱: Kolmogorov -5/3 定律
        k = torch.arange(1, len(gen_spectrum) + 1, device=device).float()
        reference_spectrum = k ** (-5/3)
        reference_spectrum = reference_spectrum / (reference_spectrum.sum() + 1e-8)
        
        # 归一化生成谱
        gen_spectrum_norm = gen_spectrum / (gen_spectrum.sum() + 1e-8)
        
        # 匹配谱
        loss = F.mse_loss(gen_spectrum_norm, reference_spectrum)
        
        return loss
