import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import os
import random
import matplotlib.pyplot as plt

# --------------------------- 1. 注意力增强模块 ------------------------------------

class DEMAttention(nn.Module):
    """
    DEM Attention: Uses elevation info to generate spatial weights for precipitation.
    地形注意力模块：利用高程信息生成空间权重。
    """
    def __init__(self, in_channels, dem_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dem_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, dem):
        attn = self.conv(dem)
        return x * attn

class LUAttention(nn.Module):
    """
    LU Attention: Provides differentiated feature weighting based on land use types.
    土地利用注意力模块：根据下垫面类型提供特征加权。
    """
    def __init__(self, in_channels, lu_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(lu_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, lu):
        attn = self.conv(lu)
        return x * attn

# --------------------------- 2. ConvLSTM 核心组件 ---------------------------------

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)
        i, f, o, g = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o), torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# --------------------------- 3. 生成器 (Generator) -------------------------------

class Generator(nn.Module):
    def __init__(self, in_channels=1, dem_channels=1, lu_channels=1, hidden_dims=[32, 64]):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.init_conv = nn.Conv2d(in_channels + dem_channels + lu_channels, hidden_dims[0], kernel_size=3, padding=1)
        self.cell1 = ConvLSTMCell(hidden_dims[0], hidden_dims[0])
        self.cell2 = ConvLSTMCell(hidden_dims[0], hidden_dims[1])
        self.dem_attn = DEMAttention(hidden_dims[1], dem_channels)
        self.lu_attn = LUAttention(hidden_dims[1], lu_channels)
        self.post_process = nn.Sequential(
            nn.Conv2d(hidden_dims[1], 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True) 
        )

    def forward(self, rain_lr, dem, lu):
        B, T, C, H_lr, W_lr = rain_lr.shape
        _, _, H_hr, W_hr = dem.shape
        device = rain_lr.device
        
        rain_hr_interp = F.interpolate(rain_lr.view(B*T, C, H_lr, W_lr), 
                                       size=(H_hr, W_hr), mode='bilinear', align_corners=False).view(B, T, C, H_hr, W_hr)
        
        h1 = torch.zeros(B, self.hidden_dims[0], H_hr, W_hr, device=device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros(B, self.hidden_dims[1], H_hr, W_hr, device=device)
        c2 = torch.zeros_like(h2)
        
        outputs = []
        for t in range(T):
            x_t = torch.cat([rain_hr_interp[:, t], dem, lu], dim=1)
            x_t = F.relu(self.init_conv(x_t))
            h1, c1 = self.cell1(x_t, h1, c1)
            h2, c2 = self.cell2(h1, h2, c2)
            feat = self.dem_attn(h2, dem)
            feat = self.lu_attn(feat, lu)
            out_t = self.post_process(feat)
            outputs.append(out_t.unsqueeze(1))
        return torch.cat(outputs, dim=1)

# --------------------------- 4. 判别器 (Discriminator) ---------------------------

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.main(x).squeeze(1)

# --------------------------- 5. 论文评估指标模块 (Scientific Metrics) -------------

class Metrics:
    @staticmethod
    def calculate_rmse(pred, target):
        return torch.sqrt(F.mse_loss(pred, target))

    @staticmethod
    def calculate_cc(pred, target):
        """相关系数 Correlation Coefficient"""
        p_m = torch.mean(pred)
        t_m = torch.mean(target)
        num = torch.sum((pred - p_m) * (target - t_m))
        den = torch.sqrt(torch.sum((pred - p_m)**2) * torch.sum((target - t_m)**2))
        return num / (den + 1e-8)

    @staticmethod
    def calculate_ts(pred, target, threshold=0.1):
        """Threat Score (CSI) 降水关键指标"""
        pred_bin = (pred >= threshold).float()
        target_bin = (target >= threshold).float()
        hits = torch.sum(pred_bin * target_bin)
        false_alarms = torch.sum(pred_bin * (1 - target_bin))
        misses = torch.sum((1 - pred_bin) * target_bin)
        return hits / (hits + false_alarms + misses + 1e-8)

# --------------------------- 6. 组合损失函数 --------------------------------------

class CombinedLoss(nn.Module):
    def __init__(self, lambda_gan=1.0, lambda_conserve=10.0, lambda_tv=0.1, lambda_pixel=5.0):
        super().__init__()
        self.lambda_gan = lambda_gan
        self.lambda_conserve = lambda_conserve
        self.lambda_tv = lambda_tv
        self.lambda_pixel = lambda_pixel
        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()

    def tv_loss(self, y):
        return torch.mean(torch.abs(y[:, :, :, 1:, :] - y[:, :, :, :-1, :])) + \
               torch.mean(torch.abs(y[:, :, :, :, 1:] - y[:, :, :, :, :-1]))

    def conservation_loss(self, pred, lr_input):
        B, T, C, H_hr, W_hr = pred.shape
        H_lr, W_lr = lr_input.shape[-2:]
        pred_lr = F.adaptive_avg_pool2d(pred.view(B*T, C, H_hr, W_hr), (H_lr, W_lr)).view(B, T, C, H_lr, W_lr)
        return self.l1(pred_lr, lr_input)

    def forward(self, fake_hr, lr_input, d_fake_score, real_hr=None):
        loss_gan = self.bce(d_fake_score, torch.ones_like(d_fake_score))
        loss_conserve = self.conservation_loss(fake_hr, lr_input)
        loss_tv = self.tv_loss(fake_hr)
        
        loss_pixel = 0
        if real_hr is not None:
            weight = torch.where(real_hr > 5.0, 2.0, 1.0) 
            loss_pixel = torch.mean(weight * torch.abs(fake_hr - real_hr))
        
        total_loss = (self.lambda_gan * loss_gan + 
                      self.lambda_conserve * loss_conserve + 
                      self.lambda_tv * loss_tv + 
                      self.lambda_pixel * loss_pixel)
        
        return total_loss, {"gan": loss_gan, "conserve": loss_conserve, "tv": loss_tv, "pixel": loss_pixel}

# --------------------------- 7. 数据集与增强 --------------------------------------

class FenheDataset(Dataset):
    def __init__(self, rain_path, dem_path, lucc_path, T=5, augment=True):
        self.rain_data = np.load(rain_path)
        self.dem = np.load(dem_path)
        self.lucc = np.load(lucc_path)
        self.T = T
        self.augment = augment
        
        self.rain_data = np.nan_to_num(self.rain_data)
        self.dem_norm = (self.dem - self.dem.min()) / (self.dem.max() - self.dem.min() + 1e-7)
        self.lucc_norm = self.lucc / 10.0

    def __len__(self):
        return self.rain_data.shape[0] - self.T

    def __getitem__(self, idx):
        x_lr = self.rain_data[idx : idx + self.T, np.newaxis, ...]
        dem = self.dem_norm[np.newaxis, ...]
        lu = self.lucc_norm[np.newaxis, ...]
        
        if self.augment and random.random() > 0.5:
            x_lr = np.flip(x_lr, axis=-1).copy()
            dem = np.flip(dem, axis=-1).copy()
            lu = np.flip(lu, axis=-1).copy()

        return torch.FloatTensor(x_lr), torch.FloatTensor(dem), torch.FloatTensor(lu)

# --------------------------- 8. 可视化函数 (Visualization) -----------------------

def plot_comparison(lr, hr_pred, dem, save_path="comparison.png"):
    """
    绘制对比图：LR输入 vs HR预测 vs DEM地形参考
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("LR Input (25km)")
    plt.imshow(lr[0, 0], cmap='YlGnBu')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("SR Output (1km)")
    plt.imshow(hr_pred[0, 0], cmap='YlGnBu')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("DEM Reference")
    plt.imshow(dem[0], cmap='terrain')
    plt.colorbar()

    plt.savefig(save_path)
    plt.close()

# --------------------------- 9. 训练逻辑 ------------------------------------------

def train_model(epochs=30, batch_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = FenheDataset(
        rain_path="data/cmorph-2021/daily/fenhe_hydro_08-08_2021.npy",
        dem_path="data/static_features_1km/dem_1km.npy",
        lucc_path="data/static_features_1km/lucc_1km.npy",
        augment=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator(hidden_dims=[16, 32]).to(device)
    D = Discriminator().to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
    
    loss_module = CombinedLoss()
    criterion_bce = nn.BCELoss()

    for epoch in range(epochs):
        for i, (rain_lr, dem, lu) in enumerate(dataloader):
            rain_lr, dem, lu = rain_lr.to(device), dem.to(device), lu.to(device)
            
            # Train D
            d_optimizer.zero_grad()
            fake_hr = G(rain_lr, dem, lu).detach()
            d_loss = criterion_bce(D(fake_hr), torch.zeros((rain_lr.size(0), 1), device=device))
            d_loss.backward()
            d_optimizer.step()

            # Train G
            g_optimizer.zero_grad()
            current_fake_hr = G(rain_lr, dem, lu)
            g_loss, loss_dict = loss_module(current_fake_hr, rain_lr, D(current_fake_hr))
            g_loss.backward()
            g_optimizer.step()

            if i == 0: # 每个Epoch保存第一张对比图
                plot_comparison(rain_lr[0, 0].cpu().numpy(), current_fake_hr[0, 0].detach().cpu().numpy(), dem[0, 0].cpu().numpy(), f"epoch_{epoch}.png")
                
        print(f"Epoch {epoch} finished.")

    torch.save(G.state_dict(), "generator_final.pth")

if __name__ == "__main__":
    train_model()