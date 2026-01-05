import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# ==================== 1. 注意力模块 ====================
class DEMAttention(nn.Module):
    def __init__(self, in_channels, dem_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(dem_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, dem):
        # features: (B, C, H, W), dem: (B, 1, H, W)
        attn = self.sigmoid(self.conv(dem))
        return features * attn

class LUAttention(nn.Module):
    def __init__(self, in_channels, lu_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(lu_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, lu):
        attn = self.sigmoid(self.conv(lu))
        return features * attn

# ==================== 2. ConvLSTM 核心 ====================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)
        i, f, o, g = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o), torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# ==================== 3. 生成器 (Generator) ====================
class Generator(nn.Module):
    def __init__(self, in_channels=1, dem_channels=1, lu_channels=1, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 输入：降水(1) + DEM(1) + LUCC(1)
        self.convlstm_cell = ConvLSTMCell(input_dim=in_channels + dem_channels + lu_channels, hidden_dim=hidden_dim)
        
        self.dem_attn = DEMAttention(hidden_dim, dem_channels)
        self.lu_attn = LUAttention(hidden_dim, lu_channels)
        
        self.post_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True) # 降水非负
        )

    def forward(self, rain_lr, dem, lu):
        B, T, _, H_lr, W_lr = rain_lr.shape
        _, _, H_hr, W_hr = dem.shape
        device = rain_lr.device

        # 1. 尺度对齐 (25km -> 1km)
        rain_hr_interp = F.interpolate(rain_lr.view(B*T, 1, H_lr, W_lr), 
                                     size=(H_hr, W_hr), mode='bilinear').view(B, T, 1, H_hr, W_hr)

        # 2. 初始化状态
        h = torch.zeros(B, self.hidden_dim, H_hr, W_hr).to(device)
        c = torch.zeros_like(h)
        
        outs = []
        for t in range(T):
            x_t = torch.cat([rain_hr_interp[:, t], dem, lu], dim=1)
            h, c = self.convlstm_cell(x_t, h, c)
            
            # 物理注意力修正
            feat = self.dem_attn(h, dem)
            feat = self.lu_attn(feat, lu)
            
            out_t = self.post_conv(feat)
            outs.append(out_t.unsqueeze(1))
            
        return torch.cat(outs, dim=1)

# ==================== 4. 判别器 (Discriminator) ====================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, 3, stride=(1,2,2), padding=1), nn.BatchNorm3d(32), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, 1, H, W) -> (B, 1, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.net(x).squeeze(1)

# ==================== 5. 真实数据加载类 ====================
class FenheDataset(Dataset):
    def __init__(self, precip_path, dem_path, lucc_path, T=5):
        # 加载数据
        self.precip_all = np.load(precip_path) # (Day, Lat, Lon)
        self.dem = np.load(dem_path)           # (458, 306)
        self.lucc = np.load(lucc_path)         # (458, 306)
        
        # 预处理
        self.dem_norm = (self.dem - self.dem.min()) / (self.dem.max() - self.dem.min() + 1e-6)
        self.lucc_norm = self.lucc / 8.0
        self.precip_all = np.nan_to_num(self.precip_all)
        self.T = T

    def __len__(self):
        return len(self.precip_all) - self.T

    def __getitem__(self, idx):
        # 低分辨率降水输入 (T, 1, H_lr, W_lr)
        x_lr = self.precip_all[idx : idx + self.T, np.newaxis, ...]
        # 静态特征
        dem = self.dem_norm[np.newaxis, ...]
        lu = self.lucc_norm[np.newaxis, ...]
        # 训练时通常需要高分辨率真值 y_hr，此处模拟返回 x_lr 的插值作为占位符
        return torch.from_numpy(x_lr).float(), torch.from_numpy(dem).float(), torch.from_numpy(lu).float()

# ==================== 6. 损失函数 ====================
def region_mean_loss(pred, lr_input):
    """
    质量守恒约束：超分后的平均降水应等于低分辨率观测
    """
    B, T, _, H_hr, W_hr = pred.shape
    H_lr, W_lr = lr_input.shape[-2:]
    # 自适应下采样回低分辨率尺寸
    pred_lr = F.adaptive_avg_pool2d(pred.view(B*T, 1, H_hr, W_hr), (H_lr, W_lr)).view(B, T, 1, H_lr, W_lr)
    return F.l1_loss(pred_lr, lr_input)

# ==================== 7. 主运行程序 ====================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    config = {
        "precip_path": "data/cmorph-2021/daily/fenhe_hydro_08-08_2021.npy",
        "dem_path": "data/static_features_1km/dem_1km.npy",
        "lucc_path": "data/static_features_1km/lucc_1km.npy"
    }

    # 数据准备
    dataset = FenheDataset(**config)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 模型初始化
    G = Generator(hidden_dim=16).to(device) # 考虑显存，hidden_dim设小一点
    D = Discriminator().to(device)
    
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-4)
    d_opt = torch.optim.Adam(D.parameters(), lr=1e-4)

    print(f"开始在设备 {device} 上进行真实数据前向测试...")

    for step, (x_lr, dem, lu) in enumerate(loader):
        x_lr, dem, lu = x_lr.to(device), dem.to(device), lu.to(device)

        # 训练步骤模拟
        G.train()
        fake_hr = G(x_lr, dem, lu)
        
        # 计算损失
        conserve_loss = region_mean_loss(fake_hr, x_lr)
        
        print(f"Step {step} | 输出形状: {fake_hr.shape} | 守恒损失: {conserve_loss.item():.4f}")
        
        if step >= 2: break # 仅做演示运行

    print("测试成功！模型已跑通真实数据链。")

if __name__ == "__main__":
    train()