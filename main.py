import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# --------------------------- ConvLSTMCell ------------------------------------
# ConvLSTM单元格实现，结合了CNN的空间特征提取能力和LSTM的时间序列建模能力
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2  # 计算填充大小以保持空间维度不变
        self.hidden_dim = hidden_dim  # 隐藏状态维度
        # 卷积层，同时计算输入门、遗忘门、输出门和候选细胞状态
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,  # 输入通道为输入特征+上一时刻隐藏状态
                              out_channels=4 * hidden_dim,  # 输出通道为4倍隐藏状态（对应四个门）
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, x, h_cur, c_cur):
        # 合并当前输入和上一时刻隐藏状态 (在通道维度上拼接)
        combined = torch.cat([x, h_cur], dim=1)  
        # 通过卷积层得到四个门和候选状态的组合特征
        conv_out = self.conv(combined)
        # 分割输出为四个部分，分别对应输入门、遗忘门、输出门和候选细胞状态
        (cc_i, cc_f, cc_o, cc_g) = torch.split(conv_out, self.hidden_dim, dim=1)
        # 输入门：控制新信息进入细胞状态的程度
        i = torch.sigmoid(cc_i)
        # 遗忘门：控制保留多少上一时刻的细胞状态
        f = torch.sigmoid(cc_f)
        # 输出门：控制细胞状态有多少输出到隐藏状态
        o = torch.sigmoid(cc_o)
        # 候选细胞状态：包含当前输入的新信息
        g = torch.tanh(cc_g)
        # 更新细胞状态：遗忘部分旧状态 + 加入部分新状态
        c_next = f * c_cur + i * g
        # 更新隐藏状态：基于输出门和新细胞状态
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# --------------------------- ConvLSTM Module ---------------------------------
# 多层ConvLSTM模块实现，由多个ConvLSTMCell堆叠组成
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=3):
        super().__init__()
        self.cells = nn.ModuleList([ConvLSTMCell(input_dim if i==0 else hidden_dims[i-1], h, kernel_size) 
                                   for i, h in enumerate(hidden_dims)])

    def forward(self, input_seq):
        # input_seq: (B, T, C, H, W) 格式，其中B=批次大小，T=时间步长，C=通道数，H=高度，W=宽度
        B, T, C, H, W = input_seq.shape
        device = input_seq.device
        # 初始化每层的隐藏状态和细胞状态为零
        h = [torch.zeros(B, cell.hidden_dim, H, W, device=device) for cell in self.cells]
        c = [torch.zeros_like(h_l) for h_l in h]
        outputs = []
        # 处理序列中的每个时间步
        for t in range(T):
            x = input_seq[:, t]  # 获取当前时间步的输入
            # 逐层处理当前时间步的输入
            for l, cell in enumerate(self.cells):
                h[l], c[l] = cell(x, h[l], c[l])
                x = h[l]  # 上一层的输出作为下一层的输入
            # 保存最后一层的隐藏状态作为输出
            outputs.append(h[-1].unsqueeze(1))  
        # 将所有时间步的输出合并为一个序列
        out = torch.cat(outputs, dim=1)  # (B, T, hidden_dim, H, W)
        return out

# --------------------------- DEM Attention Module ----------------------------
class DEMAttention(nn.Module):
    """
    对 ConvLSTM 输出特征进行 DEM 注意力加权
    """
    def __init__(self, in_channels, dem_channels=1):
        super().__init__()
        # 使用 1x1 卷积提取 DEM 注意力权重
        self.conv1 = nn.Conv2d(dem_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, dem):
        """
        features: (B, C, H, W) ConvLSTM 输出特征
        dem: (B, dem_channels, H, W) DEM 高程图
        """
        attn = self.sigmoid(self.conv1(dem))  # (B, C, H, W)
        return features * attn  # 特征加权 

# --------------------------- LU Attention Module ----------------------------
class LUAttention(nn.Module):
    """
    对 ConvLSTM 输出特征进行土地利用(LU)注意力加权
    """
    def __init__(self, in_channels, lu_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(lu_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, lu):
        """
        features: (B, C, H, W) ConvLSTM 输出特征
        lu: (B, lu_channels, H, W) 土地利用图
        """
        attn = self.sigmoid(self.conv1(lu))  # (B, C, H, W)
        return features * attn  # 特征加权

# --------------------------- Generator (ConvLSTM + Upsample) -----------------
# 生成器网络：使用ConvLSTM提取时序特征，然后通过上采样模块提高空间分辨率
class Generator(nn.Module):
    def __init__(self, in_channels=1, dem_channels=1, lu_channels=1, hidden_dims=[32, 32], upscale=4):
        """
        in_channels: 雨量低分辨率输入通道
        dem_channels: 地形/DEM 输入通道
        lu_channels: 土地利用输入通道
        hidden_dims: ConvLSTM 隐藏维度列表
        upscale: 上采样倍数
        """
        super().__init__()
        # 初始化ConvLSTM模块用于时序特征提取
        self.convlstm = ConvLSTM(input_dim=in_channels+dem_channels+lu_channels, hidden_dims=hidden_dims, kernel_size=3)

        # 后处理卷积层，将ConvLSTM的输出映射到更高维度的特征空间
        self.post_conv = nn.Conv2d(hidden_dims[-1], 64, kernel_size=3, padding=1)
        # 注意力模块
        self.dem_attn = DEMAttention(64, dem_channels)
        self.lu_attn = LUAttention(64, lu_channels)

        self.upscale = upscale  # 上采样倍数
        self.upsamples = nn.ModuleList()  # 存储上采样模块
        times = int(np.log2(upscale))  # 计算需要多少次2倍上采样
        ch = 64
        
        # 创建多级上采样模块，每级包括卷积、PixelShuffle和激活函数
        for _ in range(times):
            self.upsamples.append(nn.Sequential(
                nn.Conv2d(ch, ch * 4, kernel_size=3, padding=1),  # 通道数扩大4倍为PixelShuffle做准备
                nn.PixelShuffle(2),  # 将空间分辨率扩大2倍，通道数减小到原来的1/4
                nn.ReLU(inplace=True)  # 激活函数
            ))
        # 最终卷积层，将特征映射到输出通道数
        self.final_conv = nn.Conv2d(ch, 1, kernel_size=3, padding=1)

    def forward(self, rain_lr, dem, lu):
            """
            rain_lr: (B, T, 1, H, W) 低分辨率雨量
            dem: (B, 1, H, W) 高程图/地形信息，固定帧
            lu: (B, lu_channels, H, W) 土地利用信息，固定帧
            """
            B, T, _, H, W = rain_lr.shape
            # 扩展 DEM 和 LU 为时间序列
            dem_seq = dem.unsqueeze(1).expand(B, T, -1, -1, -1)
            lu_seq = lu.unsqueeze(1).expand(B, T, -1, -1, -1)
            # 拼接雨量和地形输入
            x = torch.cat([rain_lr, dem_seq, lu_seq], dim=2)  # (B, T, in+dem+lu, H, W)
            
            # ConvLSTM 提取时空特征
            out_seq = self.convlstm(x)  # (B, T, hidden_dim, H, W)
            
            outs = []
            for t in range(T):
                h = out_seq[:, t]
                h = F.relu(self.post_conv(h))
                # DEM 注意力
                h = self.dem_attn(h, dem)
                # LU 注意力
                h = self.lu_attn(h, lu)
                for up in self.upsamples:
                    h = up(h)
                y = F.relu(self.final_conv(h))
                outs.append(y.unsqueeze(1))
            out = torch.cat(outs, dim=1)
            return out

# --------------------------- Discriminator (3D Conv) -------------------------
# 判别器网络：使用3D卷积同时处理空间和时间维度，用于区分真实序列和生成的序列
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels,32,3,padding=1), nn.LeakyReLU(0.2),
            nn.Conv3d(32,64,3,stride=(1,2,2),padding=1), nn.BatchNorm3d(64), nn.LeakyReLU(0.2),
            nn.Conv3d(64,128,3,stride=(1,2,2),padding=1), nn.BatchNorm3d(128), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, C, H, W) -> 转换为3D卷积所需格式 (B, C, T, H, W)
        x = x.permute(0,2,1,3,4)
        # 通过判别器网络并挤压多余的维度
        return self.net(x).squeeze(1)  # (B,)

# --------------------------- 数据合成 --------------------------------
class SyntheticDataset(Dataset):
    def __init__(self, n_samples=200, T=5, low_res=(8,8), upscale=4, lu_channels=1):
        self.n = n_samples  # 样本数量
        self.T = T  # 时间步长
        self.lr_h, self.lr_w = low_res  # 低分辨率尺寸
        self.upscale = upscale  # 上采样倍数
        self.hr_h = self.lr_h * upscale
        self.hr_w = self.lr_w * upscale
        self.lu_channels = lu_channels

        # 初始化低分辨率输入、高分辨率目标、DEM 和 LU 数据
        self.X = np.zeros((n_samples, T, 1, self.lr_h, self.lr_w), dtype=np.float32)  # 雨量
        self.Y = np.zeros((n_samples, T, 1, self.hr_h, self.hr_w), dtype=np.float32)  # 高分辨率雨量
        self.DEM = np.random.rand(n_samples, 1, self.lr_h, self.lr_w).astype(np.float32)  # DEM
        self.LU = np.zeros((n_samples, lu_channels, self.lr_h, self.lr_w), dtype=np.float32)
        
        rng = np.random.RandomState(42)
        # 每个像素随机一个类别
        for c in range(lu_channels):
            self.LU[:,c,:,:] = rng.randint(0,2,(n_samples, self.lr_h, self.lr_w))  # 土地利用

        # 生成雨量序列
        for i in range(n_samples):
            base = rng.rand(self.lr_h, self.lr_w) * 5.0
            for t in range(T):
                noise = rng.randn(self.lr_h, self.lr_w) * 0.2
                field = base + 0.5 * np.sin((t + i) * 0.3) + noise
                self.X[i, t, 0] = field
                # 高分辨率上采样
                up = np.kron(field, np.ones((upscale, upscale)))
                bumps = (rng.rand(self.hr_h, self.hr_w) - 0.5) * 0.3
                self.Y[i, t, 0] = np.clip(up + bumps, 0.0, None)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # 返回雨量序列、高分辨率目标、DEM 和 LU
        return self.X[idx], self.Y[idx], self.DEM[idx], self.LU[idx]

# ==================== 约束函数 ====================
def station_interpolate(pred, station_coords):
    B,T,_,H,W = pred.shape
    coords = torch.tensor(station_coords, device=pred.device).float()*2-1
    vals = []
    for t in range(T):
        frame = pred[:,t]
        sample = F.grid_sample(frame, coords.view(1,-1,1,2).expand(B,-1,-1,-1), align_corners=True)
        vals.append(sample.squeeze(-1).squeeze(1))
    return torch.stack(vals,dim=1)

def region_mean_loss(pred, lr_input, dem, lu, upscale):
    """
    使用 DEM + LU 注意力加权的区域平均损失
    """
    B, T, _, H_hr, W_hr = pred.shape
    H_lr, W_lr = lr_input.shape[-2:]

    # 将高分辨率预测降采样到低分辨率
    pooled = F.avg_pool2d(pred.view(B*T,1,H_hr,W_hr), kernel_size=upscale).view(B,T,1,H_lr,W_lr)

    # 上采样 DEM 和 LU 到低分辨率尺寸
    dem_up = F.interpolate(dem, size=(H_lr,W_lr), mode='bilinear', align_corners=False)
    lu_up = F.interpolate(lu, size=(H_lr,W_lr), mode='nearest')  # LU 通常是分类数据，用 nearest 插值

    # 权重：可以简单叠加 DEM + LU，或者使用加权和
    dem_norm = dem_up / (dem_up.max() + 1e-6)
    lu_norm = lu_up / (lu_up.max() + 1e-6)
    weight = dem_norm + lu_norm
    
    # 计算加权绝对误差
    weighted_loss = (pooled - lr_input).abs() * weight.unsqueeze(1)  # (B, T, 1, H_lr, W_lr)

    return weighted_loss.mean()

def smoothness_loss(pred):
    dx = pred[:,:,:,1:,:] - pred[:,:,:,:-1,:]
    dy = pred[:,:,:,:,1:] - pred[:,:,:,:,:-1]
    return (dx.abs().mean() + dy.abs().mean())

# --------------------------- Training loop (demo) -----------------------------
# 训练演示函数
def train_demo():
    # 选择运行设备（GPU如果可用，否则CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 创建合成数据集和数据加载器
    dataset = SyntheticDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 初始化生成器和判别器网络
    G = Generator().to(device)
    D = Discriminator().to(device)

    # 初始化优化器
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))

    # 定义损失函数
    bce_loss = nn.BCELoss()  # 二元交叉熵损失，用于对抗训练
    
    station_coords = [(0.3,0.4),(0.7,0.8)]  # 示例站点坐标

    epochs = 30  # 训练轮数（演示用，实际应用中应设置更大的值）
    for epoch in range(epochs):
        for step, (x_batch, y_batch, dem_batch, lu_batch) in enumerate(dataloader):
            x, y, dem, lu = x_batch.to(device), y_batch.to(device), dem_batch.to(device), lu_batch.to(device)
            B = x.shape[0]  # 批次大小

            # 训练判别器
            d_opt.zero_grad()  # 清除梯度
            with torch.no_grad():
                # 在不计算生成器梯度的情况下生成假样本
                fake = G(x, dem, lu)

            # 判别器对真实样本的预测
            d_real = D(y)
            # 判别器对生成样本的预测
            d_fake = D(fake)

            # 创建真实和假样本的标签
            real_labels = torch.ones_like(d_real, device=device)
            fake_labels = torch.zeros_like(d_fake, device=device)

            # 计算判别器损失：真实样本损失+假样本损失
            d_loss = bce_loss(d_real, real_labels) + bce_loss(d_fake, fake_labels)
            d_loss.backward()  # 反向传播计算梯度
            d_opt.step()  # 更新判别器参数

            # 训练生成器
            g_opt.zero_grad()  # 清除梯度
            fake = G(x, dem, lu)  # 生成假样本

            d_fake = D(fake)  # 判别器对假样本的预测

            # 计算对抗损失：希望生成的样本被判别器认为是真实的
            adv_loss = bce_loss(d_fake, real_labels)

            # 站点约束
            fake_station = station_interpolate(fake, station_coords)
            real_station = station_interpolate(y, station_coords)
            station_loss = F.mse_loss(fake_station, real_station)

            # 守恒约束
            conserve_loss = region_mean_loss(fake, x, dem, lu, dataset.upscale)

            # 平滑约束
            smooth_loss = smoothness_loss(fake)

            # 生成器总损失：对抗损失和内容损失的加权和
            g_loss = 0.01*adv_loss + 1.0*station_loss + 0.1*conserve_loss + 0.01*smooth_loss
            g_loss.backward()  # 反向传播计算梯度
            g_opt.step()  # 更新生成器参数

            # 定期打印损失信息
            if step % 10 == 0:
                print(f"Epoch {epoch} Step {step} | "
                    f"D_loss {d_loss.item():.4f} | "
                    f"G_loss {g_loss.item():.4f} (Adv {adv_loss.item():.4f}, "
                    f"Station {station_loss.item():.4f}, Conserve {conserve_loss.item():.4f}, "
                    f"Smooth {smooth_loss.item():.4f})")

    print("训练完成！")

if __name__ == "__main__":
    train_demo()