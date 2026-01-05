import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import rasterio

# --------------------------- 1. 注意力增强模块 ------------------------------------
class DEMAttention(nn.Module):
    def __init__(self, in_channels, dem_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dem_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, dem):
        return x * self.conv(dem)

class LUAttention(nn.Module):
    def __init__(self, in_channels, lu_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(lu_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, lu):
        return x * self.conv(lu)

# --------------------------- 2. ConvLSTM 核心组件 ---------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        cc_i, cc_f, cc_o, cc_g = torch.split(self.conv(combined), self.hidden_dim, dim=1)
        i, f, o, g = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o), torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# --------------------------- 3. 生成器 (Generator) -------------------------------
class Generator(nn.Module):
    def __init__(self, in_channels=1, dem_channels=1, lu_channels=1, hidden_dims=[32, 64]):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.init_conv = nn.Conv2d(in_channels + dem_channels + lu_channels + 2, hidden_dims[0], 3, padding=1)  # +2 for CoordConv
        self.cell1 = ConvLSTMCell(hidden_dims[0], hidden_dims[0])
        self.cell2 = ConvLSTMCell(hidden_dims[0], hidden_dims[1])
        self.dem_attn = DEMAttention(hidden_dims[1], dem_channels)
        self.lu_attn = LUAttention(hidden_dims[1], lu_channels)
        self.post_process = nn.Sequential(
            nn.Conv2d(hidden_dims[1], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, rain_lr, dem, lu):
        B, T, C, H, W = rain_lr.shape
        device = rain_lr.device

        # 双线性插值到高分辨率
        rain_hr = F.interpolate(rain_lr.view(B*T, C, H, W), size=dem.shape[-2:], mode='bilinear', align_corners=False)
        rain_hr = rain_hr.view(B, T, C, *dem.shape[-2:])

        # CoordConv 特征
        H_hr, W_hr = dem.shape[-2:]
        coord_h = torch.linspace(0, 1, H_hr, device=device).view(1, 1, H_hr, 1).repeat(B, 1, 1, W_hr)
        coord_w = torch.linspace(0, 1, W_hr, device=device).view(1, 1, 1, W_hr).repeat(B, 1, H_hr, 1)

        h1 = torch.zeros(B, self.hidden_dims[0], H_hr, W_hr, device=device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros(B, self.hidden_dims[1], H_hr, W_hr, device=device)
        c2 = torch.zeros_like(h2)

        outputs = []
        for t in range(T):
            x_t = torch.cat([rain_hr[:, t], dem, lu, coord_h, coord_w], dim=1)
            x_t = F.relu(self.init_conv(x_t))
            h1, c1 = self.cell1(x_t, h1, c1)
            h2, c2 = self.cell2(h1, h2, c2)
            feat = self.dem_attn(h2, dem)
            feat = self.lu_attn(feat, lu)
            out_t = self.post_process(feat)
            outputs.append(out_t.unsqueeze(1))
        return torch.cat(outputs, dim=1)

# --------------------------- 4. 损失函数 (只用站点真值) ----------------------------
class PointLoss(nn.Module):
    def __init__(self, lambda_point=20.0):
        super().__init__()
        self.lambda_point = lambda_point

    def forward(self, pred, s_coords, s_values):
        B, T, _, _, _ = pred.shape
        loss = 0
        for b in range(B):
            for t in range(T):
                rows, cols = s_coords[:, 0], s_coords[:, 1]
                loss += F.mse_loss(pred[b, t, 0, rows, cols], s_values[b, t])
        return self.lambda_point * loss / (B * T)

# --------------------------- 5. 数据集 ------------------------------------------
def get_dem_extent_from_tif(dem_tif_path):
    with rasterio.open(dem_tif_path) as src:
        width, height = src.width, src.height
        lon_left, lat_top = src.transform * (0, 0)
        lon_right, lat_bottom = src.transform * (width, height)
        return [lat_bottom, lat_top, lon_left, lon_right]

def latlon_to_pixel(lat, lon, extent, shape):
    min_lat, max_lat, min_lon, max_lon = extent
    n_rows, n_cols = shape
    row = int((max_lat - lat) / (max_lat - min_lat) * (n_rows - 1))
    col = int((lon - min_lon) / (max_lon - min_lon) * (n_cols - 1))
    return row, col

class FenheDataset(Dataset):
    def __init__(self, rain_lr_path, dem_tif, lucc_path, meta_path, rain_excel_path, T=5):
        self.rain_lr = np.load(rain_lr_path)
        self.lucc = np.load(lucc_path) / 10.0
        self.dem = rasterio.open(dem_tif).read(1)
        self.dem_norm = (self.dem - self.dem.min()) / (self.dem.max() - self.dem.min() + 1e-7)
        self.T = T

        # DEM范围
        self.extent = get_dem_extent_from_tif(dem_tif)
        rows_total, cols_total = self.dem.shape

        # 读取站点元数据
        df_meta = pd.read_excel(meta_path, usecols=["F_站号", "经度", "纬度"])
        df_rain = pd.read_excel(rain_excel_path)
        df_rain = df_rain.sort_values(["year", "month", "day"]).reset_index(drop=True)

        # 将站点映射到像素
        self.s_coords = []
        self.s_values_list = []
        for _, row in df_meta.iterrows():
            lat, lon, st_id = row["纬度"], row["经度"], str(row["F_站号"])
            r_idx, c_idx = latlon_to_pixel(lat, lon, self.extent, (rows_total, cols_total))
            if st_id in df_rain.columns:
                self.s_coords.append([r_idx, c_idx])
                self.s_values_list.append(df_rain[st_id].to_numpy())

        self.s_coords = np.array(self.s_coords)
        self.s_values = np.stack(self.s_values_list, axis=1)  # shape: [time, num_stations]

    def __len__(self):
        return self.rain_lr.shape[0] - self.T

    def __getitem__(self, idx):
        x_lr = torch.FloatTensor(self.rain_lr[idx:idx+self.T, np.newaxis, ...])
        dem = torch.FloatTensor(self.dem_norm[np.newaxis, ...])
        lu = torch.FloatTensor(self.lucc[np.newaxis, ...])
        s_vals = torch.FloatTensor(self.s_values[idx:idx+self.T])
        s_coords = torch.LongTensor(self.s_coords)
        return x_lr, dem, lu, s_coords, s_vals

# --------------------------- 6. 可视化函数 ---------------------------------------
def plot_comparison(lr, hr_pred, dem, save_path="comparison.png"):
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.title("LR Input")
    plt.imshow(lr, cmap='YlGnBu')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.title("SR Output")
    plt.imshow(hr_pred, cmap='YlGnBu')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.title("DEM")
    plt.imshow(dem, cmap='terrain')
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()

# --------------------------- 7. 训练逻辑 ----------------------------------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FenheDataset(
        rain_lr_path="data/cmorph_lr.npy",
        dem_tif="data/dem_1km.tif",
        lucc_path="data/lucc_1km.npy",
        meta_path="data/climate/meta.xlsx",
        rain_excel_path="data/climate/rain.xlsx",
        T=5
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    G = Generator(hidden_dims=[16,32]).to(device)
    optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
    criterion = PointLoss(lambda_point=20.0)

    for epoch in range(50):
        for i, (lr, dem, lu, s_coords, s_vals) in enumerate(dataloader):
            lr, dem, lu = lr.to(device), dem.to(device), lu.to(device)
            s_vals, s_coords = s_vals.to(device), s_coords[0]  # batch 0的站点索引

            optimizer.zero_grad()
            pred = G(lr, dem, lu)
            loss = criterion(pred, s_coords, s_vals)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch} | Step {i} | Point Loss: {loss.item():.4f}")
                plot_comparison(lr[0,0,0].cpu().numpy(), pred[0,0,0].detach().cpu().numpy(), dem[0,0].cpu().numpy(), f"epoch_{epoch}_step_{i}.png")

    torch.save(G.state_dict(), "generator_point_only.pth")

if __name__ == "__main__":
    train_model()
