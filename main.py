import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os

# 创建输出目录
output_dir = "/root/pi-conv/output"
os.makedirs(output_dir, exist_ok=True)

# --------------------------- 8. 可视化函数 ------------------------------------
def plot_stations_vs_pred(s_coords, true_vals, pred_vals, save_path="station_comparison.png"):
    """
    绘制站点观测值和预测值的对比图
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制散点图
    plt.scatter(true_vals, pred_vals, alpha=0.7)
    
    # 绘制对角线表示理想预测
    max_val = max(np.max(true_vals), np.max(pred_vals))
    min_val = min(np.min(true_vals), np.min(pred_vals))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
    
    # 设置图表属性
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Station Observed vs Predicted Precipitation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=300)
    plt.close()


# --------------------------- 1. CoordConv 辅助函数 ------------------------------
def add_coord_channels(x):
    """
    在输入特征图中加入标准化行列坐标 (CoordConv)
    x: (B, C, H, W)
    返回: (B, C+2, H, W)
    """
    B, C, H, W = x.shape
    device = x.device

    row_coords = torch.linspace(0, 1, H, device=device).unsqueeze(1).repeat(1, W)
    col_coords = torch.linspace(0, 1, W, device=device).unsqueeze(0).repeat(H, 1)
    row_coords = row_coords.unsqueeze(0).unsqueeze(0).repeat(B,1,1,1)
    col_coords = col_coords.unsqueeze(0).unsqueeze(0).repeat(B,1,1,1)

    return torch.cat([x, row_coords, col_coords], dim=1)

# --------------------------- 2. 注意力增强模块 -----------------------------------
class DEMAttention(nn.Module):
    def __init__(self, in_channels, dem_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dem_channels, in_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x, dem):
        attn = self.conv(dem)
        return x * attn

class LUAttention(nn.Module):
    def __init__(self, in_channels, lu_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(lu_channels, in_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x, lu):
        attn = self.conv(lu)
        return x * attn

# --------------------------- 3. ConvLSTM 核心组件 --------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding, bias=bias)
    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)
        i, f, o, g = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o), torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# --------------------------- 4. 生成器 (Generator) ------------------------------
class Generator(nn.Module):
    def __init__(self, in_channels=1, dem_channels=1, lu_channels=0, hidden_dims=[32,64]):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.lu_channels = lu_channels
        self.init_conv = nn.Conv2d(in_channels + dem_channels + lu_channels + 2, hidden_dims[0], kernel_size=3, padding=1)
        self.cell1 = ConvLSTMCell(hidden_dims[0], hidden_dims[0])
        self.cell2 = ConvLSTMCell(hidden_dims[0], hidden_dims[1])
        self.dem_attn = DEMAttention(hidden_dims[1], dem_channels)
        self.lu_attn = LUAttention(hidden_dims[1], lu_channels)
        self.post_process = nn.Sequential(
            nn.Conv2d(hidden_dims[1], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1)
        )
    def forward(self, rain_lr, dem, lu):
        B, T, C, H, W = rain_lr.shape
        h1 = torch.zeros(B, self.hidden_dims[0], H, W, device=rain_lr.device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros(B, self.hidden_dims[1], H, W, device=rain_lr.device)
        c2 = torch.zeros_like(h2)
        outputs = []
        for t in range(T):
            x_t = torch.cat([rain_lr[:,t], dem, lu], dim=1)
            x_t = add_coord_channels(x_t)
            x_t = F.relu(self.init_conv(x_t))
            h1, c1 = self.cell1(x_t, h1, c1)
            h2, c2 = self.cell2(h1, h2, c2)
            feat = self.dem_attn(h2, dem)
            feat = self.lu_attn(feat, lu)
            out_t = self.post_process(feat)
            outputs.append(out_t.unsqueeze(1))
        return torch.cat(outputs, dim=1)


# --------------------------- 5. 组合损失函数 ------------------------------------
class CombinedLoss(nn.Module):
    def __init__(self, lambda_point=20.0, lambda_conserve=5.0):
        super().__init__()
        self.lambda_point = lambda_point
        self.lambda_conserve = lambda_conserve
        self.l1 = nn.L1Loss()
    def conservation_loss(self, pred, lr_input):
        """物理守恒损失"""
        B,T,C,H,W = pred.shape
        H_lr,W_lr = lr_input.shape[-2:]
        pred_lr = F.interpolate(pred.view(B*T, C, H, W),size=(H_lr, W_lr),mode="area").view(B, T, C, H_lr, W_lr)
        return self.l1(pred_lr, lr_input)
    def point_supervision_loss(self, pred, s_coords, s_values):
        """点观测监督损失"""
        if s_values is None or s_coords.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        B, T, _, H, W = pred.shape
        loss = torch.tensor(0.0, device=pred.device)
        count = 0

        for b in range(B):
            for t in range(T):
                # 获取当前batch和time步的站点坐标和值
                coords = s_coords[b]  # (num_stations, 2)
                target = s_values[b, t]  # (num_stations,)

                # 【修改】全 NaN 直接跳过
                if torch.isnan(target).all():
                    continue

                # 【修改】添加边界检查，确保索引在有效范围内
                valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < H) & (coords[:, 1] >= 0) & (coords[:, 1] < W)
                valid_rows = coords[valid_mask, 0]
                valid_cols = coords[valid_mask, 1]
                valid_target = target[valid_mask]

                # 【修改】有效站点数为 0，跳过
                if valid_target.numel() == 0 or valid_mask.sum() == 0:
                    continue

                pred_at_stations = pred[b, t, 0, valid_rows, valid_cols]
                mask = ~torch.isnan(valid_target)

                if mask.sum() == 0:
                    continue

                loss += F.mse_loss(pred_at_stations[mask], valid_target[mask])
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=pred.device)

        return loss / count

    def forward(self,pred,lr_input,s_coords,s_values):
        loss_point = self.point_supervision_loss(pred,s_coords,s_values)
        loss_conserve = self.conservation_loss(pred, lr_input)
        total_loss = self.lambda_point*loss_point + self.lambda_conserve*loss_conserve
        return total_loss, {"point":loss_point,"conserve":loss_conserve}

# --------------------------- 6. 数据集 -----------------------------------------------
def get_shapefile_extent(shp_path):
    gdf = gpd.read_file(shp_path)
    minx,miny,maxx,maxy = gdf.total_bounds
    return [miny,maxy,minx,maxx]  # [min_lat,max_lat,min_lon,max_lon]

class FenheDataset(Dataset):
    def __init__(self, rain_lr_path, dem_path, lucc_path, meta_path, rain_excel_path, shp_path, T=5):
        # 卫星降水数据-将nan、∞置为0
        self.rain_lr = np.nan_to_num(
            np.load(rain_lr_path).astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        self.dem = np.load(dem_path)
        self.lucc = np.load(lucc_path).astype(int)
        self.T = T

        # DEM归一化
        self.dem_norm = (self.dem - self.dem.min()) / (self.dem.max()-self.dem.min()+1e-7)

        # LUCC独热编码
        self.lucc_onehot = self._lucc_to_onehot(self.lucc)

        # 自动获取网格范围
        self.grid_extent = get_shapefile_extent(shp_path)

        # 站点处理
        self.s_coords, self.s_values = self._prepare_stations(meta_path, rain_excel_path)

    def _lucc_to_onehot(self, lucc):
        """
        lucc: (H,W) 整数标签
        输出: (C, H, W) 独热编码
        """
        num_classes = lucc.max() + 1
        H, W = lucc.shape
        onehot = np.eye(num_classes)[lucc].transpose(2,0,1).astype(np.float32)

        return onehot

    def _prepare_stations(self, meta_path, rain_excel_path):
        df_meta = pd.read_excel(meta_path, usecols=["F_站号", "经度", "纬度"])
        df_rain = pd.read_excel(rain_excel_path).query("year==2021") \
            .sort_values(["year", "month", "day"]) \
            .reset_index(drop=True)

        min_lat, max_lat, min_lon, max_lon = self.grid_extent
        rows_total, cols_total = self.rain_lr.shape[-2:]  # 480, 1440

        coords = []
        val_list = []

        # ---------- 所有站点当日非负均值（用于空间兜底） ----------
        rain_values = df_rain.drop(columns=["year", "month", "day"], errors="ignore")
        spatial_mean = rain_values.mask(rain_values < 0).mean(axis=1).to_numpy()

        for _, row in df_meta.iterrows():
            st_id = int(row["F_站号"])
            lat, lon = row["纬度"], row["经度"]

            # 网格索引
            r_idx = int((max_lat - lat) / (max_lat - min_lat) * (rows_total - 1))
            c_idx = int((lon - min_lon) / (max_lon - min_lon) * (cols_total - 1))

            # 超出边界就跳过
            if not (0 <= r_idx < rows_total and 0 <= c_idx < cols_total):
                print(f"Station {st_id} ({lat},{lon}) out of bounds, skipped")
                continue

            col = str(st_id)
            if col not in df_rain.columns:
                continue

            series = df_rain[col].to_numpy(dtype=np.float32)

            # ---------- 核心：负值修复 ----------
            for t in range(len(series)):
                if series[t] >= 0:
                    continue

                candidates = []

                # 前一天
                if t - 1 >= 0 and series[t - 1] >= 0:
                    candidates.append(series[t - 1])

                # 后一天
                if t + 1 < len(series) and series[t + 1] >= 0:
                    candidates.append(series[t + 1])

                # 用前后天均值
                if len(candidates) > 0:
                    series[t] = np.mean(candidates)
                # 空间兜底
                elif not np.isnan(spatial_mean[t]):
                    series[t] = spatial_mean[t]
                else:
                    series[t] = 0.0

            coords.append([r_idx, c_idx])
            val_list.append(series)
        
        if len(val_list) == 0:
            raise ValueError("No valid stations found within grid bounds!")

        stacked = np.stack(val_list, axis=1).astype(np.float32)
        stacked_clean = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
        return np.array(coords), stacked_clean



    def __len__(self):
        return self.rain_lr.shape[0] - self.T


    def __getitem__(self, idx):
        x_lr = torch.FloatTensor(self.rain_lr[idx:idx+self.T, np.newaxis, ...])
        _, _, H, W = x_lr.shape  # H=480, W=1440

        dem = torch.FloatTensor(self.dem_norm[np.newaxis, ...])   # (1,458,306)
        lu  = torch.FloatTensor(self.lucc_onehot)                 # (9,458,306)

        # ---------- 空间对齐 ----------
        dem = F.interpolate(
            dem.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        lu = F.interpolate(
            lu.unsqueeze(0),
            size=(H, W),
            mode="nearest"   # one-hot 必须最近邻
        ).squeeze(0)

        s_vals = torch.FloatTensor(self.s_values[idx:idx+self.T])
        s_coords = torch.LongTensor(self.s_coords)

        return x_lr, dem, lu, s_coords, s_vals

# --------------------------- 7. 训练逻辑（带站点 RMSE 和可视化） -----------------------
def train_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FenheDataset(
        rain_lr_path="data/cmorph-2021/daily/fenhe_hydro_08-08_2021.npy", # 卫星降水数据
        dem_path="data/static_features_1km/dem_1km.npy",    # 高程数据
        lucc_path="data/static_features_1km/lucc_1km.npy",  # 下垫面数据
        meta_path="data/climate/meta.xlsx", # 雨量站点元数据
        rain_excel_path="data/climate/rain.xlsx",   # 雨量站观测数据
        shp_path="data/FenheBasin/fenhe.shp",   # 汾河流域范围
        T=5 # 时间框口大小
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    num_lu_classes = dataset.lucc_onehot.shape[0]
    G = Generator(hidden_dims=[16,32], lu_channels=num_lu_classes).to(device)
    optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    loss_module = CombinedLoss(lambda_point=0.1, lambda_conserve=1.0)

    T = dataset.T
    rmse_time_history = [[] for _ in range(T)]  # 保存每个时间步 RMSE

    for epoch in range(50):
        all_batch_rmse = []

        for i, (lr, dem, lu, s_coords, s_values) in enumerate(dataloader):
            lr, dem, lu = lr.to(device), dem.to(device), lu.to(device)
            s_values, s_coords = s_values.to(device), s_coords.to(device)

            optimizer.zero_grad()
            fake_hr = G(lr, dem, lu)
            loss, loss_dict = loss_module(fake_hr, lr, s_coords, s_values)

            if torch.isnan(loss):
                print("NaN detected, skip this batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=0.5)
            optimizer.step()

            # ---------- 全 batch 全时间步站点 RMSE ----------
            with torch.no_grad():
                B, T, C, H, W = fake_hr.shape
                pred_vals = fake_hr[:,:,0,:,:]  # (B,T,H,W)

                # 【修改】检查数据形状
                # print(f"debug: s_coords shape: {s_coords.shape}, s_values shape: {s_values.shape}")

                # 获取站点坐标
                if len(s_coords.shape) == 3:  # (B, num_stations, 2)
                    # 从第一个样本获取站点坐标（所有样本共享坐标）
                    rows = s_coords[0, :, 0]
                    cols = s_coords[0, :, 1]
                else:  # (num_stations, 2)
                    rows = s_coords[:, 0]
                    cols = s_coords[:, 1]
                    
                num_stations = rows.shape[0]

                # 【修改】添加边界检查，确保站点坐标在有效范围内
                valid_station_mask = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
                valid_rows = rows[valid_station_mask]
                valid_cols = cols[valid_station_mask]
                valid_num_stations = valid_rows.shape[0]

                if valid_num_stations > 0:
                    batch_idx = torch.arange(B, device=device).view(B,1,1).expand(B,T,valid_num_stations)
                    time_idx  = torch.arange(T, device=device).view(1,T,1).expand(B,T,valid_num_stations)
                    rows_expand = valid_rows.view(1,1,-1).expand(B,T,valid_num_stations)
                    cols_expand = valid_cols.view(1,1,-1).expand(B,T,valid_num_stations)

                    pred_at_stations = pred_vals[batch_idx, time_idx, rows_expand, cols_expand]  # (B,T,valid_num_stations)
                    
                    # 【修改】正确索引 s_values
                    if len(s_coords.shape) == 3:  # (B, num_stations, 2)
                        true_vals = s_values[:, :, valid_station_mask]  # (B,T,valid_num_stations)
                    else:  # (B, T, num_stations)
                        true_vals = s_values[:, :, valid_station_mask]  # (B,T,valid_num_stations)
                        
                    mask = ~torch.isnan(true_vals)

                    if mask.sum() > 0:
                        batch_rmse = torch.sqrt(F.mse_loss(pred_at_stations[mask], true_vals[mask]))
                    else:
                        batch_rmse = torch.tensor(0.0, device=device)
                else:
                    batch_rmse = torch.tensor(0.0, device=device)

                all_batch_rmse.append(batch_rmse.item())

                # ---------- 分时间步 RMSE ----------
                if valid_num_stations > 0:
                    for t in range(T):
                        mask_t = mask[:,t,:]
                        if mask_t.sum() > 0:
                            rmse_t = torch.sqrt(F.mse_loss(pred_at_stations[:,t,:][mask_t], true_vals[:,t,:][mask_t]))
                        else:
                            rmse_t = torch.tensor(0.0, device=device)
                        rmse_time_history[t].append(rmse_t.item())

            if i % 10 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f} | Point: {loss_dict['point']:.4f} | "
                      f"Conserve: {loss_dict['conserve']:.4f} | Batch RMSE: {batch_rmse:.4f}")

        # ---------- 保存站点对比图 ----------
        if valid_num_stations > 0:
            pred_station_mean = pred_at_stations[0].mean(dim=0).cpu().numpy()  # (valid_num_stations,)
            true_station_mean = true_vals[0].mean(dim=0).cpu().numpy()          # (valid_num_stations,)
            plot_stations_vs_pred(
                s_coords[0, valid_station_mask].cpu().numpy(),
                true_station_mean,
                pred_station_mean,
                save_path=os.path.join(output_dir, f"station_comparison_epoch_{epoch}.png")
            )

        print(f"Epoch {epoch} finished. Avg Batch RMSE: {np.mean(all_batch_rmse):.4f}")
        scheduler.step(np.mean(all_batch_rmse))

        # ---------- 绘制时间步 RMSE 折线图 ----------
        plt.figure(figsize=(6,4))
        for t in range(T):
            if len(rmse_time_history[t]) > 0:
                plt.plot(range(len(rmse_time_history[t])), rmse_time_history[t], label=f'Time step {t}')
        plt.xlabel("Batch index")
        plt.ylabel("RMSE")
        plt.title(f"RMSE per Time Step - Epoch {epoch}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"rmse_per_time_epoch_{epoch}.png"))
        plt.close()


if __name__=="__main__":
    train_model()
