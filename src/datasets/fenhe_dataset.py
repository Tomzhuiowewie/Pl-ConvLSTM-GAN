import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ========== GIS 工具 ===============

def get_shapefile_extent(shp_path):
    gdf = gpd.read_file(shp_path)
    minx, miny, maxx, maxy = gdf.total_bounds
    return [miny, maxy, minx, maxx]  # [min_lat, max_lat, min_lon, max_lon]


# =========== FenheDataset ==============

class FenheDataset(Dataset):
    def __init__(self, rain_lr_path, dem_path, lucc_path,
                 rain_meta_path, rain_station_path, shp_path, T=5):

        # ---------- 卫星降水 ----------
        self.rain_lr = np.nan_to_num(
            np.load(rain_lr_path).astype(np.float32),
            nan=0.0, posinf=0.0, neginf=0.0
        )

        # ---------- DEM ----------
        self.dem = np.load(dem_path)
        self.dem_norm = (self.dem - self.dem.min()) / (self.dem.max() - self.dem.min() + 1e-7)

        # ---------- LUCC ----------    
        self.lucc = np.load(lucc_path).astype(int)
        self.lucc_onehot = self._lucc_to_onehot(self.lucc)

        self.T = T

        # ---------- 网格范围 ----------
        self.grid_extent = get_shapefile_extent(shp_path)

        # ---------- 站点 ----------
        self.s_coords, self.s_values = self._prepare_stations(rain_meta_path, rain_station_path)

    # -----------------------------
    # LUCC → onehot
    # -----------------------------
    def _lucc_to_onehot(self, lucc, ignore_index=0):
        """使用零矩阵填充的简化实现"""
        # 获取有效类别
        valid_mask = lucc != ignore_index
        unique_labels = np.unique(lucc[valid_mask])
        num_classes = len(unique_labels)
        
        H, W = lucc.shape
        onehot = np.zeros((num_classes, H, W), dtype=np.float32)
        
        # 为每个有效类别填充1
        for i, label in enumerate(unique_labels):
            onehot[i, lucc == label] = 1.0
        
        return onehot

    # -----------------------------
    # 站点及降水处理
    # -----------------------------
    def _prepare_stations(self, rain_meta_path, rain_station_path):

        df_meta = pd.read_excel(rain_meta_path, usecols=["F_站号", "经度", "纬度"])
        df_rain = (
            pd.read_excel(rain_station_path)
            .query("year==2021")
            .sort_values(["year", "month", "day"])
            .reset_index(drop=True)
        )

        min_lat, max_lat, min_lon, max_lon = self.grid_extent
        rows_total, cols_total = self.rain_lr.shape[-2:]

        coords = [] # 站点坐标(station_num, 2)
        val_list = [] # 站点降水(365, station_num)

        # ---------- 空间兜底 ----------
        rain_values = df_rain.drop(columns=["year", "month", "day"], errors="ignore")
        spatial_mean = rain_values.mask(rain_values < 0).mean(axis=1).to_numpy()

        for _, row in df_meta.iterrows():
            st_id = int(row["F_站号"])
            lat, lon = row["纬度"], row["经度"]

            r_idx = int((max_lat - lat) / (max_lat - min_lat) * (rows_total - 1))
            c_idx = int((lon - min_lon) / (max_lon - min_lon) * (cols_total - 1))

            if not (0 <= r_idx < rows_total and 0 <= c_idx < cols_total):
                print(f"Station {st_id} ({lat},{lon}) out of bounds, skipped")
                continue

            col = str(st_id)
            if col not in df_rain.columns:
                continue

            series = df_rain[col].to_numpy(dtype=np.float32)

            # ---------- 负值修复 ----------
            for t in range(len(series)):
                if series[t] >= 0:
                    continue

                candidates = []
                if t - 1 >= 0 and series[t - 1] >= 0:
                    candidates.append(series[t - 1])
                if t + 1 < len(series) and series[t + 1] >= 0:
                    candidates.append(series[t + 1])

                if len(candidates) > 0:
                    series[t] = np.mean(candidates)
                elif not np.isnan(spatial_mean[t]):
                    series[t] = spatial_mean[t]
                else:
                    series[t] = 0.0

            coords.append([r_idx, c_idx])
            val_list.append(series)

        if len(val_list) == 0:
            raise ValueError("No valid stations found within grid bounds!")

        stacked = np.stack(val_list, axis=1).astype(np.float32)
        stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)

        return np.array(coords), stacked

    # -----------------------------
    # PyTorch 接口
    # -----------------------------
    def __len__(self):
        return self.rain_lr.shape[0] - self.T

    def __getitem__(self, idx):
        x_lr = torch.FloatTensor(self.rain_lr[idx:idx + self.T, None, ...])

        # 保持 DEM 和 LUCC 的原始高分辨率 (1km)，避免信息损失
        # Generator 会直接将它们插值到目标分辨率
        dem = torch.FloatTensor(self.dem_norm[None, ...])  # [1, H_dem, W_dem]
        lu = torch.FloatTensor(self.lucc_onehot)           # [C, H_lu, W_lu]

        s_vals = torch.FloatTensor(self.s_values[idx:idx + self.T])
        s_coords = torch.LongTensor(self.s_coords)

        return x_lr, dem, lu, s_coords, s_vals

