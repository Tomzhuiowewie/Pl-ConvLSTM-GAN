import numpy as np
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from pathlib import Path

def process_static_features_to_1km(dem_path, lucc_path, out_dir):
    """
    直接将已裁剪好的流域静态数据重采样至 1km (0.01°)。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 处理 DEM
    print("正在加载并处理 DEM...")
    dem_raw = rioxarray.open_rasterio(dem_path, chunks={'x': 2048, 'y': 2048}).squeeze()
    # 数据清理
    dem_cleaned = dem_raw.where(dem_raw < 9000, 0)
    # 使用平均值代表 1km 范围内的平均海拔
    dem_1km = dem_cleaned.coarsen(x=33, y=33, boundary='trim').mean()
    
    # 2. 处理 LUCC
    print("正在加载并处理 LUCC...")
    lucc_raw = rioxarray.open_rasterio(lucc_path, chunks={'x': 2048, 'y': 2048}).squeeze()
    # 数据清理
    lucc_cleaned = lucc_raw.where(lucc_raw != 255, 0)
    # 使用 max 取窗口内最大值，既保证了整数类型，也避免了插值带来的小数
    lucc_1km = lucc_cleaned.coarsen(x=33, y=33, boundary='trim').max()

    # 3. 强制对齐 (确保两个矩阵维度完全一致)
    # 因为 coarsen 后的尺寸由原始像素数除以 33 决定，通过对齐消除 1 个像素左右的差异
    print("同步静态特征空间维度...")
    lucc_1km = lucc_1km.rio.reproject_match(dem_1km, resampling=Resampling.nearest)

    # 4. 执行计算并保存
    print("正在执行计算并导出 NPY...")
    dem_final = dem_1km.compute()
    lucc_final = lucc_1km.compute()

    # 确保数据类型正确 (DEM 为浮点，LUCC 为整数)
    np.save(out_dir / "dem_1km.npy", dem_final.values.astype(np.float32))
    np.save(out_dir / "lucc_1km.npy", lucc_final.values.astype(np.uint8))
    
    # 坐标保存逻辑保持不变
    lon = dem_final.x.values if 'x' in dem_final.coords else dem_final.lon.values
    lat = dem_final.y.values if 'y' in dem_final.coords else dem_final.lat.values
    np.save(out_dir / "lons_1km.npy", lon)
    np.save(out_dir / "lats_1km.npy", lat)

    print(f"--- 处理完成 ---")
    print(f"DEM 形状: {dem_final.shape}, 最大海拔: {dem_final.max().item():.1f}m")
    print(f"LUCC 形状: {lucc_final.shape}, 分类数: {len(np.unique(lucc_final))}")

if __name__ == "__main__":
    config = {
        "dem_path": "data/dem/dem30.tif",
        "lucc_path": "data/lucc/fenhe_2021",
        "out_dir": "data/static_features_1km"
    }
    
    process_static_features_to_1km(**config)