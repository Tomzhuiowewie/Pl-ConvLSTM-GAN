import numpy as np
import rioxarray
from pathlib import Path

def convert_tif_to_npy(dem_path, lucc_path, out_dir, year=None, save_dem=True):
    """
    将 TIF 格式的静态数据降尺度并转换为 NPY 格式（30m -> 1km）。
    
    Args:
        dem_path: DEM 文件路径
        lucc_path: LUCC 文件路径
        out_dir: 输出目录
        year: 年份（用于区分不同年份的 LUCC 数据）
        save_dem: 是否保存 DEM（默认 True，后续循环设为 False）
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 DEM（可选）
    if save_dem:
        print("正在加载 DEM...")
        dem_raw = rioxarray.open_rasterio(dem_path, chunks={'x': 2048, 'y': 2048}).squeeze()
        # 数据清理
        dem_cleaned = dem_raw.where(dem_raw < 9000, 0)
        # 降尺度：使用平均值代表 1km 范围内的平均海拔
        print("正在降尺度 DEM (30m -> 1km)...")
        dem_processed = dem_cleaned.coarsen(x=33, y=33, boundary='trim').mean()
    else:
        print("跳过 DEM 加载（已在首次循环保存）")
        dem_processed = None
    
    # 2. 加载 LUCC
    print("正在加载 LUCC...")
    lucc_raw = rioxarray.open_rasterio(lucc_path, chunks={'x': 2048, 'y': 2048}).squeeze()
    # 数据清理
    lucc_cleaned = lucc_raw.where(lucc_raw != 255, 0)
    # 降尺度：使用 max 取窗口内最大值，保证整数类型
    print("正在降尺度 LUCC (30m -> 1km)...")
    lucc_processed = lucc_cleaned.coarsen(x=33, y=33, boundary='trim').max()
    
    # 3. 执行计算并转换为 numpy 数组
    print("正在执行计算并导出...")
    if save_dem and dem_processed is not None:
        dem_final = dem_processed.compute()
        dem_vals = dem_final.values.astype(np.float32)
        # 最终清理
        dem_vals[dem_vals > 9000] = 0
    else:
        dem_final = None
        dem_vals = None
    
    lucc_final = lucc_processed.compute()
    lucc_vals = lucc_final.values.astype(np.uint8)
    # 最终清理
    lucc_vals[lucc_vals == 255] = 0

    # 显示空间信息（仅在第一次循环时）
    if save_dem and dem_vals is not None:
        print("\n空间信息对比:")
        print(f"  DEM 形状: {dem_vals.shape}")
        print(f"  LUCC 形状: {lucc_vals.shape}")
        
        if dem_vals.shape != lucc_vals.shape:
            print(f"  ⚠️  注意：DEM 和 LUCC 形状不同，将分别保存\n")
        else:
            print(f"  ✓ 形状一致\n")

    # 4. 保存为 NPY
    print("正在保存为 NPY 格式...")
    
    # DEM 只在第一次循环时保存
    if save_dem and dem_vals is not None:
        np.save(out_dir / "dem_1km.npy", dem_vals)
        
        # 保存 DEM 坐标信息
        dem_lon = dem_final.x.values if 'x' in dem_final.coords else dem_final.lon.values
        dem_lat = dem_final.y.values if 'y' in dem_final.coords else dem_final.lat.values
        np.save(out_dir / "dem_lons_1km.npy", dem_lon)
        np.save(out_dir / "dem_lats_1km.npy", dem_lat)
        
        print(f"DEM 形状: {dem_vals.shape}, 最大海拔: {dem_vals.max():.1f}m")
        print(f"保存文件: dem_1km.npy, dem_lons_1km.npy, dem_lats_1km.npy")
    
    # LUCC 按年份保存
    if year is not None:
        lucc_filename = f"lucc_1km_{year}.npy"
    else:
        lucc_filename = "lucc_1km.npy"
    np.save(out_dir / lucc_filename, lucc_vals)
    
    # 保存 LUCC 坐标信息（只在第一次循环时保存）
    if save_dem:
        lucc_lon = lucc_final.x.values if 'x' in lucc_final.coords else lucc_final.lon.values
        lucc_lat = lucc_final.y.values if 'y' in lucc_final.coords else lucc_final.lat.values
        np.save(out_dir / "lucc_lons_1km.npy", lucc_lon)
        np.save(out_dir / "lucc_lats_1km.npy", lucc_lat)
        print(f"LUCC 坐标: lucc_lons_1km.npy, lucc_lats_1km.npy")

    print(f"--- 降尺度完成 ---")
    print(f"LUCC 形状: {lucc_vals.shape}, 分类数: {len(np.unique(lucc_vals))}")
    print(f"保存文件: {lucc_filename}")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    
    base_config = {
        "dem_path": script_dir / "../../data/raw/dem/dem30.tif",
        "lucc_base_path": script_dir / "../../data/raw/lucc",
        "out_dir": script_dir / "../../data/processed/static_features_1km"
    }
    
    # 循环处理 2012-2021 年的数据
    for idx, year in enumerate(range(2012, 2022)):
        is_first = (idx == 0)  # 判断是否为第一次循环
        
        if is_first:
            print(f"\n{'='*60}")
            print(f"开始降尺度 {year} 年数据（包含 DEM）")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"开始降尺度 {year} 年 LUCC 数据")
            print(f"{'='*60}\n")
        
        # 在基础路径上追加年份
        lucc_path = base_config["lucc_base_path"] / f"fenhe_{year}"
        
        config = {
            "dem_path": base_config["dem_path"],
            "lucc_path": lucc_path,
            "out_dir": base_config["out_dir"],
            "year": year,
            "save_dem": is_first  # 只在第一次循环时保存 DEM
        }
        
        try:
            convert_tif_to_npy(**config)
            print(f"\n✓ {year} 年数据降尺度完成\n")
        except Exception as e:
            print(f"\n✗ {year} 年数据降尺度失败: {e}\n")
            continue