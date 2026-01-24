import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import regionmask
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path


def process_cmorph_to_fenhe(nc_dir, shp_path, out_path, year, save=True):
    """
    整合后的降水数据处理流程
    """
    nc_dir = Path(nc_dir) / str(year)
    out_base_path = Path(out_path)
    out_base_path.mkdir(parents=True, exist_ok=True)
    
    files = sorted(list(nc_dir.glob(f"CMORPH_V1.0_ADJ_0.25deg-HLY_{year}*.nc")))
    
    if not files:
        print(f"未找到 {year} 年的 NC 文件")
        return

    # 1. 加载数据并修正坐标轴
    print("正在加载数据并标准化坐标...")
    # 建议先打开一个文件确定变量名，假设变量名为 'cmorph'
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')
    
    # 转换经度：0~360 -> -180~180
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
    # 必须排序，否则后续掩膜和绘图会错乱
    ds = ds.sortby(['lon', 'lat'])

    # 2. 创建流域掩膜 (使用 regionmask)
    print("正在生成流域掩膜...")
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # 创建掩膜对象，lon/lat 对应 ds 中的坐标
    mask = regionmask.mask_3D_geopandas(gdf, ds.lon, ds.lat)
    
    # 调试：检查掩膜的实际值
    mask_values = mask.isel(region=0).values
    unique_values = np.unique(mask_values[~np.isnan(mask_values)])
    print(f"掩膜唯一值（非NaN）: {unique_values}")
    print(f"掩膜形状: {mask.isel(region=0).shape}")
    print(f"流域内点数（True）: {np.sum(mask_values == True)}")
    print(f"流域外点数（False）: {np.sum(mask_values == False)}")
    
    # 3. 应用掩膜并进行日聚合
    print("正在进行空间过滤与按日求和...")
    # isel(region=0) 选取 shapefile 中的第一个多边形（汾河）
    # regionmask 返回 True 表示流域内，False 表示流域外
    masked_precip = ds["cmorph"].where(mask.isel(region=0))
    
    # 裁剪到汾河流域范围，去除流域外的网格维度
    print("正在裁剪到汾河流域范围...")
    # 获取流域内有效数据的边界
    basin_mask = mask.isel(region=0)
    # 找到有效区域的经纬度范围
    lon_valid = ds.lon.where(basin_mask.any(dim='lat')).dropna('lon')
    lat_valid = ds.lat.where(basin_mask.any(dim='lon')).dropna('lat')
    
    # 裁剪到包含流域的最小矩形
    masked_precip = masked_precip.sel(
        lon=slice(lon_valid.min(), lon_valid.max()),
        lat=slice(lat_valid.min(), lat_valid.max())
    )
    
    # 数据质量检查
    print("正在检查数据质量...")
    total_hours = len(masked_precip.time)
    
    # 判断是否为闰年
    year_int = int(year)
    is_leap = (year_int % 4 == 0 and year_int % 100 != 0) or (year_int % 400 == 0)
    expected_hours = 8784 if is_leap else 8760
    expected_days = 366 if is_leap else 365
    
    print(f"年份: {year} ({'闰年' if is_leap else '平年'})")
    print(f"小时数据总量: {total_hours} (预期: {expected_hours})")
    
    # 检查时间范围
    time_start = pd.Timestamp(masked_precip.time.values[0])
    time_end = pd.Timestamp(masked_precip.time.values[-1])
    print(f"时间范围: {time_start} ~ {time_end}")
    
    # 检查是否有重复时间戳
    time_values = pd.DatetimeIndex(masked_precip.time.values)
    duplicates = time_values[time_values.duplicated()]
    if len(duplicates) > 0:
        print(f"⚠️  警告：发现 {len(duplicates)} 个重复时间戳！")
        print(f"   重复时间: {duplicates[:5].tolist()}")  # 只显示前5个
    
    # 检查数据量
    if total_hours != expected_hours:
        diff = total_hours - expected_hours
        if diff > 0:
            print(f"⚠️  警告：数据量超出预期！多了 {diff} 小时")
        else:
            print(f"⚠️  警告：数据量不完整！缺少 {-diff} 小时")
    
    # 不同体系聚合：将 24 小时数据求和
    # 【水文体系】当日 08:00 -> 次日 08:00 (北京时间)
    # 因为 UTC 00:00 = 北京 08:00，所以直接按 UTC 自然日聚合
    daily_hydro = masked_precip.resample(time="1D").sum(min_count=24)
    
    # 【气象体系】前一日 20:00 -> 当日 20:00 (北京时间)
    # 北京 20:00 = UTC 12:00。需将 UTC 12:00 偏移至当日 00:00
    daily_cma = masked_precip.shift(time=-12).resample(time="1D").sum(min_count=24)
    
    # 检查日数据量
    expected_days = 365 if year != "2020" else 366
    print(f"水文体系日数据量: {len(daily_hydro.time)} (预期: {expected_days})")
    print(f"气象体系日数据量: {len(daily_cma.time)} (预期: {expected_days})")

    if save:
        # 4. 导出数据
        results = {
            "hydro_08-08": daily_hydro,
            "cma_20-20": daily_cma
        }

        for name, ds_res in results.items():
            # 保存 NPY
            npy_path = out_base_path / f"fenhe_{name}_{year}.npy"
            np.save(npy_path, ds_res.values)
            
            # 保存 CSV (清理 NaN)
            csv_path = out_base_path / f"fenhe_{name}_{year}.csv"
            df = ds_res.to_dataframe(name="precip").reset_index()
            df_clean = df.dropna(subset=["precip"])
            df_clean.to_csv(csv_path, index=False)
            
            # 打印检查
            lon_min, lon_max = df_clean['lon'].min(), df_clean['lon'].max()
            print(f"[{name}] 导出完成。经度范围: {lon_min:.2f}~{lon_max:.2f}, 保存至: {csv_path.name}")
        

        # 5. 可视化对比 (7月10日为例)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 水文体系降水图
            daily_hydro.sel(time=f"{year}-07-10").plot(
                ax=axes[0], 
                cmap="Blues", 
                vmin=0, vmax=50,
                add_colorbar=True
            )
            axes[0].set_title("Hydrological (08:00-08:00)")
            axes[0].set_xlabel("经度 (°)")
            axes[0].set_ylabel("纬度 (°)")
            
            # 气象体系降水图
            daily_cma.sel(time=f"{year}-07-10").plot(
                ax=axes[1], 
                cmap="Blues", 
                vmin=0, vmax=50,
                add_colorbar=True
            )
            axes[1].set_title("Meteorological (20:00-20:00)")
            axes[1].set_xlabel("经度 (°)")
            axes[1].set_ylabel("纬度 (°)")
            
            plt.tight_layout()
            plt.show()
            
            # 打印实际经纬度范围
            lon_min, lon_max = float(daily_hydro.lon.min()), float(daily_hydro.lon.max())
            lat_min, lat_max = float(daily_hydro.lat.min()), float(daily_hydro.lat.max())
            print(f"裁剪后数据范围: 经度 {lon_min:.2f}°~{lon_max:.2f}°, 纬度 {lat_min:.2f}°~{lat_max:.2f}°")
            
        except Exception as e:
            print(f"提示：该日期数据不足或绘图失败，跳过绘图。错误: {e}")
    
    else:
        return daily_hydro, daily_cma


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    # 配置路径
    base_config = {
        "nc_dir": script_dir / "../../data/raw/cmorph",
        "shp_path": script_dir / "../../data/raw/FenheBasin/fenhe.shp",
        "out_path": script_dir / "../../data/processed/daily",
        "save": True
    }
    
    # 循环处理 2012-2021 年的数据
    for year in range(2012, 2022):
        print(f"\n{'='*60}")
        print(f"开始处理 {year} 年数据")
        print(f"{'='*60}\n")
        
        config = {**base_config, "year": str(year)}
        
        try:
            process_cmorph_to_fenhe(**config)
            print(f"\n✓ {year} 年数据处理完成\n")
        except Exception as e:
            print(f"\n✗ {year} 年数据处理失败: {e}\n")
            continue