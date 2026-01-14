import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import regionmask
from pathlib import Path
import matplotlib.pyplot as plt

def process_cmorph_to_fenhe(nc_dir, shp_path, out_base_path, year="2021", save=True):
    """
    整合后的降水数据处理流程
    """
    nc_dir = Path(nc_dir)
    out_base_path = Path(out_base_path)
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
    
    # 3. 应用掩膜并进行日聚合
    print("正在进行空间过滤与按日求和...")
    # isel(region=0) 选取 shapefile 中的第一个多边形（汾河）
    masked_precip = ds["cmorph"].where(mask.isel(region=0))
    
    # 裁剪到汾河流域范围，去除流域外的网格维度
    print("正在裁剪到汾河流域范围...")
    # 获取流域内有效数据的边界
    valid_mask = mask.isel(region=0) == 0
    # 找到有效区域的经纬度范围
    lon_valid = ds.lon.where(valid_mask.any(dim='lat')).dropna('lon')
    lat_valid = ds.lat.where(valid_mask.any(dim='lon')).dropna('lat')
    
    # 裁剪到包含流域的最小矩形
    masked_precip = masked_precip.sel(
        lon=slice(lon_valid.min(), lon_valid.max()),
        lat=slice(lat_valid.min(), lat_valid.max())
    )
    
    # 不同体系聚合：将 24 小时数据求和
    # 【水文体系】当日 08:00 -> 次日 08:00 (北京时间)
    # 因为 UTC 00:00 = 北京 08:00，所以直接按 UTC 自然日聚合
    daily_hydro = masked_precip.resample(time="1D").sum(min_count=24)
    
    # 【气象体系】前一日 20:00 -> 当日 20:00 (北京时间)
    # 北京 20:00 = UTC 12:00。需将 UTC 12:00 偏移至当日 00:00
    daily_cma = masked_precip.shift(time=-12).resample(time="1D").sum(min_count=24)

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
            daily_hydro.sel(time=f"{year}-07-10").plot(ax=axes[0], cmap="Blues", vmin=0, vmax=50)
            axes[0].set_title("Hydrological (08:00-08:00)")
            
            daily_cma.sel(time=f"{year}-07-10").plot(ax=axes[1], cmap="Blues", vmin=0, vmax=50)
            axes[1].set_title("Meteorological (20:00-20:00)")
            plt.tight_layout()
            plt.show()
        except:
            print("提示：该日期数据不足，跳过绘图。")
    
    else:
        return daily_hydro, daily_cma


if __name__ == "__main__":
    # 配置路径
    config = {
        "nc_dir": "data/cmorph-2021/hourly",
        "shp_path": "data/FenheBasin/fenhe.shp",
        "out_base_path": "data/cmorph-2021/daily",
        "year": "2021",
        "save": True
    }

    process_cmorph_to_fenhe(**config)