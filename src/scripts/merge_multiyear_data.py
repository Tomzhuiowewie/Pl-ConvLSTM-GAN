import numpy as np
from pathlib import Path


def merge_rain_data(data_dir, out_path, years=range(2012, 2022), data_type="hydro_08-08"):
    """
    合并多年降水数据
    
    Args:
        data_dir: 数据目录
        out_path: 输出路径
        years: 年份范围
        data_type: 数据类型 ("hydro_08-08" 或 "cma_20-20")
    """
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    for year in years:
        file_path = data_dir / f"fenhe_{data_type}_{year}.npy"
        
        if not file_path.exists():
            print(f"⚠️  警告: {file_path} 不存在，跳过")
            continue
        
        data = np.load(file_path)
        print(f"加载 {year} 年数据: 形状 {data.shape}")
        all_data.append(data)
    
    if len(all_data) == 0:
        raise ValueError("没有找到任何数据文件！")
    
    # 沿时间维度拼接
    merged_data = np.concatenate(all_data, axis=0)
    
    # 保存
    np.save(out_path, merged_data)
    print(f"\n✓ 合并完成！")
    print(f"  总形状: {merged_data.shape}")
    print(f"  总天数: {merged_data.shape[0]}")
    print(f"  保存至: {out_path}")
    
    return merged_data


def merge_lucc_data(data_dir, out_path, years=range(2012, 2022)):
    """
    合并多年LUCC数据，创建时间序列
    
    Args:
        data_dir: 数据目录
        out_path: 输出路径
        years: 年份范围
    
    Returns:
        year_to_lucc: 字典，映射年份到LUCC数据索引
    """
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_lucc = []
    year_to_idx = {}
    
    for idx, year in enumerate(years):
        file_path = data_dir / f"lucc_1km_{year}.npy"
        
        if not file_path.exists():
            print(f"⚠️  警告: {file_path} 不存在，跳过")
            continue
        
        lucc = np.load(file_path)
        print(f"加载 {year} 年 LUCC: 形状 {lucc.shape}")
        all_lucc.append(lucc)
        year_to_idx[year] = idx
    
    if len(all_lucc) == 0:
        raise ValueError("没有找到任何LUCC文件！")
    
    # 堆叠为 (num_years, H, W)
    merged_lucc = np.stack(all_lucc, axis=0)
    
    # 保存
    np.save(out_path, merged_lucc)
    
    # 保存年份映射
    mapping_path = out_path.parent / "lucc_year_mapping.npy"
    np.save(mapping_path, np.array(list(year_to_idx.keys())))
    
    print(f"\n✓ LUCC 合并完成！")
    print(f"  总形状: {merged_lucc.shape}")
    print(f"  年份数: {merged_lucc.shape[0]}")
    print(f"  保存至: {out_path}")
    print(f"  年份映射: {mapping_path}")
    
    return merged_lucc, year_to_idx


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    
    # 配置路径
    daily_dir = script_dir / "../../data/processed/daily"
    static_dir = script_dir / "../../data/processed/static_features_1km"
    
    print("="*60)
    print("开始合并 2012-2021 年数据")
    print("="*60)
    
    # 1. 合并降水数据（水文体系）
    print("\n[1/2] 合并降水数据...")
    merge_rain_data(
        data_dir=daily_dir,
        out_path=daily_dir / "fenhe_hydro_08-08_2012-2021.npy",
        years=range(2012, 2022),
        data_type="hydro_08-08"
    )
    
    # 2. 合并 LUCC 数据
    print("\n[2/2] 合并 LUCC 数据...")
    merge_lucc_data(
        data_dir=static_dir,
        out_path=static_dir / "lucc_1km_2012-2021.npy",
        years=range(2012, 2022)
    )
    
    print("\n" + "="*60)
    print("所有数据合并完成！")
    print("="*60)
