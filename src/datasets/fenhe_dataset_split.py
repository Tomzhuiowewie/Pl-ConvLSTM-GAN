import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from pathlib import Path


# ========== GIS 工具 ===============

def get_shapefile_extent(shp_path):
    gdf = gpd.read_file(shp_path)
    minx, miny, maxx, maxy = gdf.total_bounds
    return [miny, maxy, minx, maxx]  # [min_lat, max_lat, min_lon, max_lon]


# =========== 数据集划分工具 ==============

def split_dataset_by_year(dataset, train_years, val_years, test_years):
    """
    按年份划分数据集
    
    Args:
        dataset: FenheDataset实例
        train_years: tuple (start_year, end_year) 训练集年份范围
        val_years: tuple (start_year, end_year) 验证集年份范围
        test_years: tuple (start_year, end_year) 测试集年份范围
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    def get_year_indices(start_year, end_year):
        """获取指定年份范围的样本索引"""
        indices = []
        cumulative_days = 0
        
        for year in range(dataset.start_year, dataset.end_year + 1):
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            days_in_year = 366 if is_leap else 365
            
            if start_year <= year <= end_year:
                # 该年份在目标范围内，添加所有样本索引
                # 注意：最后T-1天无法形成完整时间窗口
                year_start = cumulative_days
                year_end = cumulative_days + days_in_year - dataset.T
                indices.extend(range(year_start, year_end))
            
            cumulative_days += days_in_year
        
        return indices
    
    train_indices = get_year_indices(*train_years)
    val_indices = get_year_indices(*val_years)
    test_indices = get_year_indices(*test_years)
    
    print(f"数据集划分:")
    print(f"  训练集: {train_years[0]}-{train_years[1]}, {len(train_indices)} 样本")
    print(f"  验证集: {val_years[0]}-{val_years[1]}, {len(val_indices)} 样本")
    print(f"  测试集: {test_years[0]}-{test_years[1]}, {len(test_indices)} 样本")
    print(f"  总计: {len(train_indices) + len(val_indices) + len(test_indices)} 样本")
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def split_dataset_random(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    随机划分数据集（不推荐用于时间序列）
    
    Args:
        dataset: FenheDataset实例
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    np.random.seed(seed)
    total_samples = len(dataset)
    indices = np.random.permutation(total_samples)
    
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"随机划分数据集:")
    print(f"  训练集: {len(train_indices)} 样本 ({train_ratio*100:.1f}%)")
    print(f"  验证集: {len(val_indices)} 样本 ({val_ratio*100:.1f}%)")
    print(f"  测试集: {len(test_indices)} 样本 ({test_ratio*100:.1f}%)")
    
    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())
    test_dataset = Subset(dataset, test_indices.tolist())
    
    return train_dataset, val_dataset, test_dataset


# =========== FenheDataset ==============
# 从原始fenhe_dataset.py导入
from src.datasets.fenhe_dataset import FenheDataset
