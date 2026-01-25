#!/usr/bin/env python3
"""
测试增强版损失函数
验证方案1+3的实现
"""

import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.losses.physics_loss import TerrainPrecipitationLoss, SpatialVariabilityLoss, ExtremeValueLoss
from src.losses.statistical_loss import DistributionMatchingLoss, SpatialCorrelationLoss
from src.losses.enhanced_loss import EnhancedCombinedLoss


def test_terrain_loss():
    """测试地形-降水关系损失"""
    print("="*60)
    print("测试 1: 地形-降水关系损失")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    B, T, H, W = 2, 5, 64, 64
    rain_hr = torch.randn(B, T, 1, H, W).abs().to(device)  # 非负降水
    dem = torch.randn(B, 1, H, W).abs().to(device)  # 非负地形
    
    # 创建损失函数
    loss_fn = TerrainPrecipitationLoss()
    
    try:
        loss = loss_fn(rain_hr, dem)
        print(f"✓ 地形-降水损失: {loss.item():.4f}")
        print(f"✓ 测试通过")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_variability_loss():
    """测试空间变异性损失"""
    print("\n" + "="*60)
    print("测试 2: 空间变异性损失")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    B, T, H, W = 2, 5, 64, 64
    rain_hr = torch.randn(B, T, 1, H, W).abs().to(device)
    
    # 创建损失函数
    loss_fn = SpatialVariabilityLoss()
    
    try:
        loss = loss_fn(rain_hr)
        print(f"✓ 空间变异性损失: {loss.item():.4f}")
        print(f"✓ 测试通过")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extreme_loss():
    """测试极端值分布损失"""
    print("\n" + "="*60)
    print("测试 3: 极端值分布损失")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    B, T, H, W = 2, 5, 64, 64
    num_stations = 20
    
    rain_hr = torch.randn(B, T, 1, H, W).abs().to(device)
    s_values = torch.randn(B, T, num_stations).abs().to(device)
    
    # 创建损失函数
    loss_fn = ExtremeValueLoss()
    
    try:
        loss = loss_fn(rain_hr, s_values)
        print(f"✓ 极端值分布损失: {loss.item():.4f}")
        print(f"✓ 测试通过")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distribution_loss():
    """测试概率分布匹配损失"""
    print("\n" + "="*60)
    print("测试 4: 概率分布匹配损失")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    B, T, H, W = 2, 5, 64, 64
    num_stations = 20
    
    rain_hr = torch.randn(B, T, 1, H, W).abs().to(device)
    s_values = torch.randn(B, T, num_stations).abs().to(device)
    
    # 创建损失函数
    loss_fn = DistributionMatchingLoss()
    
    try:
        loss = loss_fn(rain_hr, s_values)
        print(f"✓ 概率分布匹配损失: {loss.item():.4f}")
        print(f"✓ 测试通过")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_correlation_loss():
    """测试空间相关性损失"""
    print("\n" + "="*60)
    print("测试 5: 空间相关性损失")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    B, T, H, W = 2, 5, 64, 64
    rain_hr = torch.randn(B, T, 1, H, W).abs().to(device)
    
    # 创建损失函数
    loss_fn = SpatialCorrelationLoss()
    
    try:
        loss = loss_fn(rain_hr)
        print(f"✓ 空间相关性损失: {loss.item():.4f}")
        print(f"✓ 测试通过")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_combined_loss():
    """测试增强版组合损失"""
    print("\n" + "="*60)
    print("测试 6: 增强版组合损失")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    B, T, H_lr, W_lr = 2, 5, 32, 32
    H_hr, W_hr = 256, 256
    num_stations = 20
    
    pred = torch.randn(B, T, 1, H_hr, W_hr).abs().to(device)
    lr_input = torch.randn(B, T, 1, H_lr, W_lr).abs().to(device)
    s_coords = torch.randint(0, H_lr, (num_stations, 2)).to(device)
    s_values = torch.randn(B, T, num_stations).abs().to(device)
    dem = torch.randn(B, 1, H_hr, W_hr).abs().to(device)
    lucc = torch.randn(B, 5, H_hr, W_hr).abs().to(device)
    
    scale_factor = H_hr / H_lr
    
    # 创建增强版损失函数
    loss_fn = EnhancedCombinedLoss(
        lambda_point=1.0,
        lambda_conserve=1.0,
        lambda_smooth=0.1,
        lambda_temporal=0.05,
        lambda_terrain=0.5,
        lambda_variability=0.2,
        lambda_extreme=0.3,
        lambda_distribution=0.4,
        lambda_correlation=0.2
    )
    
    try:
        total_loss, loss_dict = loss_fn(pred, lr_input, s_coords, s_values, scale_factor, dem, lucc)
        
        print(f"✓ 总损失: {total_loss.item():.4f}")
        print(f"\n各项损失:")
        for key, value in loss_dict.items():
            print(f"  - {key}: {value:.4f}")
        
        print(f"\n✓ 测试通过")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("开始测试增强版损失函数 (方案1+3)")
    print("="*60 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("地形-降水关系损失", test_terrain_loss()))
    results.append(("空间变异性损失", test_variability_loss()))
    results.append(("极端值分布损失", test_extreme_loss()))
    results.append(("概率分布匹配损失", test_distribution_loss()))
    results.append(("空间相关性损失", test_correlation_loss()))
    results.append(("增强版组合损失", test_enhanced_combined_loss()))
    
    # 输出结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, result in results if result)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！方案1+3实现成功。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查。")
        return 1


if __name__ == "__main__":
    exit(main())
