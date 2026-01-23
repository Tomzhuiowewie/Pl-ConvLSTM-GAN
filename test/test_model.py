#!/usr/bin/env python3
"""
测试脚本：验证修复后的模型是否正常工作
"""

import torch
import numpy as np
from src.models.generator import Generator
from src.losses.combined_loss import CombinedLoss


def test_generator_basic():
    """测试 Generator 基本功能"""
    print("=" * 60)
    print("测试 1: Generator 基本前向传播")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = Generator(
        hidden_dims=[16, 32],
        lu_channels=5,
        scale_factor=8
    ).to(device)
    
    # 创建测试数据
    B, T, H, W = 2, 5, 32, 32
    rain_lr = torch.randn(B, T, 1, H, W).to(device)
    dem = torch.randn(B, 1, H, W).to(device)
    lu = torch.randn(B, 5, H, W).to(device)
    
    # 前向传播
    try:
        output = model(rain_lr, dem, lu)
        print(f"✓ 输入形状: rain_lr={rain_lr.shape}, dem={dem.shape}, lu={lu.shape}")
        print(f"✓ 输出形状: {output.shape}")
        print(f"✓ 预期输出形状: [B={B}, T={T}, 1, H'={H*8}, W'={W*8}]")
        
        if output.shape == (B, T, 1, H*8, W*8):
            print("✓ 测试通过: 输出形状正确")
        else:
            print(f"✗ 测试失败: 输出形状不匹配")
            return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False
    
    return True


def test_generator_with_grid_size():
    """测试 Generator 使用网格尺寸配置"""
    print("\n" + "=" * 60)
    print("测试 2: Generator 使用目标网格尺寸")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型（目标网格尺寸 500m）
    model = Generator(
        hidden_dims=[16, 32],
        lu_channels=5,
        target_grid_size=(500, 500)  # 目标 500m × 500m
    ).to(device)
    
    # 创建测试数据
    B, T, H, W = 2, 5, 32, 32
    rain_lr = torch.randn(B, T, 1, H, W).to(device)
    dem = torch.randn(B, 1, H, W).to(device)
    lu = torch.randn(B, 5, H, W).to(device)
    
    # 前向传播（输入网格尺寸 4km）
    try:
        output = model(rain_lr, dem, lu, input_grid_size=(4000, 4000))
        print(f"✓ 输入网格尺寸: 4000m × 4000m")
        print(f"✓ 目标网格尺寸: 500m × 500m")
        print(f"✓ 理论缩放因子: 4000/500 = 8x")
        print(f"✓ 输出形状: {output.shape}")
        
        expected_H = int(H * 4000 / 500)
        expected_W = int(W * 4000 / 500)
        print(f"✓ 预期输出形状: [B={B}, T={T}, 1, H'={expected_H}, W'={expected_W}]")
        
        if output.shape == (B, T, 1, expected_H, expected_W):
            print("✓ 测试通过: 输出形状正确")
        else:
            print(f"✗ 测试失败: 输出形状不匹配")
            return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_loss_with_scale_factor():
    """测试损失函数的坐标缩放"""
    print("\n" + "=" * 60)
    print("测试 3: 损失函数站点坐标缩放")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建损失函数
    loss_module = CombinedLoss(lambda_point=0.1, lambda_conserve=1.0)
    
    # 创建测试数据
    B, T, H_lr, W_lr = 2, 5, 32, 32
    H_hr, W_hr = 256, 256  # 8x 超分辨率
    scale_factor = H_hr / H_lr
    
    pred = torch.randn(B, T, 1, H_hr, W_hr).to(device)
    lr_input = torch.randn(B, T, 1, H_lr, W_lr).to(device)
    
    # 创建站点坐标（低分辨率）
    num_stations = 5
    s_coords = torch.randint(0, min(H_lr, W_lr), (B, num_stations, 2)).to(device)
    s_values = torch.randn(B, T, num_stations).to(device)
    
    try:
        # 计算损失（带缩放因子）
        total_loss, loss_dict = loss_module(pred, lr_input, s_coords, s_values, scale_factor)
        
        print(f"✓ 低分辨率尺寸: {H_lr} × {W_lr}")
        print(f"✓ 高分辨率尺寸: {H_hr} × {W_hr}")
        print(f"✓ 缩放因子: {scale_factor}")
        print(f"✓ 站点数量: {num_stations}")
        print(f"✓ 总损失: {total_loss.item():.4f}")
        print(f"✓ 站点损失: {loss_dict['point'].item():.4f}")
        print(f"✓ 守恒损失: {loss_dict['conserve'].item():.4f}")
        
        if not torch.isnan(total_loss):
            print("✓ 测试通过: 损失计算正常")
        else:
            print("✗ 测试失败: 损失为 NaN")
            return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_dem_dimension():
    """测试 DEM 维度处理"""
    print("\n" + "=" * 60)
    print("测试 4: DEM 维度处理修复")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Generator(
        hidden_dims=[16, 32],
        lu_channels=5,
        scale_factor=4
    ).to(device)
    
    B, T, H, W = 1, 3, 16, 16
    rain_lr = torch.randn(B, T, 1, H, W).to(device)
    dem = torch.randn(B, 1, H, W).to(device)  # 4D 张量
    lu = torch.randn(B, 5, H, W).to(device)
    
    try:
        output = model(rain_lr, dem, lu)
        print(f"✓ DEM 输入形状: {dem.shape} (4D)")
        print(f"✓ 输出形状: {output.shape}")
        print("✓ 测试通过: DEM 维度处理正确")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试修复后的模型")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("Generator 基本功能", test_generator_basic()))
    results.append(("Generator 网格尺寸配置", test_generator_with_grid_size()))
    results.append(("损失函数坐标缩放", test_loss_with_scale_factor()))
    results.append(("DEM 维度处理", test_dem_dimension()))
    
    # 输出结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, result in results if result)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！模型修复成功。")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查。")


if __name__ == "__main__":
    main()
