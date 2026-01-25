# Pl-ConvLSTM-GAN (Enhanced Version)

基于物理信息和统计约束的降水预测深度学习模型 - 增强版本

## ✨ 新增功能

本分支在原始模型基础上增加了物理约束和统计特性:

### 🔬 物理损失函数 (`src/losses/physics_loss.py`)
- **质量守恒**: 确保降水总量守恒
- **能量守恒**: 保持物理能量平衡
- **动量守恒**: 维持动力学一致性
- **水汽平衡**: 符合大气水汽守恒定律

### 📊 统计损失函数 (`src/losses/statistical_loss.py`)
- **分布匹配**: KL散度、Wasserstein距离
- **极值保持**: 保持降水极值特征
- **空间相关性**: 维持空间自相关结构
- **时间连续性**: 确保时间序列平滑

### 🚀 增强训练器 (`src/training/enhanced_trainer.py`)
- 多损失函数动态加权
- 自适应学习率调整
- 详细的训练监控和可视化

## 📁 项目结构

```
Pl-ConvLSTM-GAN/
├── configs/
│   ├── default.yaml          # 基础配置
│   └── enhanced.yaml         # 增强版配置
├── src/
│   ├── models/               # 模型定义
│   ├── datasets/             # 数据加载
│   ├── losses/
│   │   ├── combined_loss.py  # 原始损失
│   │   ├── physics_loss.py   # 物理损失 ✨
│   │   ├── statistical_loss.py # 统计损失 ✨
│   │   └── enhanced_loss.py  # 增强损失组合 ✨
│   └── training/
│       └── enhanced_trainer.py # 增强训练器 ✨
├── main.py                   # 基础训练入口
├── main_enhanced.py          # 增强训练入口 ✨
└── ENHANCED_USAGE.md         # 详细使用文档 ✨
```

## 🚀 快速开始

### 环境配置

```bash
conda activate pi-conv
```

### 训练模型

```bash
# 使用增强版本训练
python main_enhanced.py --config configs/enhanced.yaml

# 使用基础版本训练
python main.py --config configs/default.yaml
```

### 测试模型

```bash
# 测试增强损失函数
python test/test_enhanced_loss.py
```

## 📖 详细文档

详见 [ENHANCED_USAGE.md](ENHANCED_USAGE.md)

## 🔄 分支说明

- **main**: 基础版本
- **feature-enhanced**: 增强版本(当前分支)

## 📊 性能提升

增强版本通过引入物理约束和统计特性,预期在以下方面有所提升:
- 降水总量预测精度
- 极端降水事件捕捉能力
- 空间分布合理性
- 时间序列连续性

## 🛠️ 技术栈

- PyTorch
- rioxarray
- xarray
- NumPy
- YAML

## 📝 开发说明

本增强版本包含以下核心改进:

1. **物理约束**: 通过物理定律约束模型输出,提高预测的物理合理性
2. **统计特性**: 保持降水场的统计分布特征,改善极值预测
3. **灵活配置**: 支持动态调整各损失函数权重
4. **可视化监控**: 实时监控训练过程中的各项指标
