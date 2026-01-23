# Pl-ConvLSTM-GAN: 基于 ConvLSTM 和 GAN 的降水超分辨率重建

## 项目概述

Pl-ConvLSTM-GAN 是一个用于降水超分辨率重建的深度学习模型，结合了卷积长短期记忆网络（ConvLSTM）和生成对抗网络（GAN）的优势。该模型专门针对汾河流域设计，通过融合高程数据（DEM）和土地利用数据（LUCC）作为静态特征，显著提升了低分辨率卫星降水数据的重建精度。

### 核心特性
- **时空建模**: 使用 ConvLSTM 网络捕捉降水数据的时空相关性
- **多源特征融合**: 集成高程、土地利用等地理特征增强重建效果
- **坐标感知**: 采用 CoordConv 机制提高空间感知能力
- **注意力机制**: 引入 DEM 和 LUCC 注意力模块，重点关注关键地理特征
- **站点监督**: 利用气象站点观测数据进行点对点监督学习

## 项目结构

```
├── main.py              # CLI 入口，调用模块化训练流程
├── requirements.txt     # 项目依赖包列表
├── src/
│   ├── config.py        # 配置管理模块
│   ├── preprocessing/   # 原始数据处理脚本（CMORPH、DEM、站点等）
│   │   ├── cmorph.py    # CMORPH卫星降水数据处理
│   │   ├── dem_lucc.py  # DEM和LUCC静态特征处理
│   │   └── station.py   # 气象站点数据处理
│   ├── models/          # 模型结构（ConvLSTM Generator、注意力模块等）
│   │   ├── generator.py # 生成器主模型
│   │   ├── convlstm.py  # ConvLSTM单元
│   │   ├── attention.py # 注意力机制
│   │   └── coordconv.py # 坐标感知卷积
│   ├── losses/          # 损失函数定义
│   │   └── combined_loss.py # 组合损失函数
│   ├── training/        # 训练流程
│   │   └── trainer.py   # 训练器主类
│   ├── datasets/        # 数据集加载器
│   │   └── fenhe_dataset.py # 汾河流域数据集
│   └── utils/           # 可视化与通用工具
│       ├── visualization.py # 可视化工具
│       └── compare_station_alignment.py # 站点对齐分析
├── configs/             # 配置文件目录
│   └── default.yaml     # 默认配置文件
├── data/                # 数据目录
│   ├── raw/             # 原始数据
│   │   ├── climate/     # 气象站点数据
│   │   │   ├── meta.xlsx # 站点元数据
│   │   │   └── rain.xlsx # 站点降水数据
│   │   ├── FenheBasin/  # 汾河流域边界数据
│   │   │   └── fenhe.shp # 流域边界Shapefile
│   │   └── cmorph/      # CMORPH原始数据
│   ├── interim/         # 中间处理数据
│   │   ├── daily/       # 日尺度数据
│   │   └── test/        # 测试数据
│   └── processed/       # 处理后的数据
│       ├── daily/       # 日尺度处理后数据
│       └── static_features_1km/ # 静态特征数据（1km分辨率）
│           ├── dem_1km.npy # 数字高程模型
│           ├── lucc_1km.npy # 土地利用数据
│           ├── lats_1km.npy # 纬度坐标
│           └── lons_1km.npy # 经度坐标
└── output/              # 训练及分析结果（已在 .gitignore 中排除）
```

## 安装依赖

```bash
# 从 requirements.txt 安装所需的 Python 库
pip install -r requirements.txt
```

或手动安装核心依赖：

```bash
pip install torch>=1.9.0 torchvision>=0.10.0 numpy>=1.19.0 \
            pandas>=1.3.0 matplotlib>=3.3.0 geopandas>=0.9.0 \
            shapely>=1.7.0 fiona>=1.8.0 rasterio>=1.2.0 \
            xarray>=0.18.0 netCDF4>=1.5.0
```

## 数据预处理

### 1. 卫星降水数据处理 (`cmorph.py`)

```python
# 处理 CMORPH 降水数据，支持水文和气象时间体系
python -m src.preprocessing.cmorph
```

### 2. 静态特征处理 (`dem_lucc.py`)

```python
# 处理 DEM 和 LUCC 数据，重采样到 1km 分辨率
python -m src.preprocessing.dem_lucc
```

### 3. 站点数据处理

站点数据处理已集成到训练流程中，通过 `src.datasets.fenhe_dataset.py` 自动加载。

## 训练模型

```bash
# 运行主训练脚本（使用默认配置）
python main.py

# 或指定特定配置
python main.py --config dev
```

### 训练参数

- **批次大小**: 2
- **序列长度**: 5（天）
- **学习率**: 0.0001
- **训练轮数**: 50
- **隐藏维度**: [16, 32]
- **损失权重**: λ_point = 0.1, λ_conserve = 1.0

## 模型架构

### 生成器 (Generator)

生成器负责将低分辨率降水数据转换为高分辨率降水分布：

- **ConvLSTM 单元**: 捕捉时空依赖关系
- **CoordConv**: 增强空间坐标感知
- **注意力模块**: DEM 和 LUCC 注意力机制，自适应调整特征权重
- **多尺度特征**: 支持不同分辨率输入输出

### 损失函数

组合损失函数确保重建结果的准确性和真实性：

- **点监督损失**: 基于气象站点观测数据的逐点损失
- **守恒损失**: 确保空间降水总量守恒
- **GAN 对抗损失**: 生成分布逼近真实分布

## 评估指标

- **RMSE**: 站点观测与预测的均方根误差
- **MAE**: 平均绝对误差
- **相关系数**: 空间分布相关性

## 可视化结果

训练过程中会自动生成以下可视化结果：

1. **站点观测 vs 预测对比图**: `station_comparison_epoch_*.png`
2. **时间步 RMSE 曲线图**: `rmse_per_time_epoch_*.png`

## 结果示例

- 训练过程中各时间步的 RMSE 变化
- 站点观测值与模型预测值的散点图
- 降水时空分布可视化

## 技术栈

- **深度学习框架**: PyTorch
- **数据处理**: NumPy, Pandas, GeoPandas, XArray
- **可视化**: Matplotlib
- **GIS 操作**: Shapely, Fiona, Rasterio
- **科学计算**: NetCDF4

## 配置管理

项目使用 YAML 配置文件管理超参数和数据路径：

- `configs/default.yaml`: 默认训练配置
- 通过 `src.config.py` 进行配置加载和验证

## 数据管理

- **原始数据**: `data/raw/` - 存放原始卫星和站点数据
- **中间数据**: `data/interim/` - 存放预处理过程中的中间结果
- **处理后数据**: `data/processed/` - 存放最终用于训练的数据

## 许可证

本项目采用 MIT 许可证，详情请查看 LICENSE 文件。

## 作者

- [Tomzhuiowewie](https://github.com/Tomzhuiowewie)
