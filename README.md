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
├── main.py              # 主训练脚本和模型定义
├── README.md            # 项目文档
├── .gitignore           # Git忽略文件
├── data/                # 数据目录
│   ├── climate/         # 气象站点数据
│   │   ├── meta.xlsx    # 站点元数据
│   │   └── rain.xlsx    # 站点降水数据
│   ├── FenheBasin/      # 汾河流域边界数据
│   │   └── fenhe.shp    # 流域边界Shapefile
│   ├── static_features_1km/  # 静态特征数据（1km分辨率）
│   │   ├── dem_1km.npy  # 数字高程模型
│   │   ├── lucc_1km.npy # 土地利用数据
│   │   ├── lats_1km.npy # 纬度坐标
│   │   └── lons_1km.npy # 经度坐标
│   └── cmorph-2021/     # CMORPH卫星降水数据（2021年）
│       └── daily/       # 日降水数据
├── data_process/        # 数据处理脚本目录
│   ├── cmorph_process.py    # CMORPH降水数据处理
│   ├── dem_lucc_process.py  # DEM/LUCC静态特征处理
│   └── real_rain-process.py # 站点降水数据处理
└── output/              # 训练结果存储目录
    ├── station_comparison_epoch_*.png  # 站点观测与预测对比图
    └── rmse_per_time_epoch_*.png       # 时间步RMSE曲线图
```

## 安装依赖

```bash
# 安装所需的 Python 库
pip install torch torchvision numpy pandas matplotlib geopandas
```

## 数据预处理

### 1. 卫星降水数据处理 (`cmorph_process.py`)

```python
# 处理 CMORPH 降水数据，支持水文和气象时间体系
python data_process/cmorph_process.py
```

### 2. 静态特征处理 (`dem_lucc_process.py`)

```python
# 处理 DEM 和 LUCC 数据，重采样到 1km 分辨率
python data_process/dem_lucc_process.py
```

### 3. 站点数据处理

站点数据处理已集成到 `main.py` 中，无需额外运行脚本。

## 训练模型

```bash
# 运行主训练脚本
python main.py
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
- **数据处理**: NumPy, Pandas, GeoPandas
- **可视化**: Matplotlib
- **GIS 操作**: Shapely, Fiona

## 许可证

本项目采用 MIT 许可证，详情请查看 LICENSE 文件。

## 作者

- [Tomzhuiowewie](https://github.com/Tomzhuiowewie)
