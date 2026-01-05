# 汾河流域降水超分辨率重建

## 项目概述

本项目实现了一种基于卷积长短期记忆网络（ConvLSTM）和生成对抗网络（GAN）的深度学习模型，用于将低分辨率卫星降水数据（CMORPH）超分辨率重建为高分辨率降水分布。通过融合高程数据（DEM）和土地利用数据（LUCC）作为静态特征，显著提升了重建精度。该模型特别针对汾河流域设计，能够有效捕捉降水的时空变化特征。

- **时空建模**: 使用ConvLSTM网络捕捉降水数据的时空相关性
- **多源特征融合**: 集成高程、土地利用等地理特征增强重建效果
- **坐标感知**: 采用CoordConv机制提高空间感知能力
- **注意力机制**: 引入DEM和LUCC注意力模块，重点关注关键地理特征
- **站点监督**: 利用气象站点观测数据进行点对点监督学习

## 核心架构

### 数据

- **降水数据**: CMORPH卫星降水产品 (0.25° × 0.25°)
- **高程数据**: SRTM DEM数据 (30m分辨率)
- **土地利用**: ESA CCI LUCC数据 (30m分辨率)
- **站点观测**: 汾河流域气象站点降水观测数据

### 生成器 (Generator)

生成器负责将低分辨率降水数据转换为高分辨率降水分布：

- **ConvLSTM单元**: 捕捉时空依赖关系
- **CoordConv**: 增强空间坐标感知
- **注意力模块**: DEM和LUCC注意力机制，自适应调整特征权重
- **多尺度特征**: 支持不同分辨率输入输出

### 损失函数

组合损失函数确保重建结果的准确性和真实性：

- **点监督损失**: 基于气象站点观测数据的逐点损失
- **守恒损失**: 确保空间降水总量守恒
- **GAN对抗损失**: 生成分布逼近真实分布

## 项目结构

```
├── main.py              # 主训练脚本和模型定义
├── test.py              # 测试版本（简化版）
├── data_process/        # 数据处理脚本目录
│   ├── cmorph_process.py    # CMORPH降水数据处理
│   ├── dem_lucc_process.py  # DEM/LUCC静态特征处理
│   └── real_rain-process.py # 站点降水数据处理（暂时无用）
├── result/              # 训练结果存储目录
├── README.md            # 项目文档
└── .gitignore           # Git忽略文件
```

## 数据处理流程

#### 1. 卫星降水数据处理 (`cmorph_process.py`)

```python
# 时空聚合：小时数据 → 日数据
# 支持水文体系 (08:00-08:00) 和气象体系 (20:00-20:00)
process_cmorph_to_fenhe(nc_dir, shp_path, out_base_path, year="2021")
python data_process/cmorph_process.py
```

#### 2. 静态特征处理 (`dem_lucc_process.py`)

```python
# DEM和LUCC数据重采样到1km分辨率
# 坐标对齐和数据清理
process_static_features_to_1km(dem_path, lucc_path, out_dir)
python data_process/dem_lucc_process.py
```

#### 3. 站点数据处理 (`real_rain-process.py`)

```python
# 气象站点数据预处理
# 数据清洗和插值
process_station_data(meta_path, rain_path)
- main.py脚本统一处理，无需额外
```

#### 4. 数据集构建 (`main.py`)

```python
# 序列化数据加载
# 站点观测数据预处理和插值
dataset = FenheDataset(...)
```

## 训练

### 基本训练

```python
python main.py
```

### 训练参数

- **批次大小**: 2
- **序列长度**: 5 (天)
- **学习率**: 0.0002
- **训练轮数**: 50
- **隐藏维度**: [16, 32]

### 损失权重

- **点监督损失**: λ_point = 20.0
- **守恒损失**: λ_conserve = 5.0

## 评估

### 评估指标

- **RMSE**: 站点观测与预测的均方根误差
- **MAE**: 平均绝对误差
- **相关系数**: 空间分布相关性

### 可视化

- 站点观测 vs 预测对比图 (自动生成于训练过程)
- 时空降水分布可视化
- 训练损失曲线

# 运行记录

## 2016-01-05
