# 汾河流域降水超分辨率重建

## 项目概述

本项目实现了一种基于卷积长短期记忆网络（ConvLSTM）和生成对抗网络（GAN）的深度学习模型，用于将低分辨率卫星降水数据（CMORPH）超分辨率重建为高分辨率降水分布。通过融合高程数据（DEM）和土地利用数据（LUCC）作为静态特征，显著提升了重建精度。该模型特别针对汾河流域设计，能够有效捕捉降水的时空变化特征。

## 项目特色

- **时空建模**: 使用ConvLSTM网络捕捉降水数据的时空相关性
- **多源特征融合**: 集成高程、土地利用等地理特征增强重建效果
- **坐标感知**: 采用CoordConv机制提高空间感知能力
- **注意力机制**: 引入DEM和LUCC注意力模块，重点关注关键地理特征
- **站点监督**: 利用气象站点观测数据进行点对点监督学习

## 核心架构

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

## 数据集

### 输入数据

- **降水数据**: CMORPH卫星降水产品 (0.25° × 0.25°)
- **高程数据**: SRTM DEM数据 (30m分辨率)
- **土地利用**: ESA CCI LUCC数据 (30m分辨率)
- **站点观测**: 汾河流域气象站点降水观测数据

### 数据处理流程

#### 1. 卫星降水数据处理 (`cmorph_process.py`)

```python
# 时空聚合：小时数据 → 日数据
# 支持水文体系 (08:00-08:00) 和气象体系 (20:00-20:00)
process_cmorph_to_fenhe(nc_dir, shp_path, out_base_path, year="2021")
```

#### 2. 静态特征处理 (`dem_lucc_process.py`)

```python
# DEM和LUCC数据重采样到1km分辨率
# 坐标对齐和数据清理
process_static_features_to_1km(dem_path, lucc_path, out_dir)
```

#### 3. 站点数据处理 (`real_rain-process.py`)

```python
# 气象站点数据预处理
# 数据清洗和插值
process_station_data(meta_path, rain_path)
```

#### 4. 数据集构建 (`main.py`)

```python
# 序列化数据加载
# 站点观测数据预处理和插值
dataset = FenheDataset(...)
```

## 项目结构

```
├── main.py              # 主训练脚本和模型定义
├── test.py              # 测试版本（简化版）
├── data_process/        # 数据处理脚本目录
│   ├── cmorph_process.py    # CMORPH降水数据处理
│   ├── dem_lucc_process.py  # DEM/LUCC静态特征处理
│   └── real_rain-process.py # 站点降水数据处理
├── result/              # 训练结果存储目录
├── README.md            # 项目文档
└── .gitignore           # Git忽略文件
```

## 环境要求

### 依赖包

```
torch >= 1.9.0
numpy >= 1.21.0
pandas >= 1.3.0
xarray >= 0.20.0
rioxarray >= 0.11.0
rasterio >= 1.2.0
geopandas >= 0.10.0
regionmask >= 0.9.0
matplotlib >= 3.4.0
```

## 数据准备

### 目录结构

```
data/
├── cmorph-2021/
│   ├── hourly/          # 原始CMORPH小时数据 (.nc格式)
│   └── daily/           # 处理后的日数据 (.npy格式)
├── static_features_1km/ # 1km分辨率静态特征 (.npy格式)
│   ├── dem_1km.npy
│   ├── lucc_1km.npy
│   ├── lons_1km.npy
│   └── lats_1km.npy
├── climate/             # 气象站点数据 (.xlsx格式)
│   ├── meta.xlsx        # 站点元数据 (站号、经度、纬度)
│   └── rain.xlsx        # 降水观测数据
├── dem/                 # 原始DEM数据
├── lucc/                # 原始LUCC数据
└── FenheBasin/          # 汾河流域矢量边界 (.shp格式)
    └── fenhe.shp
```

### 数据处理步骤

1. **下载原始数据**

   - CMORPH降水数据: https://www.ncei.noaa.gov/
   - SRTM DEM数据: https://www.usgs.gov/
   - ESA CCI LUCC数据: https://www.esa-landcover-cci.org/
2. **运行数据预处理**

   ```bash
   # 处理降水数据
   python data_process/cmorph_process.py

   # 处理静态特征
   python data_process/dem_lucc_process.py

   # 处理站点数据
   python data_process/real_rain-process.py
   ```

## 训练

### 基本训练

```python
from main import train_model
train_model()
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

## 使用示例

```python
import torch
from torch.utils.data import DataLoader
from main import Generator, FenheDataset

# 加载模型
model = Generator(hidden_dims=[16, 32])
model.load_state_dict(torch.load('generator_coordconv.pth'))
model.eval()

# 数据加载
dataset = FenheDataset(
    rain_lr_path="data/cmorph-2021/daily/fenhe_hydro_08-08_2021.npy",
    dem_path="data/static_features_1km/dem_1km.npy",
    lucc_path="data/static_features_1km/lucc_1km.npy",
    meta_path="data/climate/meta.xlsx",
    rain_excel_path="data/climate/rain.xlsx",
    shp_path="data/FenheBasin/fenhe.shp",
    T=5
)
dataloader = DataLoader(dataset, batch_size=1)

# 推理
with torch.no_grad():
    for lr_rain, dem, lu, s_coords, s_values in dataloader:
        pred_hr = model(lr_rain, dem, lu)
        # pred_hr: (B, T, 1, H_hr, W_hr) 格式
        print(f"预测结果形状: {pred_hr.shape}")
        break
```

# 运行记录

```

Super-resolution) zhuhao@192 code % python main.py
Epoch 0 | Loss: 547.4100 | Point: 27.3664 | Conserve: 0.0163 | RMSE: 0.0683
Epoch 0 | Loss: 548.0262 | Point: 27.4010 | Conserve: 0.0013 | RMSE: 9.7031
Epoch 0 | Loss: 0.0897 | Point: 0.0003 | Conserve: 0.0167 | RMSE: 0.0113
Epoch 0 | Loss: 936.2772 | Point: 46.8052 | Conserve: 0.0346 | RMSE: 0.0260
Epoch 0 | Loss: 34.1742 | Point: 1.6928 | Conserve: 0.0638 | RMSE: 0.0570
...
Epoch 0 finished. Avg Station RMSE: 3.6887
Epoch 1 | Loss: 334.5936 | Point: 16.2057 | Conserve: 2.0960 | RMSE: 0.9653
Epoch 1 | Loss: 150.5408 | Point: 6.9807 | Conserve: 2.1853 | RMSE: 1.0340
...
Epoch 1 finished. Avg Station RMSE: 4.1205
Epoch 2 | Loss: 932.2823 | Point: 46.1393 | Conserve: 1.8994 | RMSE: 6.0424
...
```
