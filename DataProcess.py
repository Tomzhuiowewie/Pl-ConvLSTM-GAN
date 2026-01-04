import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FenheSuperResDataset(Dataset):
    def __init__(self, precip_path, dem_path, lucc_path, seq_len=5):
        """
        precip_path: 25km 降水数据 (.npy), 形状 (T, H_lr, W_lr) -> (365, 15, 12)                
        dem_path: 1km DEM 数据 (.npy), 形状 (H_hr, W_hr) -> (458, 306)
        lucc_path: 1km LUCC 数据 (.npy), 形状 (H_hr, W_hr) -> (458, 306)
        """
        # 1. 加载并直接清理 NaN (我们在预处理脚本中已将背景转为0，这里做双重保险)
        self.precip = np.nan_to_num(np.load(precip_path).astype(np.float32), nan=0.0)
        
        # 2. 静态特征标准化优化
        dem_raw = np.load(dem_path).astype(np.float32)
        lucc_raw = np.load(lucc_path).astype(np.float32)

        # 查看所有唯一的分类编号
        unique_classes = np.unique(lucc_raw)
        print(f"分类列表: {unique_classes}")
        print(f"分类总数: {len(unique_classes)}")
        
        # 使用更稳健的 Min-Max 标准化：基于处理后的实际最大值 (2716.0m)
        dem_min = dem_raw.min()
        dem_max = dem_raw.max()
        self.dem = (dem_raw - dem_min) / (dem_max - dem_min + 1e-6)
        
        # LUCC 保持原始类别 ID 即可，通常背景已在预处理中设为 0
        self.lucc = np.nan_to_num(lucc_raw, nan=0.0)

        # 3. 组合静态特征 (C, H, W) -> (2, 458, 306)
        # 提前转为 Tensor 存储在内存中，避免在 __getitem__ 中重复计算
        static_combined = np.stack([self.dem, self.lucc], axis=0)
        self.static_features = torch.from_numpy(static_combined).float()

        self.seq_len = seq_len

    def __len__(self):
        # 确保索引不会越界
        return len(self.precip) - self.seq_len

    def __getitem__(self, idx):
        """
        返回:
        - x_precip: (seq_len, 1, 15, 12) -> 低分辨率动态序列
        - static: (2, 458, 306) -> 高分辨率静态底图
        """
        # 提取降水序列
        x_precip = self.precip[idx : idx + self.seq_len]
        
        # 转换为 Tensor: (T, H, W) -> (T, C=1, H, W)
        # 这里使用 clone() 确保内存连续，避免多进程 DataLoader 报错
        x_precip_tensor = torch.from_numpy(x_precip).unsqueeze(1).clone()
        
        # 如果你有高分辨率的真值降水 (1km)，在这里加载作为 target
        # y_true = self.precip_hr[idx + self.seq_len - 1] # 预测序列最后一天的精细图
        
        return x_precip_tensor, self.static_features



if __name__ == "__main__":
    dataset = FenheSuperResDataset(
        "data/cmorph-2021/daily/fenhe_hydro_08-08_2021.npy",
        "data/static_features_1km/dem_1km.npy",
        "data/static_features_1km/lucc_1km.npy",
        seq_len=5
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch[0].shape, batch[1].shape)