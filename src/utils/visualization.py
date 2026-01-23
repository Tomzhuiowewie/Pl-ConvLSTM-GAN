import numpy as np
import matplotlib.pyplot as plt
import os


def plot_stations_vs_pred(s_coords, true_vals, pred_vals, save_path="station_comparison.png"):
    """
    绘制站点观测值和预测值的对比图
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制散点图
    plt.scatter(true_vals, pred_vals, alpha=0.7)
    
    # 绘制对角线表示理想预测
    max_val = max(np.max(true_vals), np.max(pred_vals))
    min_val = min(np.min(true_vals), np.min(pred_vals))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
    
    # 设置图表属性
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Station Observed vs Predicted Precipitation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_rmse_per_time_step(rmse_time_history, epoch, output_dir):
    """
    绘制每个时间步的RMSE变化图
    """
    T = len(rmse_time_history)
    plt.figure(figsize=(6, 4))
    for t in range(T):
        if len(rmse_time_history[t]) > 0:
            plt.plot(range(len(rmse_time_history[t])), rmse_time_history[t], label=f'Time step {t}')
    plt.xlabel("Batch index")
    plt.ylabel("RMSE")
    plt.title(f"RMSE per Time Step - Epoch {epoch}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"rmse_per_time_epoch_{epoch}.png"))
    plt.close()
