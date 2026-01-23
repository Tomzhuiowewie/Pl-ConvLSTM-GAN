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


def plot_training_curves(history, save_path="training_curves.png"):
    """
    绘制训练收敛曲线
    
    Args:
        history: dict, 包含 'epoch', 'loss', 'rmse', 'lr' 等训练历史
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = history['epoch']
    
    # 1. 总损失曲线
    axes[0, 0].plot(epochs, history['total_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Training Loss Convergence')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. RMSE曲线
    axes[0, 1].plot(epochs, history['rmse'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('RMSE Convergence')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 各项损失分量
    axes[1, 0].plot(epochs, history['point_loss'], label='Point Loss', linewidth=1.5)
    axes[1, 0].plot(epochs, history['conserve_loss'], label='Conserve Loss', linewidth=1.5)
    axes[1, 0].plot(epochs, history['smooth_loss'], label='Smooth Loss', linewidth=1.5)
    axes[1, 0].plot(epochs, history['temporal_loss'], label='Temporal Loss', linewidth=1.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 学习率变化
    if 'learning_rate' in history:
        axes[1, 1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")
