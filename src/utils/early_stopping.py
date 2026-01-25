"""
早停机制 (Early Stopping)
用于防止过拟合,在验证集性能不再提升时提前停止训练
"""
import numpy as np
import torch


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=20, min_delta=0.0, mode='min', verbose=True):
        """
        Args:
            patience: 容忍多少个epoch验证集性能不提升
            min_delta: 最小改进阈值,小于此值不算改进
            mode: 'min'表示指标越小越好(如loss/RMSE), 'max'表示越大越好(如accuracy)
            verbose: 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0  # 计数器:多少个epoch没有改进
        self.best_score = None  # 最佳分数
        self.early_stop = False  # 是否应该早停
        self.best_epoch = 0  # 最佳epoch
        
        # 根据mode设置比较函数
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.is_better = lambda new, best: new > best + min_delta
    
    def __call__(self, score, epoch):
        """
        检查是否应该早停
        
        Args:
            score: 当前epoch的验证集分数(如RMSE或loss)
            epoch: 当前epoch编号
        
        Returns:
            bool: 是否是最佳模型(用于决定是否保存)
        """
        is_best = False
        
        if self.best_score is None:
            # 第一次调用,初始化
            self.best_score = score
            self.best_epoch = epoch
            is_best = True
            if self.verbose:
                print(f"早停: 初始化最佳分数 = {score:.6f}")
        
        elif self.is_better(score, self.best_score):
            # 性能提升
            if self.verbose:
                print(f"早停: 性能提升 {self.best_score:.6f} -> {score:.6f} (改进: {abs(score - self.best_score):.6f})")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            is_best = True
        
        else:
            # 性能未提升
            self.counter += 1
            if self.verbose:
                print(f"早停: 性能未提升 (当前: {score:.6f}, 最佳: {self.best_score:.6f}), "
                      f"计数器: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"早停触发! 验证集性能已连续 {self.patience} 个epoch未提升")
                    print(f"最佳epoch: {self.best_epoch}, 最佳分数: {self.best_score:.6f}")
                    print(f"{'='*60}\n")
        
        return is_best
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
