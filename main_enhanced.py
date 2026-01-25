"""
增强版主程序 - 使用方案1+3
author: zhuhao
version: 2.1
date: 2025-01-25
"""
import argparse
from src.training.enhanced_trainer import EnhancedTrainer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Pl-ConvLSTM Enhanced Training (方案1+3)')
    parser.add_argument('--config', type=str, default='enhanced', help='Configuration name')
    args = parser.parse_args()
    
    # 创建增强版训练器实例
    trainer = EnhancedTrainer(config_name=args.config)
    
    # 验证配置
    trainer.config.validate()
    
    # 开始训练
    print("\n" + "="*60)
    print("使用增强版训练器 (方案1: 物理约束 + 方案3: 统计匹配)")
    print("="*60 + "\n")
    
    trainer.train()


if __name__ == "__main__":
    main()
