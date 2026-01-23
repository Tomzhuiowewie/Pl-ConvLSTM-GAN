import argparse
from src.training.trainer import Trainer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Pl-ConvLSTM-GAN Training') # 创建对象
    parser.add_argument('--config', type=str, default='default', help='Configuration name') # 添加参数
    args = parser.parse_args() # 解析参数
    
    # 创建训练器实例
    trainer = Trainer(config_name=args.config)
    
    # 验证配置
    trainer.config.validate()
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()