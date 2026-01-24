import yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DataConfig:
    """数据路径配置"""
    rain_lr_path: str = ""
    dem_path: str = ""
    lucc_path: str = ""
    meta_path: str = ""
    rain_excel_path: str = ""
    shp_path: str = ""
    start_year: int = 2012
    end_year: int = 2021


@dataclass
class ModelConfig:
    """模型参数配置"""
    hidden_dims: List[int] = field(default_factory=lambda: [16, 32])
    T: int = 5
    scale_factor: int = 8
    target_grid_size: List[int] = None
    input_grid_size: List[int] = None


@dataclass
class TrainingConfig:
    """训练参数配置"""
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 0.001
    lambda_point: float = 1.0
    lambda_conserve: float = 1.0
    lambda_smooth: float = 0.1
    lambda_temporal: float = 0.05
    grad_clip_norm: float = 0.5
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    # 数据集划分
    use_split: bool = True  # 是否划分数据集
    split_method: str = "year"  # "year" 或 "random"
    train_years: List[int] = field(default_factory=lambda: [2012, 2018])
    val_years: List[int] = field(default_factory=lambda: [2019, 2020])
    test_years: List[int] = field(default_factory=lambda: [2021, 2021])


@dataclass
class OutputConfig:
    """输出配置"""
    output_dir: str = "output"
    log_interval: int = 10
    save_model_interval: int = 10
    plot_dpi: int = 300


@dataclass
class Config:
    """总配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """从YAML文件加载配置"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            output=OutputConfig(**config_dict.get('output', {}))
        )
    
    def to_yaml(self, yaml_path: str):
        """保存配置到YAML文件"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'output': self.output.__dict__
        }
        
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def validate(self):
        """验证配置的有效性"""
        # 验证路径是否存在
        for path_name, path_value in self.data.__dict__.items():
            if path_name.endswith('_path') and path_value and not os.path.exists(path_value):
                print(f"Warning: {path_name} does not exist: {path_value}")
        
        # 验证参数范围
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.model.T <= 0:
            raise ValueError("Time window T must be positive")
        if self.training.epochs <= 0:
            raise ValueError("Epochs must be positive")


def load_config(config_name: str = "default") -> Config:
    """加载配置文件"""
    config_dir = os.path.join(os.path.dirname(__file__), "../configs")
    config_path = os.path.join(config_dir, f"{config_name}.yaml")
    
    # 如果指定文件不存在，尝试环境变量
    if not os.path.exists(config_path):
        env_config = os.getenv("CONFIG_NAME", "default")
        config_path = os.path.join(config_dir, f"{env_config}.yaml")
    
    return Config.from_yaml(config_path)
