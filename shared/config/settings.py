"""
共享配置模块
基础配置类，供各个服务继承使用
"""
from pydantic_settings import BaseSettings
from typing import Dict, Any, Optional
import yaml
import os
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class Settings(BaseSettings):
    """基础配置类"""
    
    # 基础配置
    PROJECT_NAME: str = "MeekYolo Service"
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    
    # 环境配置
    DEBUG: bool = False
    ENV: str = "development"
    
    # 日志配置
    LOGGING: Dict[str, Any] = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
    
    class Config:
        env_file = ".env"
        extra = "allow"  # 允许额外字段

    @classmethod
    def load_from_yaml(cls, config_path: str = None) -> "Settings":
        """从YAML文件加载配置"""
        if not config_path:
            config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
            
        # 确保配置目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
                    return cls(**config_data)
            else:
                normal_logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
                return cls()
                
        except Exception as e:
            exception_logger.exception(f"加载配置失败: {str(e)}")
            return cls()

# 创建默认配置实例
settings = Settings.load_from_yaml() 