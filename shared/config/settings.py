"""
共享配置模块
基础配置类，供各个服务继承使用
"""
from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    """基础配置类"""

    # 基础配置
    PROJECT_NAME: str = "N-MeekYolo Analysis Service"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # 日志配置
    LOGGING: Dict[str, Any] = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }

    class Config:
        env_file = ".env"
        extra = "allow"  # 允许额外字段

# 创建默认配置实例
settings = Settings()