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

    # HTTP回调配置 (保留现有的，以支持原功能)
    # 注意: 如果任务请求中包含 callback_url，它将优先于下面的全局 CALLBACK_URL
    CALLBACK_URL: str | None = None # 全局默认HTTP回调URL (如果任务未指定)
    HTTP_CALLBACK_TIMEOUT: int = 10 # HTTP回调超时时间 (秒)

    # Socket回调配置 (新增)
    SOCKET_CALLBACK_ENABLED: bool = True # 默认启用Socket回调
    SOCKET_CALLBACK_HOST: str = "localhost" # 默认Socket主机
    SOCKET_CALLBACK_PORT: int = 8089 # 默认Socket端口
    SOCKET_CONNECT_TIMEOUT: int = 5 # Socket连接超时 (秒)
    SOCKET_SEND_TIMEOUT: int = 10 # Socket发送超时 (秒)
    SOCKET_MAX_CONNECT_ATTEMPTS: int = 3 # 最大连接尝试次数
    SOCKET_CONNECT_RETRY_DELAY: int = 5 # 连接重试延迟 (秒)

    class Config:
        env_file = ".env"
        extra = "allow"  # 允许额外字段

# 创建默认配置实例
settings = Settings()