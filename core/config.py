"""
配置模块
管理应用的配置参数
"""
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, Dict, List, Optional, Union
import os
import logging
from dotenv import load_dotenv

# 加载.env文件（如果存在）
load_dotenv()

# 简单可序列化的数据模型
class BaseSettingsModel(BaseModel):
    """基础设置模型类"""
    pass

# 输出设置
class OutputSettings(BaseSettingsModel):
    """输出设置"""
    save_dir: str = "results"
    save_txt: bool = False
    save_conf: bool = False
    save_crop: bool = False
    save_masks: bool = False
    save_annotated: bool = True
    save_frames: bool = False
    frame_interval: int = 1  # 保存帧的间隔

# 日志设置
class LoggingSettings(BaseSettingsModel):
    """日志设置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file: str = "logs/app.log"
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console: bool = True

# 缓存设置
class CacheSettings(BaseSettingsModel):
    """缓存设置"""
    enabled: bool = True
    type: str = "redis" # redis, memory
    ttl: int = 3600 # 1小时

# 流媒体设置
class StreamingSettings(BaseSettingsModel):
    """流媒体设置"""
    reconnect_attempts: int = 3
    reconnect_delay: int = 5
    read_timeout: int = 30
    connect_timeout: int = 10
    max_consecutive_errors: int = 5
    frame_buffer_size: int = 30
    log_level: str = "INFO"

# 存储配置
class StorageSettings(BaseSettingsModel):
    """存储设置"""
    base_dir: str = "data"
    model_dir: str = "models"
    temp_dir: str = "temp"
    max_size: int = 10 * 1024 * 1024 * 1024  # 10GB

# 应用配置
class Settings(BaseSettings):
    """应用配置类"""
    # 模型配置
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra='ignore',  # 忽略额外的配置项
        case_sensitive=False
    )
    
    # 基础配置
    PROJECT_NAME: str = "Analysis Service"
    DESCRIPTION: str = "Meek YOLO Analysis Service"
    VERSION: str = "0.1.0"
    DEBUG_ENABLED: bool = False
    API_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"
    
    # 服务配置
    SERVICES_HOST: str = "0.0.0.0"
    SERVICES_PORT: int = 8002
    
    # Redis配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_PREFIX: str = "analysis:"
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_RETRY_ON_TIMEOUT: bool = True
    
    # 任务队列配置
    TASK_QUEUE_MAX_SIZE: int = 1000
    TASK_QUEUE_MAX_CONCURRENT: int = 10
    TASK_QUEUE_RESULT_TTL: int = 3600  # 结果保留时间（秒）
    TASK_QUEUE_CLEANUP_INTERVAL: int = 300  # 清理间隔（秒）
    TASK_QUEUE_MAX_RETRIES: int = 3
    TASK_QUEUE_RETRY_DELAY: int = 5
    
    # 缓存配置
    CACHE: CacheSettings = CacheSettings()
    
    # 日志配置
    LOGGING: LoggingSettings = LoggingSettings()
    
    # 输出配置
    OUTPUT: OutputSettings = OutputSettings()
    
    # 流媒体配置
    STREAMING: StreamingSettings = StreamingSettings()
    
    # 存储配置
    STORAGE: StorageSettings = StorageSettings()
    
    # 默认目标检测配置
    DEFAULT_DETECTION_MODEL: str = "yolov8n.pt"
    DEFAULT_SEGMENTATION_MODEL: str = "yolov8n-seg.pt"
    DEFAULT_CLASSIFICATION_MODEL: str = "yolov8n-cls.pt"
    
    # 设置日志级别
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # 通信模式，只支持http模式
    COMMUNICATION_MODE: str = "http"

# 创建设置实例
settings = Settings()

# 设置日志级别
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(level=log_level)
