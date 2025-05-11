"""
分析服务配置
"""
from pydantic_settings import BaseSettings
from typing import Dict, Any, List
from pydantic import BaseModel
import os
import yaml
import logging

logger = logging.getLogger(__name__)

class DebugConfig(BaseModel):
    """调试配置"""
    enabled: bool = False
    log_level: str = "DEBUG"
    log_file: str = "logs/debug.log"
    log_rotation: str = "1 day"
    log_retention: str = "7 days"

class StreamingConfig(BaseModel):
    """流媒体配置"""
    reconnect_attempts: int = 5
    reconnect_delay: int = 3
    read_timeout: int = 15
    connect_timeout: int = 10
    max_consecutive_errors: int = 10
    frame_buffer_size: int = 10
    log_level: str = "INFO"

class AnalysisConfig(BaseModel):
    """分析配置"""
    confidence: float = 0.2
    iou: float = 0.45
    max_det: int = 300
    device: str = "auto"
    analyze_interval: int = 1
    alarm_interval: int = 60
    random_interval_min: int = 0
    random_interval_max: int = 0
    push_interval: int = 1
    save_dir: str = "results"
    save_txt: bool = False
    save_img: bool = True
    return_base64: bool = True

class OutputConfig(BaseModel):
    """输出配置"""
    save_dir: str = "results"
    save_txt: bool = False
    save_img: bool = True
    return_base64: bool = True

class AnalysisServiceConfig(BaseSettings):
    """分析服务配置"""
    
    # 基础信息
    PROJECT_NAME: str = "MeekYolo Analysis Service"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = "development"  # 环境: development, production, testing
    
    # 调试配置
    DEBUG_ENABLED: bool = True
    DEBUG_LOG_LEVEL: str = "DEBUG"
    DEBUG_LOG_FILE: str = "logs/debug.log"
    DEBUG_LOG_ROTATION: str = "1 day"
    DEBUG_LOG_RETENTION: str = "7 days"
    
    # # CORS配置
    # CORS_ORIGINS: List[str] = ["*"]
    # CORS_ALLOW_CREDENTIALS: bool = True
    # CORS_ALLOW_METHODS: List[str] = ["*"]
    # CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # 服务配置
    SERVICES_HOST: str = "0.0.0.0"
    SERVICES_PORT: int = 8002
    
    # 通信配置
    COMMUNICATION_MODE: str = "http"
    
    # MQTT配置
    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = 1883
    MQTT_USERNAME: str = "admin"
    MQTT_PASSWORD: str = "admin"
    MQTT_TOPIC_PREFIX: str = "meek"
    MQTT_QOS: int = 1
    MQTT_KEEPALIVE: int = 60
    MQTT_RECONNECT_INTERVAL: int = 5
    MQTT_NODE_ID: str = ""
    MQTT_SERVICE_TYPE: str = "analysis"
    
    # Redis配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "123456"
    REDIS_DB: int = 0
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_RETRY_ON_TIMEOUT: bool = True
    
    # 模型服务配置
    MODEL_SERVICE_URL: str = "http://localhost:8003"
    MODEL_SERVICE_API_PREFIX: str = "/api/v1"
    
    # 分析配置
    ANALYSIS: AnalysisConfig = AnalysisConfig()
    
    # 存储配置
    STORAGE_BASE_DIR: str = "data"
    STORAGE_MODEL_DIR: str = "models"
    STORAGE_TEMP_DIR: str = "temp"
    STORAGE_MAX_SIZE: int = 10737418240
    
    # 输出配置
    OUTPUT: OutputConfig = OutputConfig()
    
    # 任务队列配置
    TASK_QUEUE_MAX_CONCURRENT: int = 30
    TASK_QUEUE_MAX_RETRIES: int = 3
    TASK_QUEUE_RETRY_DELAY: int = 5
    TASK_QUEUE_RESULT_TTL: int = 7200
    TASK_QUEUE_CLEANUP_INTERVAL: int = 3600
    
    # Redis任务队列键配置
    REDIS_TASK_QUEUE_KEY: str = "analysis:task:queue"
    REDIS_TASK_HASH_KEY: str = "analysis:task:hash"
    REDIS_TASK_RESULT_KEY: str = "analysis:task:result"
    REDIS_TASK_STATUS_KEY: str = "analysis:task:status"
    REDIS_TASK_CALLBACK_KEY: str = "analysis:task:callback"
    REDIS_TASK_EXPIRE: int = 86400
    REDIS_RESULT_EXPIRE: int = 3600
    REDIS_CALLBACK_EXPIRE: int = 1800
    
    # 服务发现配置
    DISCOVERY_INTERVAL: int = 30
    DISCOVERY_TIMEOUT: int = 5
    DISCOVERY_RETRY: int = 3
    
    # 流媒体配置
    STREAMING: StreamingConfig = StreamingConfig()
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

    def get_debug_config(self) -> DebugConfig:
        """获取调试配置"""
        return DebugConfig(
            enabled=self.DEBUG_ENABLED,
            log_level=self.DEBUG_LOG_LEVEL,
            log_file=self.DEBUG_LOG_FILE,
            log_rotation=self.DEBUG_LOG_ROTATION,
            log_retention=self.DEBUG_LOG_RETENTION
        )

# 加载配置
try:
    settings = AnalysisServiceConfig()
    logger.debug(f"配置加载成功: {settings}")
except Exception as e:
    logger.error(f"加载配置失败: {str(e)}")
    raise
