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
    log_level: str = "INFO"
    log_file: str = "logs/debug.log"
    log_rotation: str = "1 day"
    log_retention: str = "7 days"

class AnalysisServiceConfig(BaseSettings):
    """分析服务配置"""
    
    # 基础信息
    PROJECT_NAME: str = "MeekYolo Analysis Service"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"  # 环境: development, production, testing
    DEBUG: DebugConfig = DebugConfig()  # 调试配置
    
    # CORS配置
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # 通信模式配置
    class CommunicationConfig(BaseModel):
        mode: str = "http"  # http 或 mqtt
    
    # MQTT配置
    class MQTTConfig(BaseModel):
        broker_host: str = "mqtt.yingzhu.net"
        broker_port: int = 1883
        username: str = "yolo"
        password: str = "yolo"
        topic_prefix: str = "meek/"
        qos: int = 1
        keepalive: int = 60
        reconnect_interval: int = 5
        node_id: str = ""  # 节点ID，留空自动使用MAC地址
        service_type: str = "analysis"  # 节点类型
    
    # Redis配置
    class RedisConfig(BaseModel):
        host: str = "localhost"
        port: int = 6379
        password: str = "123456"
        db: int = 0
        max_connections: int = 50
        socket_timeout: int = 5
        retry_on_timeout: bool = True
        
        # 任务队列配置
        task_queue_key: str = "analysis:task:queue"
        task_hash_key: str = "analysis:task:hash"
        task_result_key: str = "analysis:task:result"
        task_status_key: str = "analysis:task:status"
        task_callback_key: str = "analysis:task:callback"
        
        # 键过期时间（秒）
        task_expire: int = 86400  # 24小时
        result_expire: int = 3600  # 1小时
        callback_expire: int = 1800  # 30分钟
    
    # 任务队列配置
    class TaskQueueConfig(BaseModel):
        max_concurrent: int = 30  # 最大并发任务数
        max_retries: int = 3  # 最大重试次数
        retry_delay: int = 5  # 重试延迟（秒）
        cleanup_interval: int = 300  # 清理间隔（秒）
        task_timeout: int = 3600  # 任务超时时间（秒）
        batch_size: int = 10  # 批处理大小
        result_ttl: int = 3600  # 结果保存时间（秒）
    
    # 服务配置
    class ServiceConfig(BaseModel):
        host: str = "0.0.0.0"
        port: int = 8002
    
    # 模型服务配置
    class ModelServiceConfig(BaseModel):
        url: str = "http://localhost:8003"
        api_prefix: str = "/api/v1"
    
    # 分析配置
    class AnalysisConfig(BaseModel):
        confidence: float = 0.1
        iou: float = 0.45
        max_det: int = 300
        device: str = "auto"
        analyze_interval: int = 1
        alarm_interval: int = 60
        random_interval: List[int] = [0, 0]
        push_interval: int = 1
    
    # 存储配置
    class StorageConfig(BaseModel):
        base_dir: str = "data"
        model_dir: str = "models"
        temp_dir: str = "temp"
        max_size: int = 1073741824  # 1GB
        
        model_config = {"protected_namespaces": ()}

    # 输出配置
    class OutputConfig(BaseModel):
        save_dir: str = "results"
        save_txt: bool = False
        save_img: bool = True
        return_base64: bool = True
    
    # 服务发现配置
    class DiscoveryConfig(BaseModel):
        interval: int = 30
        timeout: int = 5
        retry: int = 3
    
    # 服务配置
    class ServicesConfig(BaseModel):
        host: str = "0.0.0.0"
        port: int = 8002
    
    # 配置实例
    SERVICE: ServiceConfig = ServiceConfig()
    MODEL_SERVICE: ModelServiceConfig = ModelServiceConfig()
    ANALYSIS: AnalysisConfig = AnalysisConfig()
    STORAGE: StorageConfig = StorageConfig()
    OUTPUT: OutputConfig = OutputConfig()
    DISCOVERY: DiscoveryConfig = DiscoveryConfig()
    SERVICES: ServicesConfig = ServicesConfig()
    REDIS: RedisConfig = RedisConfig()
    TASK_QUEUE: TaskQueueConfig = TaskQueueConfig()
    COMMUNICATION: CommunicationConfig = CommunicationConfig()
    MQTT: MQTTConfig = MQTTConfig()
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "allow"  # 允许额外的字段
    }
    
    @classmethod
    def load_config(cls) -> "AnalysisServiceConfig":
        """加载配置"""
        try:
            config = {}
            
            # 获取配置文件路径
            if "CONFIG_PATH" in os.environ:
                config_path = os.environ["CONFIG_PATH"]
            else:
                current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                config_path = os.path.join(current_dir, "config", "config.yaml")
            
            logger.debug(f"正在从以下路径加载配置: {config_path}")
            
            if os.path.exists(config_path):
                logger.debug(f"配置文件存在: {config_path}")
                with open(config_path, "r", encoding="utf-8") as f:
                    config_content = f.read()
                    logger.debug(f"配置文件内容:\n{config_content}")
                    config.update(yaml.safe_load(config_content))
            else:
                logger.warning(f"配置文件未找到: {config_path}, 使用默认值")
            
            logger.debug(f"最终配置字典: {config}")
            return cls(**config)
            
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            raise

# 加载配置
try:
    settings = AnalysisServiceConfig.load_config()
    logger.debug(f"配置加载成功: {settings}")
except Exception as e:
    logger.error(f"加载配置失败: {str(e)}")
    raise
