"""
优化配置模块
统一管理所有硬编码值和性能参数
"""
import os
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel


class PerformanceMode(Enum):
    """性能模式枚举"""
    HIGH_QUALITY = "high_quality"      # 高质量模式
    BALANCED = "balanced"              # 平衡模式  
    HIGH_PERFORMANCE = "high_performance"  # 高性能模式


class ZLMOptimizationConfig:
    """ZLMediaKit优化配置"""
    def __init__(self):
        # 连接配置
        self.max_retries: int = int(os.getenv("ZLM_MAX_RETRIES", "5"))
        self.retry_interval: int = int(os.getenv("ZLM_RETRY_INTERVAL", "2"))
        self.api_timeout: int = int(os.getenv("ZLM_API_TIMEOUT", "10"))
        self.startup_wait: int = int(os.getenv("ZLM_STARTUP_WAIT", "3"))

        # 健康检查配置
        self.health_check_interval: int = int(os.getenv("ZLM_HEALTH_CHECK_INTERVAL", "30"))
        self.health_check_timeout: int = int(os.getenv("ZLM_HEALTH_CHECK_TIMEOUT", "5"))

        # 流处理配置
        self.stream_timeout: int = int(os.getenv("ZLM_STREAM_TIMEOUT", "15"))
        self.stream_buffer_size: int = int(os.getenv("ZLM_STREAM_BUFFER_SIZE", "30"))


class FrameProcessingConfig:
    """帧处理优化配置"""
    def __init__(self):
        # 缓冲区配置
        self.buffer_size: int = int(os.getenv("FRAME_BUFFER_SIZE", "50"))
        self.target_fps: int = int(os.getenv("FRAME_TARGET_FPS", "25"))

        # 卡顿检测配置
        self.stall_threshold: float = float(os.getenv("FRAME_STALL_THRESHOLD", "3.0"))
        self.stall_recovery_threshold: float = float(os.getenv("FRAME_STALL_RECOVERY_THRESHOLD", "1.0"))

        # 性能优化配置
        self.max_frame_skip: int = int(os.getenv("FRAME_MAX_SKIP", "5"))
        self.adaptive_quality: bool = os.getenv("FRAME_ADAPTIVE_QUALITY", "true").lower() == "true"


class TaskManagementConfig:
    """任务管理优化配置"""
    def __init__(self):
        # 并发控制
        self.max_concurrent_tasks: int = int(os.getenv("TASK_MAX_CONCURRENT", "50"))
        self.max_queue_size: int = int(os.getenv("TASK_MAX_QUEUE_SIZE", "1000"))

        # 超时配置
        self.task_timeout: int = int(os.getenv("TASK_TIMEOUT", "7200"))  # 2小时
        self.cleanup_interval: int = int(os.getenv("TASK_CLEANUP_INTERVAL", "180"))  # 3分钟

        # 重试配置
        self.max_retries: int = int(os.getenv("TASK_MAX_RETRIES", "3"))
        self.retry_delay: int = int(os.getenv("TASK_RETRY_DELAY", "5"))
        self.retry_backoff_factor: float = float(os.getenv("TASK_RETRY_BACKOFF_FACTOR", "2.0"))


class DatabaseOptimizationConfig:
    """数据库优化配置"""
    def __init__(self):
        # 连接池配置
        self.pool_size: int = int(os.getenv("DB_POOL_SIZE", "20"))
        self.max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "50"))
        self.pool_recycle: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))  # 30分钟
        self.pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        
        # 查询优化
        self.query_timeout: int = int(os.getenv("DB_QUERY_TIMEOUT", "30"))
        self.batch_size: int = int(os.getenv("DB_BATCH_SIZE", "100"))


class RedisOptimizationConfig:
    """Redis优化配置"""
    def __init__(self):
        # 连接配置
        self.max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
        self.socket_timeout: int = int(os.getenv("REDIS_SOCKET_TIMEOUT", "10"))
        self.socket_connect_timeout: int = int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
        
        # 重试配置
        self.retry_on_timeout: bool = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
        self.max_retries: int = int(os.getenv("REDIS_MAX_RETRIES", "3"))
        
        # 缓存配置
        self.default_ttl: int = int(os.getenv("REDIS_DEFAULT_TTL", "3600"))
        self.task_result_ttl: int = int(os.getenv("REDIS_TASK_RESULT_TTL", "7200"))


class NetworkOptimizationConfig:
    """网络优化配置"""
    def __init__(self):
        # HTTP客户端配置
        self.http_timeout: int = int(os.getenv("HTTP_TIMEOUT", "30"))
        self.http_max_retries: int = int(os.getenv("HTTP_MAX_RETRIES", "3"))
        self.http_backoff_factor: float = float(os.getenv("HTTP_BACKOFF_FACTOR", "0.3"))
        
        # 连接池配置
        self.http_pool_connections: int = int(os.getenv("HTTP_POOL_CONNECTIONS", "10"))
        self.http_pool_maxsize: int = int(os.getenv("HTTP_POOL_MAXSIZE", "20"))
        
        # 发现服务配置
        self.discovery_timeout: int = int(os.getenv("DISCOVERY_TIMEOUT", "10"))
        self.discovery_max_devices: int = int(os.getenv("DISCOVERY_MAX_DEVICES", "100"))


class PerformanceModeConfig:
    """性能模式配置"""
    
    def __init__(self, mode: PerformanceMode = PerformanceMode.BALANCED):
        self.mode = mode
        self._load_mode_config()
    
    def _load_mode_config(self):
        """根据性能模式加载配置"""
        configs = {
            PerformanceMode.HIGH_QUALITY: {
                "frame_buffer_size": 100,
                "target_fps": 30,
                "max_concurrent_tasks": 20,
                "stall_threshold": 1.0,
                "adaptive_quality": True,
            },
            PerformanceMode.BALANCED: {
                "frame_buffer_size": 50,
                "target_fps": 25,
                "max_concurrent_tasks": 50,
                "stall_threshold": 3.0,
                "adaptive_quality": True,
            },
            PerformanceMode.HIGH_PERFORMANCE: {
                "frame_buffer_size": 30,
                "target_fps": 20,
                "max_concurrent_tasks": 100,
                "stall_threshold": 5.0,
                "adaptive_quality": False,
            }
        }
        
        config = configs.get(self.mode, configs[PerformanceMode.BALANCED])
        
        # 动态设置环境变量（如果未设置）
        for key, value in config.items():
            env_key = f"PERF_{key.upper()}"
            if env_key not in os.environ:
                os.environ[env_key] = str(value)


class OptimizationConfig:
    """统一优化配置"""
    
    def __init__(self, performance_mode: str = None):
        # 确定性能模式
        mode_str = performance_mode or os.getenv("PERFORMANCE_MODE", "balanced")
        try:
            self.performance_mode = PerformanceMode(mode_str)
        except ValueError:
            self.performance_mode = PerformanceMode.BALANCED
        
        # 加载性能模式配置
        self.performance_config = PerformanceModeConfig(self.performance_mode)
        
        # 初始化各模块配置
        self.zlm = ZLMOptimizationConfig()
        self.frame_processing = FrameProcessingConfig()
        self.task_management = TaskManagementConfig()
        self.database = DatabaseOptimizationConfig()
        self.redis = RedisOptimizationConfig()
        self.network = NetworkOptimizationConfig()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "performance_mode": self.performance_mode.value,
            "zlm_config": {
                "max_retries": self.zlm.max_retries,
                "api_timeout": self.zlm.api_timeout,
                "stream_buffer_size": self.zlm.stream_buffer_size,
            },
            "frame_config": {
                "buffer_size": self.frame_processing.buffer_size,
                "target_fps": self.frame_processing.target_fps,
                "stall_threshold": self.frame_processing.stall_threshold,
            },
            "task_config": {
                "max_concurrent": self.task_management.max_concurrent_tasks,
                "task_timeout": self.task_management.task_timeout,
                "max_retries": self.task_management.max_retries,
            },
            "database_config": {
                "pool_size": self.database.pool_size,
                "max_overflow": self.database.max_overflow,
            },
            "redis_config": {
                "max_connections": self.redis.max_connections,
                "default_ttl": self.redis.default_ttl,
            }
        }


# 全局优化配置实例
optimization_config = OptimizationConfig()
