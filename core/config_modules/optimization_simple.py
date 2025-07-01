"""
简化版优化配置模块
统一管理所有硬编码值和性能参数
"""
import os
from enum import Enum
from typing import Dict, Any


class PerformanceMode(Enum):
    """性能模式枚举"""
    HIGH_QUALITY = "high_quality"
    BALANCED = "balanced"
    HIGH_PERFORMANCE = "high_performance"


class OptimizationConfig:
    """统一优化配置"""
    
    def __init__(self, performance_mode: str = None):
        # 确定性能模式
        mode_str = performance_mode or os.getenv("PERFORMANCE_MODE", "balanced")
        try:
            self.performance_mode = PerformanceMode(mode_str)
        except ValueError:
            self.performance_mode = PerformanceMode.BALANCED
        
        # 初始化ZLM配置
        self.zlm = self._init_zlm_config()
        
        # 初始化帧处理配置
        self.frame_processing = self._init_frame_config()
        
        # 初始化任务管理配置
        self.task_management = self._init_task_config()
        
        # 初始化数据库配置
        self.database = self._init_database_config()
        
        # 初始化Redis配置
        self.redis = self._init_redis_config()
        
        # 初始化网络配置
        self.network = self._init_network_config()
    
    def _init_zlm_config(self):
        """初始化ZLM配置"""
        class ZLMConfig:
            def __init__(self):
                self.max_retries = int(os.getenv("ZLM_MAX_RETRIES", "5"))
                self.retry_interval = int(os.getenv("ZLM_RETRY_INTERVAL", "2"))
                self.api_timeout = int(os.getenv("ZLM_API_TIMEOUT", "10"))
                self.startup_wait = int(os.getenv("ZLM_STARTUP_WAIT", "3"))
                self.health_check_interval = int(os.getenv("ZLM_HEALTH_CHECK_INTERVAL", "30"))
                self.health_check_timeout = int(os.getenv("ZLM_HEALTH_CHECK_TIMEOUT", "5"))
                self.stream_timeout = int(os.getenv("ZLM_STREAM_TIMEOUT", "15"))
                self.stream_buffer_size = int(os.getenv("ZLM_STREAM_BUFFER_SIZE", "30"))
        
        return ZLMConfig()
    
    def _init_frame_config(self):
        """初始化帧处理配置"""
        class FrameConfig:
            def __init__(self, mode):
                # 根据性能模式设置默认值
                if mode == PerformanceMode.HIGH_QUALITY:
                    default_buffer_size = "100"
                    default_fps = "30"
                    default_threshold = "1.0"
                elif mode == PerformanceMode.HIGH_PERFORMANCE:
                    default_buffer_size = "30"
                    default_fps = "20"
                    default_threshold = "5.0"
                else:  # BALANCED
                    default_buffer_size = "50"
                    default_fps = "25"
                    default_threshold = "3.0"
                
                self.buffer_size = int(os.getenv("FRAME_BUFFER_SIZE", default_buffer_size))
                self.target_fps = int(os.getenv("FRAME_TARGET_FPS", default_fps))
                self.stall_threshold = float(os.getenv("FRAME_STALL_THRESHOLD", default_threshold))
                self.stall_recovery_threshold = float(os.getenv("FRAME_STALL_RECOVERY_THRESHOLD", "1.0"))
                self.max_frame_skip = int(os.getenv("FRAME_MAX_SKIP", "5"))
                self.adaptive_quality = os.getenv("FRAME_ADAPTIVE_QUALITY", "true").lower() == "true"
        
        return FrameConfig(self.performance_mode)
    
    def _init_task_config(self):
        """初始化任务管理配置"""
        class TaskConfig:
            def __init__(self, mode):
                # 根据性能模式设置默认值
                if mode == PerformanceMode.HIGH_QUALITY:
                    default_concurrent = "20"
                elif mode == PerformanceMode.HIGH_PERFORMANCE:
                    default_concurrent = "100"
                else:  # BALANCED
                    default_concurrent = "50"
                
                self.max_concurrent_tasks = int(os.getenv("TASK_MAX_CONCURRENT", default_concurrent))
                self.max_queue_size = int(os.getenv("TASK_MAX_QUEUE_SIZE", "1000"))
                self.task_timeout = int(os.getenv("TASK_TIMEOUT", "7200"))  # 2小时
                self.cleanup_interval = int(os.getenv("TASK_CLEANUP_INTERVAL", "180"))  # 3分钟
                self.max_retries = int(os.getenv("TASK_MAX_RETRIES", "3"))
                self.retry_delay = int(os.getenv("TASK_RETRY_DELAY", "5"))
                self.retry_backoff_factor = float(os.getenv("TASK_RETRY_BACKOFF_FACTOR", "2.0"))
        
        return TaskConfig(self.performance_mode)
    
    def _init_database_config(self):
        """初始化数据库配置"""
        class DatabaseConfig:
            def __init__(self):
                self.pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
                self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "50"))
                self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "1800"))  # 30分钟
                self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
                self.query_timeout = int(os.getenv("DB_QUERY_TIMEOUT", "30"))
                self.batch_size = int(os.getenv("DB_BATCH_SIZE", "100"))
        
        return DatabaseConfig()
    
    def _init_redis_config(self):
        """初始化Redis配置"""
        class RedisConfig:
            def __init__(self):
                self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
                self.socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "10"))
                self.socket_connect_timeout = int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
                self.retry_on_timeout = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
                self.max_retries = int(os.getenv("REDIS_MAX_RETRIES", "3"))
                self.default_ttl = int(os.getenv("REDIS_DEFAULT_TTL", "3600"))
                self.task_result_ttl = int(os.getenv("REDIS_TASK_RESULT_TTL", "7200"))
        
        return RedisConfig()
    
    def _init_network_config(self):
        """初始化网络配置"""
        class NetworkConfig:
            def __init__(self):
                self.http_timeout = int(os.getenv("HTTP_TIMEOUT", "30"))
                self.http_max_retries = int(os.getenv("HTTP_MAX_RETRIES", "3"))
                self.http_backoff_factor = float(os.getenv("HTTP_BACKOFF_FACTOR", "0.3"))
                self.http_pool_connections = int(os.getenv("HTTP_POOL_CONNECTIONS", "10"))
                self.http_pool_maxsize = int(os.getenv("HTTP_POOL_MAXSIZE", "20"))
                self.discovery_timeout = int(os.getenv("DISCOVERY_TIMEOUT", "10"))
                self.discovery_max_devices = int(os.getenv("DISCOVERY_MAX_DEVICES", "100"))
        
        return NetworkConfig()
    
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
