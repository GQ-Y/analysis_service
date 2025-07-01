"""
配置模块 - 新的统一配置
管理应用的配置参数
"""

# 直接导出新的统一配置，不再兼容旧方式
from .config.unified_config import (
    unified_settings as settings,
    UnifiedSettings,
    PerformanceMode,
    ServiceConfig,
    LoggingConfig,
    RedisConfig,
    TaskConfig,
    ZLMConfig,
    ProtocolConfig,
    StreamingConfig,
    FrameProcessingConfig,
    AnalysisConfig,
    CallbackConfig,
    StorageConfig,
    PerformanceConfig
)

__all__ = [
    'settings',
    'UnifiedSettings',
    'PerformanceMode',
    'ServiceConfig',
    'LoggingConfig', 
    'RedisConfig',
    'TaskConfig',
    'ZLMConfig',
    'ProtocolConfig',
    'StreamingConfig',
    'FrameProcessingConfig',
    'AnalysisConfig',
    'CallbackConfig',
    'StorageConfig',
    'PerformanceConfig'
]
