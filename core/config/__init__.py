"""
配置模块
统一配置管理入口
"""

from .unified_config import (
    # 配置类
    UnifiedSettings,
    ServiceConfig,
    LoggingConfig,
    RedisConfig,
    TaskConfig,
    ZLMConfig,
    ProtocolConfig,
    RTSPConfig,
    WebRTCConfig,
    ONVIFConfig,
    GStreamerConfig,
    StreamingConfig,
    FrameProcessingConfig,
    AnalysisConfig,
    CallbackConfig,
    StorageConfig,
    PerformanceConfig,
    
    # 枚举
    PerformanceMode,
    
    # 全局实例
    unified_settings,
    settings
)

# 为了保持向后兼容性
__all__ = [
    "UnifiedSettings",
    "ServiceConfig", 
    "LoggingConfig",
    "RedisConfig",
    "TaskConfig",
    "ZLMConfig",
    "ProtocolConfig",
    "RTSPConfig",
    "WebRTCConfig", 
    "ONVIFConfig",
    "GStreamerConfig",
    "StreamingConfig",
    "FrameProcessingConfig", 
    "AnalysisConfig",
    "CallbackConfig",
    "StorageConfig",
    "PerformanceConfig",
    "PerformanceMode",
    "unified_settings",
    "settings"
] 