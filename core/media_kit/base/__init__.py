"""
媒体工具包基础模块
提供流接口、配置管理和事件系统等基础功能
"""

from .stream_interface import (
    IStream, 
    IStreamFactory, 
    IStreamManager, 
    StreamStatus, 
    StreamHealthStatus
)

from .config_manager import (
    IConfig, 
    IConfigFactory, 
    BaseConfig, 
    JsonConfig, 
    IniConfig, 
    ConfigManager, 
    config_manager
)

from .event_system import (
    EventSystem, 
    event_system
)

from .base_stream import BaseStream
from .stream_manager import StreamManager

__all__ = [
    # 流接口
    'IStream',
    'IStreamFactory',
    'IStreamManager',
    'StreamStatus',
    'StreamHealthStatus',
    
    # 基础流类
    'BaseStream',
    
    # 流管理器
    'StreamManager',
    
    # 配置管理
    'IConfig',
    'IConfigFactory',
    'BaseConfig',
    'JsonConfig',
    'IniConfig',
    'ConfigManager',
    'config_manager',
    
    # 事件系统
    'EventSystem',
    'event_system'
]
