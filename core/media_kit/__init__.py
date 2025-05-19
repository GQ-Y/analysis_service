"""
媒体工具包模块
提供流媒体处理、多协议支持和分析任务协调等功能
"""

# 导入基础组件
from .base import (
    IStream,
    IStreamFactory,
    IStreamManager,
    StreamStatus,
    StreamHealthStatus,
    config_manager,
    event_system
)

# 导入工厂组件
from .factory import (
    stream_factory
)

# 导入桥接组件
from .bridge import (
    analyzer_bridge
)

# 导出公共接口
__all__ = [
    # 流接口和状态
    'IStream',
    'IStreamFactory',
    'IStreamManager',
    'StreamStatus',
    'StreamHealthStatus',
    
    # 配置管理
    'config_manager',
    
    # 事件系统
    'event_system',
    
    # 工厂
    'stream_factory',
    
    # 桥接
    'analyzer_bridge'
] 