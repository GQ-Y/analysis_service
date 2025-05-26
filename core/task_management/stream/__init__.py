"""
流管理模块
负责视频流的管理和分发
"""

from .status import StreamStatus, StreamHealthStatus
from .health_monitor import StreamHealthMonitor
from .manager import StreamManager
from .node_monitor import NodeHealthMonitor
from .stream_task_bridge import StreamTaskBridge
from core.media_kit.zlm_stream import ZLMVideoStream

# 创建全局实例
stream_manager = StreamManager()
stream_task_bridge = StreamTaskBridge()

__all__ = [
    "StreamManager",
    "stream_manager",
    "ZLMVideoStream",
    "StreamStatus",
    "StreamHealthStatus",
    "StreamHealthMonitor",
    "NodeHealthMonitor",
    "StreamTaskBridge",
    "stream_task_bridge"
]