"""
流接口定义
独立的接口模块，避免循环依赖
"""
from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from typing import Dict, Any, Optional, Tuple, Set, List
import numpy as np
import asyncio



class StreamStatus(IntEnum):
    """流状态枚举 - 与现有系统兼容"""
    INITIALIZING = 0  # 初始化中
    CONNECTING = 2    # 连接中
    ONLINE = 1        # 在线
    OFFLINE = -1      # 离线
    ERROR = -2        # 错误状态
    RUNNING = 3       # 运行中
    RECONNECTING = 4  # 重连中
    PAUSED = 5        # 已暂停
    STOPPED = 6       # 已停止
    UNKNOWN = 7       # 未知状态


class StreamHealthStatus(Enum):
    """流健康状态枚举 - 与现有系统兼容"""
    HEALTHY = "healthy"           # 健康
    GOOD = "good"                 # 良好
    DEGRADED = "degraded"         # 性能下降
    POOR = "poor"                 # 较差
    UNSTABLE = "unstable"         # 不稳定
    UNHEALTHY = "unhealthy"       # 不健康
    ERROR = "error"               # 错误
    OFFLINE = "offline"           # 离线
    UNKNOWN = "unknown"           # 未知状态


class IVideoStream(ABC):
    """视频流接口定义"""

    @property
    @abstractmethod
    def stream_id(self) -> str:
        """获取流ID"""
        pass

    @property
    @abstractmethod
    def url(self) -> str:
        """获取流URL"""
        pass

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """获取流配置"""
        pass

    @abstractmethod
    async def start(self) -> bool:
        """启动流"""
        pass

    @abstractmethod
    async def stop(self) -> bool:
        """停止流"""
        pass

    @abstractmethod
    async def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """获取帧"""
        pass

    @abstractmethod
    async def get_info(self) -> Dict[str, Any]:
        """获取流信息"""
        pass

    @abstractmethod
    def get_status(self) -> StreamStatus:
        """获取流状态"""
        pass

    @abstractmethod
    def get_health_status(self) -> StreamHealthStatus:
        """获取流健康状态"""
        pass

    @abstractmethod
    def set_status(self, status: StreamStatus) -> None:
        """设置流状态"""
        pass

    @abstractmethod
    def set_health_status(self, health_status: StreamHealthStatus) -> None:
        """设置流健康状态"""
        pass

    @property
    @abstractmethod
    def subscriber_count(self) -> int:
        """获取订阅者数量"""
        pass

    @property
    @abstractmethod
    def subscribers(self) -> Set[str]:
        """获取订阅者集合"""
        pass

    @abstractmethod
    async def subscribe(self, subscriber_id: str) -> Tuple[bool, asyncio.Queue]:
        """订阅流"""
        pass

    @abstractmethod
    async def unsubscribe(self, subscriber_id: str) -> bool:
        """取消订阅流"""
        pass 