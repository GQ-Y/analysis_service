"""
视频流接口定义
定义统一的视频流接口，用于不同流实现的统一管理
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Set

import numpy as np

from .status import StreamStatus, StreamHealthStatus

class IVideoStream(ABC):
    """视频流接口，定义统一的视频流操作方法"""
    
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
        """启动流
        
        Returns:
            bool: 是否成功启动
        """
        pass
        
    @abstractmethod
    async def stop(self) -> bool:
        """停止流
        
        Returns:
            bool: 是否成功停止
        """
        pass
        
    @abstractmethod
    async def subscribe(self, subscriber_id: str) -> Tuple[bool, Optional[asyncio.Queue]]:
        """订阅流
        
        Args:
            subscriber_id: 订阅者ID
            
        Returns:
            Tuple[bool, Optional[asyncio.Queue]]: (是否成功, 帧队列)
        """
        pass
        
    @abstractmethod
    async def unsubscribe(self, subscriber_id: str) -> bool:
        """取消订阅
        
        Args:
            subscriber_id: 订阅者ID
            
        Returns:
            bool: 是否成功取消订阅
        """
        pass
        
    @abstractmethod
    async def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """获取最新帧
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (是否成功, 帧数据)
        """
        pass
        
    @abstractmethod
    async def get_info(self) -> Dict[str, Any]:
        """获取流信息
        
        Returns:
            Dict[str, Any]: 流信息
        """
        pass
        
    @abstractmethod
    def get_status(self) -> StreamStatus:
        """获取流状态
        
        Returns:
            StreamStatus: 流状态
        """
        pass
        
    @abstractmethod
    def get_health_status(self) -> StreamHealthStatus:
        """获取流健康状态
        
        Returns:
            StreamHealthStatus: 流健康状态
        """
        pass
        
    @abstractmethod
    def set_status(self, status: StreamStatus) -> None:
        """设置流状态
        
        Args:
            status: 新状态
        """
        pass
        
    @abstractmethod
    def set_health_status(self, health_status: StreamHealthStatus) -> None:
        """设置流健康状态
        
        Args:
            health_status: 新健康状态
        """
        pass
        
    @property
    @abstractmethod
    def subscriber_count(self) -> int:
        """获取订阅者数量
        
        Returns:
            int: 订阅者数量
        """
        pass
        
    @property
    @abstractmethod
    def subscribers(self) -> Set[str]:
        """获取订阅者ID集合
        
        Returns:
            Set[str]: 订阅者ID集合
        """
        pass
