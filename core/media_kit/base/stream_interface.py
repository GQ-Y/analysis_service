"""
媒体流接口定义模块
定义统一的流接口和状态枚举，供不同协议实现
"""

import asyncio
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Optional, Tuple, Set, Callable, List, Union
import numpy as np

class StreamStatus(Enum):
    """流状态枚举"""
    UNKNOWN = auto()      # 未知状态
    INITIALIZING = auto() # 初始化中
    CONNECTING = auto()   # 连接中
    RUNNING = auto()      # 运行中
    STOPPED = auto()      # 已停止
    ERROR = auto()        # 错误状态
    OFFLINE = auto()      # 离线状态
    BUFFERING = auto()    # 缓冲中
    PAUSED = auto()       # 已暂停

class StreamHealthStatus(Enum):
    """流健康状态枚举"""
    UNKNOWN = auto()      # 未知状态
    GOOD = auto()         # 良好
    POOR = auto()         # 较差
    UNHEALTHY = auto()    # 不健康
    OFFLINE = auto()      # 离线

class IStream(ABC):
    """流接口"""
    
    @property
    @abstractmethod
    def stream_id(self) -> str:
        """获取流ID"""
        pass
    
    @property
    @abstractmethod
    def status(self) -> StreamStatus:
        """获取流状态"""
        pass
    
    @property
    @abstractmethod
    def health_status(self) -> StreamHealthStatus:
        """获取流健康状态"""
        pass
    
    @property
    @abstractmethod
    def url(self) -> str:
        """获取流URL"""
        pass
    
    @property
    @abstractmethod
    def protocol(self) -> str:
        """获取流协议类型"""
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
    async def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """获取最新的帧
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (是否成功, 帧数据)
        """
        pass
    
    @abstractmethod
    async def get_snapshot(self, width: int = 0, height: int = 0) -> Optional[bytes]:
        """获取流的快照
        
        Args:
            width: 快照宽度，0表示使用原始宽度
            height: 快照高度，0表示使用原始高度
            
        Returns:
            Optional[bytes]: 快照数据(JPEG格式)，如果失败则返回None
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
    async def get_info(self) -> Dict[str, Any]:
        """获取流信息
        
        Returns:
            Dict[str, Any]: 流信息
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

class IStreamFactory(ABC):
    """流工厂接口"""
    
    @abstractmethod
    def create_stream(self, stream_config: Dict[str, Any]) -> IStream:
        """创建流实例
        
        Args:
            stream_config: 流配置
            
        Returns:
            IStream: 流实例
        """
        pass

class IStreamManager(ABC):
    """流管理器接口"""
    
    @abstractmethod
    async def create_stream(self, stream_id: str, config: Dict[str, Any]) -> bool:
        """创建流
        
        Args:
            stream_id: 流ID
            config: 流配置
            
        Returns:
            bool: 是否成功创建
        """
        pass
    
    @abstractmethod
    async def stop_stream(self, stream_id: str) -> bool:
        """停止流
        
        Args:
            stream_id: 流ID
            
        Returns:
            bool: 是否成功停止
        """
        pass
    
    @abstractmethod
    async def get_stream(self, stream_id: str) -> Optional[IStream]:
        """获取流实例
        
        Args:
            stream_id: 流ID
            
        Returns:
            Optional[IStream]: 流实例，如果不存在则返回None
        """
        pass
    
    @abstractmethod
    async def get_stream_status(self, stream_id: str) -> Optional[StreamStatus]:
        """获取流状态
        
        Args:
            stream_id: 流ID
            
        Returns:
            Optional[StreamStatus]: 流状态，如果流不存在则返回None
        """
        pass
    
    @abstractmethod
    async def get_all_streams(self) -> List[Dict[str, Any]]:
        """获取所有流信息
        
        Returns:
            List[Dict[str, Any]]: 所有流的信息列表
        """
        pass
    
    @abstractmethod
    def register_event_handler(self, event_name: str, handler: Callable) -> None:
        """注册事件处理器
        
        Args:
            event_name: 事件名称
            handler: 处理函数
        """
        pass
    
    @abstractmethod
    def unregister_event_handler(self, event_name: str, handler: Callable) -> None:
        """取消注册事件处理器
        
        Args:
            event_name: 事件名称
            handler: 处理函数
        """
        pass
