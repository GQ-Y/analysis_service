"""
零拷贝视频流接口
扩展现有IVideoStream接口，支持帧引用返回而非帧数据拷贝
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
import asyncio
import numpy as np

from .stream_interface import IVideoStream, StreamStatus, StreamHealthStatus
from ..frame.frame_reference import FrameReference
from ..memory.memory_pool import MemoryPool


class IZeroCopyVideoStream(IVideoStream):
    """
    零拷贝视频流接口
    扩展基础视频流接口，支持零拷贝帧引用获取
    """
    
    @abstractmethod
    async def get_frame_reference(self) -> Tuple[bool, Optional[FrameReference]]:
        """
        获取帧引用（零拷贝）
        
        Returns:
            Tuple[bool, Optional[FrameReference]]: (是否成功, 帧引用)
        """
        pass
    
    @abstractmethod
    async def get_frame_references_batch(self, count: int = 1) -> Tuple[bool, List[FrameReference]]:
        """
        批量获取帧引用（零拷贝）
        
        Args:
            count: 获取的帧数量
            
        Returns:
            Tuple[bool, List[FrameReference]]: (是否成功, 帧引用列表)
        """
        pass
    
    @abstractmethod
    def set_memory_pool(self, memory_pool: MemoryPool) -> bool:
        """
        设置内存池
        
        Args:
            memory_pool: 内存池实例
            
        Returns:
            bool: 是否设置成功
        """
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        获取内存使用情况
        
        Returns:
            Dict[str, Any]: 内存使用统计
        """
        pass
    
    @abstractmethod
    def is_memory_pressure_high(self) -> bool:
        """
        检查是否存在内存压力
        
        Returns:
            bool: 是否存在高内存压力
        """
        pass
    
    # 保持向后兼容的传统接口
    async def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        获取帧数据（传统接口，兼容性）
        内部调用零拷贝接口并转换为numpy数组
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (是否成功, 帧数据)
        """
        success, frame_ref = await self.get_frame_reference()
        if not success or frame_ref is None:
            return False, None
        
        # 获取numpy数组视图（零拷贝）
        frame_data = frame_ref.get_data()
        if frame_data is None:
            return False, None
        
        return True, frame_data


class ZeroCopyStreamConfig:
    """零拷贝流配置"""
    
    def __init__(self, 
                 buffer_size: int = 10,
                 enable_batch_processing: bool = True,
                 batch_size: int = 4,
                 memory_pressure_threshold: float = 0.8,
                 enable_memory_monitoring: bool = True,
                 frame_drop_on_pressure: bool = True):
        """
        初始化零拷贝流配置
        
        Args:
            buffer_size: 缓冲区大小
            enable_batch_processing: 是否启用批处理
            batch_size: 批处理大小
            memory_pressure_threshold: 内存压力阈值
            enable_memory_monitoring: 是否启用内存监控
            frame_drop_on_pressure: 内存压力时是否丢帧
        """
        self.buffer_size = buffer_size
        self.enable_batch_processing = enable_batch_processing
        self.batch_size = batch_size
        self.memory_pressure_threshold = memory_pressure_threshold
        self.enable_memory_monitoring = enable_memory_monitoring
        self.frame_drop_on_pressure = frame_drop_on_pressure
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "buffer_size": self.buffer_size,
            "enable_batch_processing": self.enable_batch_processing,
            "batch_size": self.batch_size,
            "memory_pressure_threshold": self.memory_pressure_threshold,
            "enable_memory_monitoring": self.enable_memory_monitoring,
            "frame_drop_on_pressure": self.frame_drop_on_pressure,
        }


class AsyncFrameReferenceQueue:
    """
    异步帧引用队列
    支持零拷贝的帧引用传递
    """
    
    def __init__(self, maxsize: int = 0):
        """
        初始化异步帧引用队列
        
        Args:
            maxsize: 队列最大大小，0表示无限制
        """
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.stats = {
            "total_put": 0,
            "total_get": 0,
            "current_size": 0,
            "max_size_reached": 0,
        }
    
    async def put(self, frame_ref: FrameReference) -> bool:
        """
        放入帧引用
        
        Args:
            frame_ref: 帧引用
            
        Returns:
            bool: 是否成功放入
        """
        try:
            await self.queue.put(frame_ref)
            self.stats["total_put"] += 1
            self.stats["current_size"] = self.queue.qsize()
            self.stats["max_size_reached"] = max(
                self.stats["max_size_reached"], 
                self.stats["current_size"]
            )
            return True
        except Exception:
            return False
    
    async def get(self) -> Optional[FrameReference]:
        """
        获取帧引用
        
        Returns:
            Optional[FrameReference]: 帧引用，队列为空返回None
        """
        try:
            frame_ref = await self.queue.get()
            self.stats["total_get"] += 1
            self.stats["current_size"] = self.queue.qsize()
            return frame_ref
        except Exception:
            return None
    
    async def get_nowait(self) -> Optional[FrameReference]:
        """
        非阻塞获取帧引用
        
        Returns:
            Optional[FrameReference]: 帧引用，队列为空返回None
        """
        try:
            frame_ref = self.queue.get_nowait()
            self.stats["total_get"] += 1
            self.stats["current_size"] = self.queue.qsize()
            return frame_ref
        except asyncio.QueueEmpty:
            return None
    
    def qsize(self) -> int:
        """获取队列当前大小"""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """检查队列是否为空"""
        return self.queue.empty()
    
    def full(self) -> bool:
        """检查队列是否已满"""
        return self.queue.full()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        return self.stats.copy()
