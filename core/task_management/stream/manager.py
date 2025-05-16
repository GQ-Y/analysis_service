# core/stream_manager.py
import asyncio
import cv2
import time
import threading
import gc  # <-- 导入 gc 模块
from typing import Dict, Optional, Tuple, Any, Set, List
from asyncio import Queue, Lock, Task
import traceback
from loguru import logger

# 尝试导入共享工具和配置
# 使用 try-except 块或者调整导入路径
try:
    from shared.utils.logger import setup_logger
    from core.config import settings
except ImportError:
    # 如果直接运行此文件或在特定测试环境中，提供备选方案
    print("无法导入共享模块，使用标准日志记录和默认设置替代。")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # 提供一个临时的 settings 替代品
    class MockStreamingConfig:
        connection_timeout: int = 10
        read_timeout: int = 15
        reconnect_attempts: int = 5
        reconnect_delay: int = 2
        max_consecutive_errors: int = 10
        frame_buffer_size: int = 5
    class MockSettings:
        STREAMING = MockStreamingConfig()
    settings = MockSettings()

# 导入状态定义
from .status import StreamStatus

# 导入VideoStream类
from .stream import VideoStream

logger = setup_logger(__name__)

class ManagedStream:
    """内部类，用于管理单个流的状态"""
    def __init__(self, url: str):
        self.url: str = url
        self.subscribers: Dict[str, Queue] = {} # subscriber_id -> Queue
        self.ref_count: int = 0
        self.read_task: Optional[Task] = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.lock: Lock = Lock() # 用于保护此特定流状态的锁
        self.is_running: bool = False
        self.last_frame_time: float = 0.0
        self.consecutive_errors: int = 0
        self.error_state: bool = False # 标记流是否处于永久错误状态

    async def stop_reader(self):
        """安全地停止读取器任务并释放资源"""
        # 不再持有锁，让调用者处理锁
        # async with self.lock:
        if self.read_task and not self.read_task.done():
            logger.info(f"[{self.url}] 请求取消读取任务...")
            self.read_task.cancel()
            try:
                # 等待任务实际结束，避免资源未释放
                await asyncio.wait_for(self.read_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"[{self.url}] 等待读取任务取消超时")
            except asyncio.CancelledError:
                logger.info(f"[{self.url}] 读取任务已成功取消")
            except Exception as e:
                 logger.error(f"[{self.url}] 等待读取任务取消时发生未知错误: {e}", exc_info=True)

        if self.capture and self.capture.isOpened():
            logger.info(f"[{self.url}] 释放VideoCapture资源")
            try:
                # cap.release() 可能是阻塞的，考虑在executor中运行
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.capture.release)
            except Exception as e:
                logger.error(f"[{self.url}] 释放VideoCapture时出错: {e}")
        self.capture = None
        self.read_task = None
        self.is_running = False
        logger.info(f"[{self.url}] 读取器已停止")


_stream_manager_instance = None
_stream_manager_lock = threading.Lock() # 使用 threading.Lock 以确保跨线程安全

class StreamManager:
    """视频流管理器，负责管理所有视频流的创建、共享和生命周期"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StreamManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化流管理器"""
        if self._initialized:
            return
            
        self._initialized = True
        self._streams: Dict[str, VideoStream] = {}  # stream_id -> VideoStream
        self._stream_lock = asyncio.Lock()
        
        logger.info("流管理器初始化完成")
        
    async def get_or_create_stream(self, stream_id: str, config: Dict[str, Any]) -> VideoStream:
        """获取或创建视频流
        
        Args:
            stream_id: 流ID
            config: 流配置
            
        Returns:
            VideoStream: 视频流对象
        """
        async with self._stream_lock:
            # 检查流是否已存在
            if stream_id in self._streams:
                logger.info(f"返回已存在的视频流: {stream_id}")
                return self._streams[stream_id]
                
            # 创建新的视频流
            logger.info(f"创建新的视频流: {stream_id}")
            stream = VideoStream(stream_id, config)
            
            # 添加到管理器
            self._streams[stream_id] = stream
            
            # 启动流
            await stream.start()
            
            return stream
            
    async def release_stream(self, stream_id: str) -> bool:
        """释放流资源
        
        Args:
            stream_id: 流ID
            
        Returns:
            bool: 是否成功释放
        """
        async with self._stream_lock:
            if stream_id not in self._streams:
                logger.warning(f"尝试释放不存在的流: {stream_id}")
                return False
                
            stream = self._streams[stream_id]
            
            # 如果没有订阅者，停止并移除流
            if stream.subscriber_count == 0:
                logger.info(f"流 {stream_id} 没有订阅者，停止并移除")
                await stream.stop()
                del self._streams[stream_id]
                return True
                
            logger.info(f"流 {stream_id} 仍有 {stream.subscriber_count} 个订阅者，保持运行")
            return False
            
    async def get_stream(self, stream_id: str) -> Optional[VideoStream]:
        """获取视频流
        
        Args:
            stream_id: 流ID
            
        Returns:
            Optional[VideoStream]: 视频流对象，如果不存在则返回None
        """
        async with self._stream_lock:
            return self._streams.get(stream_id)
            
    async def subscribe_stream(self, stream_id: str, subscriber_id: str, config: Dict[str, Any]) -> Tuple[bool, Optional[asyncio.Queue]]:
        """订阅视频流
        
        Args:
            stream_id: 流ID
            subscriber_id: 订阅者ID
            config: 流配置
            
        Returns:
            Tuple[bool, Optional[asyncio.Queue]]: (是否成功, 帧队列)
        """
        # 获取或创建流
        stream = await self.get_or_create_stream(stream_id, config)
        
        # 订阅流
        queue = await stream.subscribe(subscriber_id)
        
        return True, queue
        
    async def unsubscribe_stream(self, stream_id: str, subscriber_id: str) -> bool:
        """取消订阅视频流
        
        Args:
            stream_id: 流ID
            subscriber_id: 订阅者ID
            
        Returns:
            bool: 是否成功取消订阅
        """
        async with self._stream_lock:
            if stream_id not in self._streams:
                logger.warning(f"尝试取消订阅不存在的流: {stream_id}")
                return False
                
            stream = self._streams[stream_id]
            
            # 取消订阅
            result = await stream.unsubscribe(subscriber_id)
            
            # 检查是否可以释放流
            if result and stream.subscriber_count == 0:
                await self.release_stream(stream_id)
                
            return result
            
    async def get_all_streams(self) -> List[Dict[str, Any]]:
        """获取所有流信息
        
        Returns:
            List[Dict[str, Any]]: 流信息列表
        """
        async with self._stream_lock:
            return [stream.get_info() for stream in self._streams.values()]
            
    async def get_stream_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取流信息
        
        Args:
            stream_id: 流ID
            
        Returns:
            Optional[Dict[str, Any]]: 流信息
        """
        stream = await self.get_stream(stream_id)
        if stream:
            return stream.get_info()
        return None
        
    async def get_stream_status(self, stream_id: str) -> Optional[StreamStatus]:
        """获取流状态
        
        Args:
            stream_id: 流ID
            
        Returns:
            Optional[StreamStatus]: 流状态
        """
        stream = await self.get_stream(stream_id)
        if stream:
            return stream.get_status()
        return None
        
    async def stop_all_streams(self):
        """停止所有流"""
        async with self._stream_lock:
            for stream_id, stream in list(self._streams.items()):
                logger.info(f"停止流: {stream_id}")
                await stream.stop()
            self._streams.clear()
            
    async def shutdown(self):
        """关闭流管理器"""
        await self.stop_all_streams()
        logger.info("流管理器已关闭")

# 创建单例实例
stream_manager = StreamManager()

