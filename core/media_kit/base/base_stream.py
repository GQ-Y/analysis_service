"""
基础流模块
提供流的基本功能实现，供各协议继承
"""

import asyncio
import threading
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, Set, List
from abc import ABC, abstractmethod
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

from .stream_interface import (
    IStream, 
    StreamStatus, 
    StreamHealthStatus
)

class BaseStream(IStream, ABC):
    """基础流类，实现通用功能"""
    
    def __init__(self, stream_id: str, config: Dict[str, Any]):
        """初始化基础流
        
        Args:
            stream_id: 流ID
            config: 流配置
        """
        # 基本属性
        self._stream_id = stream_id
        self._config = config
        self._url = config.get("url", "")
        self._protocol = config.get("protocol", "unknown").lower()
        
        # 状态
        self._status = StreamStatus.INITIALIZING
        self._health_status = StreamHealthStatus.UNKNOWN
        self._last_error = ""
        
        # 控制标志
        self._is_running = False
        self._stop_event = asyncio.Event()
        
        # 帧缓存
        self._frame_buffer = []
        self._frame_buffer_size = config.get("frame_buffer_size", 5)
        self._frame_lock = threading.Lock()
        
        # 订阅管理
        self._subscribers: Dict[str, asyncio.Queue] = {}
        self._subscriber_lock = asyncio.Lock()
        
        # 帧处理控制
        self._frame_processed_event = asyncio.Event()
        self._frame_processed_event.set()
        
        # 任务
        self._pull_task = None
        self._frame_task = None
        
        # 统计信息
        self._stats = {
            "frames_received": 0,
            "frames_processed": 0,
            "errors": 0,
            "reconnects": 0,
            "start_time": 0,
            "last_frame_time": 0,
            "fps": 0
        }
        
        normal_logger.info(f"创建流: {stream_id}, URL: {self._url}, 协议: {self._protocol}")
    
    @property
    def stream_id(self) -> str:
        """获取流ID"""
        return self._stream_id
    
    @property
    def status(self) -> StreamStatus:
        """获取流状态"""
        return self._status
    
    @property
    def health_status(self) -> StreamHealthStatus:
        """获取流健康状态"""
        return self._health_status
    
    @property
    def url(self) -> str:
        """获取流URL"""
        return self._url
    
    @property
    def protocol(self) -> str:
        """获取流协议类型"""
        return self._protocol
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取流配置"""
        return self._config
    
    def set_status(self, status: StreamStatus) -> None:
        """设置流状态
        
        Args:
            status: 新状态
        """
        if self._status != status:
            normal_logger.info(f"流 {self._stream_id} 状态变更: {self._status.name} -> {status.name}")
            self._status = status
    
    def set_health_status(self, health_status: StreamHealthStatus) -> None:
        """设置流健康状态
        
        Args:
            health_status: 新健康状态
        """
        if self._health_status != health_status:
            normal_logger.info(f"流 {self._stream_id} 健康状态变更: {self._health_status.name} -> {health_status.name}")
            self._health_status = health_status
    
    async def start(self) -> bool:
        """启动流
        
        Returns:
            bool: 是否成功启动
        """
        if self._is_running:
            normal_logger.warning(f"流 {self._stream_id} 已经在运行中")
            return True
        
        normal_logger.info(f"启动流 {self._stream_id}")
        
        # 设置状态
        self._status = StreamStatus.INITIALIZING
        self._health_status = StreamHealthStatus.UNKNOWN
        self._last_error = ""
        
        # 重置统计
        self._stats = {
            "frames_received": 0,
            "frames_processed": 0,
            "errors": 0,
            "reconnects": 0,
            "start_time": time.time(),
            "last_frame_time": 0,
            "fps": 0
        }
        
        # 启动拉流
        success = await self._start_pulling()
        if not success:
            exception_logger.exception(f"启动流 {self._stream_id} 失败")
            self._status = StreamStatus.ERROR
            return False
        
        # 启动任务
        self._is_running = True
        self._stop_event.clear()
        self._pull_task = asyncio.create_task(self._pull_stream_task())
        self._frame_task = asyncio.create_task(self._process_frames_task())
        
        # 设置状态为连接中
        self._status = StreamStatus.CONNECTING
        normal_logger.info(f"流 {self._stream_id} 启动成功，等待连接...")
        return True 

    async def stop(self) -> bool:
        """停止流
        
        Returns:
            bool: 是否成功停止
        """
        if not self._is_running:
            normal_logger.warning(f"流 {self._stream_id} 未运行，无需停止")
            return True
        
        normal_logger.info(f"停止流 {self._stream_id}")
        
        # 设置停止事件
        self._stop_event.set()
        self._is_running = False
        
        # 等待任务结束
        if self._pull_task:
            try:
                await asyncio.wait_for(self._pull_task, timeout=5.0)
            except asyncio.TimeoutError:
                normal_logger.warning(f"等待流 {self._stream_id} 拉流任务停止超时")
            except Exception as e:
                exception_logger.exception(f"等待流 {self._stream_id} 拉流任务停止异常: {str(e)}")
        
        if self._frame_task:
            try:
                await asyncio.wait_for(self._frame_task, timeout=5.0)
            except asyncio.TimeoutError:
                normal_logger.warning(f"等待流 {self._stream_id} 帧处理任务停止超时")
            except Exception as e:
                exception_logger.exception(f"等待流 {self._stream_id} 帧处理任务停止异常: {str(e)}")
        
        # 停止拉流
        try:
            await self._stop_pulling()
        except Exception as e:
            exception_logger.exception(f"停止拉流异常: {str(e)}")
        
        # 清理订阅者
        async with self._subscriber_lock:
            self._subscribers.clear()
        
        # 清理帧缓存
        with self._frame_lock:
            self._frame_buffer.clear()
        
        # 设置状态
        self._status = StreamStatus.STOPPED
        self._health_status = StreamHealthStatus.OFFLINE
        
        normal_logger.info(f"流 {self._stream_id} 停止成功")
        return True
    
    async def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """获取最新的帧
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (是否成功, 帧数据)
        """
        # 检查流状态
        if not self._is_running or self._status not in [StreamStatus.RUNNING, StreamStatus.CONNECTING]:
            return False, None
        
        # 获取最新帧
        with self._frame_lock:
            if not self._frame_buffer:
                return False, None
            
            # 成功获取帧
            frame = self._frame_buffer[-1].copy()
        
        return True, frame
    
    async def get_snapshot(self, width: int = 0, height: int = 0) -> Optional[bytes]:
        """获取流的快照
        
        Args:
            width: 快照宽度，0表示使用原始宽度
            height: 快照高度，0表示使用原始高度
            
        Returns:
            Optional[bytes]: 快照数据(JPEG格式)，如果失败则返回None
        """
        success, frame = await self.get_frame()
        if not success or frame is None:
            return None
        
        # 调整大小
        if width > 0 and height > 0:
            frame = cv2.resize(frame, (width, height))
        
        # 转换为JPEG
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()
        except Exception as e:
            exception_logger.exception(f"生成快照异常: {str(e)}")
            return None
    
    async def subscribe(self, subscriber_id: str) -> Tuple[bool, Optional[asyncio.Queue]]:
        """订阅流
        
        Args:
            subscriber_id: 订阅者ID
            
        Returns:
            Tuple[bool, Optional[asyncio.Queue]]: (是否成功, 帧队列)
        """
        if not self._is_running:
            normal_logger.warning(f"流 {self._stream_id} 未运行，无法订阅")
            return False, None
        
        # 创建帧队列
        buffer_size = self._config.get("queue_size", 10)
        frame_queue = asyncio.Queue(maxsize=buffer_size)
        
        # 添加订阅者
        async with self._subscriber_lock:
            self._subscribers[subscriber_id] = frame_queue
        
        normal_logger.info(f"订阅者 {subscriber_id} 已订阅流 {self._stream_id}, 当前订阅者数量: {len(self._subscribers)}")
        return True, frame_queue
    
    async def unsubscribe(self, subscriber_id: str) -> bool:
        """取消订阅
        
        Args:
            subscriber_id: 订阅者ID
            
        Returns:
            bool: 是否成功取消订阅
        """
        async with self._subscriber_lock:
            if subscriber_id in self._subscribers:
                del self._subscribers[subscriber_id]
                normal_logger.info(f"订阅者 {subscriber_id} 已取消订阅流 {self._stream_id}, 当前订阅者数量: {len(self._subscribers)}")
                return True
        
        normal_logger.warning(f"订阅者 {subscriber_id} 未订阅流 {self._stream_id}")
        return False
    
    async def get_info(self) -> Dict[str, Any]:
        """获取流信息
        
        Returns:
            Dict[str, Any]: 流信息
        """
        return {
            "stream_id": self._stream_id,
            "url": self._url,
            "protocol": self._protocol,
            "status": self._status.name,
            "health_status": self._health_status.name,
            "last_error": self._last_error,
            "subscriber_count": len(self._subscribers),
            "stats": self._stats
        }
    
    @abstractmethod
    async def _start_pulling(self) -> bool:
        """开始拉流
        
        Returns:
            bool: 是否成功启动拉流
        """
        pass
    
    @abstractmethod
    async def _stop_pulling(self) -> bool:
        """停止拉流
        
        Returns:
            bool: 是否成功停止拉流
        """
        pass
    
    async def _pull_stream_task(self) -> None:
        """拉流任务，由子类实现"""
        pass
    
    async def _process_frames_task(self) -> None:
        """帧处理任务"""
        normal_logger.info(f"启动流 {self._stream_id} 帧处理任务")
        
        try:
            while not self._stop_event.is_set():
                try:
                    # 等待帧处理事件
                    await self._frame_processed_event.wait()
                    
                    # 如果停止事件已设置，退出循环
                    if self._stop_event.is_set():
                        break
                    
                    # 获取最新帧
                    success, frame = await self.get_frame()
                    if not success or frame is None:
                        # 没有帧可处理，等待一段时间
                        await asyncio.sleep(0.01)
                        continue
                    
                    # 更新统计信息
                    now = time.time()
                    self._stats["frames_processed"] += 1
                    
                    # 计算FPS
                    if self._stats["last_frame_time"] > 0:
                        time_diff = now - self._stats["last_frame_time"]
                        if time_diff > 0:
                            self._stats["fps"] = 1.0 / time_diff
                    
                    self._stats["last_frame_time"] = now
                    
                    # 向所有订阅者发送帧
                    await self._distribute_frame(frame, now)
                    
                except asyncio.CancelledError:
                    normal_logger.info(f"流 {self._stream_id} 帧处理任务被取消")
                    break
                except Exception as e:
                    exception_logger.exception(f"流 {self._stream_id} 帧处理异常: {str(e)}")
                    self._stats["errors"] += 1
                    await asyncio.sleep(0.1)
        finally:
            normal_logger.info(f"流 {self._stream_id} 帧处理任务已停止")
    
    async def _distribute_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """向所有订阅者分发帧
        
        Args:
            frame: 帧数据
            timestamp: 时间戳
        """
        # 获取订阅者列表
        async with self._subscriber_lock:
            subscribers = {sid: queue for sid, queue in self._subscribers.items()}
        
        # 向所有订阅者发送帧
        for subscriber_id, queue in subscribers.items():
            try:
                # 如果队列已满，丢弃旧帧
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                
                # 发送帧
                await queue.put((frame.copy(), timestamp))
            except Exception as e:
                exception_logger.exception(f"向订阅者 {subscriber_id} 发送帧异常: {str(e)}")
    
    def _add_frame_to_buffer(self, frame: np.ndarray) -> None:
        """添加帧到缓冲区
        
        Args:
            frame: 帧数据
        """
        with self._frame_lock:
            self._frame_buffer.append(frame)
            while len(self._frame_buffer) > self._frame_buffer_size:
                self._frame_buffer.pop(0)
        
        # 更新统计信息
        self._stats["frames_received"] += 1
        
        # 如果状态不是RUNNING，更新为RUNNING
        if self._status != StreamStatus.RUNNING:
            self.set_status(StreamStatus.RUNNING)
            
        # 如果健康状态不是GOOD，更新为GOOD
        if self._health_status != StreamHealthStatus.GOOD:
            self.set_health_status(StreamHealthStatus.GOOD) 