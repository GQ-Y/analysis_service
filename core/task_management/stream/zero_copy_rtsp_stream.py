"""
零拷贝RTSP流实现
直接使用OpenCV/FFmpeg进行拉流，支持零拷贝架构
"""

import asyncio
import threading
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from queue import Queue, Empty
from dataclasses import dataclass

from shared.utils.logger import normal_logger, exception_logger
from ...memory.memory_pool import MemoryPool
from ...frame.frame_reference import FrameReference, FrameReferenceManager, FrameMetadata
from ...interfaces.zero_copy_stream_interface import IZeroCopyVideoStream, AsyncFrameReferenceQueue
from ...interfaces.stream_interface import StreamStatus, StreamHealthStatus


@dataclass
class ZeroCopyStreamConfig:
    """零拷贝流配置"""
    max_queue_size: int = 10
    frame_timeout: float = 5.0
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 5
    buffer_size: int = 1
    enable_threading: bool = True


class ZeroCopyRTSPStream(IZeroCopyVideoStream):
    """
    零拷贝RTSP流实现
    直接使用OpenCV进行拉流，支持零拷贝内存池
    """
    
    def __init__(self, stream_id: str, stream_url: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化零拷贝RTSP流
        
        Args:
            stream_id: 流ID
            stream_url: RTSP流地址
            config: 流配置
        """
        self._stream_id = stream_id
        self._stream_url = stream_url
        self._config = config or {}

        # 流状态
        self._status = StreamStatus.STOPPED
        self._health_status = StreamHealthStatus.UNKNOWN
        
        # 零拷贝配置
        self.zero_copy_config = ZeroCopyStreamConfig()
        
        # 内存池
        self.memory_pool: Optional[MemoryPool] = None
        self.frame_reference_manager: Optional[FrameReferenceManager] = None
        
        # OpenCV视频捕获对象
        self.cap: Optional[cv2.VideoCapture] = None
        
        # 流状态
        self.is_running = False
        self.is_connected = False
        self.last_frame_time = 0
        
        # 订阅者管理
        self._subscribers: Dict[str, AsyncFrameReferenceQueue] = {}
        self.subscriber_lock = threading.Lock()
        
        # 拉流线程
        self.pull_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 流信息
        self.width = 0
        self.height = 0
        self.fps = 0.0
        
        # 统计信息
        self.frame_count = 0
        self.error_count = 0
        self.reconnect_count = 0
        
        normal_logger.info(f"创建零拷贝RTSP流: {self._stream_id}, URL: {self._stream_url}")

    # 实现IVideoStream接口的属性
    @property
    def stream_id(self) -> str:
        """获取流ID"""
        return self._stream_id

    @property
    def url(self) -> str:
        """获取流URL"""
        return self._stream_url

    @property
    def config(self) -> Dict[str, Any]:
        """获取流配置"""
        return self._config.copy()

    @property
    def subscribers(self) -> Dict[str, Any]:
        """获取订阅者"""
        with self.subscriber_lock:
            return {k: v.get_stats() for k, v in self._subscribers.items()}

    @property
    def subscriber_count(self) -> int:
        """获取订阅者数量"""
        with self.subscriber_lock:
            return len(self._subscribers)

    def get_status(self) -> StreamStatus:
        """获取流状态"""
        return self._status

    def set_status(self, status: StreamStatus):
        """设置流状态"""
        self._status = status

    def get_health_status(self) -> StreamHealthStatus:
        """获取流健康状态"""
        return self._health_status

    def set_health_status(self, health_status: StreamHealthStatus):
        """设置流健康状态"""
        self._health_status = health_status

    def set_memory_pool(self, memory_pool: MemoryPool) -> bool:
        """
        设置内存池

        Args:
            memory_pool: 内存池实例

        Returns:
            bool: 设置是否成功
        """
        try:
            self.memory_pool = memory_pool

            # 创建帧引用管理器
            self.frame_reference_manager = FrameReferenceManager(memory_pool=memory_pool)

            normal_logger.info(f"零拷贝RTSP流 {self._stream_id} 内存池设置完成")
            return True
        except Exception as e:
            exception_logger.exception(f"设置内存池失败: {self.stream_id}, {str(e)}")
            return False

    async def get_frame_reference(self) -> Tuple[bool, Optional[FrameReference]]:
        """
        获取帧引用（零拷贝）

        Returns:
            Tuple[bool, Optional[FrameReference]]: (是否成功, 帧引用)
        """
        try:
            # 这里应该从内部帧队列获取最新帧引用
            # 暂时返回None，因为这个方法通常不会被直接调用
            # 实际的帧分发通过订阅机制进行
            return False, None
        except Exception as e:
            exception_logger.exception(f"获取帧引用失败: {self._stream_id}, {str(e)}")
            return False, None

    async def get_frame_references_batch(self, count: int = 1) -> Tuple[bool, List[FrameReference]]:
        """
        批量获取帧引用（零拷贝）

        Args:
            count: 获取的帧数量

        Returns:
            Tuple[bool, List[FrameReference]]: (是否成功, 帧引用列表)
        """
        try:
            # 暂时返回空列表，实际的帧分发通过订阅机制进行
            return False, []
        except Exception as e:
            exception_logger.exception(f"批量获取帧引用失败: {self._stream_id}, {str(e)}")
            return False, []

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        获取内存使用情况

        Returns:
            Dict[str, Any]: 内存使用统计
        """
        try:
            if not self.memory_pool:
                return {"error": "内存池未设置"}

            return {
                "memory_pool_status": self.memory_pool.get_stats(),
                "subscriber_count": len(self._subscribers),
                "frame_count": self.frame_count,
                "error_count": self.error_count,
            }
        except Exception as e:
            exception_logger.exception(f"获取内存使用情况失败: {self._stream_id}, {str(e)}")
            return {"error": str(e)}

    def is_memory_pressure_high(self) -> bool:
        """
        检查是否存在内存压力

        Returns:
            bool: 是否存在高内存压力
        """
        try:
            if not self.memory_pool:
                return True  # 没有内存池认为是高压力

            status = self.memory_pool.get_stats()
            usage_ratio = status.get("memory_usage_ratio", 0.0)

            # 如果内存使用率超过80%，认为是高压力
            return usage_ratio > 0.8
        except Exception as e:
            exception_logger.exception(f"检查内存压力失败: {self._stream_id}, {str(e)}")
            return True

    async def get_info(self) -> Dict[str, Any]:
        """
        获取流信息

        Returns:
            Dict[str, Any]: 流信息
        """
        return self.get_stream_info()

    async def start(self) -> bool:
        """
        启动流

        Returns:
            bool: 启动是否成功
        """
        try:
            if self.is_running:
                normal_logger.warning(f"流 {self._stream_id} 已经在运行")
                return True

            # 验证内存池和帧引用管理器是否已设置
            if self.memory_pool is None or self.frame_reference_manager is None:
                normal_logger.error(f"流 {self._stream_id} 内存池或帧引用管理器未设置，无法启动")
                normal_logger.error(f"memory_pool: {self.memory_pool}, frame_reference_manager: {self.frame_reference_manager}")
                return False

            # 重置停止事件
            self.stop_event.clear()
            
            # 启动拉流线程
            self.pull_thread = threading.Thread(
                target=self._pull_stream_worker,
                daemon=True
            )
            self.pull_thread.start()
            
            # 等待连接建立
            max_wait = 10  # 最多等待10秒
            wait_time = 0
            while not self.is_connected and wait_time < max_wait:
                await asyncio.sleep(0.1)
                wait_time += 0.1
            
            if not self.is_connected:
                normal_logger.error(f"流 {self._stream_id} 连接超时")
                self._status = StreamStatus.ERROR
                self._health_status = StreamHealthStatus.UNHEALTHY
                return False

            self.is_running = True
            self._status = StreamStatus.RUNNING
            self._health_status = StreamHealthStatus.HEALTHY
            normal_logger.info(f"零拷贝RTSP流 {self._stream_id} 启动成功")
            return True
            
        except Exception as e:
            exception_logger.exception(f"启动流失败: {self._stream_id}, {str(e)}")
            return False
    
    async def stop(self) -> bool:
        """
        停止流
        
        Returns:
            bool: 停止是否成功
        """
        try:
            if not self.is_running:
                return True
            
            # 设置停止事件
            self.stop_event.set()
            
            # 等待拉流线程结束
            if self.pull_thread and self.pull_thread.is_alive():
                self.pull_thread.join(timeout=5.0)
            
            # 关闭OpenCV捕获对象
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # 清理订阅者
            with self.subscriber_lock:
                for queue in self._subscribers.values():
                    try:
                        await queue.close()
                    except:
                        pass
                self._subscribers.clear()
            
            self.is_running = False
            self.is_connected = False
            self._status = StreamStatus.STOPPED
            self._health_status = StreamHealthStatus.UNKNOWN

            normal_logger.info(f"零拷贝RTSP流 {self._stream_id} 已停止")
            return True

        except Exception as e:
            exception_logger.exception(f"停止流失败: {self._stream_id}, {str(e)}")
            return False
    
    async def subscribe(self, subscriber_id: str, config: Optional[Dict[str, Any]] = None) -> Optional[AsyncFrameReferenceQueue]:
        """
        订阅流
        
        Args:
            subscriber_id: 订阅者ID
            config: 订阅配置
            
        Returns:
            AsyncFrameReferenceQueue: 帧引用队列
        """
        try:
            with self.subscriber_lock:
                if subscriber_id in self._subscribers:
                    normal_logger.warning(f"订阅者 {subscriber_id} 已经订阅了流 {self._stream_id}")
                    return self._subscribers[subscriber_id]

                # 创建帧引用队列
                queue = AsyncFrameReferenceQueue(
                    maxsize=self.zero_copy_config.max_queue_size
                )

                self._subscribers[subscriber_id] = queue
                
                normal_logger.info(f"订阅者 {subscriber_id} 已订阅零拷贝RTSP流 {self._stream_id}")
                return queue

        except Exception as e:
            exception_logger.exception(f"订阅流失败: {self._stream_id}, {subscriber_id}, {str(e)}")
            return None
    
    async def unsubscribe(self, subscriber_id: str) -> bool:
        """
        取消订阅流
        
        Args:
            subscriber_id: 订阅者ID
            
        Returns:
            bool: 取消订阅是否成功
        """
        try:
            with self.subscriber_lock:
                if subscriber_id not in self._subscribers:
                    normal_logger.warning(f"订阅者 {subscriber_id} 没有订阅流 {self._stream_id}")
                    return True

                queue = self._subscribers.pop(subscriber_id)
                # AsyncFrameReferenceQueue 没有 close 方法，直接删除即可

                normal_logger.info(f"订阅者 {subscriber_id} 已取消订阅零拷贝RTSP流 {self._stream_id}")
                return True

        except Exception as e:
            exception_logger.exception(f"取消订阅流失败: {self._stream_id}, {subscriber_id}, {str(e)}")
            return False
    
    def _pull_stream_worker(self):
        """
        拉流工作线程
        """
        normal_logger.info(f"零拷贝RTSP流 {self._stream_id} 拉流线程启动")

        # 等待内存池和帧引用管理器设置完成
        max_wait_time = 30  # 最多等待30秒
        wait_interval = 0.1  # 每次等待0.1秒
        waited_time = 0

        while (self.memory_pool is None or self.frame_reference_manager is None) and waited_time < max_wait_time:
            normal_logger.debug(f"流 {self._stream_id} 等待内存池和帧引用管理器设置...")
            time.sleep(wait_interval)
            waited_time += wait_interval

        if self.memory_pool is None or self.frame_reference_manager is None:
            normal_logger.error(f"流 {self._stream_id} 内存池或帧引用管理器设置超时，停止拉流线程")
            normal_logger.error(f"memory_pool: {self.memory_pool}, frame_reference_manager: {self.frame_reference_manager}")
            return

        normal_logger.info(f"流 {self._stream_id} 内存池和帧引用管理器设置完成，开始拉流")

        reconnect_attempts = 0
        
        while not self.stop_event.is_set():
            try:
                # 连接流
                if not self._connect_stream():
                    reconnect_attempts += 1
                    if reconnect_attempts >= self.zero_copy_config.max_reconnect_attempts:
                        normal_logger.error(f"流 {self._stream_id} 重连次数超过限制，停止拉流")
                        break

                    normal_logger.warning(f"流 {self._stream_id} 连接失败，{self.zero_copy_config.reconnect_interval}秒后重试")
                    time.sleep(self.zero_copy_config.reconnect_interval)
                    continue
                
                # 重置重连计数
                reconnect_attempts = 0
                self.is_connected = True
                
                # 拉流循环
                while not self.stop_event.is_set() and self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if not ret or frame is None:
                        normal_logger.warning(f"流 {self._stream_id} 读取帧失败")
                        self.error_count += 1
                        break
                    
                    # 处理帧
                    self._process_frame(frame)
                    
                    # 更新统计
                    self.frame_count += 1
                    self.last_frame_time = time.time()
                
                # 连接断开
                self.is_connected = False
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
            except Exception as e:
                exception_logger.exception(f"拉流线程异常: {self._stream_id}, {str(e)}")
                self.error_count += 1
                self.is_connected = False
                
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
                # 等待后重试
                time.sleep(self.zero_copy_config.reconnect_interval)
        
        normal_logger.info(f"零拷贝RTSP流 {self._stream_id} 拉流线程结束")
    
    def _connect_stream(self) -> bool:
        """
        连接流
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 创建OpenCV捕获对象
            self.cap = cv2.VideoCapture(self._stream_url)
            
            # 设置缓冲区大小
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.zero_copy_config.buffer_size)
            
            # 检查连接
            if not self.cap.isOpened():
                normal_logger.error(f"无法打开流: {self._stream_url}")
                return False
            
            # 获取流信息
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            normal_logger.info(f"流 {self._stream_id} 连接成功: {self.width}x{self.height}@{self.fps}fps")
            return True

        except Exception as e:
            exception_logger.exception(f"连接流失败: {self._stream_id}, {str(e)}")
            return False
    
    def _process_frame(self, frame: np.ndarray):
        """
        处理帧数据

        Args:
            frame: OpenCV帧数据
        """
        try:
            # 检查内存池和帧引用管理器是否已设置
            if self.memory_pool is None or self.frame_reference_manager is None:
                # 这种情况在正常启动流程中不应该发生，因为start()方法已经验证过
                normal_logger.error(f"流 {self._stream_id} 内存池或帧引用管理器未设置，这是一个严重错误")
                normal_logger.error(f"memory_pool: {self.memory_pool}, frame_reference_manager: {self.frame_reference_manager}")
                # 增加错误计数并跳过处理
                self.error_count += 1
                return

            # 获取内存块
            memory_block = self.memory_pool.allocate_frame_block(self.width, self.height)
            if not memory_block:
                normal_logger.warning(f"流 {self._stream_id} 无法获取内存块")
                return

            # 复制帧数据到内存块（零拷贝）
            memory_block.copy_from_numpy(frame)

            # 创建帧元数据
            metadata = FrameMetadata(
                frame_id=hash(f"{self._stream_id}_{self.frame_count}"),
                timestamp=time.time(),
                width=self.width,
                height=self.height,
                channels=3,
                sequence_number=self.frame_count,
                memory_block_ref=memory_block.block_id
            )

            # 通过帧引用管理器创建帧引用（这样会自动设置清理回调）
            frame_ref = self.frame_reference_manager.create_reference(memory_block, metadata)
            if frame_ref:
                # 分发给所有订阅者
                self._distribute_frame(frame_ref)

                self.frame_count += 1
                self.last_frame_time = time.time()

                # 释放我们的引用（订阅者会持有自己的引用）
                frame_ref.release()
            else:
                # 如果创建帧引用失败，手动释放内存块
                self.memory_pool.deallocate_frame_block(memory_block)

        except Exception as e:
            exception_logger.exception(f"处理帧失败: {self._stream_id}, {str(e)}")
    
    def _distribute_frame(self, frame_ref: FrameReference):
        """
        分发帧给所有订阅者
        
        Args:
            frame_ref: 帧引用
        """
        try:
            with self.subscriber_lock:
                for subscriber_id, queue in list(self._subscribers.items()):
                    try:
                        # 创建新的引用（增加引用计数）
                        subscriber_ref = frame_ref.create_reference()
                        if subscriber_ref:
                            # 异步放入队列
                            asyncio.create_task(queue.put(subscriber_ref))

                    except Exception as e:
                        exception_logger.exception(f"分发帧给订阅者失败: {subscriber_id}, {str(e)}")
            
        except Exception as e:
            exception_logger.exception(f"分发帧失败: {self._stream_id}, {str(e)}")
    
    def get_stream_info(self) -> Dict[str, Any]:
        """
        获取流信息
        
        Returns:
            Dict[str, Any]: 流信息
        """
        return {
            "stream_id": self._stream_id,
            "stream_url": self._stream_url,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "is_running": self.is_running,
            "is_connected": self.is_connected,
            "frame_count": self.frame_count,
            "error_count": self.error_count,
            "reconnect_count": self.reconnect_count,
            "subscriber_count": len(self._subscribers),
            "last_frame_time": self.last_frame_time
        }
