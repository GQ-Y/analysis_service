"""
视频流类
负责单个视频流的拉取、处理和分发
"""
import cv2
import time
import threading
import asyncio
import json
from typing import Dict, Any, Optional, Set, Tuple
from loguru import logger
from asyncio import Queue, Lock

from core.config import settings
from shared.utils.logger import setup_logger
from .status import StreamStatus

# 避免循环导入
# 直接使用健康监控器实例的变量，而不是导入健康监控器
_stream_health_monitor = None

logger = setup_logger(__name__)

class VideoStream:
    """视频流类，负责拉取视频流和分发帧给订阅者"""
    
    def __init__(self, stream_id: str, config: Dict[str, Any]):
        """初始化视频流
        
        Args:
            stream_id: 流ID
            config: 流配置
        """
        global _stream_health_monitor
        if _stream_health_monitor is None:
            # 延迟导入健康监控器，避免循环导入
            from .health_monitor import stream_health_monitor
            _stream_health_monitor = stream_health_monitor
            
        self.stream_id = stream_id
        self.config = config
        self.url = config.get("url", "")
        self.rtsp_transport = config.get("rtsp_transport", "tcp")
        self.reconnect_attempts = config.get("reconnect_attempts", settings.STREAMING.reconnect_attempts)
        self.reconnect_delay = config.get("reconnect_delay", settings.STREAMING.reconnect_delay)
        self.frame_buffer_size = config.get("frame_buffer_size", settings.STREAMING.frame_buffer_size)
        
        # 线程和状态
        self.running = False
        self.thread = None
        self.stream_lock = Lock()
        self.status = StreamStatus.INITIALIZING
        self.last_frame_time = None
        self.start_time = None
        self.frame_count = 0
        self.error_count = 0
        
        # 帧队列和订阅者
        self.frame_queue = asyncio.Queue(maxsize=self.frame_buffer_size)
        self.subscribers: Dict[str, Queue] = {}  # subscriber_id -> Queue
        self.subscriber_lock = Lock()
        
        # 注册到健康监控器
        _stream_health_monitor.register_stream(self.stream_id)
        
        logger.info(f"视频流 {self.stream_id} 初始化完成: {self.url}")
        
    async def start(self):
        """启动视频流拉取"""
        if self.running:
            logger.info(f"视频流 {self.stream_id} 已在运行中")
            return True
            
        async with self.stream_lock:
            if self.running:
                return True
                
            self.running = True
            self.start_time = time.time()
            
            # 启动拉流线程
            self.thread = threading.Thread(
                target=self._stream_loop,
                daemon=True,
                name=f"Stream-{self.stream_id}"
            )
            self.thread.start()
            
            logger.info(f"视频流 {self.stream_id} 已启动")
            return True
            
    async def stop(self):
        """停止视频流拉取"""
        if not self.running:
            return True
            
        async with self.stream_lock:
            if not self.running:
                return True
                
            self.running = False
            
            # 等待线程结束
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)
                
            # 清空帧队列
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
                    
            # 从健康监控器中注销
            _stream_health_monitor.unregister_stream(self.stream_id)
            
            logger.info(f"视频流 {self.stream_id} 已停止")
            return True
            
    def _stream_loop(self):
        """拉流循环，在单独线程中运行"""
        cap = None
        reconnect_count = 0
        
        while self.running:
            try:
                # 如果捕获对象不存在或未打开，创建新的捕获对象
                if cap is None or not cap.isOpened():
                    # 释放可能存在的旧捕获对象
                    if cap is not None:
                        cap.release()
                        
                    logger.info(f"视频流 {self.stream_id} 尝试连接: {self.url}")
                    
                    # 为RTSP流配置传输协议
                    if self.url.startswith("rtsp://"):
                        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # 设置较小的缓冲区
                        if self.rtsp_transport == "tcp":
                            cap.set(cv2.CAP_PROP_RTSP_TRANSPORT_FLAGS, 0)  # 强制TCP
                    else:
                        cap = cv2.VideoCapture(self.url)
                        
                    # 检查连接是否成功
                    if not cap.isOpened():
                        reconnect_count += 1
                        if reconnect_count > self.reconnect_attempts:
                            logger.error(f"视频流 {self.stream_id} 连接失败，超过最大重试次数: {self.reconnect_attempts}")
                            _stream_health_monitor.report_error(self.stream_id, "连接失败")
                            time.sleep(self.reconnect_delay * 2)  # 延长等待时间
                            reconnect_count = 0  # 重置重试计数，给下一轮机会
                        else:
                            logger.warning(f"视频流 {self.stream_id} 连接失败，尝试重连 ({reconnect_count}/{self.reconnect_attempts})")
                            time.sleep(self.reconnect_delay)
                        continue
                    else:
                        reconnect_count = 0
                        logger.info(f"视频流 {self.stream_id} 连接成功")
                        
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    reconnect_count += 1
                    logger.warning(f"视频流 {self.stream_id} 读取帧失败，尝试重连 ({reconnect_count}/{self.reconnect_attempts})")
                    time.sleep(1)
                    
                    if reconnect_count > self.reconnect_attempts:
                        cap.release()
                        cap = None
                        _stream_health_monitor.report_error(self.stream_id, "读取帧失败")
                    
                    continue
                    
                # 重置重连计数
                reconnect_count = 0
                
                # 更新状态
                self.last_frame_time = time.time()
                self.frame_count += 1
                
                # 通知健康监控器
                _stream_health_monitor.update_frame_received(self.stream_id)
                
                # 将帧放入队列
                try:
                    # 如果队列已满，丢弃最旧的帧
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                            
                    # 将新帧放入队列
                    asyncio.run_coroutine_threadsafe(
                        self.frame_queue.put((frame, self.last_frame_time)),
                        asyncio.get_event_loop()
                    )
                    
                    # 将帧分发给所有订阅者
                    asyncio.run_coroutine_threadsafe(
                        self._distribute_frame(frame, self.last_frame_time),
                        asyncio.get_event_loop()
                    )
                    
                except Exception as e:
                    logger.error(f"视频流 {self.stream_id} 处理帧异常: {str(e)}")
                    
            except Exception as e:
                logger.error(f"视频流 {self.stream_id} 处理异常: {str(e)}")
                time.sleep(1)
                
        # 关闭视频捕获
        if cap is not None and cap.isOpened():
            cap.release()
            
        logger.info(f"视频流 {self.stream_id} 拉流线程已退出")
        
    async def _distribute_frame(self, frame, timestamp):
        """分发帧给所有订阅者"""
        async with self.subscriber_lock:
            # 复制订阅者列表，避免在分发过程中修改
            subscribers = dict(self.subscribers)
            
        # 分发给每个订阅者
        for subscriber_id, queue in subscribers.items():
            try:
                # 如果队列已满，丢弃最旧的帧
                if queue.full():
                    try:
                        queue.get_nowait()
                    except:
                        pass
                        
                # 将新帧放入订阅者队列
                await queue.put((frame.copy(), timestamp))
            except Exception as e:
                logger.error(f"向订阅者 {subscriber_id} 分发帧失败: {str(e)}")
                
    async def subscribe(self, subscriber_id: str) -> Queue:
        """订阅视频流
        
        Args:
            subscriber_id: 订阅者ID
            
        Returns:
            Queue: 帧队列
        """
        async with self.subscriber_lock:
            # 检查是否已订阅
            if subscriber_id in self.subscribers:
                logger.info(f"订阅者 {subscriber_id} 已订阅视频流 {self.stream_id}")
                return self.subscribers[subscriber_id]
                
            # 创建订阅者帧队列
            subscriber_queue = asyncio.Queue(maxsize=self.frame_buffer_size)
            self.subscribers[subscriber_id] = subscriber_queue
            
            # 注册到健康监控器
            _stream_health_monitor.add_subscriber(self.stream_id, subscriber_id)
            
            logger.info(f"订阅者 {subscriber_id} 已订阅视频流 {self.stream_id}, 当前订阅者数: {len(self.subscribers)}")
            return subscriber_queue
            
    async def unsubscribe(self, subscriber_id: str) -> bool:
        """取消订阅视频流
        
        Args:
            subscriber_id: 订阅者ID
            
        Returns:
            bool: 是否成功取消订阅
        """
        async with self.subscriber_lock:
            if subscriber_id not in self.subscribers:
                logger.warning(f"订阅者 {subscriber_id} 未订阅视频流 {self.stream_id}")
                return False
                
            # 移除订阅者队列
            del self.subscribers[subscriber_id]
            
            # 从健康监控器中移除订阅者
            _stream_health_monitor.remove_subscriber(self.stream_id, subscriber_id)
            
            logger.info(f"订阅者 {subscriber_id} 已取消订阅视频流 {self.stream_id}, 当前订阅者数: {len(self.subscribers)}")
            return True
            
    async def get_frame(self) -> Tuple[Optional[Any], Optional[float]]:
        """获取最新帧，用于直接访问
        
        Returns:
            Tuple[Optional[Any], Optional[float]]: (帧, 时间戳)
        """
        try:
            return await self.frame_queue.get()
        except Exception as e:
            logger.error(f"获取帧失败: {str(e)}")
            return None, None
            
    @property
    def subscriber_count(self) -> int:
        """获取订阅者数量"""
        return len(self.subscribers)
        
    def get_status(self) -> StreamStatus:
        """获取流状态"""
        return _stream_health_monitor.get_stream_status(self.stream_id) or StreamStatus.INITIALIZING
        
    def get_info(self) -> Dict[str, Any]:
        """获取流信息"""
        return {
            "stream_id": self.stream_id,
            "url": self.url,
            "status": int(self.get_status()),
            "status_text": self._get_status_text(self.get_status()),
            "subscriber_count": self.subscriber_count,
            "frame_count": self.frame_count,
            "fps": self._calculate_fps(),
            "start_time": self.start_time,
            "running_time": time.time() - (self.start_time or time.time()),
            "last_frame_time": self.last_frame_time
        }
        
    def _calculate_fps(self) -> float:
        """计算帧率"""
        if self.start_time is None or self.frame_count == 0:
            return 0.0
            
        running_time = time.time() - self.start_time
        if running_time <= 0:
            return 0.0
            
        return self.frame_count / running_time
        
    def _get_status_text(self, status: StreamStatus) -> str:
        """获取状态文本"""
        status_texts = {
            StreamStatus.INITIALIZING: "初始化中",
            StreamStatus.ONLINE: "在线",
            StreamStatus.OFFLINE: "离线",
            StreamStatus.ERROR: "错误"
        }
        return status_texts.get(status, "未知") 