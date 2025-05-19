"""
视频流健康监控模块
负责监控视频流的健康状态，并通知订阅者
"""
import time
import threading
import asyncio
import json
from typing import Dict, Any, Optional, Set

from core.redis_manager import RedisManager
from shared.utils.logger import get_normal_logger, get_exception_logger
from .status import StreamStatus, StreamHealthStatus

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class StreamHealthMonitor:
    """视频流健康监控器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StreamHealthMonitor, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化健康监控器"""
        if self._initialized:
            return

        self._initialized = True
        self._streams: Dict[str, Dict[str, Any]] = {}  # stream_id -> 状态信息
        self._subscribers: Dict[str, Set[str]] = {}  # stream_id -> 订阅者集合
        self._stop_event = threading.Event()
        self._check_interval = 5  # 检查间隔(秒)
        self._max_offline_time = 10  # 离线阈值(秒)
        self._redis = None
        self._monitor_thread = None

        # 启动监控线程
        self._start_monitor()

        normal_logger.info("流健康监控器初始化完成")

    def _start_monitor(self):
        """启动监控线程"""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        normal_logger.info("流健康监控线程已启动")

    def _monitor_loop(self):
        """监控循环"""
        while not self._stop_event.is_set():
            try:
                self._check_streams_health()
            except Exception as e:
                exception_logger.exception(f"流健康监控异常: {str(e)}")

            time.sleep(self._check_interval)

    def _check_streams_health(self):
        """检查所有流的健康状态"""
        current_time = time.time()

        with self._lock:
            # 遍历所有注册的流
            for stream_id, info in list(self._streams.items()):
                if stream_id not in self._streams:
                    continue

                # 获取最后一帧的时间
                last_frame_time = info.get("last_frame_time", 0)
                current_status = info.get("status", StreamStatus.INITIALIZING)

                # 检查是否需要更新状态
                if current_status == StreamStatus.ONLINE:
                    # 如果在线状态下长时间未收到帧，标记为离线
                    if current_time - last_frame_time > self._max_offline_time:
                        new_status = StreamStatus.OFFLINE
                        normal_logger.warning(f"流 {stream_id} 已 {self._max_offline_time} 秒未收到新帧，状态更新为离线")
                        self._update_stream_status(stream_id, new_status)

    def _update_stream_status(self, stream_id: str, new_status: StreamStatus):
        """更新流状态并通知订阅者"""
        with self._lock:
            if stream_id not in self._streams:
                return

            # 只有当状态变化时才通知
            old_status = self._streams[stream_id].get("status")
            if old_status == new_status:
                return

            # 更新状态
            self._streams[stream_id]["status"] = new_status
            self._streams[stream_id]["status_updated_at"] = time.time()

            # 获取订阅者
            subscribers = self._subscribers.get(stream_id, set())

        # 通知所有订阅者
        if subscribers:
            status_data = {
                "stream_id": stream_id,
                "status": int(new_status),
                "status_text": self._get_status_text(new_status),
                "timestamp": time.time()
            }

            # 使用异步运行Redis发布
            asyncio.run_coroutine_threadsafe(
                self._notify_subscribers(stream_id, subscribers, status_data),
                asyncio.get_event_loop()
            )

    async def _notify_subscribers(self, stream_id: str, subscribers: Set[str], status_data: Dict[str, Any]):
        """通知订阅者流状态变化"""
        try:
            # 延迟初始化Redis
            if self._redis is None:
                self._redis = RedisManager()

            # 将状态数据转换为JSON
            status_json = json.dumps(status_data)

            # 为每个订阅者发布消息
            for subscriber_id in subscribers:
                channel = f"stream_status:{subscriber_id}"
                await self._redis.redis.publish(channel, status_json)
                normal_logger.info(f"已通知订阅者 {subscriber_id} 流 {stream_id} 状态变化: {self._get_status_text(status_data['status'])}")

        except Exception as e:
            exception_logger.exception(f"通知订阅者流状态变化失败: {str(e)}")

    def _get_status_text(self, status: StreamStatus) -> str:
        """获取状态文本"""
        status_texts = {
            StreamStatus.INITIALIZING: "初始化中",
            StreamStatus.CONNECTING: "连接中",
            StreamStatus.ONLINE: "在线",
            StreamStatus.OFFLINE: "离线",
            StreamStatus.ERROR: "错误",
            StreamStatus.RUNNING: "运行中",
            StreamStatus.RECONNECTING: "重连中",
            StreamStatus.PAUSED: "已暂停",
            StreamStatus.STOPPED: "已停止",
            StreamStatus.UNKNOWN: "未知"
        }
        return status_texts.get(status, "未知")

    def register_stream(self, stream_id: str):
        """注册流进行监控"""
        with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = {
                    "status": StreamStatus.INITIALIZING,
                    "registered_at": time.time(),
                    "last_frame_time": 0,
                    "status_updated_at": time.time(),
                    "health_status": StreamHealthStatus.HEALTHY,
                    "error_count": 0,
                    "frame_count": 0
                }
                self._subscribers[stream_id] = set()
                normal_logger.info(f"流 {stream_id} 已注册到健康监控器")

    def unregister_stream(self, stream_id: str):
        """取消流监控"""
        with self._lock:
            if stream_id in self._streams:
                del self._streams[stream_id]

            if stream_id in self._subscribers:
                del self._subscribers[stream_id]

            normal_logger.info(f"流 {stream_id} 已从健康监控器中移除")

    def add_subscriber(self, stream_id: str, subscriber_id: str):
        """添加订阅者"""
        with self._lock:
            # 确保流已注册
            if stream_id not in self._streams:
                self.register_stream(stream_id)

            if stream_id not in self._subscribers:
                self._subscribers[stream_id] = set()

            self._subscribers[stream_id].add(subscriber_id)
            normal_logger.info(f"订阅者 {subscriber_id} 已添加到流 {stream_id}")

    def remove_subscriber(self, stream_id: str, subscriber_id: str):
        """移除订阅者"""
        with self._lock:
            if stream_id in self._subscribers and subscriber_id in self._subscribers[stream_id]:
                self._subscribers[stream_id].remove(subscriber_id)
                normal_logger.info(f"订阅者 {subscriber_id} 已从流 {stream_id} 中移除")

                # 如果没有订阅者，考虑取消注册流
                if not self._subscribers[stream_id]:
                    normal_logger.info(f"流 {stream_id} 没有订阅者，考虑取消注册")

    def update_frame_received(self, stream_id: str):
        """更新流已接收到新帧"""
        with self._lock:
            if stream_id in self._streams:
                current_status = self._streams[stream_id].get("status")
                self._streams[stream_id]["last_frame_time"] = time.time()
                self._streams[stream_id]["frame_count"] += 1

                # 如果当前状态不是在线，更新为在线
                if current_status != StreamStatus.ONLINE:
                    normal_logger.info(f"流 {stream_id} 已接收到新帧，状态更新为在线")
                    self._update_stream_status(stream_id, StreamStatus.ONLINE)

                    # 通知流管理器
                    asyncio.run_coroutine_threadsafe(
                        self._notify_stream_manager(stream_id, StreamStatus.ONLINE, StreamHealthStatus.HEALTHY),
                        asyncio.get_event_loop()
                    )

    def get_stream_status(self, stream_id: str) -> Optional[StreamStatus]:
        """获取流状态"""
        with self._lock:
            if stream_id in self._streams:
                return self._streams[stream_id].get("status")
            return None

    def get_stream_health(self, stream_id: str) -> Optional[StreamHealthStatus]:
        """获取流健康状态"""
        with self._lock:
            if stream_id in self._streams:
                return self._streams[stream_id].get("health_status")
            return None

    def report_error(self, stream_id: str, error: str):
        """报告流错误"""
        with self._lock:
            if stream_id in self._streams:
                self._streams[stream_id]["error_count"] += 1
                error_count = self._streams[stream_id]["error_count"]

                # 如果错误次数过多，更新状态为错误
                if error_count >= 5:
                    exception_logger.error(f"流 {stream_id} 错误次数过多 ({error_count})，状态更新为错误: {error}")
                    self._update_stream_status(stream_id, StreamStatus.ERROR)
                else:
                    normal_logger.warning(f"流 {stream_id} 报告错误: {error}, 错误计数: {error_count}")

    def report_recovery(self, stream_id: str):
        """报告流恢复"""
        with self._lock:
            if stream_id in self._streams:
                self._streams[stream_id]["error_count"] = 0
                current_status = self._streams[stream_id].get("status")

                # 如果当前状态是错误或离线，更新为初始化中
                if current_status in [StreamStatus.ERROR, StreamStatus.OFFLINE]:
                    normal_logger.info(f"流 {stream_id} 报告恢复，状态更新为初始化中")
                    self._update_stream_status(stream_id, StreamStatus.INITIALIZING)

    def stop(self):
        """停止监控"""
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        normal_logger.info("流健康监控器已停止")

    async def _notify_stream_manager(self, stream_id: str, status: StreamStatus, health_status: StreamHealthStatus):
        """通知流管理器更新流状态

        Args:
            stream_id: 流ID
            status: 新状态
            health_status: 新健康状态
        """
        try:
            # 延迟导入流管理器，避免循环导入
            from .manager import stream_manager

            # 更新流状态
            await stream_manager.update_stream_status(stream_id, status, health_status)
        except Exception as e:
            exception_logger.exception(f"通知流管理器更新流状态失败: {str(e)}")

    async def shutdown(self):
        """关闭监控器"""
        self.stop()
        normal_logger.info("流健康监控器已关闭")

# 创建单例实例
stream_health_monitor = StreamHealthMonitor()