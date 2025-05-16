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
        use_zlmediakit: bool = False  # 是否使用ZLMediaKit
    class MockSettings:
        STREAMING = MockStreamingConfig()
    settings = MockSettings()

# 导入状态定义
from .status import StreamStatus, StreamHealthStatus

# 导入流接口和实现
from .interface import IVideoStream
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
        self._streams: Dict[str, IVideoStream] = {}  # stream_id -> IVideoStream
        self._stream_lock = asyncio.Lock()

        # ZLMediaKit相关
        self._use_zlmediakit = getattr(settings.STREAMING, "use_zlmediakit", False)
        self._zlm_bridge = None

        logger.info(f"流管理器初始化完成，ZLMediaKit支持: {'启用' if self._use_zlmediakit else '禁用'}")

    async def initialize(self):
        """初始化流管理器，包括ZLMediaKit桥接器"""
        # 如果启用ZLMediaKit，初始化桥接器
        if self._use_zlmediakit:
            try:
                # 导入ZLMediaKit桥接器
                from core.media_kit.zlm_bridge import zlm_bridge
                self._zlm_bridge = zlm_bridge

                # 初始化桥接器
                await self._zlm_bridge.initialize()

                logger.info("ZLMediaKit桥接器初始化成功")
            except Exception as e:
                logger.error(f"初始化ZLMediaKit桥接器失败: {str(e)}")
                logger.error(traceback.format_exc())
                # 关闭ZLMediaKit支持
                self._use_zlmediakit = False

        logger.info("流管理器初始化完成")

    async def get_or_create_stream(self, stream_id: str, config: Dict[str, Any]) -> IVideoStream:
        """获取或创建视频流

        Args:
            stream_id: 流ID
            config: 流配置

        Returns:
            IVideoStream: 视频流对象
        """
        # 确保有效的stream_id
        if not stream_id:
            # 如果没有提供stream_id，尝试从配置中生成一个
            task_id = config.get("task_id", "")
            if task_id:
                stream_id = f"stream_{task_id}"
                logger.info(f"流ID为空，使用任务ID({task_id})生成流ID: {stream_id}")
            else:
                # 如果没有任务ID，则使用随机ID
                stream_id = f"stream_{time.time()}"
                logger.info(f"流ID为空，生成临时ID: {stream_id}")
            # 更新配置中的stream_id
            config["stream_id"] = stream_id

        async with self._stream_lock:
            # 检查流是否已存在
            if stream_id in self._streams:
                logger.info(f"返回已存在的视频流: {stream_id}")
                return self._streams[stream_id]

            # 如果启用ZLMediaKit，尝试创建ZLM流
            zlm_bridge_result = False
            if self._use_zlmediakit and self._zlm_bridge:
                try:
                    # 创建ZLM桥接
                    zlm_bridge_result = await self._zlm_bridge.create_bridge(stream_id, config)
                    if not zlm_bridge_result:
                        logger.warning(f"创建ZLM流桥接失败，将回退到直接使用OpenCV: {stream_id}")
                except Exception as e:
                    logger.error(f"创建ZLM流桥接时出错: {str(e)}")
                    logger.error(traceback.format_exc())
                    zlm_bridge_result = False

            # 创建新的视频流
            logger.info(f"创建新的视频流: {stream_id}")
            stream = VideoStream(stream_id, config)

            # 添加到管理器
            self._streams[stream_id] = stream

            # 启动流
            try:
                await stream.start()
            except Exception as e:
                logger.error(f"启动流失败: {str(e)}")
                logger.error(traceback.format_exc())
                # 从管理器中移除失败的流
                if stream_id in self._streams:
                    del self._streams[stream_id]
                raise

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

                # 停止流
                await stream.stop()

                # 停止ZLM桥接
                if self._use_zlmediakit and self._zlm_bridge:
                    await self._zlm_bridge.stop_bridge(stream_id)

                # 从管理器移除
                del self._streams[stream_id]
                return True

            logger.info(f"流 {stream_id} 仍有 {stream.subscriber_count} 个订阅者，保持运行")
            return False

    async def get_stream(self, stream_id: str) -> Optional[IVideoStream]:
        """获取视频流

        Args:
            stream_id: 流ID

        Returns:
            Optional[IVideoStream]: 视频流对象，如果不存在则返回None
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
        # 确保有效的stream_id
        if not stream_id:
            stream_id = f"stream_{subscriber_id}"
            logger.info(f"流ID为空，使用订阅者ID({subscriber_id})生成流ID: {stream_id}")
            # 更新配置中的stream_id
            config["stream_id"] = stream_id

        # 如果启用ZLMediaKit，尝试使用ZLM订阅
        if self._use_zlmediakit and self._zlm_bridge:
            try:
                # 判断是否有ZLM桥接
                bridge_status = await self._zlm_bridge.get_bridge_status(stream_id)

                if bridge_status:
                    # 使用ZLM订阅
                    success, queue = await self._zlm_bridge.subscribe_zlm_stream(stream_id, subscriber_id)
                    if success:
                        logger.info(f"使用ZLMediaKit订阅流: {stream_id}, 订阅者: {subscriber_id}")
                        return True, queue
                    # 如果失败，回退到直接使用VideoStream
                    logger.warning(f"ZLMediaKit订阅失败，回退到直接使用VideoStream: {stream_id}")
            except Exception as e:
                logger.error(f"使用ZLMediaKit订阅流 {stream_id} 时出错: {str(e)}")
                logger.error(traceback.format_exc())

        # 获取或创建流
        try:
            stream = await self.get_or_create_stream(stream_id, config)

            # 订阅流
            success, queue = await stream.subscribe(subscriber_id)

            return success, queue
        except Exception as e:
            logger.error(f"获取或订阅流 {stream_id} 时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None

    async def unsubscribe_stream(self, stream_id: str, subscriber_id: str) -> bool:
        """取消订阅视频流

        Args:
            stream_id: 流ID
            subscriber_id: 订阅者ID

        Returns:
            bool: 是否成功取消订阅
        """
        # 如果启用ZLMediaKit，尝试使用ZLM取消订阅
        if self._use_zlmediakit and self._zlm_bridge:
            try:
                # 判断是否有ZLM桥接
                bridge_status = await self._zlm_bridge.get_bridge_status(stream_id)

                if bridge_status:
                    # 使用ZLM取消订阅
                    success = await self._zlm_bridge.unsubscribe_zlm_stream(stream_id, subscriber_id)
                    if success:
                        logger.info(f"使用ZLMediaKit取消订阅流: {stream_id}, 订阅者: {subscriber_id}")

                        # 检查是否需要释放桥接
                        await self.check_and_release_bridge(stream_id)

                        return True
                    # 如果失败，回退到直接使用VideoStream
                    logger.warning(f"ZLMediaKit取消订阅失败，回退到直接使用VideoStream: {stream_id}")
            except Exception as e:
                logger.error(f"使用ZLMediaKit取消订阅流 {stream_id} 时出错: {str(e)}")
                logger.error(traceback.format_exc())

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

    async def check_and_release_bridge(self, stream_id: str) -> None:
        """检查并释放ZLM桥接

        Args:
            stream_id: 流ID
        """
        if not self._use_zlmediakit or not self._zlm_bridge:
            return

        # 如果没有订阅者且没有对应的VideoStream，则释放ZLM桥接
        bridge_status = await self._zlm_bridge.get_bridge_status(stream_id)
        if not bridge_status:
            return

        # 检查是否有对应的VideoStream
        async with self._stream_lock:
            if stream_id not in self._streams or self._streams[stream_id].subscriber_count == 0:
                # 停止ZLM桥接
                await self._zlm_bridge.stop_bridge(stream_id)
                logger.info(f"无订阅者，释放ZLM桥接: {stream_id}")

    async def update_stream_status(self, stream_id: str, status: StreamStatus, health_status: StreamHealthStatus, error_msg: str = "") -> None:
        """更新流状态

        Args:
            stream_id: 流ID
            status: 流状态
            health_status: 健康状态
            error_msg: 错误信息
        """
        async with self._stream_lock:
            if stream_id not in self._streams:
                logger.warning(f"尝试更新不存在的流状态: {stream_id}")
                return

            stream = self._streams[stream_id]

            # 更新状态
            stream.set_status(status)
            stream.set_health_status(health_status)

            # 记录错误
            if error_msg:
                stream.set_last_error(error_msg)

            logger.info(f"更新流状态: {stream_id}, 状态: {status.name}, 健康状态: {health_status.name}")

    async def get_all_streams(self) -> List[Dict[str, Any]]:
        """获取所有流信息

        Returns:
            List[Dict[str, Any]]: 流信息列表
        """
        async with self._stream_lock:
            result = []
            for stream in self._streams.values():
                info_method = stream.get_info
                if asyncio.iscoroutinefunction(info_method):
                    result.append(await info_method())
                else:
                    result.append(info_method())
            return result

    async def get_stream_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取流信息

        Args:
            stream_id: 流ID

        Returns:
            Optional[Dict[str, Any]]: 流信息
        """
        try:
            stream = await self.get_stream(stream_id)
            if stream:
                # 获取流信息
                return await stream.get_info()

            # 如果启用ZLMediaKit，尝试从ZLM桥接获取状态
            if self._use_zlmediakit and self._zlm_bridge:
                bridge_status = await self._zlm_bridge.get_bridge_status(stream_id)
                if bridge_status:
                    logger.info(f"从ZLM桥接获取流状态: {stream_id}")
                    return bridge_status

            logger.warning(f"流 {stream_id} 不存在或无法获取流信息")
            return None

        except Exception as e:
            logger.error(f"获取流信息时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
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
            # 检查是否为协程方法
            status_method = stream.get_status
            if asyncio.iscoroutinefunction(status_method):
                return await status_method()
            else:
                return status_method()
        return None

    async def stop_all_streams(self):
        """停止所有流"""
        async with self._stream_lock:
            for stream_id, stream in list(self._streams.items()):
                logger.info(f"停止流: {stream_id}")
                await stream.stop()

                # 如果启用ZLMediaKit，停止ZLM桥接
                if self._use_zlmediakit and self._zlm_bridge:
                    await self._zlm_bridge.stop_bridge(stream_id)

            self._streams.clear()

    async def shutdown(self):
        """关闭流管理器"""
        # 停止所有流
        await self.stop_all_streams()

        # 如果启用ZLMediaKit，关闭ZLM桥接器
        if self._use_zlmediakit and self._zlm_bridge:
            try:
                await self._zlm_bridge.shutdown()
                logger.info("ZLMediaKit桥接器已关闭")
            except Exception as e:
                logger.error(f"关闭ZLMediaKit桥接器时出错: {str(e)}")
                logger.error(traceback.format_exc())

        logger.info("流管理器已关闭")

# 创建单例实例
stream_manager = StreamManager()

