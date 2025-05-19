# core/stream_manager.py
import asyncio
import time
import threading
from typing import Dict, Optional, Tuple, Any, List
import traceback

# 导入共享工具
from shared.utils.logger import get_normal_logger, get_exception_logger

# 导入状态定义
from .status import StreamStatus, StreamHealthStatus

# 导入流接口
from .interface import IVideoStream

# 导入ZLMediaKit流实现
from core.media_kit.zlm_stream import ZLMVideoStream

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)


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
        
        normal_logger.info("流管理器初始化完成，使用ZLMediaKit作为流媒体引擎")

    async def initialize(self):
        """初始化流管理器，包括ZLMediaKit管理器"""
        try:
            # 导入ZLMediaKit管理器
            from core.media_kit.zlm_manager import zlm_manager

            # 初始化ZLMediaKit管理器
            await zlm_manager.initialize()

            normal_logger.info("ZLMediaKit管理器初始化成功")
        except Exception as e:
            exception_logger.exception(f"初始化ZLMediaKit管理器失败: {str(e)}")

        normal_logger.info("流管理器初始化完成")

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
                normal_logger.info(f"流ID为空，使用任务ID({task_id})生成流ID: {stream_id}")
            else:
                # 如果没有任务ID，则使用随机ID
                stream_id = f"stream_{time.time()}"
                normal_logger.info(f"流ID为空，生成临时ID: {stream_id}")
            # 更新配置中的stream_id
            config["stream_id"] = stream_id

        # 创建当前事件循环的锁
        async with asyncio.Lock():
            # 检查流是否已存在
            if stream_id in self._streams:
                normal_logger.info(f"返回已存在的视频流: {stream_id}")
                return self._streams[stream_id]

            # 导入ZLMediaKit管理器
            from core.media_kit.zlm_manager import zlm_manager

            # 创建新的ZLMediaKit视频流
            normal_logger.info(f"创建新的ZLMediaKit视频流: {stream_id}")
            stream = ZLMVideoStream(stream_id, config, zlm_manager)

            # 添加到管理器
            self._streams[stream_id] = stream

            # 启动流
            try:
                await stream.start()
            except Exception as e:
                exception_logger.exception(f"启动流失败: {str(e)}")
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
        # 创建当前事件循环的锁
        async with asyncio.Lock():
            if stream_id not in self._streams:
                normal_logger.warning(f"尝试释放不存在的流: {stream_id}")
                return False

            stream = self._streams[stream_id]

            # 如果没有订阅者，停止并移除流
            if stream.subscriber_count == 0:
                normal_logger.info(f"流 {stream_id} 没有订阅者，停止并移除")

                # 停止流
                await stream.stop()

                # 从管理器移除
                del self._streams[stream_id]
                return True

            normal_logger.info(f"流 {stream_id} 仍有 {stream.subscriber_count} 个订阅者，保持运行")
            return False

    async def get_stream(self, stream_id: str) -> Optional[IVideoStream]:
        """获取视频流

        Args:
            stream_id: 流ID

        Returns:
            Optional[IVideoStream]: 视频流对象，如果不存在则返回None
        """
        # 创建当前事件循环的锁
        async with asyncio.Lock():
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
            normal_logger.info(f"流ID为空，使用订阅者ID({subscriber_id})生成流ID: {stream_id}")
            # 更新配置中的stream_id
            config["stream_id"] = stream_id

        # 直接使用ZLMediaKit流

        # 获取或创建流
        try:
            stream = await self.get_or_create_stream(stream_id, config)

            # 订阅流
            success, queue = await stream.subscribe(subscriber_id)

            return success, queue
        except Exception as e:
            exception_logger.exception(f"获取或订阅流 {stream_id} 时出错: {str(e)}")
            return False, None

    async def unsubscribe_stream(self, stream_id: str, subscriber_id: str) -> bool:
        """取消订阅视频流

        Args:
            stream_id: 流ID
            subscriber_id: 订阅者ID

        Returns:
            bool: 是否成功取消订阅
        """
        # 直接使用ZLMediaKit流

        async with asyncio.Lock():
            if stream_id not in self._streams:
                normal_logger.warning(f"尝试取消订阅不存在的流: {stream_id}")
                return False

            stream = self._streams[stream_id]

            # 取消订阅
            result = await stream.unsubscribe(subscriber_id)

            # 检查是否可以释放流
            if result and stream.subscriber_count == 0:
                await self.release_stream(stream_id)

            return result



    async def update_stream_status(self, stream_id: str, status: StreamStatus, health_status: StreamHealthStatus, error_msg: str = "") -> None:
        """更新流状态

        Args:
            stream_id: 流ID
            status: 流状态
            health_status: 健康状态
            error_msg: 错误信息
        """
        async with asyncio.Lock():
            if stream_id not in self._streams:
                normal_logger.warning(f"尝试更新不存在的流状态: {stream_id}")
                return

            stream = self._streams[stream_id]

            # 更新状态
            stream.set_status(status)
            stream.set_health_status(health_status)

            # 记录错误
            if error_msg:
                stream.set_last_error(error_msg)

            normal_logger.info(f"更新流状态: {stream_id}, 状态: {status.name}, 健康状态: {health_status.name}")

    async def reconnect_stream(self, stream_id: str) -> bool:
        """重新连接流

        Args:
            stream_id: 流ID

        Returns:
            bool: 是否成功重新连接
        """
        async with asyncio.Lock():
            if stream_id not in self._streams:
                normal_logger.warning(f"尝试重新连接不存在的流: {stream_id}")
                return False

            stream = self._streams[stream_id]

            # 获取流信息
            info = await stream.get_info()
            stream_config = {
                "url": info.get("url", ""),
                "stream_id": stream_id
            }

            try:
                # 先停止流
                normal_logger.info(f"重新连接流 {stream_id}，先停止当前流")
                await stream.stop()

                # 等待一段时间
                await asyncio.sleep(1)

                # 重新启动流
                normal_logger.info(f"重新启动流 {stream_id}")
                success = await stream.start()

                if success:
                    normal_logger.info(f"成功重新连接流 {stream_id}")
                    return True
                else:
                    normal_logger.error(f"重新连接流 {stream_id} 失败")
                    return False

            except Exception as e:
                exception_logger.exception(f"重新连接流 {stream_id} 时出错: {str(e)}")
                return False

    async def get_all_streams(self) -> List[Dict[str, Any]]:
        """获取所有流信息

        Returns:
            List[Dict[str, Any]]: 流信息列表
        """
        async with asyncio.Lock():
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

            # 流不存在

            normal_logger.warning(f"流 {stream_id} 不存在或无法获取流信息")
            return None

        except Exception as e:
            exception_logger.exception(f"获取流信息时发生错误: {str(e)}")
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
        # 创建当前事件循环的锁
        async with asyncio.Lock():
            for stream_id, stream in list(self._streams.items()):
                normal_logger.info(f"停止流: {stream_id}")
                await stream.stop()

            self._streams.clear()

    async def shutdown(self):
        """关闭流管理器"""
        # 停止所有流
        await self.stop_all_streams()

        # 关闭ZLMediaKit管理器
        try:
            from core.media_kit.zlm_manager import zlm_manager
            await zlm_manager.shutdown()
            normal_logger.info("ZLMediaKit管理器已关闭")
        except Exception as e:
            exception_logger.exception(f"关闭ZLMediaKit管理器时出错: {str(e)}")

        normal_logger.info("流管理器已关闭")

# 创建单例实例
stream_manager = StreamManager()

