"""
ZLMediaKit流模块
负责管理单个媒体流的拉取、推送和分发
"""
import time
import asyncio
import threading
import traceback
import ctypes
from typing import Dict, Any, Optional, Tuple, Set
import cv2
import numpy as np
from asyncio import Lock

from loguru import logger
from shared.utils.logger import setup_logger
from core.task_management.stream.status import StreamStatus, StreamHealthStatus
from core.task_management.stream.interface import IVideoStream
from core.task_management.stream.base_stream import BaseVideoStream

logger = setup_logger(__name__)

class ZLMVideoStream(BaseVideoStream):
    """ZLMediaKit视频流类，负责管理单个流的生命周期和帧处理"""

    def __init__(self, stream_id: str, config: Dict[str, Any], manager, player_handle=None):
        """初始化ZLMediaKit流

        Args:
            stream_id: 流ID
            config: 流配置
            manager: ZLMediaKit管理器实例
            player_handle: ZLMediaKit播放器句柄，如果为None则会在start时创建
        """
        # 调用基类初始化
        super().__init__(stream_id, config)

        self.manager = manager
        self.stream_type = config.get("type", "rtsp")

        # 播放器句柄
        self._player_handle = player_handle
        self._use_sdk = player_handle is not None or (self.manager._lib and hasattr(self.manager._lib, 'mk_player_create'))

        # 推流相关
        self.app = config.get("app", "live")
        self.vhost = config.get("vhost", "__defaultVhost__")
        self.stream_name = config.get("stream_name", stream_id)

        # 帧缓存
        self._frame_buffer = []
        self._frame_buffer_size = config.get("frame_buffer_size", 5)
        self._frame_lock = threading.Lock()

        # 流控制
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

        # 帧回调相关
        self._frame_callback_registered = False
        self._frame_callback_ref = None  # 保持回调函数的引用，防止被垃圾回收

        logger.info(f"创建ZLM流: {stream_id}, URL: {self._url}, 类型: {self.stream_type}, 使用SDK: {self._use_sdk}")

    async def start(self) -> bool:
        """启动流

        Returns:
            bool: 是否成功启动
        """
        if self._is_running:
            logger.warning(f"流 {self.stream_id} 已经在运行中")
            return True

        logger.info(f"启动流 {self.stream_id}")

        # 设置状态
        self._status = StreamStatus.INITIALIZING  # 使用INITIALIZING代替CONNECTING
        self._health_status = StreamHealthStatus.UNHEALTHY  # 使用UNHEALTHY代替UNKNOWN
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

        # 启动流程
        try:
            # 根据流类型选择不同的方法拉流
            success = False
            if self.stream_type == "rtsp":
                success = await self._start_rtsp()
            elif self.stream_type == "rtmp":
                success = await self._start_rtmp()
            elif self.stream_type == "hls":
                success = await self._start_hls()
            elif self.stream_type == "http":
                success = await self._start_http()
            else:
                logger.error(f"不支持的流类型: {self.stream_type}")
                self._status = StreamStatus.ERROR
                self._last_error = f"不支持的流类型: {self.stream_type}"
                return False

            if not success:
                logger.error(f"启动流 {self.stream_id} 失败")
                self._status = StreamStatus.ERROR
                return False

            # 启动帧处理任务
            self._is_running = True
            self._stop_event.clear()
            self._pull_task = asyncio.create_task(self._pull_stream_task())
            self._frame_task = asyncio.create_task(self._process_frames_task())

            # 设置状态为运行中
            self._status = StreamStatus.ONLINE  # 使用ONLINE代替RUNNING
            logger.info(f"流 {self.stream_id} 启动成功")
            return True
        except Exception as e:
            logger.error(f"启动流 {self.stream_id} 时出错: {str(e)}")
            self._status = StreamStatus.ERROR
            self._last_error = str(e)
            return False

    async def stop(self) -> bool:
        """停止流

        Returns:
            bool: 是否成功停止
        """
        if not self._is_running:
            logger.warning(f"流 {self.stream_id} 未运行，无需停止")
            return True

        logger.info(f"停止流 {self.stream_id}")

        # 设置停止事件
        self._stop_event.set()
        self._is_running = False

        # 等待任务结束
        try:
            if self._pull_task:
                await asyncio.wait_for(self._pull_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"等待流 {self.stream_id} 拉流任务停止超时")
        except Exception as e:
            logger.error(f"等待流 {self.stream_id} 拉流任务停止时出错: {str(e)}")

        try:
            if self._frame_task:
                await asyncio.wait_for(self._frame_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"等待流 {self.stream_id} 帧处理任务停止超时")
        except Exception as e:
            logger.error(f"等待流 {self.stream_id} 帧处理任务停止时出错: {str(e)}")

        # 停止ZLM播放器
        if self._player_handle:
            try:
                # 使用C API释放播放器
                if self.manager._lib and hasattr(self.manager._lib, 'mk_player_release'):
                    logger.info(f"使用C API释放播放器: {self.stream_id}")
                    self.manager._lib.mk_player_release(self._player_handle)
                else:
                    logger.error(f"无法释放播放器: {self.stream_id}，C API不可用")

                # 清除播放器句柄
                self._player_handle = None
                self._frame_callback_registered = False
                self._frame_callback_ref = None
            except Exception as e:
                logger.error(f"停止ZLM播放器时出错: {str(e)}")
                logger.error(traceback.format_exc())

        # 清理订阅者
        async with self._subscriber_lock:
            self._subscribers.clear()

        # 清理帧缓存
        with self._frame_lock:
            self._frame_buffer.clear()

        # 设置状态
        self._status = StreamStatus.OFFLINE  # 使用OFFLINE代替STOPPED

        logger.info(f"流 {self.stream_id} 停止成功")
        return True

    async def get_info(self) -> Dict[str, Any]:
        """获取流信息

        Returns:
            Dict[str, Any]: 流信息
        """
        return {
            "stream_id": self.stream_id,
            "url": self.url,
            "type": self.stream_type,
            "status": self._status.value,
            "health_status": self._health_status.value,
            "last_error": self._last_error,
            "subscriber_count": len(self._subscribers),
            "stats": self._stats
        }

    def get_status(self) -> StreamStatus:
        """获取流状态"""
        return self._status

    def get_health_status(self) -> StreamHealthStatus:
        """获取流健康状态"""
        return self._health_status

    def set_status(self, status: StreamStatus) -> None:
        """设置流状态"""
        if self._status != status:
            logger.info(f"流 {self.stream_id} 状态变更: {self._status.name} -> {status.name}")
            self._status = status

    def set_health_status(self, health_status: StreamHealthStatus) -> None:
        """设置流健康状态"""
        if self._health_status != health_status:
            logger.info(f"流 {self.stream_id} 健康状态变更: {self._health_status.name} -> {health_status.name}")
            self._health_status = health_status

    @property
    def subscriber_count(self) -> int:
        """获取订阅者数量"""
        return len(self._subscribers)

    @property
    def subscribers(self) -> Set[str]:
        """获取订阅者ID集合"""
        return set(self._subscribers.keys())

    async def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """获取最新的帧

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (是否成功, 帧数据)
        """
        # 检查流状态
        if not self._is_running or self._status != StreamStatus.RUNNING:
            return False, None

        # 获取最新帧
        with self._frame_lock:
            if not self._frame_buffer:
                return False, None
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
            logger.error(f"生成快照时出错: {str(e)}")
            return None

    async def subscribe(self, subscriber_id: str) -> Tuple[bool, asyncio.Queue]:
        """订阅流

        Args:
            subscriber_id: 订阅者ID

        Returns:
            Tuple[bool, asyncio.Queue]: (是否成功, 帧队列)
        """
        if not self._is_running:
            logger.warning(f"流 {self.stream_id} 未运行，无法订阅")
            return False, None

        logger.info(f"订阅者 {subscriber_id} 已订阅流 {self.stream_id}")

        # 检查流状态
        current_status = self._status
        logger.info(f"流 {self.stream_id} 当前状态: {current_status}")

        # 创建帧队列
        buffer_size = self.config.get("queue_size", 10)
        logger.info(f"创建帧队列，队列大小: {buffer_size}")
        frame_queue = asyncio.Queue(maxsize=buffer_size)

        # 添加订阅者
        async with self._subscriber_lock:
            self._subscribers[subscriber_id] = frame_queue
            logger.info(f"当前订阅者数量: {len(self._subscribers)}")

        # 检查帧缓存
        with self._frame_lock:
            frame_count = len(self._frame_buffer)
            logger.info(f"当前帧缓存帧数: {frame_count}")

        logger.info(f"订阅者 {subscriber_id} 已订阅流 {self.stream_id}")
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
                logger.info(f"订阅者 {subscriber_id} 已取消订阅流 {self.stream_id}")
                return True

        logger.warning(f"订阅者 {subscriber_id} 未订阅流 {self.stream_id}")
        return False

    def _register_frame_callback(self) -> bool:
        """注册帧回调函数

        Returns:
            bool: 是否成功注册
        """
        if not self._player_handle or not self.manager._lib or self._frame_callback_registered:
            return False

        try:
            # 检查是否支持帧回调
            if not hasattr(self.manager._lib, 'mk_player_set_on_data'):
                logger.warning("ZLMediaKit库不支持帧回调")
                return False

            # 定义帧回调函数
            @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_uint64)
            def on_frame(user_data, track_type, data, len_data, pts):
                try:
                    # 只处理视频帧
                    if track_type != 0:  # 0表示视频
                        return

                    # 将数据转换为numpy数组
                    if data and len_data > 0:
                        # 复制数据，避免数据被释放
                        buffer = ctypes.string_at(data, len_data)

                        # 解码帧
                        try:
                            # 使用OpenCV解码
                            frame_array = np.frombuffer(buffer, dtype=np.uint8)
                            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                            if frame is not None:
                                # 更新缓存
                                with self._frame_lock:
                                    self._frame_buffer.append(frame)
                                    if len(self._frame_buffer) > self._frame_buffer_size:
                                        self._frame_buffer.pop(0)

                                # 更新统计
                                now = time.time()
                                self._stats["frames_received"] += 1
                                self._stats["last_frame_time"] = now
                                self._last_frame_time = now

                                # 更新状态
                                self._status = StreamStatus.ONLINE
                                self._health_status = StreamHealthStatus.HEALTHY

                                # 通知帧处理任务
                                asyncio.run_coroutine_threadsafe(
                                    self._notify_frame_processed(frame, now),
                                    asyncio.get_event_loop()
                                )
                        except Exception as e:
                            logger.error(f"解码帧时出错: {str(e)}")
                except Exception as e:
                    logger.error(f"处理帧回调时出错: {str(e)}")

            # 保存回调引用
            self._frame_callback_ref = on_frame

            # 注册回调
            self.manager._lib.mk_player_set_on_data(self._player_handle, self._frame_callback_ref, None)

            # 标记为已注册
            self._frame_callback_registered = True

            logger.info(f"成功注册帧回调: {self.stream_id}")
            return True
        except Exception as e:
            logger.error(f"注册帧回调时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    async def _notify_frame_processed(self, frame, timestamp):
        """通知帧处理完成

        Args:
            frame: 视频帧
            timestamp: 时间戳
        """
        try:
            # 设置事件
            self._frame_processed_event.set()

            # 分发帧给订阅者
            subscribers = {}
            async with self._subscriber_lock:
                subscribers = self._subscribers.copy()

            for subscriber_id, queue in subscribers.items():
                try:
                    # 如果队列已满，则丢弃旧帧
                    if queue.full():
                        try:
                            _ = queue.get_nowait()
                        except:
                            pass

                    # 放入新帧
                    await queue.put((frame.copy(), timestamp))
                except Exception as e:
                    logger.error(f"向订阅者 {subscriber_id} 分发帧时出错: {str(e)}")
        except Exception as e:
            logger.error(f"通知帧处理完成时出错: {str(e)}")

    async def _start_rtsp(self) -> bool:
        """启动RTSP流

        Returns:
            bool: 是否成功启动
        """
        try:
            # 如果已经有播放器句柄，直接返回成功
            if self._player_handle:
                logger.info(f"流 {self.stream_id} 已有播放器句柄，直接使用")
                return True

            # 尝试使用C API创建播放器
            if self._use_sdk and self.manager._lib:
                try:
                    # 转换为C字符串
                    url_c = ctypes.c_char_p(self.url.encode('utf-8'))

                    # 创建播放器
                    logger.info(f"使用C API创建播放器: {self.url}")
                    self._player_handle = self.manager._lib.mk_player_create()

                    if not self._player_handle:
                        logger.error("创建播放器失败")
                        return False

                    # 注册帧回调
                    self._register_frame_callback()

                    # 设置播放器参数
                    if hasattr(self.manager._lib, 'mk_player_set_option'):
                        # 设置RTSP传输模式
                        rtsp_transport = self.config.get("rtsp_transport", "tcp")
                        if rtsp_transport == "tcp":
                            self.manager._lib.mk_player_set_option(self._player_handle, b"rtsp_transport", b"tcp")

                        # 设置缓冲时间
                        buffer_ms = str(self.config.get("buffer_ms", 200)).encode('utf-8')
                        self.manager._lib.mk_player_set_option(self._player_handle, b"buffer_ms", buffer_ms)

                    # 播放URL
                    result = self.manager._lib.mk_player_play_url(self._player_handle, url_c)
                    if result != 0:
                        logger.error(f"播放URL失败: {result}")
                        self.manager._lib.mk_player_release(self._player_handle)
                        self._player_handle = None
                        return False

                    logger.info(f"成功使用C API创建播放器: {self.url}")
                    return True

                except Exception as e:
                    logger.error(f"使用C API创建播放器时出错: {str(e)}")
                    logger.error(traceback.format_exc())

                    # 清理资源
                    if self._player_handle:
                        try:
                            self.manager._lib.mk_player_release(self._player_handle)
                        except:
                            pass
                        self._player_handle = None

                    logger.error("C API创建流失败，无法继续")
                    return False

            # C API不可用，无法创建流
            logger.error(f"无法创建流 {self.stream_name}，C API不可用")
            return False
        except Exception as e:
            logger.error(f"启动RTSP流时出错: {str(e)}")
            return False

    async def _start_rtmp(self) -> bool:
        """启动RTMP流

        Returns:
            bool: 是否成功启动
        """
        # 与RTSP逻辑相似，使用ZLMediaKit API
        return await self._start_rtsp()  # 复用RTSP启动逻辑

    async def _start_hls(self) -> bool:
        """启动HLS流

        Returns:
            bool: 是否成功启动
        """
        # HLS流处理逻辑
        # 使用ZLMediaKit API
        return await self._start_rtsp()  # 复用RTSP启动逻辑

    async def _start_http(self) -> bool:
        """启动HTTP流(如MJPEG)

        Returns:
            bool: 是否成功启动
        """
        # HTTP流处理逻辑
        return await self._start_rtsp()  # 复用RTSP启动逻辑

    async def _pull_stream_task(self) -> None:
        """拉流任务，负责从ZLMediaKit获取视频帧"""
        try:
            # 配置重连参数
            max_retry = self.config.get("max_retry", 3)
            retry_interval = self.config.get("retry_interval", 5)
            retry_count = 0

            # 创建OpenCV捕获对象
            # 这里我们暂时使用OpenCV直接拉流，在实际集成中应该使用ZLMediaKit API
            proxied_url = f"rtsp://{self.manager._config.server_address}:{self.manager._config.rtsp_port}/{self.app}/{self.stream_name}"

            logger.info(f"拉流URL: {proxied_url}")

            while not self._stop_event.is_set():
                try:
                    logger.info(f"开始拉流 {proxied_url}")
                    cap = cv2.VideoCapture(proxied_url)

                    if not cap.isOpened():
                        logger.error(f"无法打开流: {proxied_url}")
                        self._status = StreamStatus.ERROR
                        self._health_status = StreamHealthStatus.OFFLINE
                        self._last_error = "无法打开流"

                        # 重试
                        retry_count += 1
                        if retry_count > max_retry:
                            logger.error(f"超过最大重试次数({max_retry})，停止拉流")
                            break

                        logger.info(f"等待 {retry_interval} 秒后重试 ({retry_count}/{max_retry})...")
                        await asyncio.sleep(retry_interval)
                        continue

                    # 重置重试计数
                    retry_count = 0
                    self._status = StreamStatus.ONLINE  # 使用ONLINE代替RUNNING
                    self._health_status = StreamHealthStatus.HEALTHY  # 使用HEALTHY代替GOOD

                    # 读取帧
                    frame_count = 0
                    error_count = 0
                    max_consecutive_errors = self.config.get("max_consecutive_errors", 10)
                    frame_interval = 0.01  # 10ms
                    last_frame_time = time.time()

                    while not self._stop_event.is_set():
                        current_time = time.time()

                        # 控制帧率
                        time_diff = current_time - last_frame_time
                        if time_diff < frame_interval:
                            await asyncio.sleep(frame_interval - time_diff)
                            continue

                        # 读取帧
                        ret, frame = cap.read()
                        last_frame_time = time.time()

                        if not ret:
                            error_count += 1
                            logger.warning(f"读取帧失败，连续错误: {error_count}/{max_consecutive_errors}")

                            if error_count >= max_consecutive_errors:
                                logger.error(f"连续错误达到阈值({max_consecutive_errors})，重新连接")
                                break

                            await asyncio.sleep(0.1)
                            continue

                        # 重置错误计数
                        error_count = 0

                        # 处理帧
                        now = time.time()

                        # 更新缓存
                        with self._frame_lock:
                            self._frame_buffer.append(frame)
                            if len(self._frame_buffer) > self._frame_buffer_size:
                                self._frame_buffer.pop(0)

                        # 更新统计
                        self._stats["frames_received"] += 1
                        self._stats["last_frame_time"] = now

                        # 计算FPS
                        frame_count += 1
                        if frame_count % 30 == 0:
                            elapsed = now - self._stats.get("fps_calc_time", self._stats["start_time"])
                            if elapsed > 0:
                                self._stats["fps"] = frame_count / elapsed
                                self._stats["fps_calc_time"] = now
                                frame_count = 0

                        # 通知帧处理任务
                        self._frame_processed_event.set()

                        # 分发帧给订阅者
                        subscribers = {}
                        async with self._subscriber_lock:
                            subscribers = self._subscribers.copy()

                        timestamp = now
                        for subscriber_id, queue in subscribers.items():
                            try:
                                # 如果队列已满，则丢弃旧帧
                                if queue.full():
                                    try:
                                        _ = queue.get_nowait()
                                    except:
                                        pass

                                # 放入新帧 - 注意这里调整为(frame, timestamp)的元组格式，以匹配任务处理器期望的格式
                                logger.debug(f"分发帧给订阅者 {subscriber_id}, 帧大小: {frame.shape}")
                                await queue.put((frame.copy(), timestamp))
                            except Exception as e:
                                logger.error(f"向订阅者 {subscriber_id} 分发帧时出错: {str(e)}")

                    # 关闭捕获
                    cap.release()

                    # 如果是停止事件触发的，则退出循环
                    if self._stop_event.is_set():
                        break

                    # 否则重新连接
                    logger.info("重新连接流...")
                    self._stats["reconnects"] += 1
                    self._status = StreamStatus.INITIALIZING  # 使用INITIALIZING代替RECONNECTING
                    await asyncio.sleep(retry_interval)

                except Exception as e:
                    logger.error(f"拉流任务异常: {str(e)}")
                    self._stats["errors"] += 1
                    self._status = StreamStatus.ERROR  # ERROR是正确的
                    self._health_status = StreamHealthStatus.UNHEALTHY  # 使用UNHEALTHY代替ERROR
                    self._last_error = str(e)

                    # 重试
                    retry_count += 1
                    if retry_count > max_retry:
                        logger.error(f"超过最大重试次数({max_retry})，停止拉流")
                        break

                    logger.info(f"等待 {retry_interval} 秒后重试 ({retry_count}/{max_retry})...")
                    await asyncio.sleep(retry_interval)

            # 任务结束
            logger.info(f"拉流任务结束: {self.stream_id}")
            self._status = StreamStatus.OFFLINE  # 使用OFFLINE代替STOPPED

        except Exception as e:
            logger.error(f"拉流任务发生异常: {str(e)}")
            self._status = StreamStatus.ERROR  # ERROR是正确的
            self._health_status = StreamHealthStatus.UNHEALTHY  # 使用UNHEALTHY代替ERROR
            self._last_error = str(e)

    async def _process_frames_task(self) -> None:
        """帧处理任务，负责将帧分发给订阅者"""
        try:
            last_frame_index = -1

            while not self._stop_event.is_set():
                # 等待新帧
                await self._frame_processed_event.wait()
                self._frame_processed_event.clear()

                # 获取最新帧索引
                with self._frame_lock:
                    current_frame_count = len(self._frame_buffer)

                # 如果没有新帧，则等待
                if current_frame_count <= last_frame_index:
                    await asyncio.sleep(0.01)
                    continue

                # 获取订阅者列表
                subscribers = {}
                async with self._subscriber_lock:
                    subscribers = self._subscribers.copy()

                # 分发帧给订阅者
                for i in range(last_frame_index + 1, current_frame_count):
                    # 获取帧
                    frame = None
                    timestamp = time.time()
                    with self._frame_lock:
                        if i >= len(self._frame_buffer):
                            break
                        frame = self._frame_buffer[i].copy()

                    # 分发给所有订阅者
                    for subscriber_id, queue in subscribers.items():
                        try:
                            # 如果队列已满，则丢弃旧帧
                            if queue.full():
                                try:
                                    _ = queue.get_nowait()
                                except:
                                    pass

                            # 放入新帧 - 注意这里调整为(frame, timestamp)的元组格式，以匹配任务处理器期望的格式
                            await queue.put((frame, timestamp))
                        except Exception as e:
                            logger.error(f"向订阅者 {subscriber_id} 分发帧时出错: {str(e)}")

                    # 更新统计
                    self._stats["frames_processed"] += 1

                # 更新最后处理的帧索引
                last_frame_index = current_frame_count - 1

                # 短暂休眠，避免CPU占用过高
                await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"帧处理任务异常: {str(e)}")
            self._status = StreamStatus.ERROR
            self._last_error = str(e)