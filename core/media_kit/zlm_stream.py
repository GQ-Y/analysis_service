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

from shared.utils.logger import get_normal_logger, get_exception_logger
from core.task_management.stream.status import StreamStatus, StreamHealthStatus
from core.task_management.stream.interface import IVideoStream
from core.task_management.stream.base_stream import BaseVideoStream

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

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
        self._frame_callback_ref = None  # 保持帧回调函数的引用，防止被垃圾回收
        self._play_result_callback_ref = None  # 保持播放结果回调函数的引用，防止被垃圾回收
        self._play_shutdown_callback_ref = None  # 保持播放中断回调函数的引用，防止被垃圾回收

        normal_logger.info(f"创建ZLM流: {stream_id}, URL: {self._url}, 类型: {self.stream_type}, 使用SDK: {self._use_sdk}")

    async def start(self) -> bool:
        """启动流

        Returns:
            bool: 是否成功启动
        """
        if self._is_running:
            normal_logger.warning(f"流 {self.stream_id} 已经在运行中")
            return True

        normal_logger.info(f"启动流 {self.stream_id}")

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
                normal_logger.error(f"不支持的流类型: {self.stream_type}")
                self._status = StreamStatus.ERROR
                self._last_error = f"不支持的流类型: {self.stream_type}"
                return False

            if not success:
                normal_logger.error(f"启动流 {self.stream_id} 失败")
                self._status = StreamStatus.ERROR
                return False

            # 启动帧处理任务
            self._is_running = True
            self._stop_event.clear()
            self._pull_task = asyncio.create_task(self._pull_stream_task())
            self._frame_task = asyncio.create_task(self._process_frames_task())

            # 设置状态为连接中，等待实际收到帧后再更新为ONLINE
            self._status = StreamStatus.CONNECTING
            normal_logger.info(f"流 {self.stream_id} 启动成功，等待连接...")
            return True
        except Exception as e:
            normal_logger.error(f"启动流 {self.stream_id} 时出错: {str(e)}")
            self._status = StreamStatus.ERROR
            self._last_error = str(e)
            return False

    async def stop(self) -> bool:
        """停止流

        Returns:
            bool: 是否成功停止
        """
        if not self._is_running:
            normal_logger.warning(f"流 {self.stream_id} 未运行，无需停止")
            return True

        normal_logger.info(f"停止流 {self.stream_id}")

        # 设置停止事件
        self._stop_event.set()
        self._is_running = False

        # 等待任务结束
        try:
            if self._pull_task:
                await asyncio.wait_for(self._pull_task, timeout=5.0)
        except asyncio.TimeoutError:
            normal_logger.warning(f"等待流 {self.stream_id} 拉流任务停止超时")
        except Exception as e:
            normal_logger.error(f"等待流 {self.stream_id} 拉流任务停止时出错: {str(e)}")

        try:
            if self._frame_task:
                await asyncio.wait_for(self._frame_task, timeout=5.0)
        except asyncio.TimeoutError:
            normal_logger.warning(f"等待流 {self.stream_id} 帧处理任务停止超时")
        except Exception as e:
            normal_logger.error(f"等待流 {self.stream_id} 帧处理任务停止时出错: {str(e)}")

        # 停止ZLM播放器
        if self._player_handle:
            try:
                # 注意：由于在释放播放器时遇到段错误，我们暂时跳过播放器释放步骤
                # 只清除引用，让Python的垃圾回收机制处理
                normal_logger.info(f"跳过播放器释放步骤，只清除引用: {self.stream_id}")

                # 如果在未来需要重新启用播放器释放，可以取消注释以下代码
                """
                # 使用C API释放播放器
                if self.manager._lib:
                    try:
                        # 先尝试停止播放
                        if hasattr(self.manager._lib, 'mk_player_stop'):
                            # 设置mk_player_stop的参数类型
                            self.manager._lib.mk_player_stop.argtypes = [ctypes.c_void_p]

                            normal_logger.info(f"使用C API停止播放器: {self.stream_id}")
                            # 检查播放器句柄是否有效
                            if self._player_handle and int(self._player_handle) != 0:
                                self.manager._lib.mk_player_stop(self._player_handle)
                                # 等待一小段时间，确保停止操作完成
                                time.sleep(0.5)
                            else:
                                normal_logger.warning(f"播放器句柄无效，跳过停止: {self.stream_id}")

                        # 然后释放播放器
                        if hasattr(self.manager._lib, 'mk_player_release'):
                            # 设置mk_player_release的参数类型
                            self.manager._lib.mk_player_release.argtypes = [ctypes.c_void_p]

                            normal_logger.info(f"使用C API释放播放器: {self.stream_id}")
                            # 检查播放器句柄是否有效
                            if self._player_handle and int(self._player_handle) != 0:
                                # 将播放器句柄转换为整数，然后再转换为c_void_p
                                handle_int = int(self._player_handle)
                                handle_ptr = ctypes.c_void_p(handle_int)
                                self.manager._lib.mk_player_release(handle_ptr)
                            else:
                                normal_logger.warning(f"播放器句柄无效，跳过释放: {self.stream_id}")
                    except Exception as e:
                        normal_logger.error(f"释放播放器时出错: {str(e)}")
                        normal_logger.error(traceback.format_exc())
                else:
                    normal_logger.error(f"无法释放播放器: {self.stream_id}，C API不可用")
                """

                # 清除播放器句柄
                self._player_handle = None
                self._frame_callback_registered = False
                self._frame_callback_ref = None
                self._play_result_callback_ref = None
                self._play_shutdown_callback_ref = None
            except Exception as e:
                normal_logger.error(f"停止ZLM播放器时出错: {str(e)}")
                normal_logger.error(traceback.format_exc())

        # 清理订阅者
        async with self._subscriber_lock:
            self._subscribers.clear()

        # 清理帧缓存
        with self._frame_lock:
            self._frame_buffer.clear()

        # 设置状态
        self._status = StreamStatus.OFFLINE  # 使用OFFLINE代替STOPPED
        self._health_status = StreamHealthStatus.OFFLINE  # 确保健康状态也被更新
        self._is_running = False  # 确保运行状态被更新

        # 通知健康监控器
        try:
            from core.task_management.stream.health_monitor import stream_health_monitor
            stream_health_monitor.report_error(self.stream_id, "流已停止")
        except Exception as e:
            normal_logger.error(f"通知健康监控器时出错: {str(e)}")

        normal_logger.info(f"流 {self.stream_id} 停止成功")
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
            normal_logger.info(f"流 {self.stream_id} 状态变更: {self._status.name} -> {status.name}")
            self._status = status

    def set_health_status(self, health_status: StreamHealthStatus) -> None:
        """设置流健康状态"""
        if self._health_status != health_status:
            normal_logger.info(f"流 {self.stream_id} 健康状态变更: {self._health_status.name} -> {health_status.name}")
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
        # 检查流状态 - 允许多种有效状态
        valid_statuses = [StreamStatus.RUNNING, StreamStatus.ONLINE]
        if not self._is_running or self._status not in valid_statuses:
            normal_logger.debug(f"流 {self.stream_id} 状态不可用: is_running={self._is_running}, status={self._status}")

            # 如果流状态是CONNECTING或INITIALIZING，记录详细信息但仍返回标准格式
            if self._status in [StreamStatus.CONNECTING, StreamStatus.INITIALIZING]:
                normal_logger.debug(f"流 {self.stream_id} 正在连接中")

            return False, None

        # 获取最新帧
        with self._frame_lock:
            if not self._frame_buffer:
                normal_logger.debug(f"流 {self.stream_id} 帧缓冲区为空")
                # 检查帧缓冲区为空的原因
                if self._stats.get("frames_received", 0) == 0:
                    # 从未接收到帧
                    normal_logger.debug(f"流 {self.stream_id} 从未接收到帧")
                else:
                    # 曾经接收到帧，但当前缓冲区为空
                    normal_logger.debug(f"流 {self.stream_id} 曾经接收到帧，但当前缓冲区为空")

                return False, None

            # 成功获取帧
            frame = self._frame_buffer[-1].copy()
            normal_logger.debug(f"流 {self.stream_id} 成功获取帧，大小: {frame.shape}")

            # 更新最后获取帧的时间
            self._stats["last_frame_get_time"] = time.time()

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
            normal_logger.error(f"生成快照时出错: {str(e)}")
            return None

    async def subscribe(self, subscriber_id: str) -> Tuple[bool, asyncio.Queue]:
        """订阅流

        Args:
            subscriber_id: 订阅者ID

        Returns:
            Tuple[bool, asyncio.Queue]: (是否成功, 帧队列)
        """
        if not self._is_running:
            normal_logger.warning(f"流 {self.stream_id} 未运行，无法订阅")
            return False, None

        normal_logger.info(f"订阅者 {subscriber_id} 已订阅流 {self.stream_id}")

        # 检查流状态
        current_status = self._status
        normal_logger.info(f"流 {self.stream_id} 当前状态: {current_status}")

        # 创建帧队列
        buffer_size = self.config.get("queue_size", 10)
        normal_logger.info(f"创建帧队列，队列大小: {buffer_size}")
        frame_queue = asyncio.Queue(maxsize=buffer_size)

        # 添加订阅者
        async with self._subscriber_lock:
            self._subscribers[subscriber_id] = frame_queue
            normal_logger.info(f"当前订阅者数量: {len(self._subscribers)}")

        # 检查帧缓存
        with self._frame_lock:
            frame_count = len(self._frame_buffer)
            normal_logger.info(f"当前帧缓存帧数: {frame_count}")

        normal_logger.info(f"订阅者 {subscriber_id} 已订阅流 {self.stream_id}")
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
                normal_logger.info(f"订阅者 {subscriber_id} 已取消订阅流 {self.stream_id}")
                return True

        normal_logger.warning(f"订阅者 {subscriber_id} 未订阅流 {self.stream_id}")
        return False

    def _register_frame_callback(self) -> bool:
        """注册帧回调函数

        注意：在ZLMediaKit C API中，帧回调不是通过mk_player_set_on_data注册的，
        而是通过mk_player_set_on_result注册播放结果回调，然后在播放成功后
        通过mk_track_add_delegate为每个track注册帧回调。

        这个函数现在只负责注册播放结果回调，实际的帧回调会在播放成功后注册。

        Returns:
            bool: 是否成功注册
        """
        if not self._player_handle or not self.manager._lib or self._frame_callback_registered:
            return False

        try:
            # 检查是否支持播放结果回调
            if not hasattr(self.manager._lib, 'mk_player_set_on_result'):
                normal_logger.warning("ZLMediaKit库不支持播放结果回调")
                return False

            # 定义帧回调函数 - 这个会在播放成功后通过mk_track_add_delegate注册
            @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
            def on_track_frame(user_data, frame_ptr):
                try:
                    # 这里的frame_ptr是mk_frame类型，需要使用mk_frame_xxx函数获取数据
                    if not frame_ptr:
                        return

                    # 检查是否为视频帧
                    if hasattr(self.manager._lib, 'mk_frame_is_video'):
                        # 设置mk_frame_is_video的参数类型和返回值类型
                        self.manager._lib.mk_frame_is_video.argtypes = [ctypes.c_void_p]
                        self.manager._lib.mk_frame_is_video.restype = ctypes.c_int

                        is_video = self.manager._lib.mk_frame_is_video(frame_ptr)
                        if not is_video:
                            return

                    # 获取帧数据
                    if hasattr(self.manager._lib, 'mk_frame_get_data') and hasattr(self.manager._lib, 'mk_frame_get_data_size'):
                        # 设置mk_frame_get_data的参数类型和返回值类型
                        self.manager._lib.mk_frame_get_data.argtypes = [ctypes.c_void_p]
                        self.manager._lib.mk_frame_get_data.restype = ctypes.c_char_p

                        # 设置mk_frame_get_data_size的参数类型和返回值类型
                        self.manager._lib.mk_frame_get_data_size.argtypes = [ctypes.c_void_p]
                        self.manager._lib.mk_frame_get_data_size.restype = ctypes.c_size_t

                        data_ptr = self.manager._lib.mk_frame_get_data(frame_ptr)
                        data_size = self.manager._lib.mk_frame_get_data_size(frame_ptr)

                        if data_ptr and data_size > 0:
                            # 复制数据，避免数据被释放
                            buffer = ctypes.string_at(data_ptr, data_size)

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
                                normal_logger.error(f"解码帧时出错: {str(e)}")
                    else:
                        normal_logger.warning("ZLMediaKit库不支持获取帧数据")
                except Exception as e:
                    normal_logger.error(f"处理帧回调时出错: {str(e)}")
                    normal_logger.error(traceback.format_exc())

                return True

            # 定义播放结果回调函数
            @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int)
            def on_play_result(user_data, err_code, err_msg, tracks, track_count):
                try:
                    if err_code == 0:
                        # 播放成功
                        normal_logger.info(f"播放成功: {self.stream_id}")

                        # 为每个track注册帧回调
                        if track_count > 0 and tracks:
                            for i in range(track_count):
                                track = tracks[i]

                                # 检查是否为视频track
                                if hasattr(self.manager._lib, 'mk_track_is_video'):
                                    # 设置mk_track_is_video的参数类型和返回值类型
                                    self.manager._lib.mk_track_is_video.argtypes = [ctypes.c_void_p]
                                    self.manager._lib.mk_track_is_video.restype = ctypes.c_int

                                    # 检查是否为视频track
                                    if self.manager._lib.mk_track_is_video(track):
                                        normal_logger.info(f"找到视频track: {self.stream_id}")

                                        # 注册帧回调
                                        if hasattr(self.manager._lib, 'mk_track_add_delegate'):
                                            # 设置mk_track_add_delegate的参数类型
                                            self.manager._lib.mk_track_add_delegate.argtypes = [
                                                ctypes.c_void_p,  # track句柄
                                                ctypes.c_void_p,  # 回调函数
                                                ctypes.c_void_p   # 用户数据
                                            ]
                                            self.manager._lib.mk_track_add_delegate.restype = ctypes.c_void_p

                                            # 注册回调
                                            self.manager._lib.mk_track_add_delegate(track, on_track_frame, None)
                                            normal_logger.info(f"成功为视频track注册帧回调: {self.stream_id}")
                    else:
                        # 播放失败
                        err_msg_str = err_msg.decode('utf-8') if err_msg else "未知错误"
                        normal_logger.error(f"播放失败: {self.stream_id}, 错误码: {err_code}, 错误信息: {err_msg_str}")

                        # 更新状态
                        self._status = StreamStatus.ERROR
                        self._health_status = StreamHealthStatus.UNHEALTHY
                        self._last_error = f"播放失败: 错误码 {err_code}, 错误信息: {err_msg_str}"
                except Exception as e:
                    normal_logger.error(f"处理播放结果回调时出错: {str(e)}")
                    normal_logger.error(traceback.format_exc())

            # 定义播放中断回调函数
            @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int)
            def on_play_shutdown(user_data, err_code, err_msg, tracks, track_count):
                try:
                    # 播放中断
                    err_msg_str = err_msg.decode('utf-8') if err_msg else "未知错误"
                    normal_logger.warning(f"播放中断: {self.stream_id}, 错误码: {err_code}, 错误信息: {err_msg_str}")

                    # 更新状态
                    self._status = StreamStatus.ERROR
                    self._health_status = StreamHealthStatus.UNHEALTHY
                    self._last_error = f"播放中断: 错误码 {err_code}, 错误信息: {err_msg_str}"
                except Exception as e:
                    normal_logger.error(f"处理播放中断回调时出错: {str(e)}")
                    normal_logger.error(traceback.format_exc())

            # 保存回调引用，防止被垃圾回收
            self._frame_callback_ref = on_track_frame
            self._play_result_callback_ref = on_play_result
            self._play_shutdown_callback_ref = on_play_shutdown

            # 设置回调函数参数类型
            if hasattr(self.manager._lib, 'mk_player_set_on_result'):
                self.manager._lib.mk_player_set_on_result.argtypes = [
                    ctypes.c_void_p,  # 播放器句柄
                    ctypes.c_void_p,  # 回调函数
                    ctypes.c_void_p   # 用户数据
                ]
                # 注册播放结果回调
                self.manager._lib.mk_player_set_on_result(self._player_handle, self._play_result_callback_ref, None)
            else:
                normal_logger.warning("ZLMediaKit库不支持mk_player_set_on_result函数")

            # 设置中断回调函数参数类型
            if hasattr(self.manager._lib, 'mk_player_set_on_shutdown'):
                self.manager._lib.mk_player_set_on_shutdown.argtypes = [
                    ctypes.c_void_p,  # 播放器句柄
                    ctypes.c_void_p,  # 回调函数
                    ctypes.c_void_p   # 用户数据
                ]
                # 注册播放中断回调
                self.manager._lib.mk_player_set_on_shutdown(self._player_handle, self._play_shutdown_callback_ref, None)
            else:
                normal_logger.warning("ZLMediaKit库不支持mk_player_set_on_shutdown函数")

            # 标记为已注册
            self._frame_callback_registered = True

            normal_logger.info(f"成功注册播放结果回调: {self.stream_id}")
            return True
        except Exception as e:
            normal_logger.error(f"注册帧回调时出错: {str(e)}")
            normal_logger.error(traceback.format_exc())
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
                    normal_logger.error(f"向订阅者 {subscriber_id} 分发帧时出错: {str(e)}")
        except Exception as e:
            normal_logger.error(f"通知帧处理完成时出错: {str(e)}")

    async def _start_rtsp(self) -> bool:
        """启动RTSP流

        Returns:
            bool: 是否成功启动
        """
        try:
            # 如果已经有播放器句柄，直接返回成功
            if self._player_handle:
                normal_logger.info(f"流 {self.stream_id} 已有播放器句柄，直接使用")
                return True

            # 尝试使用C API创建播放器
            if self._use_sdk and self.manager._lib:
                try:
                    # 转换为C字符串
                    url_c = ctypes.c_char_p(self.url.encode('utf-8'))

                    # 设置函数参数和返回值类型
                    if not hasattr(self.manager._lib, 'mk_player_create'):
                        normal_logger.error("ZLMediaKit库不支持mk_player_create函数")
                        return False

                    # 设置mk_player_create的返回值类型
                    self.manager._lib.mk_player_create.restype = ctypes.c_void_p

                    # 创建播放器
                    normal_logger.info(f"使用C API创建播放器: {self.url}")
                    self._player_handle = self.manager._lib.mk_player_create()

                    if not self._player_handle:
                        normal_logger.error("创建播放器失败")
                        return False

                    # 注册帧回调
                    self._register_frame_callback()

                    # 设置播放器参数
                    if hasattr(self.manager._lib, 'mk_player_set_option'):
                        # 设置mk_player_set_option的参数类型
                        self.manager._lib.mk_player_set_option.argtypes = [
                            ctypes.c_void_p,  # 播放器句柄
                            ctypes.c_char_p,  # 选项名
                            ctypes.c_char_p   # 选项值
                        ]

                        # 设置RTSP传输模式
                        rtsp_transport = self.config.get("rtsp_transport", "tcp")
                        if rtsp_transport == "tcp":
                            self.manager._lib.mk_player_set_option(
                                self._player_handle,
                                "rtsp_transport".encode('utf-8'),
                                "tcp".encode('utf-8')
                            )

                        # 设置缓冲时间
                        buffer_ms = str(self.config.get("buffer_ms", 200)).encode('utf-8')
                        self.manager._lib.mk_player_set_option(
                            self._player_handle,
                            "buffer_ms".encode('utf-8'),
                            buffer_ms
                        )

                    # 设置mk_player_play的参数类型
                    if hasattr(self.manager._lib, 'mk_player_play'):
                        self.manager._lib.mk_player_play.argtypes = [
                            ctypes.c_void_p,  # 播放器句柄
                            ctypes.c_char_p   # URL
                        ]

                        # 播放URL - 使用正确的API函数名称
                        # 官方API是mk_player_play而不是mk_player_play_url
                        self.manager._lib.mk_player_play(self._player_handle, url_c)

                        # mk_player_play没有返回值，不需要检查结果
                        normal_logger.info(f"播放URL成功: {self.url}")
                    else:
                        normal_logger.error("ZLMediaKit库不支持mk_player_play函数")
                        return False

                    normal_logger.info(f"C API创建播放器成功，等待验证流可用性: {self.url}")

                    # 等待一小段时间，让ZLMediaKit有时间处理流
                    await asyncio.sleep(1)

                    # 验证流是否真的可用
                    proxied_url = f"rtsp://{self.manager._config.server_address}:{self.manager._config.rtsp_port}/{self.app}/{self.stream_name}"

                    # 使用OpenCV尝试打开流并读取一帧
                    try:
                        # 在线程池中执行OpenCV操作，避免阻塞
                        loop = asyncio.get_event_loop()

                        def verify_stream():
                            try:
                                cap = cv2.VideoCapture(proxied_url)
                                if not cap.isOpened():
                                    normal_logger.warning(f"C API创建流成功，但OpenCV无法打开流: {proxied_url}")
                                    return False

                                # 尝试读取一帧
                                ret, _ = cap.read()
                                cap.release()

                                if not ret:
                                    normal_logger.warning(f"C API创建流成功，但无法读取帧: {proxied_url}")
                                    return False

                                normal_logger.info(f"流验证成功，可以通过OpenCV访问: {proxied_url}")
                                return True
                            except Exception as e:
                                normal_logger.error(f"验证流时出错: {str(e)}")
                                return False

                        # 在线程池中执行验证
                        is_valid = await loop.run_in_executor(None, verify_stream)

                        if not is_valid:
                            normal_logger.warning(f"C API创建的流无法通过OpenCV访问，但仍然继续: {self.url}")
                            # 我们仍然返回True，因为C API创建成功，后续的_pull_stream_task会继续尝试
                    except Exception as e:
                        normal_logger.error(f"验证流时发生异常: {str(e)}")
                        # 继续执行，不影响返回结果

                    return True

                except Exception as e:
                    normal_logger.error(f"使用C API创建播放器时出错: {str(e)}")
                    normal_logger.error(traceback.format_exc())

                    # 清理资源
                    if self._player_handle:
                        try:
                            self.manager._lib.mk_player_release(self._player_handle)
                        except:
                            pass
                        self._player_handle = None

                    normal_logger.error("C API创建流失败，无法继续")
                    return False

            # C API不可用，无法创建流
            normal_logger.error(f"无法创建流 {self.stream_name}，C API不可用")
            return False
        except Exception as e:
            normal_logger.error(f"启动RTSP流时出错: {str(e)}")
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
            # 尝试使用原始URL
            proxied_url = self.url

            # 如果需要，也可以使用代理URL
            # proxied_url = f"rtsp://{self.manager._config.server_address}:{self.manager._config.rtsp_port}/{self.app}/{self.stream_name}"

            normal_logger.info(f"拉流URL: {proxied_url}")

            while not self._stop_event.is_set():
                try:
                    normal_logger.info(f"开始拉流 {proxied_url}")
                    cap = cv2.VideoCapture(proxied_url)

                    if not cap.isOpened():
                        normal_logger.error(f"无法打开流: {proxied_url}")
                        self._status = StreamStatus.ERROR
                        self._health_status = StreamHealthStatus.OFFLINE
                        self._last_error = "无法打开流"

                        # 重试
                        retry_count += 1
                        if retry_count > max_retry:
                            normal_logger.error(f"超过最大重试次数({max_retry})，停止拉流")
                            break

                        normal_logger.info(f"等待 {retry_interval} 秒后重试 ({retry_count}/{max_retry})...")
                        await asyncio.sleep(retry_interval)
                        continue

                    # 重置重试计数
                    retry_count = 0

                    # 先尝试读取一帧，确认流真的可用
                    ret, test_frame = cap.read()
                    if not ret:
                        normal_logger.error(f"流可以打开但无法读取第一帧: {proxied_url}")
                        self._status = StreamStatus.ERROR
                        self._health_status = StreamHealthStatus.OFFLINE
                        self._last_error = "流可以打开但无法读取第一帧"
                        cap.release()

                        # 重试
                        retry_count += 1
                        if retry_count > max_retry:
                            normal_logger.error(f"超过最大重试次数({max_retry})，停止拉流")
                            break

                        normal_logger.info(f"等待 {retry_interval} 秒后重试 ({retry_count}/{max_retry})...")
                        await asyncio.sleep(retry_interval)
                        continue

                    # 成功读取第一帧，更新状态
                    normal_logger.info(f"成功读取第一帧，流 {self.stream_id} 连接成功")
                    self._status = StreamStatus.ONLINE
                    self._health_status = StreamHealthStatus.HEALTHY

                    # 处理第一帧
                    with self._frame_lock:
                        self._frame_buffer.append(test_frame)

                    # 更新统计
                    self._stats["frames_received"] += 1
                    self._stats["last_frame_time"] = time.time()

                    # 通知帧处理任务
                    self._frame_processed_event.set()

                    # 读取后续帧
                    frame_count = 1
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
                            normal_logger.warning(f"读取帧失败，连续错误: {error_count}/{max_consecutive_errors}")

                            if error_count >= max_consecutive_errors:
                                normal_logger.error(f"连续错误达到阈值({max_consecutive_errors})，重新连接")
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
                                # 移除频繁的帧分发日志，减少日志量
                                await queue.put((frame.copy(), timestamp))
                            except Exception as e:
                                normal_logger.error(f"向订阅者 {subscriber_id} 分发帧时出错: {str(e)}")

                    # 关闭捕获
                    cap.release()

                    # 如果是停止事件触发的，则退出循环
                    if self._stop_event.is_set():
                        break

                    # 否则重新连接
                    normal_logger.info("重新连接流...")
                    self._stats["reconnects"] += 1
                    self._status = StreamStatus.INITIALIZING  # 使用INITIALIZING代替RECONNECTING
                    await asyncio.sleep(retry_interval)

                except Exception as e:
                    normal_logger.error(f"拉流任务异常: {str(e)}")
                    self._stats["errors"] += 1
                    self._status = StreamStatus.ERROR  # ERROR是正确的
                    self._health_status = StreamHealthStatus.UNHEALTHY  # 使用UNHEALTHY代替ERROR
                    self._last_error = str(e)

                    # 重试
                    retry_count += 1
                    if retry_count > max_retry:
                        normal_logger.error(f"超过最大重试次数({max_retry})，停止拉流")
                        break

                    normal_logger.info(f"等待 {retry_interval} 秒后重试 ({retry_count}/{max_retry})...")
                    await asyncio.sleep(retry_interval)

            # 任务结束
            normal_logger.info(f"拉流任务结束: {self.stream_id}")
            self._status = StreamStatus.OFFLINE  # 使用OFFLINE代替STOPPED

        except Exception as e:
            normal_logger.error(f"拉流任务发生异常: {str(e)}")
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
                            normal_logger.error(f"向订阅者 {subscriber_id} 分发帧时出错: {str(e)}")

                    # 更新统计
                    self._stats["frames_processed"] += 1

                # 更新最后处理的帧索引
                last_frame_index = current_frame_count - 1

                # 短暂休眠，避免CPU占用过高
                await asyncio.sleep(0.001)

        except Exception as e:
            normal_logger.error(f"帧处理任务异常: {str(e)}")
            self._status = StreamStatus.ERROR
            self._last_error = str(e)