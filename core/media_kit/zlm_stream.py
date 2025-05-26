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

    def __init__(self, stream_key_from_manager: str, config: Dict[str, Any], manager, player_handle=None):
        """初始化ZLMediaKit流

        Args:
            stream_key_from_manager: 由StreamManager根据URL哈希生成的唯一流密钥.
            config: 流配置
            manager: ZLMediaKit管理器实例
            player_handle: 不再使用，保留参数兼容性
        """
        # 调用基类初始化. BaseVideoStream的stream_id现在将是stream_key_from_manager
        super().__init__(stream_key_from_manager, config)
        self.manager = manager # ZLMediaKitManager实例
        self.stream_type = config.get("type", "rtsp") # rtsp, rtmp, hls, http etc.
        self._url = config.get("url", "")  # 源流URL

        # ZLM相关配置
        self.vhost = config.get("vhost", "__defaultVhost__") # 新逻辑：直接使用ZLM的默认vhost标识
        self.app = config.get("app", "live") # 应用名
        self.stream_name = self.stream_id # 使用从manager传入的stream_key作为ZLM中的stream名
        self._zlm_stream_key: Optional[str] = config.get("stream_key") # 由ZLM API调用返回的key, 在addStreamProxy后设置

        # ZLM API调用返回的内部key，用于后续操作如delStreamProxy
        self._zlm_internal_api_key: Optional[str] = None 

        # 帧缓存 - 增加缓冲区大小提高稳定性
        self._frame_buffer = []
        self._frame_buffer_size = config.get("frame_buffer_size", 30)  # 从5增加到30
        self._frame_lock = threading.Lock()

        # 流控制
        self._frame_processed_event = asyncio.Event()
        self._frame_processed_event.set()
        self._stop_lock = threading.Lock()
        self._is_stopping = False

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

        # 移除C API回调相关属性
        self._frame_callback_registered = False

        # 代理URL（由ZLM HTTP API创建的流地址） - 这个可能由 _start_stream_with_http_api 设置
        # self._proxy_url = "" # 移除，因为它会在 _start_stream_with_http_api 中从API响应获取
        
        # self._stream_key = config.get("stream_key", "") # 移除，因为我们现在传入 stream_key_from_manager

        normal_logger.info(f"创建ZLM共享流 (key/stream_id: {self.stream_id}), 源URL: {self._url}, ZLM App: {self.app}, ZLM Stream Name: {self.stream_name}, 类型: {self.stream_type}")

    # 重写url属性，确保能够正确获取和设置url值
    @property
    def url(self) -> str:
        """获取流URL"""
        return self._url

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
        self._status = StreamStatus.INITIALIZING
        self._health_status = StreamHealthStatus.UNHEALTHY
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
            
            # 使用HTTP API创建流，所有流类型使用相同的方法
            if self.stream_type in ["rtsp", "rtmp", "hls", "http"]:
                success = await self._start_stream_with_http_api()
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
        with self._stop_lock:
            if self._is_stopping:
                normal_logger.info(f"流 {self.stream_id} 正在停止中，跳过重复停止")
                return True
            self._is_stopping = True

        try:
            normal_logger.info(f"正在停止流: {self.stream_id}")

            # 1. 首先标记状态
            self._status = StreamStatus.STOPPED
            self._is_running = False

            # 2. 取消所有任务
            if self._pull_task and not self._pull_task.done():
                self._pull_task.cancel()
                try:
                    await self._pull_task
                except asyncio.CancelledError:
                    normal_logger.debug(f"拉流任务已取消: {self.stream_id}") # 使用 debug 级别
                except Exception as e:
                    normal_logger.error(f"取消拉流任务时出错: {str(e)}")

            if self._frame_task and not self._frame_task.done():
                self._frame_task.cancel()
                try:
                    await self._frame_task
                except asyncio.CancelledError:
                    normal_logger.debug(f"帧处理任务已取消: {self.stream_id}") # 使用 debug 级别
                except Exception as e:
                    normal_logger.error(f"取消帧处理任务时出错: {str(e)}")

            # 3. 使用HTTP API关闭流
            # stream_key = self.config.get("stream_key", self._stream_key) # 旧逻辑
            # 使用 self._zlm_internal_api_key (addStreamProxy返回的key)
            if self._zlm_internal_api_key:
                try:
                    # 使用delStreamProxy关闭流代理
                    normal_logger.info(f"使用HTTP API (delStreamProxy) 关闭流代理 (ZLM internal key: {self._zlm_internal_api_key}), stream_id: {self.stream_id}")
                    result = self.manager.call_api("delStreamProxy", {"key": self._zlm_internal_api_key})
                    
                    if result.get("code") == 0:
                        normal_logger.info(f"HTTP API (delStreamProxy) 关闭流代理成功, stream_id: {self.stream_id}")
                    else:
                        error_msg = result.get("msg", "未知错误")
                        normal_logger.warning(f"HTTP API (delStreamProxy) 关闭流代理失败: {error_msg}, stream_id: {self.stream_id}")
                        
                        # 备用：尝试使用close_streams API (如果delStreamProxy失败或key未知)
                        await self._fallback_close_stream_api()
                except Exception as e:
                    normal_logger.error(f"通过 delStreamProxy 关闭流代理 (ZLM internal key: {self._zlm_internal_api_key}) 时出错: {str(e)}, stream_id: {self.stream_id}")
                    await self._fallback_close_stream_api() # 尝试备用关闭
            else:
                # 如果没有 _zlm_internal_api_key (例如，addStreamProxy未成功或未返回key)
                normal_logger.warning(f"停止流 {self.stream_id} 时 _zlm_internal_api_key 未知，尝试使用 close_streams API。")
                await self._fallback_close_stream_api()

            # 4. 清理订阅者
            async with self._subscriber_lock:
                self._subscribers.clear()

            # 5. 清理帧缓存
            with self._frame_lock:
                self._frame_buffer.clear()

            # 6. 设置最终状态
            self._status = StreamStatus.OFFLINE
            self._health_status = StreamHealthStatus.OFFLINE
            self._is_running = False
            self._is_stopping = False

            normal_logger.info(f"流 {self.stream_id} 已成功停止")
            return True

        except Exception as e:
            exception_logger.exception(f"停止流 {self.stream_id} 时出错: {str(e)}")
            self._is_stopping = False
            return False

    async def get_info(self) -> Dict[str, Any]:
        """获取流信息

        Returns:
            Dict[str, Any]: 流信息
        """
        # 构建基本信息
        info = {
            "stream_id": self.stream_id,
            "url": self.url,
            "type": self.stream_type,
            "status": self._status.value,
            "status_text": self._status.name,
            "health_status": self._health_status.value,
            "health_status_text": self._health_status.name,
            "last_error": self._last_error,
            "subscriber_count": len(self._subscribers),
            "stats": self._stats,
            "connection_info": {
                "proxy_url": self._proxy_url,
                "original_url": self.url,
                "using_proxy": self._stats.get("using_proxy", True),
                "connected_url": self._stats.get("connected_url", ""),
                "c_api_available": self._use_sdk,
                "player_handle_active": self._player_handle is not None,
                "frame_callback_registered": self._frame_callback_registered
            },
            "diagnostics": {
                "timestamps": {
                    "start_time": self._stats.get("start_time", 0),
                    "last_frame_time": self._stats.get("last_frame_time", 0),
                    "current_time": time.time()
                },
                "frames": {
                    "received": self._stats.get("frames_received", 0),
                    "processed": self._stats.get("frames_processed", 0),
                    "buffer_size": len(self._frame_buffer),
                    "max_buffer_size": self._frame_buffer_size
                },
                "errors": self._stats.get("errors", 0),
                "reconnects": self._stats.get("reconnects", 0),
                "fps": self._stats.get("fps", 0)
            }
        }
        
        # 计算流健康度指标
        if self._stats.get("frames_received", 0) > 0:
            # 计算距离最后一帧的时间
            last_frame_time = self._stats.get("last_frame_time", 0)
            time_since_last_frame = time.time() - last_frame_time if last_frame_time > 0 else float('inf')
            
            # 添加到诊断信息
            info["diagnostics"]["health_metrics"] = {
                "time_since_last_frame": time_since_last_frame,
                "healthy": time_since_last_frame < 5.0,  # 如果5秒内有帧，认为是健康的
            }
        
        return info

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
        valid_statuses = [StreamStatus.RUNNING, StreamStatus.ONLINE, StreamStatus.CONNECTING]
        if not self._is_running:
            normal_logger.debug(f"流 {self.stream_id} 状态不可用: is_running={self._is_running}, status={self._status}")
            return False, None

        # 对于CONNECTING或INITIALIZING状态，给予更长的等待时间
        if self._status in [StreamStatus.CONNECTING, StreamStatus.INITIALIZING]:
            # 检查是否有帧正在接收中
            if self._stats.get("frames_received", 0) > 0:
                normal_logger.debug(f"流 {self.stream_id} 正在连接中，已接收 {self._stats.get('frames_received', 0)} 帧")
            else:
                normal_logger.debug(f"流 {self.stream_id} 正在连接中，尚未接收到帧")
            
            # 如果状态是CONNECTING但已经接收了帧，则更新状态为ONLINE
            if self._stats.get("frames_received", 0) > 0 and self._status == StreamStatus.CONNECTING:
                self._status = StreamStatus.ONLINE
                self._health_status = StreamHealthStatus.HEALTHY
                normal_logger.info(f"流 {self.stream_id} 状态由CONNECTING更新为ONLINE")
        
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
                    # 检查最后一帧时间，判断是否需要重连
                    last_frame_time = self._stats.get("last_frame_time", 0)
                    if last_frame_time > 0 and time.time() - last_frame_time > 5.0:
                        normal_logger.warning(f"流 {self.stream_id} 超过5秒未接收到新帧，可能需要重连")

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

    async def _start_stream_with_http_api(self) -> bool:
        """使用HTTP API启动流
        
        通过ZLMediaKit HTTP API创建流代理

        Returns:
            bool: 是否成功启动
        """
        try:
            # 验证URL格式
            if not self.url.startswith(("rtsp://", "rtmp://", "http://", "https://", "hls://")):
                normal_logger.error(f"URL格式不正确: {self.url}")
                self._last_error = "URL格式不正确"
                return False
            
            # 使用HTTP API创建流代理
            normal_logger.info(f"使用HTTP API创建流代理: {self.url}")
            
            # 准备API参数 - 根据ZLMediaKit API文档格式
            api_params = {
                "vhost": self.vhost,
                "app": self.app,
                "stream": self.stream_name,
                "url": self.url
            }
            
            # 添加可选参数
            if self.config.get("enable_rtsp") is not None:
                api_params["enable_rtsp"] = self.config.get("enable_rtsp", 1)
            
            if self.config.get("enable_rtmp") is not None:
                api_params["enable_rtmp"] = self.config.get("enable_rtmp", 1)
                
            if self.config.get("enable_hls") is not None:
                api_params["enable_hls"] = self.config.get("enable_hls", 1)
                
            if self.config.get("enable_mp4") is not None:
                api_params["enable_mp4"] = self.config.get("enable_mp4", 0)
            
            # RTSP特有参数
            if self.stream_type == "rtsp":
                rtsp_transport = self.config.get("rtsp_transport", "tcp")
                api_params["rtp_type"] = 0 if rtsp_transport.lower() == "tcp" else 1
            
            # 通用参数
            if self.config.get("timeout_sec") is not None:
                api_params["timeout_sec"] = self.config.get("timeout_sec", 10)
                
            if self.config.get("retry_count") is not None:
                api_params["retry_count"] = self.config.get("retry_count", -1)  # -1表示无限重试
            
            # 调用ZLM HTTP API创建流代理 - 添加错误处理
            normal_logger.info(f"调用addStreamProxy API, 参数: {api_params}")
            result = self.manager.call_api("addStreamProxy", api_params)
            
            if result.get("code") == 0:
                normal_logger.info(f"HTTP API创建流代理成功: {self.stream_id}")
                self._zlm_internal_api_key = result.get("data", {}).get("key", "")
                self.config["stream_key"] = self._zlm_internal_api_key
                self._proxy_url = self._get_zlm_proxy_url()
                self._stats["proxy_url"] = self._proxy_url
                self._stats["stream_key"] = self._zlm_internal_api_key
                self._stats["using_proxy"] = True # Standard ZLM proxying
                self._stats["connected_url"] = self.url
                from shared.utils.logger import TEST_LOG_MARKER
                normal_logger.info(f"{TEST_LOG_MARKER} STREAM_PULL_SUCCESS")
                normal_logger.info(f"{TEST_LOG_MARKER} ZLM_PROCESS_SUCCESS")
                normal_logger.info(f"{TEST_LOG_MARKER} RTSP_STREAM_CONNECTED")
                normal_logger.info(f"{TEST_LOG_MARKER} STREAM_CONNECTED")
                self._is_running = True
                self._status = StreamStatus.ONLINE
                self._health_status = StreamHealthStatus.HEALTHY
                return True
            else:
                error_msg = result.get("msg", "未知错误")
                normal_logger.error(f"HTTP API创建流代理失败: {error_msg}")
                self._last_error = f"HTTP API创建流代理失败: {error_msg}"

                # 新增：如果是因为 schema 不支持，并且原始URL是HTTP(S)开头的，
                # 尝试直接使用原始URL给OpenCV (假设它可能是OpenCV能直接播放的格式)
                if "not supported play schema" in error_msg.lower() and \
                   (self.url.lower().startswith("http://") or self.url.lower().startswith("https://")):
                    normal_logger.warning(f"流 {self.stream_id}: addStreamProxy 不支持该 schema。将尝试直接使用原始URL '{self.url}' 进行 OpenCV 拉流。")
                    # self._proxy_url = self.url # _proxy_url 通常指ZLM生成的代理地址，这里不设置
                    self.config["force_direct_opencv_pull"] = True 
                    self._stats["using_proxy"] = False # 明确标记不通过ZLM代理，而是直接拉源
                    self._stats["connected_url"] = self.url
                    self._is_running = True 
                    self._status = StreamStatus.CONNECTING 
                    self._health_status = StreamHealthStatus.UNKNOWN
                    return True # 返回True，让后续的 _pull_task 尝试用原始URL
                
                if "已经存在" in error_msg or "already exists" in error_msg.lower():
                    normal_logger.warning(f"流 {self.stream_id} 已经存在，尝试复用")
                    
                    # 尝试获取已存在流的信息
                    stream_info_params = {
                        "vhost": self.vhost,
                        "app": self.app,
                        "stream": self.stream_name
                    }
                    
                    info_result = self.manager.call_api("getMediaInfo", stream_info_params)
                    if info_result.get("code") == 0:
                        normal_logger.info(f"成功获取已存在流信息: {self.stream_id}")
                        
                        # 构建代理URL
                        self._proxy_url = self._get_zlm_proxy_url()
                        
                        # 更新统计信息
                        self._stats["proxy_url"] = self._proxy_url
                        self._stats["using_proxy"] = True
                        self._stats["connected_url"] = self.url
                        
                        # 更新流状态
                        self._status = StreamStatus.ONLINE
                        self._health_status = StreamHealthStatus.HEALTHY
                        self._is_running = True
                        
                        # 添加日志标记
                        from shared.utils.logger import TEST_LOG_MARKER
                        normal_logger.info(f"{TEST_LOG_MARKER} STREAM_CONNECTED")
                        
                        return True
                
                return False
                
        except Exception as e:
            normal_logger.error(f"启动流 {self.stream_id} 时出错: {str(e)}")
            normal_logger.error(traceback.format_exc())
            self._last_error = f"启动流时出错: {str(e)}"
            return False

    def _get_zlm_proxy_url(self) -> str:
        """获取ZLMediaKit代理URL
        
        根据配置文件构建完整的代理URL，供外部播放器使用
        
        Returns:
            str: ZLMediaKit代理URL
        """
        # 获取ZLM配置
        server_address = self.manager._config.server_address
        rtsp_port = self.manager._config.rtsp_port
        
        # 如果server_address是localhost或127.0.0.1，尝试获取本机实际IP
        if server_address in ['localhost', '127.0.0.1']:
            try:
                import socket
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                normal_logger.info(f"ZLM服务器地址是本地地址，使用本机IP: {local_ip}")
                server_address = local_ip
            except Exception as e:
                normal_logger.warning(f"无法获取本机IP，使用配置的服务器地址: {server_address}")
        
        # 获取应用名和流名称
        app = self.app
        stream_name = self.stream_name
        
        # 根据ZLM配置的路径格式构建URL
        # 默认情况下，ZLM的RTSP代理格式为 rtsp://server:port/app/stream_name
        proxy_url = f"rtsp://{server_address}:{rtsp_port}/{app}/{stream_name}"
        
        # 将代理URL保存到统计信息中
        self._stats["proxy_url"] = proxy_url
        self._stats["proxy_server"] = server_address
        self._stats["proxy_port"] = rtsp_port
        
        return proxy_url

    async def _pull_stream_task(self) -> None:
        """拉流任务，负责从ZLMediaKit获取视频帧
        
        使用OpenCV从ZLM代理URL拉取视频流
        """
        cap = None # Initialize cap here to ensure it's defined in finally/except blocks
        zlm_pull_url_for_opencv = None
        try:
            normal_logger.info(f"OpenCV拉流任务开始 for stream_id: {self.stream_id}")
            
            # 步骤 0: 检查是否要强制使用原始URL (例如当addStreamProxy因schema不支持而失败时)
            if self.config.get("force_direct_opencv_pull"):
                zlm_pull_url_for_opencv = self.url # self.url 就是 _url, 即原始输入 URL
                normal_logger.info(f"流 {self.stream_id}: 强制使用原始URL进行OpenCV拉流: {zlm_pull_url_for_opencv}")
            # 步骤 1: 检查配置中是否直接指定了 OpenCV 拉流地址 (用户覆盖，次高优先级)
            elif self.config.get("opencv_pull_url"):
                zlm_pull_url_for_opencv = self.config["opencv_pull_url"]
                normal_logger.info(f"流 {self.stream_id}: 使用配置中指定的 OpenCV 拉流地址: {zlm_pull_url_for_opencv}")
            else:
                # 步骤 2: 调用 get_zlm_player_proxy_info 主要用于确认流状态和获取元数据
                try:
                    normal_logger.info(f"流 {self.stream_id}: 尝试从 ZLM API 获取播放信息 (元数据检查)...")
                    proxy_info = await self.get_zlm_player_proxy_info()
                    if proxy_info and proxy_info.get("success"):
                        normal_logger.info(f"流 {self.stream_id}: ZLM API元数据检查成功. Online: {proxy_info.get('online_status_from_zlm')}. Raw info: {proxy_info.get('raw_media_info')}")
                    elif proxy_info:
                        normal_logger.warning(f"流 {self.stream_id}: ZLM API元数据检查失败或未返回成功。Response: {proxy_info}")
                    else: 
                        normal_logger.warning(f"流 {self.stream_id}: ZLM API元数据检查返回 None.")
                except Exception as e_player_info:
                    exception_logger.error(f"流 {self.stream_id}: ZLM API元数据检查时发生异常: {e_player_info}")

                # 步骤 3: 使用 get_opencv_pull_url() 构建指向本地ZLM代理的拉流地址 (标准流程)
                normal_logger.info(f"流 {self.stream_id}: 使用 get_opencv_pull_url() 构建指向本地ZLM的 OpenCV 拉流地址...")
                zlm_pull_url_for_opencv = self.get_opencv_pull_url()
            
            if not zlm_pull_url_for_opencv: # 如果最终还是没有URL
                error_message = f"流 {self.stream_id}: 无法确定有效的 OpenCV 拉流地址。原始URL: {self.url}"
                normal_logger.error(error_message)
                self._status = StreamStatus.ERROR
                self._health_status = StreamHealthStatus.UNHEALTHY
                self._last_error = error_message
                # 考虑是否触发重连或彻底失败
                return

            normal_logger.info(f"流 {self.stream_id}: 最终选定 OpenCV 拉流地址: {zlm_pull_url_for_opencv}")

            # 记录使用的是OpenCV方式
            self._stats["using_proxy"] = True # 这个标记可能需要重新审视其含义，因为我们总是通过ZLM拉流
            self._stats["proxy_url"] = zlm_pull_url_for_opencv # 记录实际使用的URL
            self._stats["original_url"] = self.url # 记录原始输入URL
            
            # 添加日志标记，用于测试检测
            from shared.utils.logger import TEST_LOG_MARKER
            normal_logger.info(f"{TEST_LOG_MARKER} STREAM_PULL_SUCCESS")
            
            # 设置OpenCV参数
            cap = None
            max_open_retries = 10
            open_retry_count = 0
            
            # 尝试多次打开视频流
            while open_retry_count < max_open_retries and not self._stop_event.is_set():
                try:
                    # 使用OpenCV打开视频流
                    cap = cv2.VideoCapture(zlm_pull_url_for_opencv) # 使用最终确定的URL
                    if not cap.isOpened():
                        open_retry_count += 1
                        normal_logger.warning(f"无法打开最终确定的拉流URL (尝试 {open_retry_count}/{max_open_retries}): {zlm_pull_url_for_opencv}")
                        
                        # 关闭失败的cap
                        if cap:
                            cap.release()
                            cap = None
                            
                        # 短暂等待后重试
                        await asyncio.sleep(2.0)
                        continue
                    
                    # 成功打开
                    break
                except Exception as e:
                    open_retry_count += 1
                    normal_logger.error(f"打开视频流时出错 (尝试 {open_retry_count}/{max_open_retries}): {str(e)}")
                    
                    # 关闭失败的cap
                    if cap:
                        cap.release()
                        cap = None
                        
                    # 短暂等待后重试
                    await asyncio.sleep(2.0)
            
            # 检查是否成功打开
            if not cap or not cap.isOpened():
                normal_logger.error(f"多次尝试后仍无法打开最终确定的拉流URL: {zlm_pull_url_for_opencv}")
                self._status = StreamStatus.ERROR
                self._health_status = StreamHealthStatus.UNHEALTHY
                self._last_error = f"无法打开最终确定的拉流URL: {zlm_pull_url_for_opencv}"
                return
                
            # 设置OpenCV参数
            # 缓冲区大小 - 1表示不缓冲，立即返回最新帧
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 记录视频流信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            normal_logger.info(f"视频流信息 - 宽度: {width}, 高度: {height}, FPS: {fps}")
            self._stats["width"] = width
            self._stats["height"] = height
            self._stats["fps"] = fps
            
            # 添加ZLM处理成功的日志标记
            normal_logger.info(f"{TEST_LOG_MARKER} ZLM_PROCESS_SUCCESS")
            
            # 设置状态为运行中
            self._status = StreamStatus.RUNNING
            self._health_status = StreamHealthStatus.HEALTHY
            
            # 主循环 - 从OpenCV获取帧
            last_stats_time = time.time()
            frames_count = 0
            retry_count = 0
            max_retries = 5
            
            # 读取帧间隔控制，避免过快读取
            frame_interval = 1.0 / max(fps, 25.0)  # 默认最低FPS为25
            last_frame_time = 0
            
            while not self._stop_event.is_set():
                try:
                    # 控制读取帧的间隔，避免CPU过高使用
                    current_time = time.time()
                    time_since_last_frame = current_time - last_frame_time
                    if time_since_last_frame < frame_interval:
                        # 未到达下一帧时间，短暂休眠
                        await asyncio.sleep(max(0.001, frame_interval - time_since_last_frame))
                        continue
                    
                    # 尝试读取帧
                    ret, frame = cap.read()
                    last_frame_time = time.time()  # 更新最后一帧读取时间
                    
                    if ret:
                        # 成功读取帧
                        retry_count = 0  # 重置重试计数
                        
                        # 更新缓存 - 使用更高效的锁策略
                        with self._frame_lock:
                            # 如果缓冲区已满，则移除最旧的一半帧，提高处理效率
                            if len(self._frame_buffer) >= self._frame_buffer_size:
                                # 丢弃旧帧而不是逐个移除
                                self._frame_buffer = self._frame_buffer[len(self._frame_buffer)//2:]
                            
                            # 添加新帧
                            self._frame_buffer.append(frame)
                        
                        # 更新统计
                        now = time.time()
                        self._stats["frames_received"] += 1
                        self._stats["last_frame_time"] = now
                        
                        # 设置状态为在线
                        if self._status != StreamStatus.ONLINE:
                            self._status = StreamStatus.ONLINE
                            self._health_status = StreamHealthStatus.HEALTHY
                            normal_logger.info(f"流 {self.stream_id} 已连接并接收到首帧")
                            normal_logger.info(f"{TEST_LOG_MARKER} FRAME_RECEIVED_SUCCESS")
                        
                        # 通知帧处理任务
                        self._frame_processed_event.set()
                        
                        # 每10秒输出一次状态信息
                        if now - last_stats_time >= 10.0:
                            total_received = self._stats.get("frames_received", 0)
                            current_fps = (total_received - frames_count) / (now - last_stats_time)
                            self._stats["fps"] = current_fps
                            
                            normal_logger.info(f"流 {self.stream_id} 状态: 已接收 {total_received} 帧, "
                                            f"缓冲区: {len(self._frame_buffer)}/{self._frame_buffer_size}, "
                                            f"FPS: {current_fps:.2f}")
                            
                            frames_count = total_received
                            last_stats_time = now
                    else:
                        # 读取帧失败
                        retry_count += 1
                        normal_logger.warning(f"读取帧失败，重试 {retry_count}/{max_retries}")
                        
                        if retry_count >= max_retries:
                            normal_logger.error(f"连续 {max_retries} 次读取帧失败，尝试重新连接")
                            
                            # 关闭当前连接
                            cap.release()
                            
                            # 等待一段时间再重连
                            await asyncio.sleep(1.0)
                            
                            # 重新连接
                            cap = cv2.VideoCapture(zlm_pull_url_for_opencv) # 使用相同的最终URL
                            if not cap.isOpened():
                                normal_logger.error(f"重新连接失败: {zlm_pull_url_for_opencv}")
                                self._stats["errors"] += 1
                                self._status = StreamStatus.ERROR
                                self._health_status = StreamHealthStatus.UNHEALTHY
                                break
                            
                            # 重置重试计数
                            retry_count = 0
                            self._stats["reconnects"] += 1
                            
                            # 重新设置缓冲区
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            normal_logger.info(f"重新连接成功: {zlm_pull_url_for_opencv}")
                        else:
                            # 短暂休眠，避免立即重试
                            await asyncio.sleep(0.1)
                except Exception as frame_err:
                    # 捕获读取帧时的异常
                    normal_logger.error(f"读取帧时出错: {str(frame_err)}")
                    retry_count += 1
                    
                    if retry_count >= max_retries:
                        # 太多错误，尝试重连
                        try:
                            if cap:
                                cap.release()
                            
                            await asyncio.sleep(1.0)
                            
                            # 重新连接
                            cap = cv2.VideoCapture(zlm_pull_url_for_opencv) # 使用相同的最终URL
                            if not cap.isOpened():
                                normal_logger.error(f"重新连接失败: {zlm_pull_url_for_opencv}")
                                self._stats["errors"] += 1
                                break
                            
                            # 重置重试计数
                            retry_count = 0
                            self._stats["reconnects"] += 1
                            
                            # 重新设置缓冲区
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            normal_logger.info(f"重新连接成功: {zlm_pull_url_for_opencv}")
                        except Exception as reconnect_err:
                            normal_logger.error(f"重连过程中出错: {str(reconnect_err)}")
                            self._stats["errors"] += 1
                            break
                    else:
                        # 短暂休眠，避免立即重试
                        await asyncio.sleep(0.1)
            
            # 关闭OpenCV连接
            if cap:
                cap.release()
            normal_logger.info(f"OpenCV拉流任务结束 for stream_id: {self.stream_id}")
            
        except Exception as e:
            self._stats["errors"] += 1
            self._status = StreamStatus.ERROR
            self._health_status = StreamHealthStatus.UNHEALTHY
            self._last_error = f"拉流任务异常: {str(e)}"
            
            if cap: # 确保在异常情况下也释放cap
                cap.release()
            exception_logger.exception(f"拉流任务异常 for stream_id {self.stream_id}: {str(e)}")
        
        normal_logger.info(f"拉流任务退出: {self.url}, stream_id: {self.stream_id}")

    async def _process_frames_task(self) -> None:
        """帧处理任务，负责将帧分发给订阅者"""
        try:
            last_frame_index = -1
            last_distribute_time = time.time()
            min_frame_interval = 0.02  # 50fps的最小帧间隔

            while not self._stop_event.is_set():
                # 等待新帧，添加超时机制避免永久阻塞
                try:
                    await asyncio.wait_for(self._frame_processed_event.wait(), timeout=1.0)
                    self._frame_processed_event.clear()
                except asyncio.TimeoutError:
                    # 超时但没有新帧，短暂休眠后继续
                    await asyncio.sleep(0.05)
                    continue

                # 获取最新帧索引
                current_frame_count = 0
                with self._frame_lock:
                    current_frame_count = len(self._frame_buffer)

                # 如果没有新帧但已经超过50ms未分发帧，也尝试分发当前帧
                # 降低从100ms到50ms，增加分发频率
                current_time = time.time()
                if current_frame_count <= last_frame_index and current_time - last_distribute_time < 0.05:
                    await asyncio.sleep(0.01)
                    continue

                # 获取订阅者列表
                subscribers = {}
                async with self._subscriber_lock:
                    subscribers = self._subscribers.copy()

                # 如果没有订阅者，跳过
                if not subscribers:
                    await asyncio.sleep(0.01)
                    continue

                # 分发帧给订阅者
                # 如果没有新帧但超过50ms未分发，分发最后一帧
                if current_frame_count <= last_frame_index:
                    if current_frame_count > 0:
                        with self._frame_lock:
                            if len(self._frame_buffer) > 0:
                                frame = self._frame_buffer[-1].copy()
                                timestamp = time.time()
                                
                                # 分发给所有订阅者
                                for subscriber_id, queue in subscribers.items():
                                    try:
                                        # 如果队列已满，则丢弃一半的旧帧以释放空间
                                        if queue.full():
                                            try:
                                                # 移除一半旧帧而不是只移除一帧
                                                for _ in range(queue.qsize() // 2):
                                                    _ = queue.get_nowait()
                                            except:
                                                pass

                                        # 放入最后一帧
                                        await queue.put((frame, timestamp))
                                    except Exception as e:
                                        normal_logger.error(f"向订阅者 {subscriber_id} 分发帧时出错: {str(e)}")
                    
                    last_distribute_time = current_time
                    await asyncio.sleep(min_frame_interval)  # 使用最小帧间隔控制分发速率
                    continue

                # 有新帧，分发新帧
                # 每个订阅者只分发最新的帧，避免队列堆积
                # 如果帧太多，跳过中间帧，只发送最新的帧
                if current_frame_count - last_frame_index > 5:  # 如果有超过5帧的积压
                    normal_logger.info(f"帧积压过多 ({current_frame_count - last_frame_index} 帧)，跳过中间帧")
                    # 只分发最新的帧
                    with self._frame_lock:
                        if len(self._frame_buffer) > 0:
                            frame = self._frame_buffer[-1].copy()
                            timestamp = time.time()
                            
                            # 分发给所有订阅者
                            for subscriber_id, queue in subscribers.items():
                                try:
                                    # 如果队列已满，则清空队列
                                    if queue.full():
                                        try:
                                            while not queue.empty():
                                                _ = queue.get_nowait()
                                        except:
                                            pass

                                    # 放入最新帧
                                    await queue.put((frame, timestamp))
                                except Exception as e:
                                    normal_logger.error(f"向订阅者 {subscriber_id} 分发帧时出错: {str(e)}")
                    
                    # 更新索引和时间
                    last_frame_index = current_frame_count - 1
                    last_distribute_time = time.time()
                else:
                    # 正常分发每一帧
                    for i in range(last_frame_index + 1, current_frame_count):
                        # 获取帧
                        frame = None
                        timestamp = time.time()
                        with self._frame_lock:
                            if i < len(self._frame_buffer):
                                frame = self._frame_buffer[i].copy()
                            else:
                                break

                        if frame is None:
                            continue

                        # 分发给所有订阅者
                        for subscriber_id, queue in subscribers.items():
                            try:
                                # 如果队列已满，则丢弃旧帧
                                if queue.full():
                                    try:
                                        _ = queue.get_nowait()
                                    except:
                                        pass

                                # 放入新帧
                                await queue.put((frame, timestamp))
                            except Exception as e:
                                normal_logger.error(f"向订阅者 {subscriber_id} 分发帧时出错: {str(e)}")

                        # 更新统计
                        self._stats["frames_processed"] += 1

                    # 更新最后处理的帧索引和分发时间
                    last_frame_index = current_frame_count - 1
                    last_distribute_time = time.time()

                # 短暂休眠，避免CPU占用过高，但保持较高响应性
                await asyncio.sleep(min_frame_interval)

        except Exception as e:
            normal_logger.error(f"帧处理任务异常: {str(e)}")
            normal_logger.error(f"异常详情: {traceback.format_exc()}")
            self._status = StreamStatus.ERROR
            self._last_error = str(e)

    async def get_proxy_url_info(self) -> Dict[str, Any]:
        """获取此流在ZLMediaKit中的代理播放地址信息 (如果已通过addStreamProxy创建)
        返回包含各种协议播放地址的字典。
        主要用于确认流是否存在于ZLM以及获取其元数据。
        """
        if not self.manager:
            return {"success": False, "error": "ZLMediaKit管理器未初始化"} # 添加 success: False

        try:
            params = {
                "vhost": self.vhost,
                "app": self.app,
                "schema": "rtmp", # ZLM 要求 schema 参数。rtmp 或 rtsp 都可以。
                "stream": self.stream_name
            }
            media_info = self.manager.call_api("getMediaInfo", params)
            normal_logger.debug(f"getMediaInfo for {self.vhost}/{self.app}/{self.stream_name} response: {media_info}")

            if media_info and media_info.get("code") == 0:
                is_online = media_info.get("online")
                if is_online is None:
                    normal_logger.info(f"ZLM getMediaInfo (code 0) for {self.stream_id} did not return 'online' status, assuming stream info is available.")
                elif not is_online:
                    normal_logger.warning(f"ZLM getMediaInfo表示流 {self.vhost}/{self.app}/{self.stream_name} (key: {self.stream_id}) 当前不在线 (online: {is_online})。")
                
                # 不再尝试构建 player_urls 字典，因为 ZLM API 不直接提供这些。
                # ZLM 的播放地址需要客户端根据规则自行构建。
                return {
                    "success": True,
                    "stream_id": self.stream_id,
                    "vhost": self.vhost,
                    "app": self.app,
                    "stream_name": self.stream_name,
                    "zlm_internal_api_key": self._zlm_internal_api_key,
                    "online_status_from_zlm": is_online,
                    "origin_url": media_info.get("originUrl"),
                    "origin_type": media_info.get("originTypeStr"),
                    "tracks": media_info.get("tracks", []),
                    "raw_media_info": media_info # 返回原始的media_info
                }
            elif media_info and media_info.get("code") == -2: # Stream not found
                normal_logger.warning(f"ZLM中未找到流 {self.vhost}/{self.app}/{self.stream_name} (getMediaInfo code -2). stream_id(key): {self.stream_id}")
                return {"success": False, "error": f"ZLM中未找到流 {self.vhost}/{self.app}/{self.stream_name}", "code": -2, "response": media_info}
            else:
                error_msg = media_info.get("msg", "获取媒体信息失败") if media_info else "API无响应"
                normal_logger.error(f"调用getMediaInfo获取播放地址失败 for {self.vhost}/{self.app}/{self.stream_name}: {error_msg}. Response: {media_info}")
                return {"success": False, "error": error_msg, "response": media_info}

        except Exception as e:
            exception_logger.exception(f"获取流 {self.stream_id} 的代理播放地址信息时出错: {str(e)}")
            return {"success": False, "error": str(e)} # 添加 success: False

    async def get_zlm_player_proxy_info(self) -> Optional[Dict[str, Any]]:
        """公共方法，调用 get_proxy_url_info 获取播放地址信息。"""
        return await self.get_proxy_url_info()

    async def reconnect(self) -> bool:
        """重新连接流（先停止，再启动）"""
        normal_logger.info(f"请求重新连接流: {self.stream_id} (URL: {self._url})")
        if self._is_stopping:
            normal_logger.warning(f"流 {self.stream_id} 正在停止中，无法执行重连。")
            return False

        # 确保同一时间只有一个重连操作
        # if self._reconnect_lock.locked():
        #     normal_logger.warning(f"流 {self.stream_id} 的重连操作已在进行中，跳过此次请求。")
        #     return False # Or wait for the lock

        # async with self._reconnect_lock:
        normal_logger.info(f"开始执行流 {self.stream_id} 的重连逻辑。")
        # 停止当前流（如果正在运行）
        if self._is_running:
            normal_logger.info(f"重连流 {self.stream_id}: 先停止当前活动。")
            # 调用 stop 但不希望它再次触发重连
            # 这里的 stop 应该是一个彻底的清理，以便 start 可以重新开始
            # 需要确保 stop 内部的重连逻辑不会被这里的调用触发
            # current_reconnect_attempts = self.config.get("reconnect_attempts", 3)
            # self.config["reconnect_attempts"] = 0 # 临时禁用stop内部的重连
            await self.stop() # stop 应该处理任务取消和资源清理
            # self.config["reconnect_attempts"] = current_reconnect_attempts # 恢复
            await asyncio.sleep(1)  # 等待资源完全释放

        # 重置状态以准备重新启动
        self._status = StreamStatus.INITIALIZING
        self._health_status = StreamHealthStatus.UNHEALTHY
        self._last_error = "发起重连"
        self._is_running = False # 确保标记为未运行
        self._zlm_internal_api_key = None # 清除旧的API key
        # self._reconnect_attempts_done = 0 # start 会重置

        normal_logger.info(f"流 {self.stream_id}: 状态已重置，准备重新启动以进行重连。")
        # 重新启动流
        success = await self.start()
        if success:
            normal_logger.info(f"流 {self.stream_id} 重新连接成功 (通过stop/start)。")
            return True
        else:
            normal_logger.error(f"流 {self.stream_id} 重新连接失败 (start调用失败)。")
            self._status = StreamStatus.ERROR # 确保标记为错误
            return False

    def get_opencv_pull_url(self) -> Optional[str]:
        """获取供OpenCV直接拉流的URL (通常是RTMP或RTSP)
           优先使用ZLM getMediaInfo API返回的内网播放地址
        """
        if not (self._status == StreamStatus.ONLINE or self._status == StreamStatus.CONNECTING or self._zlm_internal_api_key):
            normal_logger.warning(f"尝试获取OpenCV拉流地址时，流 {self.stream_id} (app:{self.app}, stream:{self.stream_name}) 可能未成功添加到ZLM或状态不佳。Status: {self._status.name}")
            # return None # 不应立即返回None，因为get_proxy_url_info可能会在首次连接时被调用

        # 尝试从ZLM获取最新的播放地址信息
        # 注意：get_proxy_url_info 是异步的，但此方法是同步的。这是一个潜在问题。
        # 为了解决这个问题，_pull_stream_task 中应该异步调用 get_proxy_url_info。
        # 此方法可以保留一个简化的、基于已知配置的备用URL构造。
        
        # 优先的逻辑应该是 _pull_stream_task 内部异步调用 get_proxy_url_info
        # 并从其结果中选择合适的 internal URL。
        # 这里提供一个基于配置的备用，但不推荐作为主要方式。

        # 构造一个可能的内网播放地址 (如果ZLM和分析服务在同一网络)
        # 这部分逻辑需要 ZLMManager 提供 ZLM 的 host 和 port
        zlm_host = self.manager.zlm_internal_host if self.manager else "127.0.0.1"
        
        # 尝试从config获取覆盖的OpenCV拉流地址
        opencv_pull_url_override = self.config.get("opencv_pull_url")
        if opencv_pull_url_override:
            normal_logger.info(f"使用配置中提供的OpenCV拉流地址: {opencv_pull_url_override} for stream {self.stream_id}")
            return opencv_pull_url_override

        # 默认尝试RTMP，然后RTSP
        # 这些端口应该是ZLM实际监听的端口
        rtmp_port = self.manager.rtmp_port if self.manager else 1935
        rtsp_port = self.manager.rtsp_port if self.manager else 554

        # 优先使用 self.stream_name (即 stream_key_from_manager)
        stream_identifier = self.stream_name 

        # 尝试RTMP内网地址
        # ZLM的播放地址通常是 vhost/app/stream_identifier
        # 如果vhost是__defaultVhost__，实际播放时通常可以省略vhost
        # 但如果OpenCV需要完整的路径，且ZLM配置了多vhost，则需要精确
        # 假设单vhost或__defaultVhost__可以省略
        pull_url_rtmp = f"rtmp://{zlm_host}:{rtmp_port}/{self.app}/{stream_identifier}"
        # 尝试RTSP内网地址
        pull_url_rtsp = f"rtsp://{zlm_host}:{rtsp_port}/{self.app}/{stream_identifier}"

        # 默认返回RTMP，因为OpenCV对RTMP支持较好
        # 调用者（_pull_stream_task）应尝试从 get_proxy_url_info 获取更准确的地址
        normal_logger.info(f"为流 {self.stream_id} 生成的备用OpenCV拉流地址 (RTMP): {pull_url_rtmp}")
        return pull_url_rtmp

    async def _fallback_close_stream_api(self):
        """使用 close_streams API 作为备用关闭方法。"""
        try:
            close_params = {
                "vhost": self.vhost,
                "app": self.app,
                "stream": self.stream_name, # self.stream_name 就是 self.stream_id (即 stream_key_from_manager)
                "force": 1 # 强制关闭
            }
            normal_logger.info(f"尝试使用 close_streams API 关闭流: vhost={self.vhost}, app={self.app}, stream={self.stream_name}, stream_id(key): {self.stream_id}")
            result = self.manager.call_api("close_streams", close_params)
            if result.get("code") == 0:
                normal_logger.info(f"close_streams API 关闭流成功: stream_id(key): {self.stream_id}")
            else:
                normal_logger.warning(f"close_streams API 关闭流失败: {result}, stream_id(key): {self.stream_id}")
        except Exception as e:
            normal_logger.error(f"通过 close_streams API 关闭流时出错: {str(e)}, stream_id(key): {self.stream_id}")