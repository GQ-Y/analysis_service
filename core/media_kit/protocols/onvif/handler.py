"""
ONVIF协议处理模块
实现ONVIF设备接入和控制
"""

import asyncio
import time
import re
import cv2
import numpy as np
import urllib.parse
from typing import Dict, Any, Optional, Tuple, List, Union, Set
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

try:
    # 尝试导入ONVIF相关模块
    import onvif
    from onvif import ONVIFCamera
except ImportError:
    normal_logger.warning("onvif模块未安装，ONVIF功能将不可用")
    onvif = None
    ONVIFCamera = None

from ...base.base_stream import BaseStream
from ...base.stream_interface import StreamStatus, StreamHealthStatus
from .config import OnvifConfig, onvif_config

class OnvifDeviceManager:
    """ONVIF设备管理器，负责管理和缓存ONVIF设备连接"""

    def __init__(self):
        """初始化设备管理器"""
        self.devices = {}  # 缓存的设备实例
        self._cache_lock = asyncio.Lock()

    async def get_device(self,
                    host: str,
                    port: int,
                    username: str,
                    password: str,
                    timeout: int = 10) -> Optional['ONVIFCamera']:
        """获取ONVIF设备实例

        Args:
            host: 设备主机名或IP
            port: 设备端口
            username: 用户名
            password: 密码
            timeout: 超时时间(秒)

        Returns:
            Optional[ONVIFCamera]: ONVIF设备实例，如果创建失败则返回None
        """
        if ONVIFCamera is None:
            normal_logger.error("onvif模块未安装，无法创建设备实例")
            return None

        device_key = f"{host}:{port}:{username}"

        async with self._cache_lock:
            # 检查缓存中是否已存在设备实例
            if device_key in self.devices:
                normal_logger.debug(f"使用缓存的ONVIF设备实例: {host}:{port}")
                return self.devices[device_key]

            # 创建新的设备实例
            try:
                # 使用线程池执行阻塞操作
                loop = asyncio.get_event_loop()
                device = await loop.run_in_executor(
                    None,
                    lambda: ONVIFCamera(
                        host=host,
                        port=port,
                        user=username,
                        passwd=password,
                        wsdl_dir=None,
                        encrypt=True,
                        no_cache=False,
                        timeout=timeout
                    )
                )

                normal_logger.info(f"成功创建ONVIF设备实例: {host}:{port}")
                self.devices[device_key] = device
                return device
            except Exception as e:
                normal_logger.error(f"创建ONVIF设备实例失败: {host}:{port}, 错误: {str(e)}")
                return None

    async def get_device_profiles(self, device: 'ONVIFCamera') -> List[Dict[str, Any]]:
        """获取设备支持的媒体配置文件

        Args:
            device: ONVIF设备实例

        Returns:
            List[Dict[str, Any]]: 媒体配置文件列表
        """
        try:
            # 初始化媒体服务
            media_service = device.create_media_service()

            # 获取配置文件
            loop = asyncio.get_event_loop()
            profiles = await loop.run_in_executor(
                None,
                media_service.GetProfiles
            )

            # 转换为可读格式
            result = []
            for profile in profiles:
                profile_info = {
                    "token": profile.token,
                    "name": profile.Name,
                    "fixed": profile.fixed,
                    "video_encoder": None,
                    "video_source": None,
                    "ptz": None
                }

                # 如果有视频编码器配置
                if hasattr(profile, "VideoEncoderConfiguration"):
                    profile_info["video_encoder"] = {
                        "name": profile.VideoEncoderConfiguration.Name,
                        "token": profile.VideoEncoderConfiguration.token,
                        "encoding": profile.VideoEncoderConfiguration.Encoding,
                        "width": profile.VideoEncoderConfiguration.Resolution.Width,
                        "height": profile.VideoEncoderConfiguration.Resolution.Height,
                        "quality": profile.VideoEncoderConfiguration.Quality,
                        "framerate": profile.VideoEncoderConfiguration.RateControl.FrameRateLimit,
                        "bitrate": profile.VideoEncoderConfiguration.RateControl.BitrateLimit
                    }

                # 如果有视频源配置
                if hasattr(profile, "VideoSourceConfiguration"):
                    profile_info["video_source"] = {
                        "name": profile.VideoSourceConfiguration.Name,
                        "token": profile.VideoSourceConfiguration.token,
                        "source_token": profile.VideoSourceConfiguration.SourceToken,
                        "bounds": {
                            "x": profile.VideoSourceConfiguration.Bounds.x,
                            "y": profile.VideoSourceConfiguration.Bounds.y,
                            "width": profile.VideoSourceConfiguration.Bounds.width,
                            "height": profile.VideoSourceConfiguration.Bounds.height
                        }
                    }

                # 如果有PTZ配置
                if hasattr(profile, "PTZConfiguration"):
                    profile_info["ptz"] = {
                        "name": profile.PTZConfiguration.Name,
                        "token": profile.PTZConfiguration.token,
                        "node_token": profile.PTZConfiguration.NodeToken
                    }

                result.append(profile_info)

            return result
        except Exception as e:
            normal_logger.error(f"获取ONVIF设备配置文件失败: {str(e)}")
            return []

    async def get_stream_uri(self,
                      device: 'ONVIFCamera',
                      profile_token: str,
                      protocol: str = "RTSP") -> Optional[str]:
        """获取媒体流URI

        Args:
            device: ONVIF设备实例
            profile_token: 配置文件Token
            protocol: 流协议类型 (RTSP/HTTP/UDP)

        Returns:
            Optional[str]: 媒体流URI，如果失败则返回None
        """
        try:
            # 初始化媒体服务
            media_service = device.create_media_service()

            # 创建请求对象
            request = media_service.create_type('GetStreamUri')
            request.ProfileToken = profile_token
            request.StreamSetup = {
                'Stream': 'RTP-Unicast',
                'Transport': {
                    'Protocol': protocol
                }
            }

            # 获取流URI
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: media_service.GetStreamUri(request)
            )

            return response.Uri
        except Exception as e:
            normal_logger.error(f"获取ONVIF媒体流URI失败: {str(e)}")
            return None

    async def control_ptz(self,
                    device: 'ONVIFCamera',
                    profile_token: str,
                    x_speed: float,
                    y_speed: float,
                    zoom_speed: float,
                    timeout: float = 1.0,
                    continuous: bool = False) -> bool:
        """控制PTZ

        Args:
            device: ONVIF设备实例
            profile_token: 配置文件Token
            x_speed: X轴速度 (-1.0 ~ 1.0)
            y_speed: Y轴速度 (-1.0 ~ 1.0)
            zoom_speed: 缩放速度 (-1.0 ~ 1.0)
            timeout: 超时时间(秒)，当continuous=False时使用
            continuous: 是否连续移动

        Returns:
            bool: 操作是否成功
        """
        try:
            # 初始化PTZ服务
            ptz_service = device.create_ptz_service()

            if continuous:
                # 创建连续移动请求
                request = ptz_service.create_type('ContinuousMove')
                request.ProfileToken = profile_token
                request.Velocity = {
                    'PanTilt': {
                        'x': x_speed,
                        'y': y_speed
                    },
                    'Zoom': {
                        'x': zoom_speed
                    }
                }

                # 执行连续移动
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: ptz_service.ContinuousMove(request)
                )
            else:
                # 创建相对移动请求
                request = ptz_service.create_type('RelativeMove')
                request.ProfileToken = profile_token
                request.Translation = {
                    'PanTilt': {
                        'x': x_speed,
                        'y': y_speed
                    },
                    'Zoom': {
                        'x': zoom_speed
                    }
                }

                # 执行相对移动
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: ptz_service.RelativeMove(request)
                )

                # 如果不是连续移动，等待超时后停止
                await asyncio.sleep(timeout)

                # 停止移动
                stop_request = ptz_service.create_type('Stop')
                stop_request.ProfileToken = profile_token
                await loop.run_in_executor(
                    None,
                    lambda: ptz_service.Stop(stop_request)
                )

            return True
        except Exception as e:
            normal_logger.error(f"控制PTZ失败: {str(e)}")
            return False

    async def stop_ptz(self, device: 'ONVIFCamera', profile_token: str) -> bool:
        """停止PTZ移动

        Args:
            device: ONVIF设备实例
            profile_token: 配置文件Token

        Returns:
            bool: 操作是否成功
        """
        try:
            # 初始化PTZ服务
            ptz_service = device.create_ptz_service()

            # 创建停止请求
            request = ptz_service.create_type('Stop')
            request.ProfileToken = profile_token

            # 执行停止请求
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: ptz_service.Stop(request)
            )

            return True
        except Exception as e:
            normal_logger.error(f"停止PTZ移动失败: {str(e)}")
            return False

# 单例设备管理器
device_manager = OnvifDeviceManager()

class OnvifStream(BaseStream):
    """ONVIF流类，实现ONVIF协议的流处理和设备控制"""

    def __init__(self, config: Dict[str, Any]):
        """初始化ONVIF流

        Args:
            config: 流配置
        """
        # 检查onvif模块是否可用
        if onvif is None:
            normal_logger.warning("onvif模块未安装，ONVIF功能将受限")

        # 调用基类初始化
        super().__init__(config.get("stream_id", ""), config)

        # ONVIF特有配置 - 使用统一配置
        try:
            from core.config import settings
            self._onvif_config = OnvifConfig()
            # 从统一配置加载
            self._onvif_config.retry_count = settings.PROTOCOLS.retry_count
            self._onvif_config.retry_interval = settings.PROTOCOLS.retry_interval
            self._onvif_config.auth_enable = settings.PROTOCOLS.onvif_auth_enable
            self._onvif_config.auth_username = settings.PROTOCOLS.onvif_auth_username
            self._onvif_config.auth_password = settings.PROTOCOLS.onvif_auth_password
            self._onvif_config.connection_timeout = settings.PROTOCOLS.onvif_connection_timeout
            self._onvif_config.receive_timeout = settings.PROTOCOLS.onvif_receive_timeout
            self._onvif_config.prefer_profile_type = settings.PROTOCOLS.onvif_prefer_profile_type
            self._onvif_config.prefer_h264 = settings.PROTOCOLS.onvif_prefer_h264
            self._onvif_config.prefer_tcp = settings.PROTOCOLS.onvif_prefer_tcp
            self._onvif_config.buffer_size = settings.PROTOCOLS.onvif_buffer_size
        except ImportError:
            # 如果无法导入配置，使用默认值
            self._onvif_config = OnvifConfig()

        # 如果配置中有ONVIF特有配置，更新（优先级更高）
        onvif_extra = config.get("onvif", {})
        if onvif_extra:
            for key, value in onvif_extra.items():
                if hasattr(self._onvif_config, key):
                    setattr(self._onvif_config, key, value)

        # 解析设备信息
        self._parse_device_info()

        # ONVIF设备实例
        self._device = None

        # 配置文件信息
        self._profiles = []
        self._selected_profile = None

        # 媒体流URL
        self._media_url = ""

        # OpenCV视频捕获对象
        self._cap = None

        # 重连配置
        self._retry_count = 0
        self._max_retries = self._onvif_config.retry_count
        self._retry_interval = self._onvif_config.retry_interval / 1000.0  # 转换为秒

        normal_logger.info(f"创建ONVIF流: {self._stream_id}, IP: {self._device_ip}, 端口: {self._device_port}")

    def _parse_device_info(self) -> None:
        """解析设备信息，从URL中提取IP、端口、用户名和密码"""
        # 假设URL格式为onvif://username:password@ip:port/profile_token
        url = self._url

        if url.startswith("onvif://"):
            parsed = urllib.parse.urlparse(url)

            # 提取用户信息
            if parsed.username:
                self._onvif_config.auth_username = parsed.username
            if parsed.password:
                self._onvif_config.auth_password = parsed.password

            # 提取主机和端口
            self._device_ip = parsed.hostname or "127.0.0.1"
            self._device_port = parsed.port or 80

            # 从路径中提取配置文件Token
            if parsed.path and parsed.path != "/":
                self._onvif_config.profile_token = parsed.path.strip("/")
        else:
            # 如果URL格式不符合期望，尝试从配置中获取
            self._device_ip = self._url
            self._device_port = 80

    async def _connect_to_device(self) -> bool:
        """连接到ONVIF设备

        Returns:
            bool: 是否成功连接
        """
        try:
            # 创建设备实例
            self._device = await device_manager.get_device(
                host=self._device_ip,
                port=self._device_port,
                username=self._onvif_config.auth_username,
                password=self._onvif_config.auth_password,
                timeout=self._onvif_config.connection_timeout // 1000  # 转换为秒
            )

            if not self._device:
                normal_logger.error(f"无法连接到ONVIF设备: {self._device_ip}:{self._device_port}")
                self._last_error = "无法连接到ONVIF设备"
                return False

            # 获取设备配置文件
            self._profiles = await device_manager.get_device_profiles(self._device)
            if not self._profiles:
                normal_logger.error(f"无法获取ONVIF设备配置文件: {self._device_ip}:{self._device_port}")
                self._last_error = "无法获取设备配置文件"
                return False

            # 选择配置文件
            self._selected_profile = await self._select_profile()
            if not self._selected_profile:
                normal_logger.error(f"无法选择合适的ONVIF配置文件: {self._device_ip}:{self._device_port}")
                self._last_error = "无法选择合适的配置文件"
                return False

            normal_logger.info(f"已连接到ONVIF设备: {self._device_ip}:{self._device_port}, 使用配置文件: {self._selected_profile['name']}")
            return True
        except Exception as e:
            normal_logger.error(f"连接ONVIF设备异常: {str(e)}")
            self._last_error = f"连接设备异常: {str(e)}"
            return False

    async def _select_profile(self) -> Optional[Dict[str, Any]]:
        """选择合适的配置文件

        Returns:
            Optional[Dict[str, Any]]: 选择的配置文件，如果没有合适的则返回None
        """
        # 如果指定了配置文件Token，直接使用
        if self._onvif_config.profile_token:
            for profile in self._profiles:
                if profile["token"] == self._onvif_config.profile_token:
                    return profile

        # 根据偏好选择配置文件
        prefer_type = self._onvif_config.prefer_profile_type.lower()
        prefer_h264 = self._onvif_config.prefer_h264

        # 首先尝试按名称匹配
        for profile in self._profiles:
            name = profile["name"].lower()

            # 检查名称中是否包含偏好类型
            if prefer_type in name:
                # 检查是否有视频编码器
                if profile["video_encoder"]:
                    # 如果偏好H264，检查编码类型
                    if not prefer_h264 or profile["video_encoder"]["encoding"] == "H264":
                        return profile

        # 如果按名称未找到匹配的，选择第一个可用的配置文件
        for profile in self._profiles:
            if profile["video_encoder"]:
                # 如果偏好H264，检查编码类型
                if not prefer_h264 or profile["video_encoder"]["encoding"] == "H264":
                    return profile

        # 如果仍未找到，选择第一个配置文件
        return self._profiles[0] if self._profiles else None

    async def _start_pulling(self) -> bool:
        """开始拉流

        Returns:
            bool: 是否成功启动拉流
        """
        try:
            # 连接到设备
            if not await self._connect_to_device():
                return False

            # 获取媒体流URI
            protocol = "RTSP" if self._onvif_config.prefer_tcp else "UDP"
            self._media_url = await device_manager.get_stream_uri(
                self._device,
                self._selected_profile["token"],
                protocol
            )

            if not self._media_url:
                normal_logger.error(f"无法获取ONVIF媒体流URI: {self._device_ip}:{self._device_port}")
                self._last_error = "无法获取媒体流URI"
                return False

            normal_logger.info(f"获取到ONVIF媒体流URL: {self._media_url}")

            # 如果是RTSP协议，确保使用TCP传输
            if protocol == "RTSP" and self._onvif_config.prefer_tcp:
                # 打开 RTSP 流，使用 TCP 传输
                self._cap = cv2.VideoCapture(self._media_url, cv2.CAP_FFMPEG)
                if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._onvif_config.buffer_size)
                # 设置RTSP传输为TCP
                if hasattr(cv2, 'CAP_PROP_RTSP_TRANSPORT'):
                    self._cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, cv2.CAP_RTSP_TRANSPORT_TCP)
            else:
                # 使用普通方式打开
                self._cap = cv2.VideoCapture(self._media_url)
                if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._onvif_config.buffer_size)

            # 验证捕获对象是否成功创建
            if not self._cap.isOpened():
                normal_logger.error(f"无法打开ONVIF媒体流: {self._media_url}")
                self._last_error = "无法打开媒体流"
                return False

            normal_logger.info(f"成功创建ONVIF流捕获对象: {self._media_url}")
            return True
        except Exception as e:
            normal_logger.error(f"创建ONVIF流捕获对象异常: {str(e)}")
            self._last_error = f"创建流异常: {str(e)}"
            return False

    async def _stop_pulling(self) -> bool:
        """停止拉流

        Returns:
            bool: 是否成功停止拉流
        """
        try:
            # 释放OpenCV捕获对象
            if self._cap:
                self._cap.release()
                self._cap = None

            self._media_url = ""

            normal_logger.info(f"成功停止ONVIF流: {self._device_ip}:{self._device_port}")
            return True
        except Exception as e:
            normal_logger.error(f"停止ONVIF流异常: {str(e)}")
            return False

    async def _pull_stream_task(self) -> None:
        """拉流任务，从ONVIF媒体流拉取视频帧"""
        normal_logger.info(f"启动ONVIF流拉流任务: {self._device_ip}:{self._device_port}")

        # 重置重试计数
        self._retry_count = 0

        try:
            while not self._stop_event.is_set():
                try:
                    # 如果捕获对象未初始化或已关闭，重新打开
                    if self._cap is None or not self._cap.isOpened():
                        if not await self._reconnect():
                            # 重连失败，停止任务
                            break

                    # 读取一帧
                    ret, frame = self._cap.read()

                    if not ret or frame is None:
                        # 读取失败，尝试重连
                        normal_logger.warning(f"读取ONVIF流帧失败: {self._media_url}")
                        self._last_error = "读取ONVIF流帧失败"

                        if not await self._reconnect():
                            # 重连失败，停止任务
                            break

                        continue

                    # 读取成功，重置重试计数
                    self._retry_count = 0

                    # 添加帧到缓冲区
                    self._add_frame_to_buffer(frame)

                    # 帧处理完成，设置事件
                    self._frame_processed_event.set()

                    # 控制读取速度，避免占用太多CPU
                    await asyncio.sleep(0.001)

                except asyncio.CancelledError:
                    normal_logger.info(f"ONVIF流 {self._stream_id} 拉流任务被取消")
                    break
                except Exception as e:
                    normal_logger.error(f"ONVIF流 {self._stream_id} 拉流异常: {str(e)}")
                    self._last_error = f"拉流异常: {str(e)}"
                    self._stats["errors"] += 1

                    # 尝试重连
                    if not await self._reconnect():
                        # 重连失败，停止任务
                        break
        except asyncio.CancelledError:
            normal_logger.info(f"ONVIF流 {self._stream_id} 拉流任务被取消")
        except Exception as e:
            normal_logger.error(f"ONVIF流 {self._stream_id} 拉流任务异常: {str(e)}")
            self._last_error = f"拉流任务异常: {str(e)}"
        finally:
            # 释放资源
            if self._cap:
                self._cap.release()
                self._cap = None

            normal_logger.info(f"ONVIF流 {self._stream_id} 拉流任务已停止")

            # 设置状态
            self.set_status(StreamStatus.STOPPED)
            self.set_health_status(StreamHealthStatus.OFFLINE)

    async def _reconnect(self) -> bool:
        """重新连接ONVIF流

        Returns:
            bool: 是否成功重连
        """
        # 检查重试次数
        if self._retry_count >= self._max_retries:
            normal_logger.error(f"ONVIF流 {self._stream_id} 重试次数已达上限 ({self._max_retries}次)")
            self._last_error = f"重试次数已达上限 ({self._max_retries}次)"
            self.set_status(StreamStatus.ERROR)
            self.set_health_status(StreamHealthStatus.UNHEALTHY)
            return False

        # 增加重试次数和统计
        self._retry_count += 1
        self._stats["reconnects"] += 1

        normal_logger.info(f"ONVIF流 {self._stream_id} 尝试重连 (第 {self._retry_count}/{self._max_retries} 次)")

        # 释放旧的捕获对象
        if self._cap:
            self._cap.release()
            self._cap = None

        # 等待重试间隔
        await asyncio.sleep(self._retry_interval)

        # 设置状态
        self.set_status(StreamStatus.CONNECTING)

        # 重新创建捕获对象
        try:
            return await self._start_pulling()
        except Exception as e:
            normal_logger.error(f"ONVIF流 {self._stream_id} 重连异常: {str(e)}")
            self._last_error = f"重连异常: {str(e)}"
            return False

    async def ptz_control(self,
                     x_speed: float,
                     y_speed: float,
                     zoom_speed: float = 0.0,
                     continuous: bool = False,
                     timeout: float = 1.0) -> bool:
        """控制PTZ

        Args:
            x_speed: X轴速度 (-1.0 ~ 1.0)
            y_speed: Y轴速度 (-1.0 ~ 1.0)
            zoom_speed: 缩放速度 (-1.0 ~ 1.0)
            continuous: 是否连续移动
            timeout: 操作超时时间(秒)，当continuous=False时使用

        Returns:
            bool: 操作是否成功
        """
        if not self._device or not self._selected_profile:
            normal_logger.error("设备未连接，无法控制PTZ")
            return False

        # 检查是否有PTZ能力
        if not self._selected_profile.get("ptz"):
            normal_logger.error("当前配置文件不支持PTZ控制")
            return False

        # 调用设备管理器进行PTZ控制
        return await device_manager.control_ptz(
            device=self._device,
            profile_token=self._selected_profile["token"],
            x_speed=x_speed * self._onvif_config.ptz_speed_x,
            y_speed=y_speed * self._onvif_config.ptz_speed_y,
            zoom_speed=zoom_speed * self._onvif_config.ptz_speed_zoom,
            timeout=timeout,
            continuous=continuous
        )

    async def ptz_stop(self) -> bool:
        """停止PTZ移动

        Returns:
            bool: 操作是否成功
        """
        if not self._device or not self._selected_profile:
            normal_logger.error("设备未连接，无法停止PTZ")
            return False

        # 检查是否有PTZ能力
        if not self._selected_profile.get("ptz"):
            normal_logger.error("当前配置文件不支持PTZ控制")
            return False

        # 调用设备管理器停止PTZ
        return await device_manager.stop_ptz(
            device=self._device,
            profile_token=self._selected_profile["token"]
        )
