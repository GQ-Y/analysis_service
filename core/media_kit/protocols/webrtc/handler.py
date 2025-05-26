"""
WebRTC协议处理模块
实现WebRTC流的拉取和处理
"""

import asyncio
import time
import cv2
import numpy as np
import json
import uuid
import aiohttp
from typing import Dict, Any, Optional, Tuple, List, Union
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

try:
    import aiortc
    from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaBlackhole
    from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack
    from av import VideoFrame
except ImportError:
    normal_logger.warning("aiortc模块未安装，WebRTC功能将不可用")
    aiortc = None
    MediaPlayer = None
    MediaRecorder = None
    MediaBlackhole = None
    MediaStreamTrack = None
    VideoStreamTrack = None
    VideoFrame = None

from ...base.base_stream import BaseStream
from ...base.stream_interface import StreamStatus, StreamHealthStatus
from .config import WebRTCConfig, webrtc_config

class FrameTransformer(VideoStreamTrack):
    """帧转换器，用于将WebRTC帧转换为OpenCV格式"""

    def __init__(self):
        super().__init__()
        self.frame_queue = asyncio.Queue(maxsize=10)
        self._last_frame = None

    async def recv(self):
        """接收帧数据

        Returns:
            VideoFrame: 视频帧
        """
        if self._last_frame is None:
            # 创建黑帧
            width, height = 640, 480
            img = np.zeros((height, width, 3), np.uint8)
            frame = VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts = 0
            frame.time_base = 1 / 30
            self._last_frame = frame

        try:
            # 非阻塞方式获取帧
            try:
                img = self.frame_queue.get_nowait()
                self.frame_queue.task_done()

                # 转换为VideoFrame
                frame = VideoFrame.from_ndarray(img, format="bgr24")
                frame.pts = self._last_frame.pts + 1
                frame.time_base = self._last_frame.time_base
                self._last_frame = frame
                return frame
            except asyncio.QueueEmpty:
                # 如果队列为空，返回上一帧
                return self._last_frame
        except Exception as e:
            exception_logger.exception(f"WebRTC帧转换异常: {str(e)}")
            return self._last_frame

class VideoFrameConsumer:
    """视频帧消费者，用于接收WebRTC帧"""

    def __init__(self, max_queue_size: int = 10):
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.track = None

    async def consume_frame(self, frame: np.ndarray) -> None:
        """消费一帧

        Args:
            frame: 视频帧
        """
        try:
            if not self.queue.full():
                await self.queue.put(frame)
        except Exception as e:
            exception_logger.exception(f"WebRTC帧消费异常: {str(e)}")

    async def get_frame(self) -> Optional[np.ndarray]:
        """获取一帧

        Returns:
            Optional[np.ndarray]: 视频帧
        """
        try:
            frame = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            self.queue.task_done()
            return frame
        except (asyncio.QueueEmpty, asyncio.TimeoutError):
            return None
        except Exception as e:
            exception_logger.exception(f"WebRTC获取帧异常: {str(e)}")
            return None

class WebRTCStream(BaseStream):
    """WebRTC流类，实现WebRTC协议的流处理"""

    def __init__(self, config: Dict[str, Any]):
        """初始化WebRTC流

        Args:
            config: 流配置
        """
        # 检查aiortc模块是否可用
        if aiortc is None:
            raise RuntimeError("aiortc模块未安装，WebRTC功能不可用")

        # 调用基类初始化
        super().__init__(config.get("stream_id", ""), config)

        # WebRTC特有配置 - 使用统一配置
        try:
            from core.config import settings
            self._webrtc_config = WebRTCConfig()
            # 从统一配置加载
            self._webrtc_config.retry_count = settings.PROTOCOLS.retry_count
            self._webrtc_config.retry_interval = settings.PROTOCOLS.retry_interval
            self._webrtc_config.timeout = settings.PROTOCOLS.timeout
            self._webrtc_config.enable_audio = settings.PROTOCOLS.webrtc_enable_audio
            self._webrtc_config.video_codec = settings.PROTOCOLS.webrtc_video_codec
            self._webrtc_config.max_bitrate = settings.PROTOCOLS.webrtc_max_bitrate
            self._webrtc_config.force_tcp = settings.PROTOCOLS.webrtc_force_tcp
            self._webrtc_config.local_tcp_port = settings.PROTOCOLS.webrtc_local_tcp_port
            self._webrtc_config.use_whip = settings.PROTOCOLS.webrtc_use_whip
            self._webrtc_config.use_whep = settings.PROTOCOLS.webrtc_use_whep
            # ZLM API URL会自动从配置获取
        except ImportError:
            # 如果无法导入配置，使用默认值
            self._webrtc_config = WebRTCConfig()

        # 如果配置中有WebRTC特有配置，更新（优先级更高）
        webrtc_extra = config.get("webrtc", {})
        if webrtc_extra:
            for key, value in webrtc_extra.items():
                if hasattr(self._webrtc_config, key):
                    setattr(self._webrtc_config, key, value)

        # WebRTC对象
        self._pc = None  # PeerConnection
        self._session = None  # HTTP会话
        self._track_consumer = VideoFrameConsumer()  # 视频帧消费者

        # ZLM相关
        self._zlm_api = self._webrtc_config.zlm_api_url
        self._use_zlm = self._webrtc_config.use_zlm_webrtc

        # 检查URL类型，判断是否为WHIP/WHEP
        if '/whip' in self._url.lower():
            self._webrtc_config.use_whip = True
            normal_logger.info(f"检测到WHIP URL: {self._url}")
        elif '/whep' in self._url.lower():
            self._webrtc_config.use_whep = True
            normal_logger.info(f"检测到WHEP URL: {self._url}")

        # 重连配置
        self._retry_count = 0
        self._max_retries = self._webrtc_config.retry_count
        self._retry_interval = self._webrtc_config.retry_interval / 1000.0  # 转换为秒

        # 超时配置
        self._timeout = self._webrtc_config.timeout / 1000.0  # 转换为秒

        normal_logger.info(f"创建WebRTC流: {self._stream_id}, URL: {self._url}")

    async def _start_pulling(self) -> bool:
        """开始拉流

        Returns:
            bool: 是否成功启动拉流
        """
        try:
            # 创建HTTP会话（用于信令）
            self._session = aiohttp.ClientSession()

            # 根据URL类型选择不同的连接方式
            if self._webrtc_config.use_whep:
                # 使用WHEP协议
                return await self._start_whep()
            elif self._webrtc_config.use_whip:
                # 使用WHIP协议
                return await self._start_whip()
            # 如果使用ZLM的WebRTC功能
            elif self._use_zlm and self._zlm_api:
                return await self._start_zlm_webrtc()
            else:
                # 普通WebRTC连接
                return await self._start_direct_webrtc()
        except Exception as e:
            exception_logger.error(f"创建WebRTC流异常: {str(e)}")
            self._last_error = f"创建WebRTC流异常: {str(e)}"
            return False

    async def _start_zlm_webrtc(self) -> bool:
        """通过ZLM启动WebRTC

        Returns:
            bool: 是否成功
        """
        try:
            # 解析URL，提取流ID
            stream_path = self._url.split('/')[-1] if '/' in self._url else self._url

            # 创建WebRTC连接，配置ICE服务器
            self._pc = aiortc.RTCPeerConnection(configuration={
                "iceServers": self._webrtc_config.ice_servers
            })

            # 添加视频轨道接收回调
            @self._pc.on("track")
            async def on_track(track):
                if track.kind == "video":
                    normal_logger.info(f"收到WebRTC视频轨道")

                    # 设置视频轨道
                    self._track_consumer.track = track

                    # 处理视频帧
                    while True:
                        try:
                            # 接收帧
                            frame = await track.recv()

                            # 将帧转换为OpenCV格式
                            if hasattr(frame, "to_ndarray"):
                                # 视频帧转换
                                img = frame.to_ndarray(format="bgr24")

                                # 消费帧
                                await self._track_consumer.consume_frame(img)
                            else:
                                normal_logger.warning("收到非视频帧")
                        except Exception as e:
                            exception_logger.error(f"处理WebRTC帧异常: {str(e)}")
                            break
                elif track.kind == "audio" and self._webrtc_config.enable_audio:
                    normal_logger.info(f"收到WebRTC音频轨道")

            # 创建传输
            self._pc.addTransceiver("video", direction="recvonly")
            if self._webrtc_config.enable_audio:
                self._pc.addTransceiver("audio", direction="recvonly")

            # 创建Offer
            offer = await self._pc.createOffer()
            await self._pc.setLocalDescription(offer)

            # 将Offer发送到ZLM
            api_url = f"{self._zlm_api}/index/api/webrtc"
            params = {
                "type": "play",
                "stream_id": stream_path,
                "sdp": self._pc.localDescription.sdp
            }

            # 如果需要强制TCP连接
            if self._webrtc_config.force_tcp:
                params["transport"] = "tcp"

            async with self._session.post(api_url, json=params) as response:
                if response.status != 200:
                    exception_logger.error(f"ZLM WebRTC请求失败: {response.status}")
                    self._last_error = f"ZLM WebRTC请求失败: {response.status}"
                    return False

                # 解析响应
                result = await response.json()
                if not result.get("code", -1) == 0:
                    exception_logger.error(f"ZLM WebRTC响应错误: {result.get('msg', '未知错误')}")
                    self._last_error = f"ZLM WebRTC响应错误: {result.get('msg', '未知错误')}"
                    return False

                # 获取Answer
                sdp = result.get("sdp", "")
                if not sdp:
                    exception_logger.error("ZLM未返回SDP")
                    self._last_error = "ZLM未返回SDP"
                    return False

                # 设置Answer
                answer = aiortc.RTCSessionDescription(sdp=sdp, type="answer")
                await self._pc.setRemoteDescription(answer)

                # 获取ICE服务器信息（如果有）
                ice_servers = result.get("ice_servers", [])
                if ice_servers:
                    normal_logger.info(f"接收到ZLM ICE服务器配置: {ice_servers}")
                    # 可以在这里更新ICE服务器配置

                normal_logger.info(f"成功创建ZLM WebRTC连接: {stream_path}")
                return True
        except Exception as e:
            exception_logger.error(f"ZLM WebRTC连接异常: {str(e)}")
            self._last_error = f"ZLM WebRTC连接异常: {str(e)}"
            return False

    async def _start_direct_webrtc(self) -> bool:
        """直接通过WebRTC URL启动连接

        Returns:
            bool: 是否成功
        """
        try:
            # 创建WebRTC连接
            self._pc = aiortc.RTCPeerConnection(configuration={
                "iceServers": self._webrtc_config.ice_servers
            })

            # 添加视频轨道接收回调
            @self._pc.on("track")
            async def on_track(track):
                if track.kind == "video":
                    normal_logger.info(f"收到WebRTC视频轨道")

                    # 设置视频轨道
                    self._track_consumer.track = track

                    # 处理视频帧
                    while True:
                        try:
                            # 接收帧
                            frame = await track.recv()

                            # 将帧转换为OpenCV格式
                            if hasattr(frame, "to_ndarray"):
                                # 视频帧转换
                                img = frame.to_ndarray(format="bgr24")

                                # 消费帧
                                await self._track_consumer.consume_frame(img)
                            else:
                                normal_logger.warning("收到非视频帧")
                        except Exception as e:
                            exception_logger.error(f"处理WebRTC帧异常: {str(e)}")
                            break
                elif track.kind == "audio" and self._webrtc_config.enable_audio:
                    normal_logger.info(f"收到WebRTC音频轨道")

            # 创建传输
            self._pc.addTransceiver("video", direction="recvonly")
            if self._webrtc_config.enable_audio:
                self._pc.addTransceiver("audio", direction="recvonly")

            # 创建Offer
            offer = await self._pc.createOffer()
            await self._pc.setLocalDescription(offer)

            # 假设URL是WebRTC信令服务器地址
            async with self._session.post(
                self._url,
                json={"sdp": self._pc.localDescription.sdp, "type": self._pc.localDescription.type},
                timeout=self._timeout
            ) as response:
                if response.status != 200:
                    exception_logger.error(f"WebRTC信令请求失败: {response.status}")
                    self._last_error = f"WebRTC信令请求失败: {response.status}"
                    return False

                # 解析响应
                data = await response.json()
                if "sdp" not in data or "type" not in data:
                    exception_logger.error("信令服务器响应格式错误")
                    self._last_error = "信令服务器响应格式错误"
                    return False

                # 设置Answer
                answer = aiortc.RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                await self._pc.setRemoteDescription(answer)

                normal_logger.info("成功创建直接WebRTC连接")
                return True
        except Exception as e:
            exception_logger.error(f"直接WebRTC连接异常: {str(e)}")
            self._last_error = f"直接WebRTC连接异常: {str(e)}"
            return False

    async def _start_whep(self) -> bool:
        """通过WHEP协议启动WebRTC

        Returns:
            bool: 是否成功
        """
        try:
            # 创建WebRTC连接
            self._pc = aiortc.RTCPeerConnection(configuration={
                "iceServers": self._webrtc_config.ice_servers
            })

            # 添加视频轨道接收回调
            @self._pc.on("track")
            async def on_track(track):
                if track.kind == "video":
                    normal_logger.info(f"收到WebRTC视频轨道")

                    # 设置视频轨道
                    self._track_consumer.track = track

                    # 处理视频帧
                    while True:
                        try:
                            # 接收帧
                            frame = await track.recv()

                            # 将帧转换为OpenCV格式
                            if hasattr(frame, "to_ndarray"):
                                # 视频帧转换
                                img = frame.to_ndarray(format="bgr24")

                                # 消费帧
                                await self._track_consumer.consume_frame(img)
                            else:
                                normal_logger.warning("收到非视频帧")
                        except Exception as e:
                            exception_logger.error(f"处理WebRTC帧异常: {str(e)}")
                            break
                elif track.kind == "audio" and self._webrtc_config.enable_audio:
                    normal_logger.info(f"收到WebRTC音频轨道")

            # 创建传输
            self._pc.addTransceiver("video", direction="recvonly")
            if self._webrtc_config.enable_audio:
                self._pc.addTransceiver("audio", direction="recvonly")

            # 创建Offer
            offer = await self._pc.createOffer()
            await self._pc.setLocalDescription(offer)

            # 发送WHEP请求
            headers = {
                "Content-Type": "application/sdp"
            }

            # 发送Offer到WHEP端点
            async with self._session.post(
                self._url,
                headers=headers,
                data=self._pc.localDescription.sdp,
                timeout=self._timeout
            ) as response:
                if response.status >= 300:
                    exception_logger.error(f"WHEP请求失败: {response.status}")
                    self._last_error = f"WHEP请求失败: {response.status}"
                    return False

                # 读取SDP Answer
                sdp_answer = await response.text()
                if not sdp_answer:
                    exception_logger.error("WHEP服务器未返回SDP")
                    self._last_error = "WHEP服务器未返回SDP"
                    return False

                # 获取ICE服务器信息（从链接头，如果有）
                if response.headers.get("Link"):
                    normal_logger.info(f"接收到WHEP ICE服务器配置: {response.headers.get('Link')}")
                    # TODO: 从Link头解析ICE服务器配置

                # 设置Answer
                answer = aiortc.RTCSessionDescription(sdp=sdp_answer, type="answer")
                await self._pc.setRemoteDescription(answer)

                normal_logger.info(f"成功创建WHEP连接")
                return True
        except Exception as e:
            exception_logger.error(f"WHEP连接异常: {str(e)}")
            self._last_error = f"WHEP连接异常: {str(e)}"
            return False

    async def _start_whip(self) -> bool:
        """通过WHIP协议启动WebRTC推流

        Returns:
            bool: 是否成功
        """
        normal_logger.warning(f"WHIP推流暂不支持")
        self._last_error = f"WHIP推流暂不支持"
        return False

    async def _stop_pulling(self) -> bool:
        """停止拉流

        Returns:
            bool: 是否成功停止拉流
        """
        try:
            # 关闭WebRTC连接
            if self._pc:
                await self._pc.close()
                self._pc = None

            # 关闭HTTP会话
            if self._session:
                await self._session.close()
                self._session = None

            normal_logger.info(f"成功停止WebRTC流: {self._url}")
            return True
        except Exception as e:
            exception_logger.error(f"停止WebRTC流异常: {str(e)}")
            return False

    async def _pull_stream_task(self) -> None:
        """拉流任务，从WebRTC服务器拉取视频帧"""
        normal_logger.info(f"启动WebRTC流拉流任务: {self._url}")

        # 重置重试计数
        self._retry_count = 0

        try:
            # 循环获取视频帧
            while not self._stop_event.is_set():
                try:
                    # 检查连接状态
                    if not self._pc or self._pc.connectionState == "failed" or self._pc.connectionState == "closed":
                        if not await self._reconnect():
                            # 重连失败，停止任务
                            break

                    # 获取帧
                    frame = await self._track_consumer.get_frame()

                    if frame is None:
                        # 获取帧失败
                        normal_logger.warning(f"获取WebRTC帧失败: {self._url}")
                        continue

                    # 添加帧到缓冲区
                    self._add_frame_to_buffer(frame)

                    # 帧处理完成，设置事件
                    self._frame_processed_event.set()

                    # 控制读取速度
                    await asyncio.sleep(0.001)

                except asyncio.CancelledError:
                    normal_logger.info(f"WebRTC流 {self._stream_id} 拉流任务被取消")
                    break
                except Exception as e:
                    exception_logger.error(f"WebRTC流 {self._stream_id} 拉流异常: {str(e)}")
                    self._last_error = f"拉流异常: {str(e)}"
                    self._stats["errors"] += 1

                    # 尝试重连
                    if not await self._reconnect():
                        # 重连失败，停止任务
                        break
        except asyncio.CancelledError:
            normal_logger.info(f"WebRTC流 {self._stream_id} 拉流任务被取消")
        except Exception as e:
            exception_logger.error(f"WebRTC流 {self._stream_id} 拉流任务异常: {str(e)}")
            self._last_error = f"拉流任务异常: {str(e)}"
        finally:
            # 释放资源
            if self._pc:
                await self._pc.close()
                self._pc = None

            if self._session:
                await self._session.close()
                self._session = None

            normal_logger.info(f"WebRTC流 {self._stream_id} 拉流任务已停止")

            # 设置状态
            self.set_status(StreamStatus.STOPPED)
            self.set_health_status(StreamHealthStatus.OFFLINE)

    async def _reconnect(self) -> bool:
        """重新连接WebRTC流

        Returns:
            bool: 是否成功重连
        """
        # 检查重试次数
        if self._retry_count >= self._max_retries:
            exception_logger.error(f"WebRTC流 {self._stream_id} 重试次数已达上限 ({self._max_retries}次)")
            self._last_error = f"重试次数已达上限 ({self._max_retries}次)"
            self.set_status(StreamStatus.ERROR)
            self.set_health_status(StreamHealthStatus.UNHEALTHY)
            return False

        # 增加重试次数和统计
        self._retry_count += 1
        self._stats["reconnects"] += 1

        normal_logger.info(f"WebRTC流 {self._stream_id} 尝试重连 (第 {self._retry_count}/{self._max_retries} 次)")

        # 释放旧的资源
        if self._pc:
            await self._pc.close()
            self._pc = None

        if self._session:
            await self._session.close()
            self._session = None

        # 等待重试间隔
        await asyncio.sleep(self._retry_interval)

        # 设置状态
        self.set_status(StreamStatus.CONNECTING)

        # 重新创建连接
        try:
            return await self._start_pulling()
        except Exception as e:
            exception_logger.error(f"WebRTC流 {self._stream_id} 重连异常: {str(e)}")
            self._last_error = f"重连异常: {str(e)}"
            return False
