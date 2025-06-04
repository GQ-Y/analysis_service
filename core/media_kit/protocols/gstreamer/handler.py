"""
GStreamer协议处理模块
实现高性能、低延迟的视频流拉取和处理
"""

import asyncio
import time
import threading
import queue
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import cv2

# GStreamer imports
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstApp', '1.0')
    from gi.repository import Gst, GstApp, GLib
    GST_AVAILABLE = True
except ImportError:
    GST_AVAILABLE = False
    print("警告: GStreamer Python绑定未安装，将回退到OpenCV模式")

from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

from ...base.base_stream import BaseStream
from ...base.stream_interface import StreamStatus, StreamHealthStatus
from .config import GStreamerConfig, gstreamer_config

class GStreamerStream(BaseStream):
    """GStreamer流类，实现高性能视频流处理"""

    def __init__(self, config: Dict[str, Any]):
        """初始化GStreamer流

        Args:
            config: 流配置
        """
        # 检查GStreamer可用性
        if not GST_AVAILABLE:
            raise ImportError("GStreamer Python绑定未安装，请安装 python3-gi 和 gstreamer1.0-python3-plugin-loader")
        
        # 初始化GStreamer
        Gst.init(None)
        
        # 调用基类初始化
        super().__init__(config.get("stream_id", ""), config)

        # GStreamer特有配置
        try:
            from core.config import settings
            self._gst_config = GStreamerConfig()
            # 从统一配置加载
            self._gst_config.retry_count = settings.PROTOCOLS.retry_count
            self._gst_config.retry_interval = settings.PROTOCOLS.retry_interval
            self._gst_config.timeout = settings.PROTOCOLS.timeout
        except ImportError:
            # 如果无法导入配置，使用默认值
            self._gst_config = GStreamerConfig()

        # 如果配置中有GStreamer特有配置，更新（优先级更高）
        gst_extra = config.get("gstreamer", {})
        if gst_extra:
            for key, value in gst_extra.items():
                if hasattr(self._gst_config, key):
                    setattr(self._gst_config, key, value)

        # GStreamer对象
        self._pipeline = None
        self._appsink = None
        self._main_loop = None
        self._loop_thread = None
        
        # 帧队列
        self._frame_queue = queue.Queue(maxsize=self._gst_config.buffer_size)
        
        # 重连配置
        self._retry_count = 0
        self._max_retries = self._gst_config.retry_count
        self._retry_interval = self._gst_config.retry_interval / 1000.0  # 转换为秒

        # 超时配置
        self._timeout = self._gst_config.timeout / 1000.0  # 转换为秒
        
        # 统计信息
        self._frame_count = 0
        self._last_frame_time = 0
        self._pipeline_errors = 0

        normal_logger.info(f"创建GStreamer流: {self._stream_id}, URL: {self._url}")

    async def _start_pulling(self) -> bool:
        """开始拉流

        Returns:
            bool: 是否成功启动拉流
        """
        try:
            # 构建GStreamer管道
            pipeline_str = self._build_pipeline()
            
            if not pipeline_str:
                self._last_error = "无法构建GStreamer管道"
                return False

            normal_logger.info(f"GStreamer管道: {pipeline_str}")
            
            # 创建管道
            self._pipeline = Gst.parse_launch(pipeline_str)
            
            if not self._pipeline:
                self._last_error = "创建GStreamer管道失败"
                return False
            
            # 获取appsink元素
            self._appsink = self._pipeline.get_by_name("appsink")
            if not self._appsink:
                self._last_error = "获取appsink元素失败"
                return False
            
            # 配置appsink
            self._appsink.set_property("emit-signals", True)
            self._appsink.set_property("drop", self._gst_config.drop_on_latency)
            self._appsink.set_property("max-buffers", self._gst_config.buffer_size)
            
            # 连接信号
            self._appsink.connect("new-sample", self._on_new_sample)
            
            # 设置管道消息处理
            bus = self._pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)
            
            # 启动GLib主循环
            self._main_loop = GLib.MainLoop()
            self._loop_thread = threading.Thread(target=self._run_main_loop)
            self._loop_thread.daemon = True
            self._loop_thread.start()
            
            # 启动管道
            ret = self._pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                self._last_error = "启动GStreamer管道失败"
                return False
            
            normal_logger.info(f"成功启动GStreamer流: {self._url}")
            return True
            
        except Exception as e:
            exception_logger.exception(f"创建GStreamer流异常: {str(e)}")
            self._last_error = f"创建GStreamer流异常: {str(e)}"
            return False

    async def _stop_pulling(self) -> bool:
        """停止拉流

        Returns:
            bool: 是否成功停止拉流
        """
        try:
            # 停止管道
            if self._pipeline:
                self._pipeline.set_state(Gst.State.NULL)
                self._pipeline = None
            
            # 停止主循环
            if self._main_loop:
                self._main_loop.quit()
                self._main_loop = None
            
            # 等待线程结束
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=2.0)
                self._loop_thread = None
            
            # 清空帧队列
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break

            normal_logger.info(f"成功停止GStreamer流: {self._url}")
            return True
            
        except Exception as e:
            exception_logger.exception(f"停止GStreamer流异常: {str(e)}")
            return False

    def _build_pipeline(self) -> Optional[str]:
        """构建GStreamer管道字符串
        
        Returns:
            Optional[str]: 管道字符串，失败返回None
        """
        try:
            # 如果有自定义管道，直接使用
            if self._gst_config.custom_pipeline:
                return self._gst_config.custom_pipeline
            
            # 解析URL协议
            url_lower = self._url.lower()
            
            if url_lower.startswith("rtsp://"):
                return self._build_rtsp_pipeline()
            elif url_lower.startswith("rtmp://"):
                return self._build_rtmp_pipeline()
            elif url_lower.startswith(("http://", "https://")):
                if ".m3u8" in url_lower:
                    return self._build_hls_pipeline()
                else:
                    return self._build_http_pipeline()
            elif url_lower.startswith("rtp://"):
                return self._build_rtp_pipeline()
            else:
                # 尝试通用管道
                return self._build_generic_pipeline()
                
        except Exception as e:
            exception_logger.exception(f"构建GStreamer管道失败: {str(e)}")
            return None

    def _build_rtsp_pipeline(self) -> str:
        """构建RTSP管道"""
        # 基础管道组件，优先使用TCP传输避免UDP问题
        source = f"rtspsrc location={self._url}"
        
        # 添加RTSP特有配置，强制使用TCP
        source += f" protocols=tcp"  # 只使用TCP
        source += f" latency={self._gst_config.rtsp_latency}"
        source += f" timeout={self._gst_config.network_timeout * 1000000}"  # 转换为纳秒
        source += f" user-agent=\"{self._gst_config.user_agent}\""
        source += f" drop-on-latency=true"  # 启用丢帧
        
        # 添加认证信息
        if self._gst_config.auth_enable and self._gst_config.auth_user:
            auth_url = self._url.replace("rtsp://", f"rtsp://{self._gst_config.auth_user}:{self._gst_config.auth_password}@")
            source = f"rtspsrc location={auth_url}"
        
        # 解码管道 - 使用更稳定的配置
        decode_pipeline = self._build_decode_pipeline()
        
        # 完整管道
        pipeline = f"{source} ! {decode_pipeline} ! appsink name=appsink"
        
        return pipeline

    def _build_rtmp_pipeline(self) -> str:
        """构建RTMP管道"""
        source = f"rtmpsrc location={self._url}"
        source += f" timeout={self._gst_config.network_timeout}"
        
        # RTMP解封装和解码
        demux = "flvdemux name=demux"
        decode_pipeline = self._build_decode_pipeline()
        
        pipeline = f"{source} ! {demux} demux.video ! {decode_pipeline} ! appsink name=appsink"
        
        return pipeline

    def _build_hls_pipeline(self) -> str:
        """构建HLS管道"""
        source = f"souphttpsrc location={self._url}"
        source += f" timeout={self._gst_config.network_timeout}"
        source += f" user-agent=\"{self._gst_config.user_agent}\""
        
        # HLS解封装和解码
        demux = "hlsdemux"
        decode_pipeline = self._build_decode_pipeline()
        
        pipeline = f"{source} ! {demux} ! {decode_pipeline} ! appsink name=appsink"
        
        return pipeline

    def _build_http_pipeline(self) -> str:
        """构建HTTP管道"""
        source = f"souphttpsrc location={self._url}"
        source += f" timeout={self._gst_config.network_timeout}"
        source += f" user-agent=\"{self._gst_config.user_agent}\""
        
        # 自动检测格式并解码
        decode_pipeline = f"decodebin ! {self._build_video_convert_pipeline()}"
        
        pipeline = f"{source} ! {decode_pipeline} ! appsink name=appsink"
        
        return pipeline

    def _build_rtp_pipeline(self) -> str:
        """构建RTP管道"""
        source = f"udpsrc uri={self._url}"
        
        # RTP解封装和解码
        decode_pipeline = f"application/x-rtp ! rtpjitterbuffer ! rtph264depay ! {self._build_decode_pipeline()}"
        
        pipeline = f"{source} ! {decode_pipeline} ! appsink name=appsink"
        
        return pipeline

    def _build_generic_pipeline(self) -> str:
        """构建通用管道"""
        source = f"uridecodebin uri={self._url}"
        
        # 通用解码
        convert_pipeline = self._build_video_convert_pipeline()
        
        pipeline = f"{source} ! {convert_pipeline} ! appsink name=appsink"
        
        return pipeline

    def _build_decode_pipeline(self) -> str:
        """构建解码管道"""
        if not self._gst_config.enable_hardware_decode or self._gst_config.hardware_decoder == "none":
            # 软件解码 - 添加更多容错处理
            return "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! videoscale"
        
        # 硬件解码
        hw_decoder = self._gst_config.hardware_decoder
        
        if hw_decoder == "auto":
            # 自动检测硬件解码器
            hw_decoder = self._detect_hardware_decoder()
        
        if hw_decoder == "nvdec":
            # NVIDIA GPU解码
            return "rtph264depay ! h264parse ! nvh264dec ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR"
        elif hw_decoder == "vaapi":
            # Intel GPU解码
            return "rtph264depay ! h264parse ! vaapih264dec ! vaapipostproc ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR"
        elif hw_decoder == "qsv":
            # Intel Quick Sync解码
            return "rtph264depay ! h264parse ! msdkh264dec ! msdkvpp ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR"
        else:
            # 回退到软件解码
            return "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! videoscale"

    def _build_video_convert_pipeline(self) -> str:
        """构建视频转换管道"""
        return "videoconvert ! videoscale ! video/x-raw,format=BGR"

    def _detect_hardware_decoder(self) -> str:
        """检测可用的硬件解码器"""
        try:
            # 检查NVIDIA GPU
            registry = Gst.Registry.get()
            if registry.find_plugin("nvcodec"):
                normal_logger.info("检测到NVIDIA GPU，使用nvdec硬件解码")
                return "nvdec"
            
            # 检查Intel GPU
            if registry.find_plugin("vaapi"):
                normal_logger.info("检测到Intel GPU，使用vaapi硬件解码")
                return "vaapi"
            
            # 检查Intel Quick Sync
            if registry.find_plugin("msdk"):
                normal_logger.info("检测到Intel Quick Sync，使用qsv硬件解码")
                return "qsv"
            
            normal_logger.info("未检测到硬件解码器，使用软件解码")
            return "none"
            
        except Exception as e:
            normal_logger.warning(f"检测硬件解码器失败: {str(e)}，使用软件解码")
            return "none"

    def _run_main_loop(self):
        """运行GLib主循环"""
        try:
            self._main_loop.run()
        except Exception as e:
            normal_logger.error(f"GLib主循环异常: {str(e)}")

    def _on_new_sample(self, appsink):
        """处理新帧样本"""
        try:
            # 获取样本
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR
            
            # 获取缓冲区
            buffer = sample.get_buffer()
            if not buffer:
                return Gst.FlowReturn.ERROR
            
            # 获取帧数据
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR
            
            try:
                # 获取视频信息
                caps = sample.get_caps()
                structure = caps.get_structure(0)
                width = structure.get_int("width")[1]
                height = structure.get_int("height")[1]
                format_str = structure.get_string("format")
                
                # 根据格式确定通道数
                if format_str == "BGR":
                    channels = 3
                elif format_str == "BGRA" or format_str == "BGRx":
                    channels = 4
                else:
                    normal_logger.warning(f"未知的视频格式: {format_str}，假设为BGR")
                    channels = 3
                
                # 计算期望的数据大小
                expected_size = width * height * channels
                actual_size = len(map_info.data)
                
                if actual_size != expected_size:
                    normal_logger.warning(f"帧数据大小不匹配: 期望{expected_size}, 实际{actual_size}")
                    # 尝试重新计算尺寸
                    if channels == 3:
                        # 尝试不同的尺寸组合
                        total_pixels = actual_size // 3
                        height = int((total_pixels / width) ** 0.5) * 2  # 取最接近的偶数
                        width = total_pixels // height
                
                # 转换为numpy数组
                frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
                
                try:
                    if channels == 4:
                        # BGRA或BGRx格式，转换为BGR
                        frame = frame_data.reshape((height, width, 4))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    else:
                        # BGR格式
                        frame = frame_data.reshape((height, width, 3))
                except ValueError as e:
                    normal_logger.error(f"帧重塑失败: {str(e)}, 数据大小: {len(frame_data)}, 尺寸: {width}x{height}x{channels}")
                    return Gst.FlowReturn.ERROR
                
                # 添加帧到队列（非阻塞）
                try:
                    self._frame_queue.put_nowait((frame.copy(), time.time()))
                    self._frame_count += 1
                    self._last_frame_time = time.time()
                except queue.Full:
                    # 队列满，丢弃旧帧
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait((frame.copy(), time.time()))
                    except queue.Empty:
                        pass
                
            finally:
                buffer.unmap(map_info)
            
            return Gst.FlowReturn.OK
            
        except Exception as e:
            normal_logger.error(f"处理帧样本异常: {str(e)}")
            return Gst.FlowReturn.ERROR

    def _on_bus_message(self, bus, message):
        """处理总线消息"""
        try:
            message_type = message.type
            
            if message_type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                normal_logger.error(f"GStreamer错误: {error.message}, 调试信息: {debug}")
                self._last_error = f"GStreamer错误: {error.message}"
                self._pipeline_errors += 1
                
                # 设置状态为错误
                self.set_status(StreamStatus.ERROR)
                self.set_health_status(StreamHealthStatus.UNHEALTHY)
                
            elif message_type == Gst.MessageType.WARNING:
                warning, debug = message.parse_warning()
                normal_logger.warning(f"GStreamer警告: {warning.message}, 调试信息: {debug}")
                
            elif message_type == Gst.MessageType.EOS:
                normal_logger.info("GStreamer流结束")
                self.set_status(StreamStatus.STOPPED)
                
            elif message_type == Gst.MessageType.STATE_CHANGED:
                if message.src == self._pipeline:
                    old_state, new_state, pending_state = message.parse_state_changed()
                    normal_logger.debug(f"管道状态变化: {old_state.value_nick} -> {new_state.value_nick}")
                    
                    if new_state == Gst.State.PLAYING:
                        self.set_status(StreamStatus.RUNNING)
                        self.set_health_status(StreamHealthStatus.HEALTHY)
                        
        except Exception as e:
            normal_logger.error(f"处理总线消息异常: {str(e)}")

    async def _pull_stream_task(self) -> None:
        """拉流任务，从GStreamer管道获取视频帧"""
        normal_logger.info(f"启动GStreamer流拉流任务: {self._url}")

        # 重置重试计数
        self._retry_count = 0

        try:
            while not self._stop_event.is_set():
                try:
                    # 如果管道未运行，尝试重连
                    if not self._pipeline or self._pipeline.get_state(0).state != Gst.State.PLAYING:
                        if not await self._reconnect():
                            # 重连失败，停止任务
                            break

                    # 从队列获取帧
                    try:
                        frame_data = self._frame_queue.get(timeout=1.0)
                        frame, timestamp = frame_data
                        
                        # 读取成功，重置重试计数
                        self._retry_count = 0
                        
                        # 添加帧到缓冲区
                        self._add_frame_to_buffer(frame)
                        
                        # 帧处理完成，设置事件
                        self._frame_processed_event.set()
                        
                    except queue.Empty:
                        # 超时，检查管道状态
                        if self._pipeline_errors > 0:
                            normal_logger.warning(f"GStreamer流存在错误，尝试重连: {self._url}")
                            if not await self._reconnect():
                                break
                        continue

                except asyncio.CancelledError:
                    normal_logger.info(f"GStreamer流 {self._stream_id} 拉流任务被取消")
                    break
                except Exception as e:
                    exception_logger.exception(f"GStreamer流 {self._stream_id} 拉流异常: {str(e)}")
                    self._last_error = f"拉流异常: {str(e)}"
                    self._stats["errors"] += 1

                    # 尝试重连
                    if not await self._reconnect():
                        # 重连失败，停止任务
                        break

        except asyncio.CancelledError:
            normal_logger.info(f"GStreamer流 {self._stream_id} 拉流任务被取消")
        except Exception as e:
            exception_logger.exception(f"GStreamer流 {self._stream_id} 拉流任务异常: {str(e)}")
            self._last_error = f"拉流任务异常: {str(e)}"
        finally:
            normal_logger.info(f"GStreamer流 {self._stream_id} 拉流任务已停止")

            # 设置状态
            self.set_status(StreamStatus.STOPPED)
            self.set_health_status(StreamHealthStatus.OFFLINE)

    async def _reconnect(self) -> bool:
        """重新连接GStreamer流

        Returns:
            bool: 是否成功重连
        """
        # 检查重试次数
        if self._retry_count >= self._max_retries:
            normal_logger.error(f"GStreamer流 {self._stream_id} 重试次数已达上限 ({self._max_retries}次)")
            self._last_error = f"重试次数已达上限 ({self._max_retries}次)"
            self.set_status(StreamStatus.ERROR)
            self.set_health_status(StreamHealthStatus.UNHEALTHY)
            return False

        # 增加重试次数和统计
        self._retry_count += 1
        self._stats["reconnects"] += 1

        normal_logger.info(f"GStreamer流 {self._stream_id} 尝试重连 (第 {self._retry_count}/{self._max_retries} 次)")

        # 停止当前管道
        await self._stop_pulling()

        # 等待重试间隔
        await asyncio.sleep(self._retry_interval)

        # 设置状态
        self.set_status(StreamStatus.CONNECTING)

        # 重新启动
        try:
            # 重置错误计数
            self._pipeline_errors = 0
            return await self._start_pulling()
        except Exception as e:
            exception_logger.exception(f"GStreamer流 {self._stream_id} 重连异常: {str(e)}")
            self._last_error = f"重连异常: {str(e)}"
            return False

    async def get_info(self) -> Dict[str, Any]:
        """获取流信息"""
        info = await super().get_info()
        
        # 添加GStreamer特有信息
        info.update({
            "gstreamer_version": Gst.version_string() if GST_AVAILABLE else "未安装",
            "hardware_decoder": self._gst_config.hardware_decoder,
            "pipeline_errors": self._pipeline_errors,
            "frame_count": self._frame_count,
            "last_frame_time": self._last_frame_time
        })
        
        return info 