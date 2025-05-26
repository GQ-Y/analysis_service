"""
RTMP协议处理模块
实现RTMP流的拉取和处理
"""

import asyncio
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

from ...base.base_stream import BaseStream
from ...base.stream_interface import StreamStatus, StreamHealthStatus
from .config import RtmpConfig, rtmp_config

class RtmpStream(BaseStream):
    """RTMP流类，实现RTMP协议的流处理"""

    def __init__(self, config: Dict[str, Any]):
        """初始化RTMP流

        Args:
            config: 流配置
        """
        # 调用基类初始化
        super().__init__(config.get("stream_id", ""), config)

        # RTMP特有配置 - 使用统一配置
        try:
            from core.config import settings
            self._rtmp_config = RtmpConfig()
            # 从统一配置加载
            self._rtmp_config.retry_count = settings.PROTOCOLS.retry_count
            self._rtmp_config.retry_interval = settings.PROTOCOLS.retry_interval
            self._rtmp_config.timeout = settings.PROTOCOLS.timeout
            self._rtmp_config.port = settings.PROTOCOLS.rtmp_port
            self._rtmp_config.ssl_port = settings.PROTOCOLS.rtmp_ssl_port
            self._rtmp_config.auth_enable = settings.PROTOCOLS.rtmp_auth_enable
            self._rtmp_config.auth_user = settings.PROTOCOLS.rtmp_auth_user
            self._rtmp_config.auth_password = settings.PROTOCOLS.rtmp_auth_password
            self._rtmp_config.max_buffer_ms = settings.PROTOCOLS.rtmp_max_buffer_ms
        except ImportError:
            # 如果无法导入配置，使用默认值
            self._rtmp_config = RtmpConfig()

        # 如果配置中有RTMP特有配置，更新（优先级更高）
        rtmp_extra = config.get("rtmp", {})
        if rtmp_extra:
            for key, value in rtmp_extra.items():
                if hasattr(self._rtmp_config, key):
                    setattr(self._rtmp_config, key, value)

        # OpenCV视频捕获对象
        self._cap = None

        # 重连配置
        self._retry_count = 0
        self._max_retries = self._rtmp_config.retry_count
        self._retry_interval = self._rtmp_config.retry_interval / 1000.0  # 转换为秒

        # 超时配置
        self._timeout = self._rtmp_config.timeout / 1000.0  # 转换为秒

        normal_logger.info(f"创建RTMP流: {self._stream_id}, URL: {self._url}")

    async def _start_pulling(self) -> bool:
        """开始拉流

        Returns:
            bool: 是否成功启动拉流
        """
        try:
            # 构建OpenCV捕获参数
            cap_params = [
                f"timeout={int(self._timeout * 1000000)}",  # 微秒
                "buffer_size=0"  # 禁用缓冲
            ]

            # 如果需要认证
            auth_url = self._url
            if self._rtmp_config.auth_enable and self._rtmp_config.auth_user and self._rtmp_config.auth_password:
                # 在URL中添加认证参数（RTMP可能使用不同的方式）
                # 这取决于RTMP服务器的实现
                if "?" in auth_url:
                    auth_url += f"&user={self._rtmp_config.auth_user}&password={self._rtmp_config.auth_password}"
                else:
                    auth_url += f"?user={self._rtmp_config.auth_user}&password={self._rtmp_config.auth_password}"
                normal_logger.info(f"使用认证URL: {auth_url}")

            # 构建完整的捕获URL
            if cap_params:
                if "?" in auth_url:
                    cap_url = f"{auth_url}&{'&'.join(cap_params)}"
                else:
                    cap_url = f"{auth_url}?{'&'.join(cap_params)}"
            else:
                cap_url = auth_url

            normal_logger.debug(f"捕获URL: {cap_url}")

            # 创建捕获对象
            self._cap = cv2.VideoCapture(cap_url)

            # 验证捕获对象是否成功创建
            if not self._cap.isOpened():
                exception_logger.exception(f"无法打开RTMP流: {cap_url}")
                self._last_error = "无法打开RTMP流"
                return False

            normal_logger.info(f"成功创建RTMP流捕获对象: {cap_url}")
            return True
        except Exception as e:
            exception_logger.exception(f"创建RTMP流捕获对象异常: {str(e)}")
            self._last_error = f"创建RTMP流异常: {str(e)}"
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

            normal_logger.info(f"成功停止RTMP流: {self._url}")
            return True
        except Exception as e:
            exception_logger.exception(f"停止RTMP流异常: {str(e)}")
            return False

    async def _pull_stream_task(self) -> None:
        """拉流任务，从RTMP服务器拉取视频帧"""
        normal_logger.info(f"启动RTMP流拉流任务: {self._url}")

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
                        normal_logger.warning(f"读取RTMP流帧失败: {self._url}")
                        self._last_error = "读取RTMP流帧失败"

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
                    normal_logger.info(f"RTMP流 {self._stream_id} 拉流任务被取消")
                    break
                except Exception as e:
                    exception_logger.exception(f"RTMP流 {self._stream_id} 拉流异常: {str(e)}")
                    self._last_error = f"拉流异常: {str(e)}"
                    self._stats["errors"] += 1

                    # 尝试重连
                    if not await self._reconnect():
                        # 重连失败，停止任务
                        break
        except asyncio.CancelledError:
            normal_logger.info(f"RTMP流 {self._stream_id} 拉流任务被取消")
        except Exception as e:
            exception_logger.exception(f"RTMP流 {self._stream_id} 拉流任务异常: {str(e)}")
            self._last_error = f"拉流任务异常: {str(e)}"
        finally:
            # 释放资源
            if self._cap:
                self._cap.release()
                self._cap = None

            normal_logger.info(f"RTMP流 {self._stream_id} 拉流任务已停止")

            # 设置状态
            self.set_status(StreamStatus.STOPPED)
            self.set_health_status(StreamHealthStatus.OFFLINE)

    async def _reconnect(self) -> bool:
        """重新连接RTMP流

        Returns:
            bool: 是否成功重连
        """
        # 检查重试次数
        if self._retry_count >= self._max_retries:
            normal_logger.error(f"RTMP流 {self._stream_id} 重试次数已达上限 ({self._max_retries}次)")
            self._last_error = f"重试次数已达上限 ({self._max_retries}次)"
            self.set_status(StreamStatus.ERROR)
            self.set_health_status(StreamHealthStatus.UNHEALTHY)
            return False

        # 增加重试次数和统计
        self._retry_count += 1
        self._stats["reconnects"] += 1

        normal_logger.info(f"RTMP流 {self._stream_id} 尝试重连 (第 {self._retry_count}/{self._max_retries} 次)")

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
            exception_logger.exception(f"RTMP流 {self._stream_id} 重连异常: {str(e)}")
            self._last_error = f"重连异常: {str(e)}"
            return False
