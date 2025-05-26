"""
GB28181协议处理模块
通过ZLMediaKit集成实现GB28181国标设备接入和流处理
"""

import asyncio
import re
import subprocess
import time
import requests
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional, Tuple, List, Union, Set
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

from ...base.base_stream import BaseStream
from ...base.stream_interface import StreamStatus, StreamHealthStatus
from .config import Gb28181Config, gb28181_config

class Gb28181Stream(BaseStream):
    """GB28181流类，通过ZLMediaKit实现GB28181协议的流处理"""

    def __init__(self, config: Dict[str, Any]):
        """初始化GB28181流

        Args:
            config: 流配置
        """
        # 调用基类初始化
        super().__init__(config.get("stream_id", ""), config)

        # GB28181特有配置 - 使用统一配置
        try:
            from core.config import settings
            self._gb28181_config = Gb28181Config()
            # 从统一配置加载
            self._gb28181_config.retry_count = settings.PROTOCOLS.retry_count
            self._gb28181_config.retry_interval = settings.PROTOCOLS.retry_interval
            self._gb28181_config.sip_port = settings.PROTOCOLS.gb28181_sip_port
            self._gb28181_config.device_id = settings.PROTOCOLS.gb28181_device_id
            self._gb28181_config.server_id = settings.PROTOCOLS.gb28181_server_id
            self._gb28181_config.domain = settings.PROTOCOLS.gb28181_domain
            self._gb28181_config.password = settings.PROTOCOLS.gb28181_password
            self._gb28181_config.prefer_stream_type = settings.PROTOCOLS.gb28181_prefer_stream_type
            self._gb28181_config.auto_switch_sub_stream = settings.PROTOCOLS.gb28181_auto_switch_sub_stream
            self._gb28181_config.buffer_size = settings.PROTOCOLS.gb28181_buffer_size

            # ZLM相关配置从统一配置获取
            self._zlm_host = settings.STREAMING.zlm_server_address
            self._zlm_http_port = settings.STREAMING.zlm_http_port
            self._zlm_secret = settings.STREAMING.zlm_api_secret
            self._zlm_rtp_port = 0  # 0表示自动选择
        except ImportError:
            # 如果无法导入配置，使用默认值
            self._gb28181_config = Gb28181Config()
            self._zlm_host = "127.0.0.1"
            self._zlm_http_port = 8088
            self._zlm_secret = ""
            self._zlm_rtp_port = 0

        # 如果配置中有GB28181特有配置，更新（优先级更高）
        gb28181_extra = config.get("gb28181", {})
        if gb28181_extra:
            for key, value in gb28181_extra.items():
                if hasattr(self._gb28181_config, key):
                    setattr(self._gb28181_config, key, value)

        # ZLM配置也可以从config中覆盖
        self._zlm_host = config.get("zlm_host", self._zlm_host)
        self._zlm_http_port = config.get("zlm_http_port", self._zlm_http_port)
        self._zlm_secret = config.get("zlm_secret", self._zlm_secret)
        self._zlm_rtp_port = config.get("zlm_rtp_port", self._zlm_rtp_port)

        # 解析设备信息
        self._parse_device_info(config)

        # 流媒体信息
        self._media_url = ""
        self._rtp_server_id = None

        # 重连配置
        self._retry_count = 0
        self._max_retries = self._gb28181_config.retry_count
        self._retry_interval = self._gb28181_config.retry_interval / 1000.0  # 转换为秒

        # 状态追踪
        self._is_streaming = False

        normal_logger.info(f"创建GB28181流: {self._stream_id}, 设备ID: {self._device_id}, 通道ID: {self._channel_id}")

    def _parse_device_info(self, config: Dict[str, Any]) -> None:
        """解析设备信息

        Args:
            config: 流配置
        """
        # 从URL解析设备ID和通道ID
        # 假设URL格式为gb28181://device_id:channel_id@server
        url = self._url

        # 提取设备ID和通道ID
        pattern = r'gb28181://([^:]+):([^@]+)@'
        match = re.match(pattern, url)
        if match:
            self._device_id = match.group(1)
            self._channel_id = match.group(2)
        else:
            # 使用备用方法解析
            self._device_id = config.get("device_id", "")
            self._channel_id = config.get("channel_id", "")

        # 如果没有设备ID或通道ID，使用流ID作为通道ID
        if not self._device_id:
            self._device_id = config.get("device_id", "")
        if not self._channel_id:
            self._channel_id = config.get("channel_id", "") or self._stream_id

        # 码流类型
        self._stream_type = config.get("stream_type", self._gb28181_config.prefer_stream_type)

        # 国标平台相关配置 (从url或config中提取)
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)

        # 获取SIP服务器信息
        self._sip_server_ip = parsed_url.hostname or config.get("sip_host", self._gb28181_config.sip_host)
        self._sip_server_port = parsed_url.port or config.get("sip_port", self._gb28181_config.sip_port)

    async def _start_pulling(self) -> bool:
        """开始拉流，通过ZLMediaKit创建GB28181 RTP接收服务器

        Returns:
            bool: 是否成功启动拉流
        """
        try:
            # 1. 创建RTP接收服务器 (使用ZLM的API)
            api_url = f"http://{self._zlm_host}:{self._zlm_http_port}/index/api/openRtpServer"

            params = {
                "secret": self._zlm_secret,
                "port": self._zlm_rtp_port,  # 0表示随机端口
                "tcp_mode": 0,  # UDP模式
                "stream_id": f"{self._device_id}_{self._channel_id}"
            }

            async def execute_request():
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(api_url, params=params)
                )
                return response

            response = await execute_request()

            if response.status_code != 200:
                exception_logger.error(f"创建RTP服务器失败: HTTP {response.status_code}")
                self._last_error = f"创建RTP服务器失败: HTTP {response.status_code}"
                return False

            result = response.json()
            if result.get("code") != 0:
                exception_logger.error(f"创建RTP服务器失败: {result.get('msg', '未知错误')}")
                self._last_error = f"创建RTP服务器失败: {result.get('msg', '未知错误')}"
                return False

            # 获取分配的RTP端口和服务器ID
            port = result.get("data", {}).get("port", 0)
            self._rtp_server_id = result.get("data", {}).get("id", "")

            if not port or not self._rtp_server_id:
                exception_logger.error("无法获取RTP服务器端口或ID")
                self._last_error = "无法获取RTP服务器端口或ID"
                return False

            normal_logger.info(f"成功创建RTP服务器: ID={self._rtp_server_id}, 端口={port}")

            # 2. 请求设备推流 (这里需要集成SIP命令发送逻辑)
            # 在实际应用中，这里应该向设备发送SIP Invite请求，请求设备向RTP服务器推流
            # 由于SIP命令发送需要完整的GB28181客户端实现，这里使用模拟代码
            success = await self._request_device_stream(port)

            if not success:
                exception_logger.error(f"请求设备推流失败: {self._device_id}/{self._channel_id}")
                self._last_error = "请求设备推流失败"
                # 关闭RTP服务器
                await self._close_rtp_server()
                return False

            # 3. 构建媒体URL
            self._media_url = f"rtsp://{self._zlm_host}:554/{self._device_id}_{self._channel_id}"

            normal_logger.info(f"成功建立GB28181流: {self._media_url}")
            self._is_streaming = True
            return True

        except Exception as e:
            exception_logger.error(f"启动GB28181流异常: {str(e)}")
            self._last_error = f"启动流异常: {str(e)}"
            return False

    async def _request_device_stream(self, rtp_port: int) -> bool:
        """请求设备推流到指定RTP端口

        这里应该包含向GB28181设备发送SIP Invite命令的逻辑
        由于完整实现超出了本示例范围，这里使用模拟代码

        Args:
            rtp_port: RTP服务器端口

        Returns:
            bool: 是否成功请求设备推流
        """
        # 模拟发送SIP命令
        normal_logger.info(f"向设备 {self._device_id} 请求通道 {self._channel_id} 推流到 {self._zlm_host}:{rtp_port}")

        # 在实际应用中，这里应该:
        # 1. 构建SIP Invite消息
        # 2. 设置SDP内容，包含媒体信息和RTP端口
        # 3. 通过SIP协议发送Invite请求
        # 4. 等待设备响应并建立会话

        # 这里等待3秒模拟建立会话的过程
        await asyncio.sleep(3)

        # 检查流是否已经建立 (查询ZLM的API)
        api_url = f"http://{self._zlm_host}:{self._zlm_http_port}/index/api/isMediaOnline"
        params = {
            "secret": self._zlm_secret,
            "schema": "rtsp",
            "vhost": "__defaultVhost__",
            "app": "",
            "stream": f"{self._device_id}_{self._channel_id}"
        }

        # 最多尝试5次，每次间隔1秒
        for i in range(5):
            try:
                async def check_stream():
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: requests.get(api_url, params=params)
                    )
                    return response

                response = await check_stream()

                if response.status_code == 200:
                    result = response.json()
                    if result.get("code") == 0 and result.get("data", 0) == 1:
                        normal_logger.info(f"设备 {self._device_id} 已经成功推流")
                        return True
            except Exception as e:
                normal_logger.warning(f"检查流状态异常: {str(e)}")

            await asyncio.sleep(1)

        # 模拟返回值
        return False

    async def _close_rtp_server(self) -> bool:
        """关闭RTP服务器

        Returns:
            bool: 是否成功关闭
        """
        if not self._rtp_server_id:
            return True

        try:
            api_url = f"http://{self._zlm_host}:{self._zlm_http_port}/index/api/closeRtpServer"
            params = {
                "secret": self._zlm_secret,
                "stream_id": f"{self._device_id}_{self._channel_id}"
            }

            async def execute_request():
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(api_url, params=params)
                )
                return response

            response = await execute_request()

            if response.status_code != 200:
                exception_logger.error(f"关闭RTP服务器失败: HTTP {response.status_code}")
                return False

            result = response.json()
            if result.get("code") != 0:
                exception_logger.error(f"关闭RTP服务器失败: {result.get('msg', '未知错误')}")
                return False

            normal_logger.info(f"成功关闭RTP服务器: ID={self._rtp_server_id}")
            self._rtp_server_id = None
            return True

        except Exception as e:
            exception_logger.error(f"关闭RTP服务器异常: {str(e)}")
            return False

    async def _stop_device_stream(self) -> bool:
        """停止设备推流

        这里应该包含向GB28181设备发送SIP Bye命令的逻辑
        由于完整实现超出了本示例范围，这里使用模拟代码

        Returns:
            bool: 是否成功停止设备推流
        """
        if not self._is_streaming:
            return True

        # 模拟发送SIP Bye命令
        normal_logger.info(f"向设备 {self._device_id} 发送停止推流命令")

        # 在实际应用中，这里应该:
        # 1. 构建SIP Bye消息
        # 2. 通过SIP协议发送Bye请求
        # 3. 等待设备响应并结束会话

        # 这里等待1秒模拟结束会话的过程
        await asyncio.sleep(1)

        self._is_streaming = False
        return True

    async def _stop_pulling(self) -> bool:
        """停止拉流

        Returns:
            bool: 是否成功停止拉流
        """
        try:
            # 1. 停止设备推流
            await self._stop_device_stream()

            # 2. 关闭RTP服务器
            await self._close_rtp_server()

            self._media_url = ""
            normal_logger.info(f"成功停止GB28181流: {self._device_id}/{self._channel_id}")
            return True
        except Exception as e:
            exception_logger.error(f"停止GB28181流异常: {str(e)}")
            return False

    async def _pull_stream_task(self) -> None:
        """拉流任务，从ZLMediaKit获取GB28181流
        此方法在基类中调用，负责监控流状态
        """
        normal_logger.info(f"启动GB28181流状态监控任务: {self._device_id}/{self._channel_id}")

        # 重置重试计数
        self._retry_count = 0

        try:
            while not self._stop_event.is_set():
                try:
                    # 检查流是否在线
                    if self._is_streaming:
                        # 检查ZLM中的流是否存在
                        api_url = f"http://{self._zlm_host}:{self._zlm_http_port}/index/api/isMediaOnline"
                        params = {
                            "secret": self._zlm_secret,
                            "schema": "rtsp",
                            "vhost": "__defaultVhost__",
                            "app": "",
                            "stream": f"{self._device_id}_{self._channel_id}"
                        }

                        async def check_stream():
                            loop = asyncio.get_event_loop()
                            response = await loop.run_in_executor(
                                None,
                                lambda: requests.get(api_url, params=params)
                            )
                            return response

                        response = await check_stream()

                        if response.status_code != 200 or response.json().get("data", 0) != 1:
                            normal_logger.warning(f"流不在线，尝试重连: {self._device_id}/{self._channel_id}")
                            self._last_error = "流不在线"

                            if not await self._reconnect():
                                # 重连失败，停止任务
                                break

                    # 周期性检查，每10秒检查一次
                    await asyncio.sleep(10)

                except asyncio.CancelledError:
                    normal_logger.info(f"GB28181流 {self._stream_id} 任务被取消")
                    break
                except Exception as e:
                    exception_logger.error(f"GB28181流 {self._stream_id} 监控异常: {str(e)}")
                    self._last_error = f"监控异常: {str(e)}"
                    self._stats["errors"] += 1

                    # 尝试重连
                    if not await self._reconnect():
                        # 重连失败，停止任务
                        break
        except asyncio.CancelledError:
            normal_logger.info(f"GB28181流 {self._stream_id} 任务被取消")
        except Exception as e:
            exception_logger.error(f"GB28181流 {self._stream_id} 任务异常: {str(e)}")
            self._last_error = f"任务异常: {str(e)}"
        finally:
            # 停止流
            await self._stop_device_stream()
            await self._close_rtp_server()

            normal_logger.info(f"GB28181流 {self._stream_id} 任务已停止")

            # 设置状态
            self.set_status(StreamStatus.STOPPED)
            self.set_health_status(StreamHealthStatus.OFFLINE)

    async def _reconnect(self) -> bool:
        """重新连接GB28181流

        Returns:
            bool: 是否成功重连
        """
        # 检查重试次数
        if self._retry_count >= self._max_retries:
            exception_logger.error(f"GB28181流 {self._stream_id} 重试次数已达上限 ({self._max_retries}次)")
            self._last_error = f"重试次数已达上限 ({self._max_retries}次)"
            self.set_status(StreamStatus.ERROR)
            self.set_health_status(StreamHealthStatus.UNHEALTHY)
            return False

        # 增加重试次数和统计
        self._retry_count += 1
        self._stats["reconnects"] += 1

        normal_logger.info(f"GB28181流 {self._stream_id} 尝试重连 (第 {self._retry_count}/{self._max_retries} 次)")

        # 停止当前流
        await self._stop_device_stream()
        await self._close_rtp_server()

        # 等待重试间隔
        await asyncio.sleep(self._retry_interval)

        # 设置状态
        self.set_status(StreamStatus.CONNECTING)

        # 重新创建流
        try:
            return await self._start_pulling()
        except Exception as e:
            exception_logger.error(f"GB28181流 {self._stream_id} 重连异常: {str(e)}")
            self._last_error = f"重连异常: {str(e)}"
            return False

    async def get_stream_url(self) -> str:
        """获取流URL

        Returns:
            str: 流URL
        """
        return self._media_url
