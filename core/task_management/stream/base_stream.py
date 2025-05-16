"""
基础视频流类
实现通用的错误处理和重连机制
"""
import asyncio
import time
import threading
from typing import Dict, Any, Optional, Tuple, Set
from asyncio import Queue, Lock
import traceback

import numpy as np
from loguru import logger

from core.config import settings
from shared.utils.logger import setup_logger
from .status import StreamStatus, StreamHealthStatus
from .interface import IVideoStream

logger = setup_logger(__name__)

class BaseVideoStream(IVideoStream):
    """基础视频流类，实现通用功能"""
    
    def __init__(self, stream_id: str, config: Dict[str, Any]):
        """初始化基础视频流
        
        Args:
            stream_id: 流ID
            config: 流配置
        """
        self._stream_id = stream_id
        self._config = config
        self._url = config.get("url", "")
        
        # 状态
        self._status = StreamStatus.INITIALIZING
        self._health_status = StreamHealthStatus.UNKNOWN
        self._last_error = ""
        self._last_frame_time = None
        self._start_time = None
        
        # 订阅者
        self._subscribers = {}  # subscriber_id -> Queue
        self._subscriber_lock = Lock()
        
        # 控制
        self._is_running = False
        self._stop_event = asyncio.Event()
        
        # 重连配置
        self._reconnect_attempts = config.get("reconnect_attempts", settings.STREAMING.reconnect_attempts)
        self._reconnect_base_delay = config.get("reconnect_delay", settings.STREAMING.reconnect_delay)
        self._reconnect_max_delay = config.get("max_reconnect_delay", 60.0)
        self._max_consecutive_errors = config.get("max_consecutive_errors", settings.STREAMING.max_consecutive_errors)
        
        logger.info(f"基础视频流 {stream_id} 初始化完成: {self._url}")
    
    @property
    def stream_id(self) -> str:
        """获取流ID"""
        return self._stream_id
        
    @property
    def url(self) -> str:
        """获取流URL"""
        return self._url
        
    @property
    def config(self) -> Dict[str, Any]:
        """获取流配置"""
        return self._config
        
    @property
    def subscriber_count(self) -> int:
        """获取订阅者数量"""
        return len(self._subscribers)
        
    @property
    def subscribers(self) -> Set[str]:
        """获取订阅者ID集合"""
        return set(self._subscribers.keys())
        
    def get_status(self) -> StreamStatus:
        """获取流状态"""
        return self._status
        
    def get_health_status(self) -> StreamHealthStatus:
        """获取流健康状态"""
        return self._health_status
        
    def set_status(self, status: StreamStatus) -> None:
        """设置流状态
        
        Args:
            status: 新状态
        """
        if self._status != status:
            logger.info(f"流 {self._stream_id} 状态变更: {self._status.name} -> {status.name}")
            self._status = status
            
    def set_health_status(self, health_status: StreamHealthStatus) -> None:
        """设置流健康状态
        
        Args:
            health_status: 新健康状态
        """
        if self._health_status != health_status:
            logger.info(f"流 {self._stream_id} 健康状态变更: {self._health_status.name} -> {health_status.name}")
            self._health_status = health_status
            
    def set_last_error(self, error_msg: str) -> None:
        """设置最后错误信息
        
        Args:
            error_msg: 错误信息
        """
        if error_msg:
            logger.error(f"流 {self._stream_id} 错误: {error_msg}")
            self._last_error = error_msg
            
    async def _reconnect_with_backoff(self) -> bool:
        """使用指数退避策略重连
        
        Returns:
            bool: 是否成功重连
        """
        attempt = 0
        while attempt < self._reconnect_attempts and not self._stop_event.is_set():
            # 计算延迟时间（指数退避）
            delay = min(self._reconnect_base_delay * (2 ** attempt), self._reconnect_max_delay)
            
            # 等待延迟时间
            logger.info(f"等待 {delay:.1f} 秒后重试连接 ({attempt+1}/{self._reconnect_attempts})...")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                if self._stop_event.is_set():
                    logger.info("重连过程中收到停止请求")
                    return False
            except asyncio.TimeoutError:
                # 超时正常，继续重连
                pass
            
            # 尝试重连
            try:
                success = await self._connect()
                if success:
                    logger.info(f"重连成功: {self._stream_id}")
                    return True
            except Exception as e:
                logger.error(f"重连时出错: {str(e)}")
                logger.error(traceback.format_exc())
                
            attempt += 1
            
        if attempt >= self._reconnect_attempts:
            logger.error(f"超过最大重试次数({self._reconnect_attempts})，停止重连")
            self.set_status(StreamStatus.ERROR)
            self.set_health_status(StreamHealthStatus.UNHEALTHY)
            self.set_last_error(f"超过最大重试次数({self._reconnect_attempts})")
            
        return False
        
    async def _connect(self) -> bool:
        """连接到流
        
        Returns:
            bool: 是否成功连接
        """
        # 子类需要实现此方法
        raise NotImplementedError("子类必须实现_connect方法")
        
    async def get_info(self) -> Dict[str, Any]:
        """获取流信息
        
        Returns:
            Dict[str, Any]: 流信息
        """
        # 子类可以扩展此方法
        return {
            "stream_id": self._stream_id,
            "url": self._url,
            "status": self._status.value,
            "status_text": self._get_status_text(),
            "health_status": self._health_status.value,
            "health_status_text": self._get_health_status_text(),
            "last_error": self._last_error,
            "subscriber_count": self.subscriber_count,
            "start_time": self._start_time,
            "last_frame_time": self._last_frame_time
        }
        
    def _get_status_text(self) -> str:
        """获取状态文本"""
        status_texts = {
            StreamStatus.INITIALIZING: "初始化中",
            StreamStatus.CONNECTING: "连接中",
            StreamStatus.ONLINE: "在线",
            StreamStatus.OFFLINE: "离线",
            StreamStatus.ERROR: "错误",
            StreamStatus.RUNNING: "运行中",
            StreamStatus.RECONNECTING: "重连中",
            StreamStatus.PAUSED: "已暂停",
            StreamStatus.STOPPED: "已停止",
            StreamStatus.UNKNOWN: "未知"
        }
        return status_texts.get(self._status, "未知")
        
    def _get_health_status_text(self) -> str:
        """获取健康状态文本"""
        health_texts = {
            StreamHealthStatus.HEALTHY: "健康",
            StreamHealthStatus.GOOD: "良好",
            StreamHealthStatus.DEGRADED: "性能下降",
            StreamHealthStatus.POOR: "较差",
            StreamHealthStatus.UNSTABLE: "不稳定",
            StreamHealthStatus.UNHEALTHY: "不健康",
            StreamHealthStatus.ERROR: "错误",
            StreamHealthStatus.OFFLINE: "离线",
            StreamHealthStatus.UNKNOWN: "未知"
        }
        return health_texts.get(self._health_status, "未知")
