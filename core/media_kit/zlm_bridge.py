"""
ZLMediaKit与OpenCV桥接模块
负责将ZLMediaKit与当前系统中的OpenCV分析流程集成
"""
import os
import cv2
import asyncio
import threading
from typing import Dict, Any, Optional, Tuple, List, Set, Callable
import time
import traceback
import numpy as np
from queue import Queue, Empty

from shared.utils.logger import get_normal_logger, get_exception_logger
from core.task_management.stream.status import StreamStatus, StreamHealthStatus
from shared.utils.app_state import app_state_manager

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class ZLMBridge:
    """ZLMediaKit与OpenCV桥接类"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ZLMBridge, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化桥接器"""
        if self._initialized:
            return
            
        # ZLM管理器引用，延迟加载避免循环导入
        self._zlm_manager = None
        
        # 流管理器引用，延迟加载避免循环导入
        self._stream_manager = None
        
        # 状态
        self._is_running = False
        
        # 桥接映射关系
        self._stream_mapping = {}  # 分析系统流ID -> ZLM流ID
        self._reverse_mapping = {}  # ZLM流ID -> 分析系统流ID
        
        # 状态同步锁
        self._mapping_lock = threading.Lock()
        
        # 设置初始化标记
        self._initialized = True
        
        normal_logger.info("ZLMediaKit桥接器初始化完成")
    
    async def initialize(self) -> None:
        """初始化桥接器
        
        Returns:
            None
        """
        try:
            # 已经运行则跳过
            if self._is_running:
                normal_logger.info("ZLMediaKit桥接器已经在运行中")
                return
                
            normal_logger.info("初始化ZLMediaKit桥接器...")
            
            # 导入ZLM管理器，延迟导入避免循环依赖
            from .zlm_manager import zlm_manager
            self._zlm_manager = zlm_manager
            
            # 获取流管理器，延迟导入避免循环依赖
            from shared.utils.app_state import app_state_manager
            self._stream_manager = app_state_manager.get_stream_manager()
            
            # 注册事件回调
            self._register_callbacks()
            
            # 标记为运行中
            self._is_running = True
            
            normal_logger.info("ZLMediaKit桥接器初始化完成")
        except Exception as e:
            exception_logger.exception(f"初始化ZLMediaKit桥接器失败: {str(e)}")
    
    async def shutdown(self) -> None:
        """关闭桥接器
        
        Returns:
            None
        """
        try:
            if not self._is_running:
                normal_logger.info("ZLMediaKit桥接器未运行，无需关闭")
                return
                
            normal_logger.info("正在关闭ZLMediaKit桥接器...")
            
            # 取消注册事件回调
            if self._zlm_manager:
                # TODO: 取消回调注册
                pass
            
            # 停止所有桥接流
            with self._mapping_lock:
                stream_ids = list(self._stream_mapping.keys())
            
            for stream_id in stream_ids:
                await self.stop_bridge(stream_id)
            
            # 标记为未运行
            self._is_running = False
            
            normal_logger.info("ZLMediaKit桥接器已关闭")
        except Exception as e:
            exception_logger.exception(f"关闭ZLMediaKit桥接器时出错: {str(e)}")
    
    def _register_callbacks(self) -> None:
        """注册事件回调"""
        # 注册ZLM事件回调
        if self._zlm_manager:
            # ZLM流状态变化回调
            self._zlm_manager.register_event_callback("stream_status_changed", self._on_zlm_stream_status_changed)
            # 其他事件...
    
    async def create_bridge(self, stream_id: str, config: Dict[str, Any]) -> bool:
        """创建流桥接
        
        Args:
            stream_id: 分析系统中的流ID
            config: 流配置
        
        Returns:
            bool: 是否成功创建桥接
        """
        try:
            # 检查是否已经存在
            with self._mapping_lock:
                if stream_id in self._stream_mapping:
                    normal_logger.info(f"流 {stream_id} 的桥接已存在")
                    return True
            
            # 创建ZLM流ID
            # 获取任务ID和视频ID
            task_id = config.get("task_id", "")
            video_id = config.get("video_id", "")
            
            # 如果有任务ID和视频ID，则使用它们构建流ID
            if task_id and video_id:
                zlm_stream_id = f"task_{task_id}_video_{video_id}"
            else:
                # 否则使用旧的命名方式
                zlm_stream_id = f"zlm_{stream_id}"
            
            # 确保配置中包含URL
            if "url" not in config:
                exception_logger.error(f"创建桥接失败: 配置中缺少URL")
                return False
            
            # 配置ZLM流
            zlm_config = config.copy()
            zlm_config["stream_name"] = zlm_stream_id
            
            # 创建ZLM流
            success = await self._zlm_manager.create_stream(zlm_stream_id, zlm_config)
            if not success:
                exception_logger.error(f"创建ZLM流 {zlm_stream_id} 失败")
                return False
            
            # 保存映射关系
            with self._mapping_lock:
                self._stream_mapping[stream_id] = zlm_stream_id
                self._reverse_mapping[zlm_stream_id] = stream_id
            
            normal_logger.info(f"成功创建流桥接: {stream_id} -> {zlm_stream_id}")
            return True
        except Exception as e:
            exception_logger.exception(f"创建流桥接时出错: {str(e)}")
            return False
    
    async def stop_bridge(self, stream_id: str) -> bool:
        """停止流桥接
        
        Args:
            stream_id: 分析系统中的流ID
        
        Returns:
            bool: 是否成功停止桥接
        """
        try:
            # 获取ZLM流ID
            zlm_stream_id = None
            with self._mapping_lock:
                if stream_id in self._stream_mapping:
                    zlm_stream_id = self._stream_mapping[stream_id]
                    del self._stream_mapping[stream_id]
                    if zlm_stream_id in self._reverse_mapping:
                        del self._reverse_mapping[zlm_stream_id]
            
            if not zlm_stream_id:
                normal_logger.warning(f"流 {stream_id} 的桥接不存在，无需停止")
                return True
            
            # 停止ZLM流
            success = await self._zlm_manager.stop_stream(zlm_stream_id)
            if not success:
                exception_logger.error(f"停止ZLM流 {zlm_stream_id} 失败")
                return False
            
            normal_logger.info(f"成功停止流桥接: {stream_id} -> {zlm_stream_id}")
            return True
        except Exception as e:
            exception_logger.exception(f"停止流桥接时出错: {str(e)}")
            return False
    
    async def get_bridge_status(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取桥接状态
        
        Args:
            stream_id: 分析系统中的流ID
        
        Returns:
            Optional[Dict[str, Any]]: 桥接状态，如果桥接不存在则返回None
        """
        try:
            # 获取ZLM流ID
            zlm_stream_id = None
            with self._mapping_lock:
                if stream_id in self._stream_mapping:
                    zlm_stream_id = self._stream_mapping[stream_id]
            
            if not zlm_stream_id:
                normal_logger.warning(f"流 {stream_id} 的桥接不存在")
                return None
            
            # 获取ZLM流信息
            info = await self._zlm_manager.get_stream_info(zlm_stream_id)
            if not info:
                exception_logger.error(f"获取ZLM流 {zlm_stream_id} 信息失败")
                return None
            
            # 转换为桥接状态
            bridge_info = {
                "stream_id": stream_id,
                "zlm_stream_id": zlm_stream_id,
                "status": info.get("status"),
                "health_status": info.get("health_status"),
                "last_error": info.get("last_error"),
                "stats": info.get("stats", {})
            }
            
            return bridge_info
        except Exception as e:
            exception_logger.exception(f"获取桥接状态时出错: {str(e)}")
            return None
    
    async def subscribe_zlm_stream(self, stream_id: str, subscriber_id: str) -> Tuple[bool, Optional[asyncio.Queue]]:
        """订阅ZLM流
        
        Args:
            stream_id: 分析系统中的流ID
            subscriber_id: 订阅者ID
        
        Returns:
            Tuple[bool, Optional[asyncio.Queue]]: (是否成功, 帧队列)
        """
        try:
            # 获取ZLM流ID
            zlm_stream_id = None
            with self._mapping_lock:
                if stream_id in self._stream_mapping:
                    zlm_stream_id = self._stream_mapping[stream_id]
            
            if not zlm_stream_id:
                normal_logger.warning(f"流 {stream_id} 的桥接不存在，无法订阅")
                return False, None
            
            # 从ZLM获取流对象
            try:
                from .zlm_stream import ZLMVideoStream
                with self._zlm_manager._stream_lock:
                    if zlm_stream_id not in self._zlm_manager._streams:
                        normal_logger.warning(f"ZLM流 {zlm_stream_id} 不存在，无法订阅")
                        return False, None
                    
                    stream = self._zlm_manager._streams[zlm_stream_id]
                
                # 订阅流
                success, frame_queue = await stream.subscribe(subscriber_id)
                if not success:
                    exception_logger.error(f"订阅ZLM流 {zlm_stream_id} 失败")
                    return False, None
                
                normal_logger.info(f"成功订阅ZLM流: {stream_id} -> {zlm_stream_id}, 订阅者: {subscriber_id}")
                return True, frame_queue
            except ImportError as e:
                exception_logger.error(f"导入ZLMVideoStream失败: {str(e)}")
                return False, None
            except Exception as e:
                exception_logger.exception(f"从ZLM获取流对象时出错: {str(e)}")
                return False, None
        except Exception as e:
            exception_logger.exception(f"订阅ZLM流时出错: {str(e)}")
            return False, None
    
    async def unsubscribe_zlm_stream(self, stream_id: str, subscriber_id: str) -> bool:
        """取消订阅ZLM流
        
        Args:
            stream_id: 分析系统中的流ID
            subscriber_id: 订阅者ID
        
        Returns:
            bool: 是否成功取消订阅
        """
        try:
            # 获取ZLM流ID
            zlm_stream_id = None
            with self._mapping_lock:
                if stream_id in self._stream_mapping:
                    zlm_stream_id = self._stream_mapping[stream_id]
            
            if not zlm_stream_id:
                normal_logger.warning(f"流 {stream_id} 的桥接不存在，无法取消订阅")
                return False
            
            # 从ZLM获取流对象
            from .zlm_stream import ZLMVideoStream
            with self._zlm_manager._stream_lock:
                if zlm_stream_id not in self._zlm_manager._streams:
                    normal_logger.warning(f"ZLM流 {zlm_stream_id} 不存在，无法取消订阅")
                    return False
                
                stream = self._zlm_manager._streams[zlm_stream_id]
            
            # 取消订阅
            success = await stream.unsubscribe(subscriber_id)
            if not success:
                exception_logger.error(f"取消订阅ZLM流 {zlm_stream_id} 失败")
                return False
            
            normal_logger.info(f"成功取消订阅ZLM流: {stream_id} -> {zlm_stream_id}, 订阅者: {subscriber_id}")
            return True
        except Exception as e:
            exception_logger.exception(f"取消订阅ZLM流时出错: {str(e)}")
            return False
    
    def _on_zlm_stream_status_changed(self, data: Dict[str, Any]) -> None:
        """ZLM流状态变化事件处理
        
        Args:
            data: 事件数据
        """
        try:
            # 获取ZLM流ID
            zlm_stream_id = data.get("stream_id")
            if not zlm_stream_id:
                return
            
            # 获取分析系统流ID
            stream_id = None
            with self._mapping_lock:
                if zlm_stream_id in self._reverse_mapping:
                    stream_id = self._reverse_mapping[zlm_stream_id]
            
            if not stream_id:
                return
            
            # 转换状态
            zlm_status = data.get("status")
            if zlm_status == "running":
                status = StreamStatus.RUNNING
            elif zlm_status == "connecting":
                status = StreamStatus.CONNECTING
            elif zlm_status == "stopped":
                status = StreamStatus.STOPPED
            elif zlm_status == "error":
                status = StreamStatus.ERROR
            else:
                status = StreamStatus.UNKNOWN
            
            # 转换健康状态
            zlm_health = data.get("health_status")
            if zlm_health == "good":
                health_status = StreamHealthStatus.GOOD
            elif zlm_health == "poor":
                health_status = StreamHealthStatus.POOR
            elif zlm_health == "error":
                health_status = StreamHealthStatus.UNHEALTHY
            elif zlm_health == "offline":
                health_status = StreamHealthStatus.OFFLINE
            elif zlm_health == "unhealthy":
                health_status = StreamHealthStatus.UNHEALTHY
            else:
                health_status = StreamHealthStatus.UNKNOWN
            
            # 通知分析系统流管理器
            if self._stream_manager:
                asyncio.create_task(self._stream_manager.update_stream_status(
                    stream_id, 
                    status, 
                    health_status, 
                    data.get("last_error", "")
                ))
        except Exception as e:
            exception_logger.exception(f"处理ZLM流状态变化事件时出错: {str(e)}")

    async def on_stream_changed(self, stream_id: str, regist: bool = True) -> None:
        """处理流注册/注销事件"""
        try:
            stream_manager = app_state_manager.get_stream_manager()
            if not stream_manager:
                normal_logger.warning("流管理器未初始化，无法处理流变更事件")
                return

            # 处理流变更事件
            if regist:
                normal_logger.info(f"流注册事件: {stream_id}")
            else:
                normal_logger.info(f"流注销事件: {stream_id}")

        except Exception as e:
            exception_logger.exception(f"处理流变更事件时出错: {str(e)}")

# 单例实例
zlm_bridge = ZLMBridge() 