"""
流管理器模块
负责管理所有流的创建、停止和状态监控
"""

import asyncio
import threading
from typing import Dict, Any, Optional, Tuple, Set, Callable, List, Union
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

from .stream_interface import (
    IStream, 
    IStreamManager, 
    StreamStatus, 
    StreamHealthStatus
)
from .event_system import event_system

class StreamManager(IStreamManager):
    """流管理器，负责管理所有流实例的生命周期"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StreamManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化流管理器"""
        if self._initialized:
            return
            
        # 流实例映射，格式: 流ID -> 流实例
        self._streams: Dict[str, IStream] = {}
        
        # 流锁，用于保护流映射和操作
        self._streams_lock = asyncio.Lock()
        
        # 事件系统
        self._event_system = event_system
        
        # 流工厂
        # 延迟导入，避免循环依赖
        self._stream_factory = None
        
        # 健康检查任务
        self._health_check_task = None
        self._health_check_interval = 10  # 健康检查间隔，单位秒
        self._stop_health_check = asyncio.Event()
        
        # 统计信息，格式: 流ID -> 统计数据
        self._stats: Dict[str, Dict[str, Any]] = {}
        
        self._initialized = True
    
    async def initialize(self) -> None:
        """初始化流管理器，启动健康检查等任务"""
        # 导入流工厂
        from ..factory.stream_factory import stream_factory
        self._stream_factory = stream_factory
        
        # 启动健康检查任务
        self._stop_health_check.clear()
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        normal_logger.info("流管理器启动完成")
    
    async def shutdown(self) -> None:
        """关闭流管理器，停止所有流和任务"""
        normal_logger.info("正在关闭流管理器...")
        
        # 停止健康检查任务
        if self._health_check_task:
            self._stop_health_check.set()
            try:
                await asyncio.wait_for(self._health_check_task, timeout=2.0)
            except asyncio.TimeoutError:
                normal_logger.warning("等待健康检查任务停止超时")
            self._health_check_task = None
        
        # 停止所有流
        async with self._streams_lock:
            stream_ids = list(self._streams.keys())
        
        for stream_id in stream_ids:
            await self.stop_stream(stream_id)
        
        normal_logger.info("流管理器已关闭")
    
    async def _health_check_loop(self) -> None:
        """健康检查循环任务"""
        normal_logger.info("启动流健康检查任务")
        
        try:
            while not self._stop_health_check.is_set():
                try:
                    await self._check_all_streams_health()
                except Exception as e:
                    exception_logger.exception(f"流健康检查异常: {str(e)}")
                
                # 等待下一次检查
                try:
                    await asyncio.wait_for(
                        self._stop_health_check.wait(),
                        timeout=self._health_check_interval
                    )
                except asyncio.TimeoutError:
                    # 超时正常，继续下一次检查
                    pass
        except asyncio.CancelledError:
            normal_logger.info("流健康检查任务被取消")
        except Exception as e:
            exception_logger.exception(f"流健康检查任务异常退出: {str(e)}")
        finally:
            normal_logger.info("流健康检查任务已停止")
    
    async def _check_all_streams_health(self) -> None:
        """检查所有流的健康状态"""
        async with self._streams_lock:
            stream_ids = list(self._streams.keys())
        
        for stream_id in stream_ids:
            try:
                await self._check_stream_health(stream_id)
            except Exception as e:
                exception_logger.exception(f"检查流 {stream_id} 健康状态异常: {str(e)}")
    
    async def _check_stream_health(self, stream_id: str) -> None:
        """检查单个流的健康状态
        
        Args:
            stream_id: 流ID
        """
        # 获取流实例
        stream = await self.get_stream(stream_id)
        if not stream:
            return
        
        # 当前状态
        current_status = stream.status
        current_health = stream.health_status
        
        # 如果流已经是ERROR或STOPPED状态，不需要再检查
        if current_status in [StreamStatus.ERROR, StreamStatus.STOPPED]:
            return
        
        # 尝试获取一帧，检查是否能正常获取
        try:
            success, frame = await stream.get_frame()
            
            # 更新健康状态
            if success and frame is not None:
                # 帧获取成功，流正常
                new_health = StreamHealthStatus.GOOD
                
                # 如果流状态不是RUNNING，更新为RUNNING
                if current_status != StreamStatus.RUNNING:
                    stream.set_status(StreamStatus.RUNNING)
                    await self._trigger_stream_status_changed(
                        stream_id, 
                        StreamStatus.RUNNING,
                        ""
                    )
            else:
                # 帧获取失败，流可能异常
                # 但可能是暂时性的，先标记为POOR
                new_health = StreamHealthStatus.POOR
        except Exception as e:
            # 获取帧异常，流健康状态不佳
            exception_logger.exception(f"流 {stream_id} 获取帧异常: {str(e)}")
            new_health = StreamHealthStatus.UNHEALTHY
        
        # 如果健康状态变化，更新并触发事件
        if new_health != current_health:
            stream.set_health_status(new_health)
            await self._trigger_stream_health_changed(stream_id, new_health)
            
            # 如果新状态是UNHEALTHY，可能需要尝试重连
            if new_health == StreamHealthStatus.UNHEALTHY:
                await self._handle_stream_unhealthy(stream_id)
    
    async def _handle_stream_unhealthy(self, stream_id: str) -> None:
        """处理流健康状态不佳的情况
        
        Args:
            stream_id: 流ID
        """
        # 获取流实例
        stream = await self.get_stream(stream_id)
        if not stream:
            return
        
        # 如果流已经是ERROR或STOPPED状态，不需要处理
        if stream.status in [StreamStatus.ERROR, StreamStatus.STOPPED]:
            return
        
        # 获取流的统计信息
        stats = self._stats.get(stream_id, {})
        
        # 检查重试次数
        retry_count = stats.get("retry_count", 0)
        max_retries = stats.get("max_retries", 3)
        
        if retry_count < max_retries:
            # 尝试重启流
            normal_logger.info(f"流 {stream_id} 健康状态不佳，尝试重启 (第 {retry_count + 1}/{max_retries} 次)")
            
            # 更新重试次数
            stats["retry_count"] = retry_count + 1
            self._stats[stream_id] = stats
            
            # 尝试重启
            try:
                # 先停止
                await stream.stop()
                # 然后重新启动
                success = await stream.start()
                
                if success:
                    normal_logger.info(f"流 {stream_id} 重启成功")
                    # 重置重试次数
                    stats["retry_count"] = 0
                    self._stats[stream_id] = stats
                else:
                    normal_logger.error(f"流 {stream_id} 重启失败")
                    # 设置状态为ERROR
                    stream.set_status(StreamStatus.ERROR)
                    await self._trigger_stream_status_changed(
                        stream_id, 
                        StreamStatus.ERROR,
                        "流重启失败"
                    )
            except Exception as e:
                exception_logger.exception(f"流 {stream_id} 重启异常: {str(e)}")
                # 设置状态为ERROR
                stream.set_status(StreamStatus.ERROR)
                await self._trigger_stream_status_changed(
                    stream_id, 
                    StreamStatus.ERROR,
                    f"流重启异常: {str(e)}"
                )
        else:
            # 重试次数已达上限，设置为ERROR状态
            normal_logger.error(f"流 {stream_id} 重试次数已达上限 ({max_retries}次)，设置为错误状态")
            stream.set_status(StreamStatus.ERROR)
            await self._trigger_stream_status_changed(
                stream_id, 
                StreamStatus.ERROR,
                "重试次数已达上限"
            )
    
    async def _trigger_stream_status_changed(self, stream_id: str, status: StreamStatus, message: str = "") -> None:
        """触发流状态变化事件
        
        Args:
            stream_id: 流ID
            status: 新状态
            message: 状态消息
        """
        event_data = {
            "stream_id": stream_id,
            "status": status,
            "message": message
        }
        
        # 触发事件
        await self._event_system.trigger_async_event("stream_status_changed", event_data)
    
    async def _trigger_stream_health_changed(self, stream_id: str, health_status: StreamHealthStatus) -> None:
        """触发流健康状态变化事件
        
        Args:
            stream_id: 流ID
            health_status: 新健康状态
        """
        event_data = {
            "stream_id": stream_id,
            "health_status": health_status
        }
        
        # 触发事件
        await self._event_system.trigger_async_event("stream_health_changed", event_data)
    
    async def create_stream(self, stream_id: str, config: Dict[str, Any]) -> bool:
        """创建流
        
        Args:
            stream_id: 流ID
            config: 流配置
            
        Returns:
            bool: 是否成功创建
        """
        # 检查是否已存在
        async with self._streams_lock:
            if stream_id in self._streams:
                normal_logger.info(f"流 {stream_id} 已存在")
                return True
            
        try:
            # 创建流实例
            if not self._stream_factory:
                from ..factory.stream_factory import stream_factory
                self._stream_factory = stream_factory
            
            # 确保配置中包含流ID
            stream_config = config.copy()
            stream_config["stream_id"] = stream_id
            
            # 使用工厂创建流
            stream = self._stream_factory.create_stream(stream_config)
            
            # 启动流
            success = await stream.start()
            if not success:
                normal_logger.error(f"启动流 {stream_id} 失败")
                return False
            
            # 保存流实例
            async with self._streams_lock:
                self._streams[stream_id] = stream
            
            # 初始化统计信息
            self._stats[stream_id] = {
                "create_time": asyncio.get_event_loop().time(),
                "retry_count": 0,
                "max_retries": config.get("max_retries", 3),
                "last_frame_time": 0,
                "total_frames": 0
            }
            
            normal_logger.info(f"成功创建流 {stream_id}")
            return True
        except Exception as e:
            exception_logger.exception(f"创建流 {stream_id} 异常: {str(e)}")
            return False
    
    async def stop_stream(self, stream_id: str) -> bool:
        """停止流
        
        Args:
            stream_id: 流ID
            
        Returns:
            bool: 是否成功停止
        """
        # 获取流实例
        stream = None
        async with self._streams_lock:
            if stream_id in self._streams:
                stream = self._streams[stream_id]
                del self._streams[stream_id]
            else:
                normal_logger.warning(f"流 {stream_id} 不存在，无需停止")
                return True
        
        if not stream:
            return True
        
        try:
            # 停止流
            success = await stream.stop()
            
            # 删除统计信息
            if stream_id in self._stats:
                del self._stats[stream_id]
            
            # 触发事件
            await self._trigger_stream_status_changed(
                stream_id, 
                StreamStatus.STOPPED,
                ""
            )
            
            normal_logger.info(f"成功停止流 {stream_id}")
            return success
        except Exception as e:
            exception_logger.exception(f"停止流 {stream_id} 异常: {str(e)}")
            return False
    
    async def get_stream(self, stream_id: str) -> Optional[IStream]:
        """获取流实例
        
        Args:
            stream_id: 流ID
            
        Returns:
            Optional[IStream]: 流实例，如果不存在则返回None
        """
        async with self._streams_lock:
            return self._streams.get(stream_id)
    
    async def get_stream_status(self, stream_id: str) -> Optional[StreamStatus]:
        """获取流状态
        
        Args:
            stream_id: 流ID
            
        Returns:
            Optional[StreamStatus]: 流状态，如果流不存在则返回None
        """
        stream = await self.get_stream(stream_id)
        if not stream:
            return None
        
        return stream.status
    
    async def get_stream_health(self, stream_id: str) -> Optional[StreamHealthStatus]:
        """获取流健康状态
        
        Args:
            stream_id: 流ID
            
        Returns:
            Optional[StreamHealthStatus]: 流健康状态，如果流不存在则返回None
        """
        stream = await self.get_stream(stream_id)
        if not stream:
            return None
        
        return stream.health_status
    
    async def get_all_streams(self) -> List[Dict[str, Any]]:
        """获取所有流信息
        
        Returns:
            List[Dict[str, Any]]: 所有流的信息列表
        """
        result = []
        
        async with self._streams_lock:
            for stream_id, stream in self._streams.items():
                try:
                    # 获取流信息
                    info = await stream.get_info()
                    
                    # 添加统计信息
                    if stream_id in self._stats:
                        info["stats"] = self._stats[stream_id]
                    
                    result.append(info)
                except Exception as e:
                    exception_logger.exception(f"获取流 {stream_id} 信息异常: {str(e)}")
        
        return result
    
    async def update_stream_stats(self, stream_id: str, stats: Dict[str, Any]) -> bool:
        """更新流统计信息
        
        Args:
            stream_id: 流ID
            stats: 统计信息
            
        Returns:
            bool: 是否成功更新
        """
        if stream_id not in self._stats:
            return False
        
        self._stats[stream_id].update(stats)
        return True
    
    async def update_stream_status(self, stream_id: str, status: StreamStatus, 
                                  health_status: Optional[StreamHealthStatus] = None,
                                  message: str = "") -> bool:
        """更新流状态
        
        Args:
            stream_id: 流ID
            status: 新状态
            health_status: 新健康状态，如果为None则不更新
            message: 状态消息
            
        Returns:
            bool: 是否成功更新
        """
        stream = await self.get_stream(stream_id)
        if not stream:
            return False
        
        # 更新状态
        if stream.status != status:
            stream.set_status(status)
            await self._trigger_stream_status_changed(stream_id, status, message)
        
        # 更新健康状态
        if health_status is not None and stream.health_status != health_status:
            stream.set_health_status(health_status)
            await self._trigger_stream_health_changed(stream_id, health_status)
        
        return True
    
    def register_event_handler(self, event_name: str, handler: Callable) -> None:
        """注册事件处理器
        
        Args:
            event_name: 事件名称
            handler: 处理函数
        """
        # 判断是否为异步处理器
        if asyncio.iscoroutinefunction(handler):
            self._event_system.register_async_handler(event_name, handler)
        else:
            self._event_system.register_sync_handler(event_name, handler)
    
    def unregister_event_handler(self, event_name: str, handler: Callable) -> None:
        """取消注册事件处理器
        
        Args:
            event_name: 事件名称
            handler: 处理函数
        """
        # 判断是否为异步处理器
        if asyncio.iscoroutinefunction(handler):
            self._event_system.unregister_async_handler(event_name, handler)
        else:
            self._event_system.unregister_sync_handler(event_name, handler)

# 单例实例将在需要时创建
