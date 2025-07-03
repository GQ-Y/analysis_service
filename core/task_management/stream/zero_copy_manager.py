"""
零拷贝流管理器
重构StreamManager支持共享内存流缓冲和零拷贝帧引用传递
"""
import asyncio
import time
import threading
from typing import Dict, Optional, Tuple, Any, List
import traceback

from ...interfaces.zero_copy_stream_interface import (
    IZeroCopyVideoStream,
    ZeroCopyStreamConfig,
    AsyncFrameReferenceQueue
)
from ...interfaces.stream_interface import StreamStatus, StreamHealthStatus
from ...frame.frame_reference import FrameReference, FrameReferenceManager
from ...memory.memory_pool import MemoryPool
from .manager import StreamManager
from .zero_copy_rtsp_stream import ZeroCopyRTSPStream

# 使用项目现有的日志系统
try:
    from shared.utils.logger import get_normal_logger, get_exception_logger, get_analysis_logger
    normal_logger = get_normal_logger(__name__)
    exception_logger = get_exception_logger(__name__)
    analysis_logger = get_analysis_logger()
except ImportError:
    import logging
    normal_logger = logging.getLogger(__name__)
    exception_logger = logging.getLogger(__name__)
    analysis_logger = logging.getLogger(__name__)


class ZeroCopyStreamManager(StreamManager):
    """
    零拷贝流管理器
    扩展基础流管理器，支持零拷贝帧引用传递和共享内存池
    """

    def __init__(self):
        """
        初始化零拷贝流管理器
        """
        super().__init__()

        # 零拷贝相关组件（延迟初始化）
        self.memory_pool: Optional[MemoryPool] = None
        self.frame_ref_manager: Optional[FrameReferenceManager] = None
        self._zero_copy_initialized = False

    def set_memory_pool(self, memory_pool: MemoryPool):
        """
        设置内存池并初始化零拷贝组件

        Args:
            memory_pool: 内存池实例
        """
        self.memory_pool = memory_pool
        self.frame_ref_manager = FrameReferenceManager()
        self._zero_copy_initialized = True
        normal_logger.info("零拷贝流管理器内存池设置完成")
        
        # 零拷贝流缓冲区
        self._zero_copy_buffers: Dict[str, AsyncFrameReferenceQueue] = {}
        self._zero_copy_configs: Dict[str, ZeroCopyStreamConfig] = {}

        # 零拷贝流实例管理
        self.zero_copy_streams: Dict[str, ZeroCopyRTSPStream] = {}
        
        # 内存监控
        self._memory_stats = {
            "total_allocations": 0,
            "allocation_failures": 0,
            "memory_pressure_events": 0,
            "frames_dropped": 0,
        }
        
        # 性能统计
        self._performance_stats = {
            "zero_copy_operations": 0,
            "traditional_operations": 0,
            "batch_operations": 0,
            "avg_allocation_time": 0.0,
        }
        
        normal_logger.info("零拷贝流管理器初始化完成")

    def _check_zero_copy_ready(self) -> bool:
        """
        检查零拷贝组件是否已准备就绪

        Returns:
            bool: 是否准备就绪
        """
        if not self._zero_copy_initialized:
            normal_logger.error("零拷贝组件未初始化，请先调用 set_memory_pool()")
            return False

        if not self.memory_pool or not self.memory_pool.initialized:
            normal_logger.error("内存池未初始化或不可用")
            return False

        return True

    async def subscribe_stream_zero_copy(self,
                                       stream_id: str, 
                                       subscriber_id: str, 
                                       config: Dict[str, Any],
                                       zero_copy_config: Optional[ZeroCopyStreamConfig] = None) -> Tuple[bool, Optional[AsyncFrameReferenceQueue]]:
        """
        订阅视频流（零拷贝模式）
        
        Args:
            stream_id: 流ID
            subscriber_id: 订阅者ID
            config: 流配置
            zero_copy_config: 零拷贝配置
            
        Returns:
            Tuple[bool, Optional[AsyncFrameReferenceQueue]]: (是否成功, 帧引用队列)
        """
        try:
            # 检查零拷贝组件是否就绪
            if not self._check_zero_copy_ready():
                return False, None

            # 使用默认零拷贝配置
            if zero_copy_config is None:
                zero_copy_config = ZeroCopyStreamConfig()

            # 直接创建零拷贝RTSP流
            stream_url = config.get("url") or config.get("stream_url")
            if not stream_url:
                normal_logger.error(f"配置中缺少流地址: {stream_id}")
                return False, None

            # 检查是否已存在零拷贝流
            if stream_id in self.zero_copy_streams:
                stream = self.zero_copy_streams[stream_id]
                normal_logger.info(f"使用已存在的零拷贝流: {stream_id}")
            else:
                # 创建新的零拷贝RTSP流
                stream = ZeroCopyRTSPStream(stream_id, stream_url, config)

                # 设置内存池
                if not stream.set_memory_pool(self.memory_pool):
                    normal_logger.error(f"无法为零拷贝流 {stream_id} 设置内存池")
                    return False, None

                # 启动流
                if not await stream.start():
                    normal_logger.error(f"无法启动零拷贝流: {stream_id}")
                    return False, None

                # 保存流引用
                self.zero_copy_streams[stream_id] = stream
                normal_logger.info(f"创建并启动零拷贝RTSP流: {stream_id}, URL: {stream_url}")
            
            # 设置内存池
            if not stream.set_memory_pool(self.memory_pool):
                normal_logger.error(f"无法为流 {stream_id} 设置内存池")
                return False, None
            
            # 创建零拷贝缓冲区（帧引用队列），并注册到流的订阅者列表中，便于拉流线程进行拥塞控制
            buffer_key = f"{stream_id}_{subscriber_id}"
            frame_queue = AsyncFrameReferenceQueue(maxsize=zero_copy_config.max_in_flight_frames)

            # 将队列注册到流内部，保证 _process_frame 能够正确统计 in_flight_frames
            try:
                with stream.subscriber_lock:
                    stream._subscribers[subscriber_id] = frame_queue  # noqa: SLF001  (内部属性，性能考虑)
            except Exception as reg_err:
                exception_logger.exception(
                    f"注册订阅者队列到流 {stream_id} 失败: {reg_err}"
                )

            # 在管理器侧保存映射，方便后续取消订阅与监控
            self._zero_copy_buffers[buffer_key] = frame_queue
            self._zero_copy_configs[buffer_key] = zero_copy_config

            # 启动帧引用分发任务 —— 将队列中的帧提供给任务处理器
            asyncio.create_task(
                self._distribute_frame_references(
                    stream, buffer_key, frame_queue, zero_copy_config
                )
            )
            
            normal_logger.info(f"零拷贝订阅成功: stream_id={stream_id}, subscriber_id={subscriber_id}")
            return True, frame_queue
            
        except Exception as e:
            exception_logger.exception(f"零拷贝订阅失败: {str(e)}")
            return False, None
    
    async def _distribute_frame_references(self, 
                                         stream: IZeroCopyVideoStream,
                                         buffer_key: str,
                                         frame_queue: AsyncFrameReferenceQueue,
                                         config: ZeroCopyStreamConfig) -> None:
        """
        分发帧引用任务
        
        Args:
            stream: 零拷贝视频流
            buffer_key: 缓冲区键
            frame_queue: 帧引用队列
            config: 零拷贝配置
        """
        normal_logger.info(f"启动帧引用分发任务: {buffer_key}")
        
        try:
            while buffer_key in self._zero_copy_buffers:
                # 检查内存压力
                if config.enable_memory_monitoring and stream.is_memory_pressure_high():
                    self._memory_stats["memory_pressure_events"] += 1
                    
                    if config.frame_drop_on_pressure:
                        # 内存压力下丢帧
                        normal_logger.warning(f"内存压力高，跳过帧获取: {buffer_key}")
                        await asyncio.sleep(0.01)  # 短暂等待
                        continue
                
                # 获取帧引用
                if config.enable_batch_processing:
                    # 批量获取
                    success, frame_refs = await stream.get_frame_references_batch(config.batch_size)
                    if success and frame_refs:
                        for frame_ref in frame_refs:
                            if frame_queue.full():
                                # 队列满时丢弃最旧的帧
                                old_ref = await frame_queue.get_nowait()
                                if old_ref:
                                    old_ref.release()
                                self._memory_stats["frames_dropped"] += 1
                            
                            await frame_queue.put(frame_ref)
                        
                        self._performance_stats["batch_operations"] += 1
                        self._performance_stats["zero_copy_operations"] += len(frame_refs)
                    else:
                        # 添加调试日志：批量获取失败
                        analysis_logger.warning(f"[帧引用分发] 批量获取帧引用失败: {buffer_key}, success={success}, frame_refs={len(frame_refs) if frame_refs else 0}")
                else:
                    # 单帧获取
                    success, frame_ref = await stream.get_frame_reference()
                    if success and frame_ref:
                        if frame_queue.full():
                            # 队列满时丢弃最旧的帧
                            old_ref = await frame_queue.get_nowait()
                            if old_ref:
                                old_ref.release()
                            self._memory_stats["frames_dropped"] += 1
                        
                        await frame_queue.put(frame_ref)
                        self._performance_stats["zero_copy_operations"] += 1
                    else:
                        # 添加调试日志：单帧获取失败
                        analysis_logger.warning(f"[帧引用分发] 单帧获取帧引用失败: {buffer_key}, success={success}, frame_ref={frame_ref is not None}")
                        
                        # 检查流状态
                        if hasattr(stream, 'get_status'):
                            stream_status = stream.get_status()
                            analysis_logger.info(f"[帧引用分发] 流状态: {buffer_key}, status={stream_status}")
                        
                        # 检查是否有订阅者
                        if hasattr(stream, '_subscribers'):
                            subscriber_count = len(stream._subscribers)
                            analysis_logger.info(f"[帧引用分发] 订阅者数量: {buffer_key}, subscribers={subscriber_count}")
                
                # 短暂等待避免CPU占用过高
                await asyncio.sleep(0.001)
                
        except Exception as e:
            exception_logger.exception(f"帧引用分发任务异常: {buffer_key}, {str(e)}")
        finally:
            normal_logger.info(f"帧引用分发任务结束: {buffer_key}")
    
    async def unsubscribe_stream_zero_copy(self, stream_id: str, subscriber_id: str) -> bool:
        """
        取消订阅视频流（零拷贝模式）
        
        Args:
            stream_id: 流ID
            subscriber_id: 订阅者ID
            
        Returns:
            bool: 是否成功取消订阅
        """
        try:
            buffer_key = f"{stream_id}_{subscriber_id}"
            
            # 清理零拷贝缓冲区
            if buffer_key in self._zero_copy_buffers:
                frame_queue = self._zero_copy_buffers[buffer_key]
                
                # 释放队列中的所有帧引用
                while not frame_queue.empty():
                    frame_ref = await frame_queue.get_nowait()
                    if frame_ref:
                        frame_ref.release()
                
                del self._zero_copy_buffers[buffer_key]
                del self._zero_copy_configs[buffer_key]
                
                normal_logger.info(f"零拷贝取消订阅成功: {buffer_key}")
            
            # 调用基类取消订阅
            return await self.unsubscribe_stream(stream_id, subscriber_id)
            
        except Exception as e:
            exception_logger.exception(f"零拷贝取消订阅失败: {str(e)}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存统计信息
        
        Returns:
            Dict[str, Any]: 内存统计信息
        """
        pool_stats = self.memory_pool.get_stats() if self.memory_pool else {}
        
        return {
            "memory_pool": pool_stats,
            "manager_stats": self._memory_stats.copy(),
            "performance_stats": self._performance_stats.copy(),
            "active_buffers": len(self._zero_copy_buffers),
            "frame_ref_manager": self.frame_ref_manager.get_manager_stats(),
        }
    
    def get_zero_copy_buffer_stats(self) -> Dict[str, Any]:
        """
        获取零拷贝缓冲区统计信息
        
        Returns:
            Dict[str, Any]: 缓冲区统计信息
        """
        buffer_stats = {}
        for buffer_key, frame_queue in self._zero_copy_buffers.items():
            buffer_stats[buffer_key] = {
                "queue_size": frame_queue.qsize(),
                "queue_stats": frame_queue.get_stats(),
                "config": self._zero_copy_configs[buffer_key].to_dict(),
            }
        
        return buffer_stats
