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
        分发帧引用任务 - 增强版本，支持重试和降级机制
        
        Args:
            stream: 零拷贝视频流
            buffer_key: 缓冲区键
            frame_queue: 帧引用队列
            config: 零拷贝配置
        """
        normal_logger.info(f"启动帧引用分发任务: {buffer_key}")
        
        # 分发状态跟踪
        consecutive_failures = 0
        last_success_time = time.time()
        total_distributed = 0
        
        try:
            while buffer_key in self._zero_copy_buffers:
                try:
                    # 检查内存压力
                    if config.enable_memory_monitoring and stream.is_memory_pressure_high():
                        self._memory_stats["memory_pressure_events"] += 1
                        
                        if config.frame_drop_on_pressure:
                            # 内存压力下丢帧
                            normal_logger.warning(f"内存压力高，跳过帧获取: {buffer_key}")
                            await asyncio.sleep(0.01)  # 短暂等待
                            continue
                    
                    success = False
                    frame_refs = []
                    
                    # 获取帧引用（支持重试）
                    for retry_count in range(3):  # 最多重试3次
                        if config.enable_batch_processing:
                            # 批量获取
                            success, frame_refs = await stream.get_frame_references_batch(config.batch_size)
                            if success and frame_refs:
                                # 批量分发成功
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
                                total_distributed += len(frame_refs)
                                break
                            else:
                                # 批量获取失败，尝试单帧获取作为降级
                                if retry_count == 0:
                                    analysis_logger.warning(f"[帧引用分发] 批量获取失败，尝试单帧降级: {buffer_key}")
                        
                        # 单帧获取（主要路径或降级路径）
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
                            total_distributed += 1
                            break
                        
                        # 重试前等待
                        if retry_count < 2:
                            await asyncio.sleep(0.005)  # 5ms
                    
                    # 处理获取结果
                    if success:
                        # 成功重置失败计数
                        if consecutive_failures > 0:
                            analysis_logger.info(f"[帧引用分发] {buffer_key} 恢复正常，连续失败 {consecutive_failures} 次后成功")
                            consecutive_failures = 0
                        
                        last_success_time = time.time()
                        
                        # 短暂等待避免CPU占用过高
                        await asyncio.sleep(0.001)
                    else:
                        # 失败处理
                        consecutive_failures += 1
                        
                        # 根据失败次数调整等待时间（指数退避）
                        if consecutive_failures <= 10:
                            wait_time = 0.01  # 10ms
                        elif consecutive_failures <= 50:
                            wait_time = 0.05  # 50ms
                        else:
                            wait_time = 0.1   # 100ms
                        
                        # 定期记录失败状态
                        if consecutive_failures % 100 == 0:
                            time_since_success = time.time() - last_success_time
                            analysis_logger.warning(f"[帧引用分发] {buffer_key} 连续失败 {consecutive_failures} 次, "
                                                   f"上次成功: {time_since_success:.1f}秒前, 总分发: {total_distributed}")
                            
                            # 检查流状态
                            if hasattr(stream, 'get_status'):
                                stream_status = stream.get_status()
                                analysis_logger.info(f"[帧引用分发] 流状态: {buffer_key}, status={stream_status}")
                            
                            # 检查连接状态
                            if hasattr(stream, 'is_connected'):
                                is_connected = stream.is_connected
                                analysis_logger.info(f"[帧引用分发] 连接状态: {buffer_key}, is_connected={is_connected}")
                        
                        await asyncio.sleep(wait_time)
                        
                except asyncio.CancelledError:
                    # 任务被取消
                    normal_logger.info(f"帧引用分发任务被取消: {buffer_key}")
                    break
                except Exception as e:
                    consecutive_failures += 1
                    exception_logger.exception(f"帧引用分发异常: {buffer_key}, {str(e)}")
                    await asyncio.sleep(0.1)  # 异常后等待更长时间
                    
        except Exception as e:
            exception_logger.exception(f"帧引用分发任务严重异常: {buffer_key}, {str(e)}")
        finally:
            normal_logger.info(f"帧引用分发任务结束: {buffer_key}, 总分发帧数: {total_distributed}, "
                             f"最终连续失败: {consecutive_failures} 次")
    
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

    async def check_zero_copy_system_health(self) -> Dict[str, Any]:
        """
        检查零拷贝系统健康状态
        
        Returns:
            Dict[str, Any]: 健康状态报告
        """
        health_report = {
            "timestamp": time.time(),
            "overall_health": "healthy",
            "streams": {},
            "memory_stats": self._memory_stats.copy(),
            "performance_stats": self._performance_stats.copy(),
            "issues": []
        }
        
        try:
            # 检查每个流的状态
            for buffer_key, buffer_info in self._zero_copy_buffers.items():
                stream = buffer_info.get("stream")
                frame_queue = buffer_info.get("frame_queue")
                
                stream_health = {
                    "stream_id": getattr(stream, '_stream_id', 'unknown'),
                    "is_running": getattr(stream, 'is_running', False),
                    "is_connected": getattr(stream, 'is_connected', False),
                    "queue_size": frame_queue.qsize() if frame_queue else 0,
                    "queue_maxsize": getattr(frame_queue, 'maxsize', 0),
                    "frame_count": getattr(stream, 'frame_count', 0),
                    "error_count": getattr(stream, 'error_count', 0),
                    "last_frame_time": getattr(stream, 'last_frame_time', 0),
                }
                
                # 检查流是否有问题
                current_time = time.time()
                time_since_last_frame = current_time - stream_health["last_frame_time"]
                
                if not stream_health["is_running"]:
                    health_report["issues"].append(f"流 {buffer_key} 未运行")
                elif not stream_health["is_connected"]:
                    health_report["issues"].append(f"流 {buffer_key} 未连接")
                elif time_since_last_frame > 30 and stream_health["last_frame_time"] > 0:
                    health_report["issues"].append(f"流 {buffer_key} 超过30秒未收到帧")
                elif stream_health["queue_size"] == 0:
                    health_report["issues"].append(f"流 {buffer_key} 帧引用队列为空")
                elif stream_health["error_count"] > 100:
                    health_report["issues"].append(f"流 {buffer_key} 错误计数过高: {stream_health['error_count']}")
                
                # 检查帧引用获取统计
                if hasattr(stream, '_get_reference_failures'):
                    stream_health["reference_failures"] = stream._get_reference_failures
                    stream_health["last_successful_get_time"] = getattr(stream, '_last_successful_get_time', 0)
                    
                    if stream_health["reference_failures"] > 1000:
                        health_report["issues"].append(f"流 {buffer_key} 帧引用获取失败次数过多: {stream_health['reference_failures']}")
                
                health_report["streams"][buffer_key] = stream_health
            
            # 检查内存压力
            if self._memory_stats.get("memory_pressure_events", 0) > 100:
                health_report["issues"].append(f"内存压力事件过多: {self._memory_stats['memory_pressure_events']}")
            
            # 检查丢帧率
            total_frames = self._performance_stats.get("zero_copy_operations", 0)
            dropped_frames = self._memory_stats.get("frames_dropped", 0)
            if total_frames > 0:
                drop_rate = (dropped_frames / total_frames) * 100
                health_report["drop_rate_percent"] = round(drop_rate, 2)
                if drop_rate > 10:  # 丢帧率超过10%
                    health_report["issues"].append(f"丢帧率过高: {drop_rate:.1f}%")
            
            # 设置整体健康状态
            if health_report["issues"]:
                if len(health_report["issues"]) > 5:
                    health_report["overall_health"] = "critical"
                else:
                    health_report["overall_health"] = "warning"
            
            return health_report
            
        except Exception as e:
            exception_logger.exception(f"零拷贝系统健康检查失败: {str(e)}")
            return {
                "timestamp": time.time(),
                "overall_health": "error",
                "error": str(e),
                "streams": {},
                "issues": ["健康检查执行失败"]
            }

    def log_system_health_summary(self):
        """
        记录系统健康状态摘要（供定期调用）
        """
        try:
            # 简化的健康状态记录
            active_streams = len(self._zero_copy_buffers)
            total_operations = self._performance_stats.get("zero_copy_operations", 0)
            total_dropped = self._memory_stats.get("frames_dropped", 0)
            
            normal_logger.info(f"[零拷贝系统] 活跃流数: {active_streams}, "
                             f"总操作数: {total_operations}, 总丢帧: {total_dropped}")
            
        except Exception as e:
            exception_logger.exception(f"记录系统健康摘要失败: {str(e)}")
