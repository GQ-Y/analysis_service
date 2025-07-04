"""
时间轴管理器 - 流水线架构核心
基于时间轴的帧调度和管理，实现高效的多核并行处理
参考timelinetool的SortedDict设计优化
"""
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import weakref
import psutil
import multiprocessing as mp
from sortedcontainers import SortedDict

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

@dataclass
class FrameTimelineEntry:
    """帧时间轴条目 - 参考timelinetool的FrameBuffer设计"""
    frame_id: str
    stream_id: str
    timestamp: float
    memory_block_id: str
    frame_index: int
    status: str = "pending"  # pending, processing, completed, expired
    assigned_processor_id: Optional[str] = None
    processing_start_time: Optional[float] = None
    processing_end_time: Optional[float] = None
    retry_count: int = 0

    def __post_init__(self):
        self.created_time = time.time()

    @property
    def age(self) -> float:
        """帧年龄（秒）"""
        return time.time() - self.created_time

    @property
    def is_expired(self) -> bool:
        """是否已过期 - 参考timelinetool的超时机制"""
        # 处理中的帧30秒超时，待处理的帧2秒超时
        if self.status == "processing" and self.processing_start_time:
            return time.time() - self.processing_start_time > 30.0
        return self.age > 2.0

@dataclass
class TimelineConfig:
    """时间轴配置 - 参考timelinetool设计"""
    frame_timeout_seconds: float = 2.0
    max_timeline_size: int = 1000
    cleanup_interval_seconds: float = 0.5
    enable_batch_processing: bool = True
    max_batch_size: int = 16
    processing_timeout_seconds: float = 30.0

@dataclass
class StreamMetrics:
    """流指标统计"""
    stream_id: str
    total_frames: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    avg_processing_time: float = 0.0
    current_fps: float = 0.0
    last_frame_time: float = 0.0

@dataclass
class ProcessorMetrics:
    """处理器指标统计"""
    processor_id: str
    total_processed: int = 0
    avg_processing_time: float = 0.0
    current_load: float = 0.0
    last_assigned_time: float = 0.0
    active_frames: List[str] = field(default_factory=list)

class TimelineManager:
    """
    时间轴管理器 - 流水线架构核心
    实现基于时间轴的帧调度和管理
    参考timelinetool的SortedDict设计优化
    """

    def __init__(self, target_cpu_utilization: float = 0.8, config: Optional[TimelineConfig] = None):
        self.target_cpu_utilization = target_cpu_utilization
        self.config = config or TimelineConfig()

        # 使用SortedDict按时间戳排序（参考timelinetool）
        self.timeline_slots = SortedDict()
        self.timeline_lock = None  # 延迟初始化，避免事件循环绑定问题

        # 流和处理器管理
        self.stream_metrics: Dict[str, StreamMetrics] = {}
        self.processor_metrics: Dict[str, ProcessorMetrics] = {}

        # 索引快速查找
        self.frame_index: Dict[str, FrameTimelineEntry] = {}  # frame_id -> entry
        self.stream_frames: Dict[str, deque] = defaultdict(deque)  # stream_id -> frame_ids

        # 处理器管理
        self.active_processors: Dict[str, Any] = {}  # processor_id -> processor_instance

        # 统计信息
        self.stats = {
            "total_frames_added": 0,
            "total_frames_processed": 0,
            "total_frames_dropped": 0,
            "pending_frames": 0,
            "processing_frames": 0,
            "expired_frames": 0,
            "cpu_utilization": 0.0
        }

        self.running = False
        self.cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动时间轴管理器"""
        if self.running:
            return

        # 在当前事件循环中初始化锁，避免事件循环绑定问题
        if self.timeline_lock is None:
            self.timeline_lock = asyncio.Lock()

        self.running = True
        normal_logger.info(f"时间轴管理器启动 - 目标CPU利用率: {self.target_cpu_utilization*100}%")

        # 启动清理任务
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_frames())
    
    async def stop(self):
        """停止时间轴管理器"""
        if not self.running:
            return

        self.running = False

        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # 重置锁对象，避免事件循环绑定问题
        self.timeline_lock = None

        normal_logger.info("时间轴管理器已停止")
    
    async def add_frame(self, frame_id: str, stream_id: str, timestamp: float,
                       memory_block_id: str, frame_index: int) -> bool:
        """添加帧到时间轴 - 参考timelinetool的add_frame"""
        try:
            # 检查锁是否有效
            if self.timeline_lock is None:
                return False

            async with self.timeline_lock:
                entry = FrameTimelineEntry(
                    frame_id=frame_id,
                    stream_id=stream_id,
                    timestamp=timestamp,
                    memory_block_id=memory_block_id,
                    frame_index=frame_index
                )

                # 使用时间戳作为键插入SortedDict（参考timelinetool）
                # 处理时间戳冲突：如果时间戳已存在，微调时间戳
                original_timestamp = timestamp
                collision_count = 0
                while timestamp in self.timeline_slots:
                    collision_count += 1
                    timestamp = original_timestamp + (collision_count * 0.000001)

                entry.timestamp = timestamp  # 更新条目的时间戳
                self.timeline_slots[timestamp] = entry

                # 更新索引
                self.frame_index[frame_id] = entry
                self.stream_frames[stream_id].append(frame_id)

                self.stats["total_frames_added"] += 1
                self.stats["pending_frames"] += 1

                # 添加调试日志
                analysis_logger.debug(f"[时间轴] 添加帧: {frame_id}, 内存块: {memory_block_id}, "
                                    f"时间轴大小: {len(self.timeline_slots)}, 待处理: {self.stats['pending_frames']}")

                # 检查时间轴大小限制（参考timelinetool）
                if len(self.timeline_slots) > self.config.max_timeline_size:
                    # 移除最旧的帧
                    oldest_timestamp, oldest_entry = self.timeline_slots.popitem(0)
                    self._remove_frame_from_indices(oldest_entry)
                    self.stats["expired_frames"] += 1
                    analysis_logger.info(f"时间轴已满，移除最旧帧: {oldest_entry.frame_id}")

                return True

        except RuntimeError as e:
            if "bound to a different event loop" in str(e):
                # 锁绑定到不同的事件循环，返回False
                return False
            exception_logger.exception(f"添加帧到时间轴失败: {frame_id}, {str(e)}")
            return False
        except Exception as e:
            exception_logger.exception(f"添加帧到时间轴失败: {frame_id}, {str(e)}")
            return False

    def _add_frame_sync(self, frame_id: str, stream_id: str, timestamp: float,
                       memory_block_id: str, frame_index: int) -> bool:
        """
        同步添加帧到时间轴 - 避免异步复杂性

        Args:
            frame_id: 帧ID
            stream_id: 流ID
            timestamp: 时间戳
            memory_block_id: 内存块ID
            frame_index: 帧索引

        Returns:
            bool: 是否成功添加
        """
        try:
            # 添加调试日志
            analysis_logger.info(f"[时间轴] _add_frame_sync被调用: frame_id={frame_id}, memory_block_id={memory_block_id}")

            # 检查是否已初始化
            if not hasattr(self, 'timeline_slots'):
                analysis_logger.error("时间轴管理器未初始化")
                return False

            entry = FrameTimelineEntry(
                frame_id=frame_id,
                stream_id=stream_id,
                timestamp=timestamp,
                memory_block_id=memory_block_id,
                frame_index=frame_index
            )

            # 使用时间戳作为键插入SortedDict
            original_timestamp = timestamp
            collision_count = 0
            while timestamp in self.timeline_slots:
                collision_count += 1
                timestamp = original_timestamp + (collision_count * 0.000001)

            entry.timestamp = timestamp
            self.timeline_slots[timestamp] = entry

            # 更新索引
            self.frame_index[frame_id] = entry
            self.stream_frames[stream_id].append(frame_id)

            self.stats["total_frames_added"] += 1
            self.stats["pending_frames"] += 1

            # 检查时间轴大小限制
            if len(self.timeline_slots) > self.config.max_timeline_size:
                oldest_timestamp, oldest_entry = self.timeline_slots.popitem(0)
                self._remove_frame_from_indices(oldest_entry)
                self.stats["expired_frames"] += 1
                analysis_logger.info(f"时间轴已满，移除最旧帧: {oldest_entry.frame_id}")

            # 添加调试日志
            analysis_logger.debug(f"[时间轴] 同步添加帧: {frame_id}, 内存块: {memory_block_id}, "
                                f"时间轴大小: {len(self.timeline_slots)}, 待处理: {self.stats['pending_frames']}")

            return True

        except Exception as e:
            exception_logger.exception(f"同步添加帧到时间轴失败: {frame_id}, {str(e)}")
            return False
    
    async def get_next_frames_for_processing(self, processor_id: str,
                                           batch_size: int = 1) -> List[FrameTimelineEntry]:
        """为处理器获取下一批待处理帧 - 参考timelinetool的SortedDict设计"""
        # 检查锁是否有效
        if self.timeline_lock is None:
            return []

        try:
            async with self.timeline_lock:
                available_frames = []
                current_time = time.time()
                timeout_threshold = current_time - self.config.frame_timeout_seconds
                timestamps_to_remove = []

                # 使用SortedDict的有序迭代（参考timelinetool的高效排序）
                for timestamp, entry in self.timeline_slots.items():
                    if len(available_frames) >= batch_size:
                        break

                    # 检查帧是否超时（参考timelinetool的超时机制）
                    if timestamp < timeout_threshold:
                        entry.status = "expired"
                        timestamps_to_remove.append(timestamp)
                        self.stats["expired_frames"] += 1
                        continue

                    if entry.status == "pending" and not entry.is_expired:
                        entry.status = "processing"
                        entry.assigned_processor_id = processor_id
                        entry.processing_start_time = current_time

                        self.stats["pending_frames"] -= 1
                        self.stats["processing_frames"] += 1

                        available_frames.append(entry)
                        timestamps_to_remove.append(timestamp)

                # 从时间轴中移除已分配或过期的帧
                for timestamp in timestamps_to_remove:
                    removed_entry = self.timeline_slots.pop(timestamp, None)
                    if removed_entry and removed_entry.status == "expired":
                        self._remove_frame_from_indices(removed_entry)

                # 添加调试日志
                if available_frames:
                    frame_ids = [f.frame_id for f in available_frames]
                    analysis_logger.debug(f"[时间轴] 处理器 {processor_id} 获取 {len(available_frames)} 帧: {frame_ids}")
                else:
                    analysis_logger.debug(f"[时间轴] 处理器 {processor_id} 未获取到帧, 时间轴大小: {len(self.timeline_slots)}")

                return available_frames
        except RuntimeError as e:
            if "bound to a different event loop" in str(e):
                # 锁绑定到不同的事件循环，返回空列表
                return []
            raise
    
    def _remove_frame_from_indices(self, entry: FrameTimelineEntry):
        """从索引中移除帧"""
        if entry.frame_id in self.frame_index:
            del self.frame_index[entry.frame_id]

        if entry.stream_id in self.stream_frames:
            try:
                self.stream_frames[entry.stream_id].remove(entry.frame_id)
            except ValueError:
                pass  # 帧ID不在队列中

    async def get_due_frames_batch(self, batch_size: int = 1) -> List[FrameTimelineEntry]:
        """
        获取到期的帧批次 - 参考timelinetool的get_due_frames
        """
        if self.timeline_lock is None:
            return []

        try:
            async with self.timeline_lock:
                due_frames = []
                current_time = time.time()
                timeout_threshold = current_time - self.config.frame_timeout_seconds
                timestamps_to_remove = []

                # 获取到期的帧（参考timelinetool的超时逻辑）
                for timestamp, entry in self.timeline_slots.items():
                    if len(due_frames) >= batch_size:
                        break

                    # 检查是否到期或超时
                    if timestamp <= timeout_threshold or entry.status == "pending":
                        if entry.status == "pending":
                            entry.status = "processing"
                            entry.processing_start_time = current_time
                            self.stats["pending_frames"] -= 1
                            self.stats["processing_frames"] += 1

                        due_frames.append(entry)
                        timestamps_to_remove.append(timestamp)

                # 从时间轴中移除已分配的帧
                for timestamp in timestamps_to_remove:
                    self.timeline_slots.pop(timestamp, None)

                return due_frames
        except RuntimeError as e:
            if "bound to a different event loop" in str(e):
                return []
            raise

    async def mark_frame_completed(self, frame_id: str, processing_time: float = 0.0) -> bool:
        """标记帧处理完成"""
        try:
            # 检查锁是否有效
            if self.timeline_lock is None:
                return False

            async with self.timeline_lock:
                if frame_id not in self.frame_index:
                    return False

                entry = self.frame_index[frame_id]
                entry.status = "completed"
                entry.processing_end_time = time.time()

                self.stats["total_frames_processed"] += 1
                return True

        except RuntimeError as e:
            if "bound to a different event loop" in str(e):
                # 锁绑定到不同的事件循环，返回False
                return False
            exception_logger.exception(f"标记帧完成失败: {frame_id}, {str(e)}")
            return False
        except Exception as e:
            exception_logger.exception(f"标记帧完成失败: {frame_id}, {str(e)}")
            return False
    
    async def _cleanup_expired_frames(self):
        """清理过期帧的后台任务 - 参考timelinetool的清理机制"""
        while self.running:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)

                # 记录清理前的统计信息
                before_stats = self.stats.copy()

                # 检查锁是否有效
                if self.timeline_lock is None:
                    continue

                try:
                    async with self.timeline_lock:
                        current_time = time.time()
                        expired_timestamps = []

                        # 使用SortedDict的高效迭代查找需要清理的帧
                        for timestamp, entry in self.timeline_slots.items():
                            # 清理已完成的帧（参考timelinetool的完成帧清理）
                            if entry.status == "completed":
                                expired_timestamps.append(timestamp)
                                self.stats["processing_frames"] -= 1
                                analysis_logger.debug(f"清理已完成帧: {entry.frame_id}")

                            # 清理超时的处理中帧
                            elif (entry.status == "processing" and
                                entry.processing_start_time and
                                current_time - entry.processing_start_time > self.config.processing_timeout_seconds):

                                expired_timestamps.append(timestamp)
                                self.stats["expired_frames"] += 1
                                self.stats["processing_frames"] -= 1
                                analysis_logger.info(f"清理超时处理帧: {entry.frame_id}")

                            # 清理过期的待处理帧
                            elif entry.is_expired and entry.status == "pending":
                                expired_timestamps.append(timestamp)
                                self.stats["expired_frames"] += 1
                                self.stats["pending_frames"] -= 1
                                analysis_logger.info(f"清理过期待处理帧: {entry.frame_id}, 年龄: {entry.age:.2f}s")

                        # 移除过期帧
                        for timestamp in expired_timestamps:
                            expired_entry = self.timeline_slots.pop(timestamp, None)
                            if expired_entry:
                                self._remove_frame_from_indices(expired_entry)
                                self.stats["total_frames_dropped"] += 1

                        # 记录清理结果（参考timelinetool的日志记录）
                        if expired_timestamps:
                            analysis_logger.info(f"时间轴清理完成: 清理了 {len(expired_timestamps)} 个帧, "
                                               f"当前时间轴大小: {len(self.timeline_slots)}")

                        # 定期记录状态
                        if hasattr(self, '_last_status_log_time'):
                            if time.time() - self._last_status_log_time > 10.0:  # 每10秒记录一次
                                self.log_status()
                                self._last_status_log_time = time.time()
                        else:
                            self._last_status_log_time = time.time()

                except RuntimeError as e:
                    if "bound to a different event loop" in str(e):
                        # 锁绑定到不同的事件循环，跳过这次清理
                        continue
                    raise

            except Exception as e:
                exception_logger.exception(f"清理过期帧异常: {str(e)}")
                await asyncio.sleep(1.0)
    
    def register_processor(self, processor_id: str, processor_instance: Any):
        """注册处理器"""
        self.active_processors[processor_id] = processor_instance
        normal_logger.info(f"注册处理器: {processor_id}")
    
    def unregister_processor(self, processor_id: str):
        """注销处理器"""
        if processor_id in self.active_processors:
            del self.active_processors[processor_id]
        normal_logger.info(f"注销处理器: {processor_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息 - 参考timelinetool的统计设计"""
        return {
            "timeline_length": len(self.timeline_slots),
            "active_processors": len(self.active_processors),
            "config": {
                "frame_timeout_seconds": self.config.frame_timeout_seconds,
                "max_timeline_size": self.config.max_timeline_size,
                "max_batch_size": self.config.max_batch_size,
                "processing_timeout_seconds": self.config.processing_timeout_seconds
            },
            **self.stats
        }

    def log_status(self):
        """记录状态 - 参考timelinetool的_log_status"""
        stats = self.get_statistics()
        normal_logger.info(
            f"[时间轴] 当前状态: "
            f"总帧数={stats['total_frames_added']}, "
            f"待处理={stats['pending_frames']}, "
            f"处理中={stats['processing_frames']}, "
            f"已完成={stats['total_frames_processed']}, "
            f"已过期={stats['expired_frames']}, "
            f"时间轴大小={stats['timeline_length']}"
        )