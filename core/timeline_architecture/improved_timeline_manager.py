"""
改进的时间轴管理器 - 参考timelinetool设计
"""
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
from sortedcontainers import SortedDict

from shared.logging.logger_config import normal_logger, analysis_logger, exception_logger


@dataclass
class FrameTimelineEntry:
    """时间轴帧条目"""
    frame_id: str
    stream_id: str
    timestamp: float
    frame_data: Any
    status: str = "pending"  # pending, processing, completed, expired
    assigned_processor_id: Optional[str] = None
    processing_start_time: Optional[float] = None
    retry_count: int = 0
    
    @property
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.processing_start_time is None:
            return False
        return time.time() - self.processing_start_time > 30.0  # 30秒超时


@dataclass
class TimelineConfig:
    """时间轴配置"""
    frame_timeout_seconds: float = 2.0
    max_timeline_size: int = 1000
    cleanup_interval_seconds: float = 5.0
    enable_batch_processing: bool = True
    max_batch_size: int = 16


class ImprovedTimelineManager:
    """
    改进的时间轴管理器 - 参考timelinetool的SortedDict设计
    """
    
    def __init__(self, config: Optional[TimelineConfig] = None):
        self.config = config or TimelineConfig()
        
        # 使用SortedDict按时间戳排序（参考timelinetool）
        self.timeline_slots = SortedDict()
        self.timeline_lock = asyncio.Lock()
        
        # 统计信息
        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "expired_frames": 0,
            "pending_frames": 0,
            "processing_frames": 0
        }
        
        # 清理任务
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
        normal_logger.info("改进的时间轴管理器初始化完成")
    
    async def start(self):
        """启动时间轴管理器"""
        self.running = True
        
        # 启动清理任务
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        normal_logger.info("时间轴管理器已启动")
    
    async def stop(self):
        """停止时间轴管理器"""
        self.running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        normal_logger.info("时间轴管理器已停止")
    
    async def add_frame(self, frame_entry: FrameTimelineEntry):
        """
        添加帧到时间轴 - 参考timelinetool的add_frame
        """
        async with self.timeline_lock:
            # 使用时间戳作为键（参考timelinetool）
            self.timeline_slots[frame_entry.timestamp] = frame_entry
            self.stats["total_frames"] += 1
            self.stats["pending_frames"] += 1
            
            # 检查时间轴大小限制
            if len(self.timeline_slots) > self.config.max_timeline_size:
                # 移除最旧的帧
                oldest_timestamp, oldest_entry = self.timeline_slots.popitem(0)
                self.stats["expired_frames"] += 1
                analysis_logger.info(f"时间轴已满，移除最旧帧: {oldest_entry.frame_id}")
    
    async def get_due_frames_batch(self, batch_size: int = 1) -> List[FrameTimelineEntry]:
        """
        获取到期的帧批次 - 参考timelinetool的get_due_frames
        """
        async with self.timeline_lock:
            due_frames = []
            current_time = time.time()
            timeout_threshold = current_time - self.config.frame_timeout_seconds
            
            # 获取到期的帧（参考timelinetool的超时逻辑）
            due_timestamps = []
            for timestamp, entry in self.timeline_slots.items():
                if len(due_frames) >= batch_size:
                    break
                
                # 检查是否到期或超时
                if (timestamp <= timeout_threshold or 
                    entry.status == "pending"):
                    
                    if entry.status == "pending":
                        entry.status = "processing"
                        entry.processing_start_time = current_time
                        self.stats["pending_frames"] -= 1
                        self.stats["processing_frames"] += 1
                    
                    due_frames.append(entry)
                    due_timestamps.append(timestamp)
            
            # 从时间轴中移除已分配的帧
            for timestamp in due_timestamps:
                self.timeline_slots.pop(timestamp, None)
            
            return due_frames
    
    async def get_next_frames_for_processing(self, processor_id: str,
                                           batch_size: int = 1) -> List[FrameTimelineEntry]:
        """
        为处理器获取下一批待处理帧 - 改进版本
        """
        async with self.timeline_lock:
            available_frames = []
            current_time = time.time()
            timeout_threshold = current_time - self.config.frame_timeout_seconds
            
            # 按时间戳排序获取最旧的可用帧（参考timelinetool的SortedDict逻辑）
            timestamps_to_remove = []
            
            for timestamp, entry in self.timeline_slots.items():
                if len(available_frames) >= batch_size:
                    break
                
                # 检查帧是否超时
                if timestamp < timeout_threshold:
                    entry.status = "expired"
                    self.stats["expired_frames"] += 1
                    timestamps_to_remove.append(timestamp)
                    continue
                
                if entry.status == "pending":
                    entry.status = "processing"
                    entry.assigned_processor_id = processor_id
                    entry.processing_start_time = current_time
                    
                    self.stats["pending_frames"] -= 1
                    self.stats["processing_frames"] += 1
                    
                    available_frames.append(entry)
                    timestamps_to_remove.append(timestamp)
            
            # 移除已分配或过期的帧
            for timestamp in timestamps_to_remove:
                self.timeline_slots.pop(timestamp, None)
            
            return available_frames
    
    async def mark_frame_completed(self, frame_id: str):
        """标记帧处理完成"""
        async with self.timeline_lock:
            self.stats["processed_frames"] += 1
            self.stats["processing_frames"] -= 1
    
    async def _cleanup_loop(self):
        """清理循环 - 定期清理过期帧"""
        while self.running:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                await self._cleanup_expired_frames()
            except asyncio.CancelledError:
                break
            except Exception as e:
                exception_logger.exception(f"时间轴清理异常: {str(e)}")
    
    async def _cleanup_expired_frames(self):
        """清理过期帧"""
        async with self.timeline_lock:
            current_time = time.time()
            expired_timestamps = []
            
            for timestamp, entry in self.timeline_slots.items():
                # 清理超时的处理中帧
                if (entry.status == "processing" and 
                    entry.processing_start_time and
                    current_time - entry.processing_start_time > 30.0):
                    
                    expired_timestamps.append(timestamp)
                    self.stats["expired_frames"] += 1
                    self.stats["processing_frames"] -= 1
            
            # 移除过期帧
            for timestamp in expired_timestamps:
                expired_entry = self.timeline_slots.pop(timestamp, None)
                if expired_entry:
                    analysis_logger.info(f"清理过期帧: {expired_entry.frame_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "timeline_size": len(self.timeline_slots),
            "config": {
                "frame_timeout_seconds": self.config.frame_timeout_seconds,
                "max_timeline_size": self.config.max_timeline_size,
                "max_batch_size": self.config.max_batch_size
            }
        }
    
    def log_status(self):
        """记录状态 - 参考timelinetool的_log_status"""
        stats = self.get_stats()
        normal_logger.info(
            f"[时间轴] 当前状态: "
            f"总帧数={stats['total_frames']}, "
            f"待处理={stats['pending_frames']}, "
            f"处理中={stats['processing_frames']}, "
            f"已完成={stats['processed_frames']}, "
            f"已过期={stats['expired_frames']}, "
            f"时间轴大小={stats['timeline_size']}"
        )
