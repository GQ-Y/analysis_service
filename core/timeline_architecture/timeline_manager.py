"""
时间轴管理器 - 流水线架构核心
基于时间轴的帧调度和管理，实现高效的多核并行处理
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
    """帧时间轴条目"""
    frame_id: str
    stream_id: str
    timestamp: float
    memory_block_id: str
    frame_index: int
    status: str = "pending"  # pending, processing, completed, expired
    assigned_processor_id: Optional[str] = None
    processing_start_time: Optional[float] = None
    processing_end_time: Optional[float] = None
    
    def __post_init__(self):
        self.created_time = time.time()
    
    @property
    def age(self) -> float:
        """帧年龄（秒）"""
        return time.time() - self.created_time
    
    @property
    def is_expired(self) -> bool:
        """是否已过期（超过1秒）"""
        return self.age > 1.0

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
    """
    
    def __init__(self, target_cpu_utilization: float = 0.8):
        self.target_cpu_utilization = target_cpu_utilization
        
        # 时间轴存储（按时间戳排序）
        self.timeline: deque[FrameTimelineEntry] = deque()
        self.timeline_lock = asyncio.Lock()
        
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
            "cpu_utilization": 0.0
        }
        
        self.running = False
        self.cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动时间轴管理器"""
        if self.running:
            return
        
        self.running = True
        print(f"时间轴管理器启动 - 目标CPU利用率: {self.target_cpu_utilization*100}%")
        
        # 启动清理任务
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_frames())
    
    async def stop(self):
        """停止时间轴管理器"""
        if not self.running:
            return
        
        self.running = False
        
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
    
    async def add_frame(self, frame_id: str, stream_id: str, timestamp: float,
                       memory_block_id: str, frame_index: int) -> bool:
        """添加帧到时间轴"""
        try:
            async with self.timeline_lock:
                entry = FrameTimelineEntry(
                    frame_id=frame_id,
                    stream_id=stream_id,
                    timestamp=timestamp,
                    memory_block_id=memory_block_id,
                    frame_index=frame_index
                )
                
                # 按时间戳顺序插入
                self.timeline.append(entry)
                
                # 更新索引
                self.frame_index[frame_id] = entry
                self.stream_frames[stream_id].append(frame_id)
                
                self.stats["total_frames_added"] += 1
                return True
                
        except Exception as e:
            print(f"添加帧到时间轴失败: {frame_id}, {e}")
            return False
    
    async def get_next_frames_for_processing(self, processor_id: str, 
                                           batch_size: int = 1) -> List[FrameTimelineEntry]:
        """为处理器获取下一批待处理帧"""
        async with self.timeline_lock:
            available_frames = []
            
            for entry in self.timeline:
                if len(available_frames) >= batch_size:
                    break
                
                if entry.status == "pending" and not entry.is_expired:
                    entry.status = "processing"
                    entry.assigned_processor_id = processor_id
                    entry.processing_start_time = time.time()
                    available_frames.append(entry)
            
            return available_frames
    
    async def mark_frame_completed(self, frame_id: str, processing_time: float) -> bool:
        """标记帧处理完成"""
        try:
            async with self.timeline_lock:
                if frame_id not in self.frame_index:
                    return False
                
                entry = self.frame_index[frame_id]
                entry.status = "completed"
                entry.processing_end_time = time.time()
                
                self.stats["total_frames_processed"] += 1
                return True
                
        except Exception as e:
            print(f"标记帧完成失败: {frame_id}, {e}")
            return False
    
    async def _cleanup_expired_frames(self):
        """清理过期帧的后台任务"""
        while self.running:
            try:
                await asyncio.sleep(0.5)
                
                async with self.timeline_lock:
                    expired_frames = []
                    
                    while self.timeline and self.timeline[0].is_expired:
                        expired_frame = self.timeline.popleft()
                        expired_frames.append(expired_frame)
                    
                    for frame in expired_frames:
                        if frame.frame_id in self.frame_index:
                            del self.frame_index[frame.frame_id]
                        
                        self.stats["total_frames_dropped"] += 1
                        print(f"丢弃过期帧: {frame.frame_id}, 年龄: {frame.age:.2f}s")
                
            except Exception as e:
                print(f"清理过期帧异常: {e}")
                await asyncio.sleep(1.0)
    
    def register_processor(self, processor_id: str, processor_instance: Any):
        """注册处理器"""
        self.active_processors[processor_id] = processor_instance
        print(f"注册处理器: {processor_id}")
    
    def unregister_processor(self, processor_id: str):
        """注销处理器"""
        if processor_id in self.active_processors:
            del self.active_processors[processor_id]
        print(f"注销处理器: {processor_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "timeline_length": len(self.timeline),
            "active_processors": len(self.active_processors),
            **self.stats
        } 