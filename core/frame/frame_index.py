"""
帧索引管理模块
提供高效的帧查找、时间范围查询和批量操作
"""
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Iterator
from collections import defaultdict, deque
import bisect

from .frame_metadata import FrameMetadata

# 使用项目现有的日志系统
try:
    from shared.utils.logger import get_normal_logger
    logger = get_normal_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class FrameIndex:
    """
    帧索引管理器
    提供高效的帧查找、时间范围查询和内存映射文件支持
    """
    
    def __init__(self, max_frames: int = 100000, enable_time_index: bool = True):
        """
        初始化帧索引管理器
        
        Args:
            max_frames: 最大帧数量
            enable_time_index: 是否启用时间索引
        """
        self.max_frames = max_frames
        self.enable_time_index = enable_time_index
        
        # 主索引：frame_id -> FrameMetadata
        self.frame_index: Dict[int, FrameMetadata] = {}
        
        # 时间索引：timestamp -> frame_id (有序列表)
        self.time_index: List[Tuple[float, int]] = []
        
        # 序列号索引：sequence_number -> frame_id
        self.sequence_index: Dict[int, int] = {}
        
        # 状态索引：analysis_status -> List[frame_id]
        self.status_index: Dict[int, List[int]] = defaultdict(list)
        
        # 内存块引用索引：memory_block_ref -> frame_id
        self.memory_ref_index: Dict[int, int] = {}
        
        # LRU缓存队列
        self.lru_queue: deque = deque(maxlen=max_frames)
        
        # 线程安全锁
        self.lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            "total_frames": 0,
            "index_hits": 0,
            "index_misses": 0,
            "time_queries": 0,
            "cleanup_count": 0,
        }
        
        logger.info(f"帧索引管理器初始化完成: 最大帧数={max_frames}, 时间索引={'启用' if enable_time_index else '禁用'}")
    
    def add_frame(self, metadata: FrameMetadata) -> bool:
        """
        添加帧到索引
        
        Args:
            metadata: 帧元数据
            
        Returns:
            bool: 是否添加成功
        """
        with self.lock:
            try:
                frame_id = metadata.frame_id
                
                # 检查是否已存在
                if frame_id in self.frame_index:
                    logger.warning(f"帧 {frame_id} 已存在，将更新")
                    self._remove_frame_from_indices(frame_id)
                
                # 检查容量限制
                if len(self.frame_index) >= self.max_frames:
                    self._evict_oldest_frame()
                
                # 添加到主索引
                self.frame_index[frame_id] = metadata
                
                # 添加到时间索引
                if self.enable_time_index:
                    bisect.insort(self.time_index, (metadata.timestamp, frame_id))
                
                # 添加到序列号索引
                if metadata.sequence_number > 0:
                    self.sequence_index[metadata.sequence_number] = frame_id
                
                # 添加到状态索引
                self.status_index[metadata.analysis_status].append(frame_id)
                
                # 添加到内存块引用索引
                if metadata.memory_block_ref > 0:
                    self.memory_ref_index[metadata.memory_block_ref] = frame_id
                
                # 更新LRU队列
                self.lru_queue.append(frame_id)
                
                # 更新统计
                self.stats["total_frames"] = len(self.frame_index)
                
                logger.debug(f"帧 {frame_id} 添加到索引成功")
                return True
                
            except Exception as e:
                logger.error(f"添加帧 {metadata.frame_id} 到索引失败: {str(e)}")
                return False
    
    def get_frame(self, frame_id: int) -> Optional[FrameMetadata]:
        """
        根据帧ID获取帧元数据
        
        Args:
            frame_id: 帧ID
            
        Returns:
            Optional[FrameMetadata]: 帧元数据，不存在返回None
        """
        with self.lock:
            metadata = self.frame_index.get(frame_id)
            
            if metadata:
                self.stats["index_hits"] += 1
                # 更新LRU
                if frame_id in self.lru_queue:
                    self.lru_queue.remove(frame_id)
                self.lru_queue.append(frame_id)
                logger.debug(f"帧 {frame_id} 索引命中")
            else:
                self.stats["index_misses"] += 1
                logger.debug(f"帧 {frame_id} 索引未命中")
            
            return metadata
    
    def get_frame_by_sequence(self, sequence_number: int) -> Optional[FrameMetadata]:
        """
        根据序列号获取帧元数据
        
        Args:
            sequence_number: 序列号
            
        Returns:
            Optional[FrameMetadata]: 帧元数据，不存在返回None
        """
        with self.lock:
            frame_id = self.sequence_index.get(sequence_number)
            if frame_id:
                return self.get_frame(frame_id)
            return None
    
    def get_frame_by_memory_ref(self, memory_block_ref: int) -> Optional[FrameMetadata]:
        """
        根据内存块引用获取帧元数据
        
        Args:
            memory_block_ref: 内存块引用ID
            
        Returns:
            Optional[FrameMetadata]: 帧元数据，不存在返回None
        """
        with self.lock:
            frame_id = self.memory_ref_index.get(memory_block_ref)
            if frame_id:
                return self.get_frame(frame_id)
            return None
    
    def get_frames_by_time_range(self, start_time: float, end_time: float) -> List[FrameMetadata]:
        """
        根据时间范围获取帧列表
        
        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳
            
        Returns:
            List[FrameMetadata]: 时间范围内的帧列表
        """
        if not self.enable_time_index:
            logger.warning("时间索引未启用，无法进行时间范围查询")
            return []
        
        with self.lock:
            self.stats["time_queries"] += 1
            
            # 使用二分查找找到时间范围
            start_idx = bisect.bisect_left(self.time_index, (start_time, 0))
            end_idx = bisect.bisect_right(self.time_index, (end_time, float('inf')))
            
            frames = []
            for i in range(start_idx, end_idx):
                timestamp, frame_id = self.time_index[i]
                metadata = self.frame_index.get(frame_id)
                if metadata:
                    frames.append(metadata)
            
            logger.debug(f"时间范围查询 [{start_time:.3f}, {end_time:.3f}] 返回 {len(frames)} 帧")
            return frames
    
    def get_frames_by_status(self, analysis_status: int) -> List[FrameMetadata]:
        """
        根据分析状态获取帧列表
        
        Args:
            analysis_status: 分析状态
            
        Returns:
            List[FrameMetadata]: 指定状态的帧列表
        """
        with self.lock:
            frame_ids = self.status_index.get(analysis_status, [])
            frames = []
            
            for frame_id in frame_ids:
                metadata = self.frame_index.get(frame_id)
                if metadata:
                    frames.append(metadata)
            
            logger.debug(f"状态查询 {analysis_status} 返回 {len(frames)} 帧")
            return frames
    
    def update_frame_status(self, frame_id: int, old_status: int, new_status: int) -> bool:
        """
        更新帧的分析状态
        
        Args:
            frame_id: 帧ID
            old_status: 旧状态
            new_status: 新状态
            
        Returns:
            bool: 是否更新成功
        """
        with self.lock:
            metadata = self.frame_index.get(frame_id)
            if not metadata:
                return False
            
            # 从旧状态索引中移除
            if frame_id in self.status_index[old_status]:
                self.status_index[old_status].remove(frame_id)
            
            # 添加到新状态索引
            self.status_index[new_status].append(frame_id)
            
            # 更新元数据
            metadata.analysis_status = new_status
            
            logger.debug(f"帧 {frame_id} 状态更新: {old_status} -> {new_status}")
            return True
    
    def remove_frame(self, frame_id: int) -> bool:
        """
        从索引中移除帧
        
        Args:
            frame_id: 帧ID
            
        Returns:
            bool: 是否移除成功
        """
        with self.lock:
            if frame_id not in self.frame_index:
                return False
            
            self._remove_frame_from_indices(frame_id)
            logger.debug(f"帧 {frame_id} 从索引中移除")
            return True
    
    def _remove_frame_from_indices(self, frame_id: int) -> None:
        """从所有索引中移除帧"""
        metadata = self.frame_index.get(frame_id)
        if not metadata:
            return
        
        # 从主索引移除
        del self.frame_index[frame_id]
        
        # 从时间索引移除
        if self.enable_time_index:
            try:
                self.time_index.remove((metadata.timestamp, frame_id))
            except ValueError:
                pass
        
        # 从序列号索引移除
        if metadata.sequence_number in self.sequence_index:
            del self.sequence_index[metadata.sequence_number]
        
        # 从状态索引移除
        if frame_id in self.status_index[metadata.analysis_status]:
            self.status_index[metadata.analysis_status].remove(frame_id)
        
        # 从内存块引用索引移除
        if metadata.memory_block_ref in self.memory_ref_index:
            del self.memory_ref_index[metadata.memory_block_ref]
        
        # 从LRU队列移除
        if frame_id in self.lru_queue:
            self.lru_queue.remove(frame_id)
        
        # 更新统计
        self.stats["total_frames"] = len(self.frame_index)
    
    def _evict_oldest_frame(self) -> None:
        """驱逐最旧的帧"""
        if self.lru_queue:
            oldest_frame_id = self.lru_queue.popleft()
            self._remove_frame_from_indices(oldest_frame_id)
            logger.debug(f"驱逐最旧帧: {oldest_frame_id}")
    
    def cleanup_old_frames(self, max_age: float) -> int:
        """
        清理超过指定年龄的帧
        
        Args:
            max_age: 最大年龄（秒）
            
        Returns:
            int: 清理的帧数量
        """
        current_time = time.time()
        cutoff_time = current_time - max_age
        
        with self.lock:
            frames_to_remove = []
            
            for frame_id, metadata in self.frame_index.items():
                if metadata.timestamp < cutoff_time:
                    frames_to_remove.append(frame_id)
            
            for frame_id in frames_to_remove:
                self._remove_frame_from_indices(frame_id)
            
            self.stats["cleanup_count"] += len(frames_to_remove)
            
            if frames_to_remove:
                logger.info(f"清理了 {len(frames_to_remove)} 个超时帧 (>{max_age}秒)")
            
            return len(frames_to_remove)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.lock:
            return {
                "config": {
                    "max_frames": self.max_frames,
                    "enable_time_index": self.enable_time_index,
                },
                "current_state": {
                    "total_frames": len(self.frame_index),
                    "time_index_size": len(self.time_index),
                    "sequence_index_size": len(self.sequence_index),
                    "memory_ref_index_size": len(self.memory_ref_index),
                    "lru_queue_size": len(self.lru_queue),
                },
                "performance": self.stats.copy(),
                "hit_rate": (
                    self.stats["index_hits"] / 
                    max(1, self.stats["index_hits"] + self.stats["index_misses"])
                ) * 100,
            }
    
    def clear(self) -> None:
        """清空所有索引"""
        with self.lock:
            self.frame_index.clear()
            self.time_index.clear()
            self.sequence_index.clear()
            self.status_index.clear()
            self.memory_ref_index.clear()
            self.lru_queue.clear()
            
            # 重置统计
            self.stats = {
                "total_frames": 0,
                "index_hits": 0,
                "index_misses": 0,
                "time_queries": 0,
                "cleanup_count": 0,
            }
            
            logger.info("帧索引已清空")
    
    def __len__(self) -> int:
        """返回索引中的帧数量"""
        return len(self.frame_index)
    
    def __contains__(self, frame_id: int) -> bool:
        """检查帧是否在索引中"""
        return frame_id in self.frame_index
    
    def __iter__(self) -> Iterator[FrameMetadata]:
        """迭代所有帧元数据"""
        with self.lock:
            for metadata in self.frame_index.values():
                yield metadata
