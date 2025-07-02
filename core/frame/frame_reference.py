"""
零拷贝帧引用模块
提供零拷贝帧引用系统，支持原子引用计数和自动回收
"""
import time
import threading
import weakref
from typing import Optional, Dict, Any, Callable
import numpy as np

from .frame_metadata import FrameMetadata
from ..memory.memory_block import MemoryBlock

# 使用项目现有的日志系统
try:
    from shared.utils.logger import get_normal_logger
    logger = get_normal_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class FrameReference:
    """
    零拷贝帧引用
    管理帧数据的引用计数和生命周期，支持零拷贝数据访问
    """
    
    def __init__(self, memory_block: MemoryBlock, metadata: FrameMetadata, 
                 cleanup_callback: Optional[Callable[['FrameReference'], None]] = None):
        """
        初始化帧引用
        
        Args:
            memory_block: 内存块
            metadata: 帧元数据
            cleanup_callback: 清理回调函数
        """
        self.memory_block = memory_block
        self.metadata = metadata
        self.cleanup_callback = cleanup_callback
        
        # 引用计数（原子操作）
        self._ref_count = 1
        self._ref_lock = threading.RLock()
        
        # 状态管理
        self._is_valid = True
        self._access_count = 0
        self._last_access_time = time.time()
        
        # 弱引用管理器
        self._weak_refs = set()
        self._weak_ref_lock = threading.Lock()
        
        # 确保内存块被正确引用
        if not self.memory_block.acquire():
            raise RuntimeError(f"无法获取内存块 {memory_block.block_id} 的引用")
        
        logger.debug(f"创建帧引用: frame_id={metadata.frame_id}, "
                    f"memory_block={memory_block.block_id}")
    
    def get_data(self) -> Optional[np.ndarray]:
        """
        获取帧数据（零拷贝）
        
        Returns:
            Optional[np.ndarray]: numpy数组视图，失败返回None
        """
        with self._ref_lock:
            if not self._is_valid:
                logger.warning(f"帧引用 {self.metadata.frame_id} 已失效")
                return None
            
            # 更新访问统计
            self._access_count += 1
            self._last_access_time = time.time()
            
            # 获取零拷贝numpy视图
            array = self.memory_block.get_numpy_view()
            if array is not None:
                logger.debug(f"帧 {self.metadata.frame_id} 零拷贝数据访问成功")
            else:
                logger.error(f"帧 {self.metadata.frame_id} 零拷贝数据访问失败")
            
            return array
    
    def get_metadata(self) -> FrameMetadata:
        """
        获取帧元数据
        
        Returns:
            FrameMetadata: 帧元数据
        """
        return self.metadata
    
    def get_memory_block(self) -> MemoryBlock:
        """
        获取内存块引用
        
        Returns:
            MemoryBlock: 内存块
        """
        return self.memory_block
    
    def clone(self) -> Optional['FrameReference']:
        """
        克隆帧引用（增加引用计数）
        
        Returns:
            Optional[FrameReference]: 新的帧引用，失败返回None
        """
        with self._ref_lock:
            if not self._is_valid:
                logger.warning(f"无法克隆已失效的帧引用 {self.metadata.frame_id}")
                return None
            
            # 增加引用计数
            self._ref_count += 1
            
            # 创建新的引用对象
            new_ref = FrameReference.__new__(FrameReference)
            new_ref.memory_block = self.memory_block
            new_ref.metadata = self.metadata
            new_ref.cleanup_callback = self.cleanup_callback
            new_ref._ref_count = 0  # 新引用从0开始，会在下面设置为1
            new_ref._ref_lock = self._ref_lock  # 共享锁
            new_ref._is_valid = True
            new_ref._access_count = 0
            new_ref._last_access_time = time.time()
            new_ref._weak_refs = set()
            new_ref._weak_ref_lock = threading.Lock()
            
            # 设置新引用的引用计数为1
            new_ref._ref_count = 1
            
            logger.debug(f"克隆帧引用: frame_id={self.metadata.frame_id}, "
                        f"总引用计数={self._ref_count}")
            
            return new_ref
    
    def create_weak_reference(self) -> Optional[weakref.ReferenceType]:
        """
        创建弱引用
        
        Returns:
            Optional[weakref.ReferenceType]: 弱引用对象
        """
        if not self._is_valid:
            return None
        
        def cleanup_weak_ref(ref):
            with self._weak_ref_lock:
                self._weak_refs.discard(ref)
        
        weak_ref = weakref.ref(self, cleanup_weak_ref)
        
        with self._weak_ref_lock:
            self._weak_refs.add(weak_ref)
        
        return weak_ref
    
    def is_valid(self) -> bool:
        """
        检查引用是否有效
        
        Returns:
            bool: 是否有效
        """
        return self._is_valid
    
    def get_ref_count(self) -> int:
        """
        获取引用计数
        
        Returns:
            int: 引用计数
        """
        with self._ref_lock:
            return self._ref_count
    
    def get_access_stats(self) -> Dict[str, Any]:
        """
        获取访问统计信息
        
        Returns:
            Dict[str, Any]: 访问统计
        """
        with self._ref_lock:
            return {
                "access_count": self._access_count,
                "last_access_time": self._last_access_time,
                "ref_count": self._ref_count,
                "is_valid": self._is_valid,
                "weak_ref_count": len(self._weak_refs),
                "age_seconds": time.time() - self._last_access_time,
            }
    
    def release(self) -> None:
        """
        释放帧引用
        减少引用计数，当引用计数为0时自动清理资源
        """
        self._release()

    def _release(self) -> None:
        """内部释放方法"""
        with self._ref_lock:
            if self._ref_count > 0:
                self._ref_count -= 1
                
                logger.debug(f"释放帧引用: frame_id={self.metadata.frame_id}, "
                            f"剩余引用计数={self._ref_count}")
                
                # 如果引用计数为0，执行清理
                if self._ref_count == 0:
                    self._cleanup()
    
    def _cleanup(self) -> None:
        """清理资源"""
        if not self._is_valid:
            return
        
        self._is_valid = False
        
        # 释放内存块引用
        if self.memory_block:
            self.memory_block.release()
        
        # 调用清理回调
        if self.cleanup_callback:
            try:
                self.cleanup_callback(self)
            except Exception as e:
                logger.error(f"帧引用清理回调失败: {str(e)}")
        
        # 清理弱引用
        with self._weak_ref_lock:
            self._weak_refs.clear()
        
        logger.debug(f"帧引用清理完成: frame_id={self.metadata.frame_id}")
    
    def __enter__(self) -> 'FrameReference':
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self._release()
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, '_is_valid') and self._is_valid:
            self._release()
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"FrameReference(frame_id={self.metadata.frame_id}, "
                f"memory_block={self.memory_block.block_id}, "
                f"refs={self._ref_count}, "
                f"valid={self._is_valid})")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()


class FrameReferenceManager:
    """
    帧引用管理器
    管理帧引用的创建、跟踪和清理
    """
    
    def __init__(self, max_references: int = 10000, memory_pool=None):
        """
        初始化帧引用管理器

        Args:
            max_references: 最大引用数量
            memory_pool: 内存池实例，用于释放内存块
        """
        self.max_references = max_references
        self.memory_pool = memory_pool

        # 活跃引用跟踪
        self.active_references: Dict[int, FrameReference] = {}  # frame_id -> FrameReference
        self.reference_stats: Dict[int, Dict[str, Any]] = {}    # frame_id -> stats

        # 线程安全
        self.lock = threading.RLock()

        # 统计信息
        self.stats = {
            "created_count": 0,
            "released_count": 0,
            "cleanup_count": 0,
            "max_concurrent": 0,
        }

        logger.info(f"帧引用管理器初始化完成: 最大引用数={max_references}")
    
    def create_reference(self, memory_block: MemoryBlock, metadata: FrameMetadata) -> Optional[FrameReference]:
        """
        创建帧引用
        
        Args:
            memory_block: 内存块
            metadata: 帧元数据
            
        Returns:
            Optional[FrameReference]: 帧引用，失败返回None
        """
        with self.lock:
            try:
                # 检查容量限制
                if len(self.active_references) >= self.max_references:
                    self._cleanup_expired_references()
                    
                    if len(self.active_references) >= self.max_references:
                        logger.warning("帧引用数量达到上限，无法创建新引用")
                        return None
                
                # 创建清理回调
                def cleanup_callback(ref: FrameReference):
                    self._on_reference_cleanup(ref)
                
                # 创建帧引用
                frame_ref = FrameReference(memory_block, metadata, cleanup_callback)
                
                # 添加到活跃引用跟踪
                frame_id = metadata.frame_id
                self.active_references[frame_id] = frame_ref
                self.reference_stats[frame_id] = {
                    "created_time": time.time(),
                    "memory_block_id": memory_block.block_id,
                    "initial_ref_count": 1,
                }
                
                # 更新统计
                self.stats["created_count"] += 1
                current_count = len(self.active_references)
                if current_count > self.stats["max_concurrent"]:
                    self.stats["max_concurrent"] = current_count
                
                logger.debug(f"创建帧引用成功: frame_id={frame_id}, "
                            f"当前活跃引用数={current_count}")
                
                return frame_ref
                
            except Exception as e:
                logger.error(f"创建帧引用失败: {str(e)}")
                return None
    
    def get_reference(self, frame_id: int) -> Optional[FrameReference]:
        """
        获取现有的帧引用
        
        Args:
            frame_id: 帧ID
            
        Returns:
            Optional[FrameReference]: 帧引用，不存在返回None
        """
        with self.lock:
            return self.active_references.get(frame_id)
    
    def _on_reference_cleanup(self, frame_ref: FrameReference) -> None:
        """引用清理回调"""
        with self.lock:
            frame_id = frame_ref.metadata.frame_id

            # 如果有内存池，调用内存池的释放方法
            if self.memory_pool and frame_ref.memory_block:
                try:
                    self.memory_pool.deallocate_frame_block(frame_ref.memory_block)
                    logger.debug(f"通过内存池释放内存块: {frame_ref.memory_block.block_id}")
                except Exception as e:
                    logger.error(f"内存池释放内存块失败: {str(e)}")

            # 从活跃引用中移除
            if frame_id in self.active_references:
                del self.active_references[frame_id]

            # 移除统计信息
            if frame_id in self.reference_stats:
                del self.reference_stats[frame_id]

            # 更新统计
            self.stats["released_count"] += 1

            logger.debug(f"帧引用清理回调: frame_id={frame_id}")
    
    def cleanup_expired_references(self, max_age: float = 300.0) -> int:
        """
        清理过期的引用
        
        Args:
            max_age: 最大年龄（秒）
            
        Returns:
            int: 清理的引用数量
        """
        with self.lock:
            return self._cleanup_expired_references(max_age)
    
    def _cleanup_expired_references(self, max_age: float = 300.0) -> int:
        """内部清理过期引用方法"""
        current_time = time.time()
        expired_refs = []
        
        for frame_id, frame_ref in self.active_references.items():
            stats = frame_ref.get_access_stats()
            age = current_time - stats["last_access_time"]
            
            if age > max_age and stats["ref_count"] <= 1:
                expired_refs.append(frame_id)
        
        # 强制清理过期引用
        for frame_id in expired_refs:
            frame_ref = self.active_references.get(frame_id)
            if frame_ref:
                frame_ref._cleanup()
        
        self.stats["cleanup_count"] += len(expired_refs)
        
        if expired_refs:
            logger.info(f"清理了 {len(expired_refs)} 个过期帧引用")
        
        return len(expired_refs)
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """
        获取管理器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.lock:
            return {
                "config": {
                    "max_references": self.max_references,
                },
                "current_state": {
                    "active_references": len(self.active_references),
                    "tracked_stats": len(self.reference_stats),
                },
                "performance": self.stats.copy(),
            }
    
    def clear_all_references(self) -> None:
        """清理所有引用"""
        with self.lock:
            # 强制清理所有活跃引用
            for frame_ref in list(self.active_references.values()):
                frame_ref._cleanup()
            
            self.active_references.clear()
            self.reference_stats.clear()
            
            logger.info("所有帧引用已清理")
    
    def __len__(self) -> int:
        """返回活跃引用数量"""
        return len(self.active_references)
