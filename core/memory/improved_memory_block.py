"""
改进的内存块设计 - 参考timelinetool的FrameBuffer引用计数机制
"""
import threading
import time
import numpy as np
from typing import Optional, Callable, Any
from dataclasses import dataclass

from shared.logging.logger_config import normal_logger, analysis_logger, exception_logger


@dataclass
class MemoryBlockStats:
    """内存块统计信息"""
    block_id: int
    width: int
    height: int
    channels: int
    size_bytes: int
    ref_count: int
    created_time: float
    last_access_time: float
    total_allocations: int = 0
    total_releases: int = 0


class ImprovedMemoryBlock:
    """
    改进的内存块 - 参考timelinetool的FrameBuffer设计
    """
    
    def __init__(self, block_id: int, width: int, height: int, channels: int = 3, 
                 alignment: int = 64, dtype=np.uint8):
        """
        初始化内存块
        
        Args:
            block_id: 内存块ID
            width: 宽度
            height: 高度
            channels: 通道数
            alignment: 内存对齐字节数
            dtype: 数据类型
        """
        self.block_id = block_id
        self.width = width
        self.height = height
        self.channels = channels
        self.alignment = alignment
        self.dtype = dtype
        
        # 计算内存大小
        self.size_bytes = width * height * channels * np.dtype(dtype).itemsize
        
        # 分配对齐内存（参考timelinetool的内存分配）
        self._allocate_aligned_memory()
        
        # 引用计数管理（参考timelinetool的FrameBuffer）
        self.ref_count = 0
        self.ref_lock = threading.Lock()
        
        # 内存池引用（用于自动回收）
        self._pool_ref: Optional[Any] = None
        self._recycle_callback: Optional[Callable] = None
        
        # 统计信息
        self.created_time = time.time()
        self.last_access_time = self.created_time
        self.total_allocations = 0
        self.total_releases = 0
        
        # 状态标记
        self.is_valid = True
        self.is_in_use = False
        
        normal_logger.debug(f"内存块创建: {self.block_id} ({width}x{height}x{channels})")
    
    def _allocate_aligned_memory(self):
        """分配对齐内存"""
        try:
            # 创建numpy数组（参考timelinetool的frame_data）
            self.data = np.empty((self.height, self.width, self.channels), dtype=self.dtype)
            
            # 确保内存对齐
            if self.alignment > 1:
                # 检查是否已对齐
                if self.data.ctypes.data % self.alignment != 0:
                    # 重新分配对齐内存
                    aligned_size = self.size_bytes + self.alignment - 1
                    raw_memory = np.empty(aligned_size, dtype=np.uint8)
                    
                    # 计算对齐地址
                    aligned_addr = (raw_memory.ctypes.data + self.alignment - 1) // self.alignment * self.alignment
                    offset = aligned_addr - raw_memory.ctypes.data
                    
                    # 创建对齐视图
                    aligned_memory = raw_memory[offset:offset + self.size_bytes]
                    self.data = aligned_memory.view(dtype=self.dtype).reshape(
                        (self.height, self.width, self.channels)
                    )
            
        except Exception as e:
            exception_logger.exception(f"内存块分配失败: {self.block_id}, {str(e)}")
            raise
    
    def add_ref(self):
        """
        增加引用计数 - 参考timelinetool的FrameBuffer.add_ref
        """
        with self.ref_lock:
            self.ref_count += 1
            self.total_allocations += 1
            self.last_access_time = time.time()
            self.is_in_use = True
            
            analysis_logger.debug(f"内存块引用+1: {self.block_id}, 当前引用数: {self.ref_count}")
    
    def release(self):
        """
        释放引用 - 参考timelinetool的FrameBuffer.release
        """
        with self.ref_lock:
            if self.ref_count <= 0:
                normal_logger.warning(f"内存块 {self.block_id} 引用计数已为0，无法继续释放")
                return
            
            self.ref_count -= 1
            self.total_releases += 1
            self.last_access_time = time.time()
            
            analysis_logger.debug(f"内存块引用-1: {self.block_id}, 当前引用数: {self.ref_count}")
            
            # 引用计数归零时自动回收（参考timelinetool的自动回收机制）
            if self.ref_count == 0:
                self.is_in_use = False
                if self._recycle_callback:
                    try:
                        self._recycle_callback(self)
                    except Exception as e:
                        exception_logger.exception(f"内存块回收回调异常: {self.block_id}, {str(e)}")
    
    def get_numpy_view(self) -> Optional[np.ndarray]:
        """
        获取numpy视图 - 参考timelinetool的frame_data访问
        """
        if not self.is_valid:
            normal_logger.error(f"内存块 {self.block_id} 已无效")
            return None
        
        self.last_access_time = time.time()
        return self.data
    
    def copy_from_numpy(self, source: np.ndarray) -> bool:
        """
        从numpy数组复制数据 - 参考timelinetool的数据复制
        """
        try:
            if not self.is_valid:
                normal_logger.error(f"内存块 {self.block_id} 已无效")
                return False
            
            # 检查形状匹配
            expected_shape = (self.height, self.width, self.channels)
            if source.shape != expected_shape:
                normal_logger.error(
                    f"内存块 {self.block_id} 形状不匹配: {source.shape} vs {expected_shape}"
                )
                return False
            
            # 复制数据（参考timelinetool的np.copyto）
            np.copyto(self.data, source)
            self.last_access_time = time.time()
            
            analysis_logger.debug(f"内存块数据复制成功: {self.block_id}")
            return True
            
        except Exception as e:
            exception_logger.exception(f"内存块数据复制失败: {self.block_id}, {str(e)}")
            return False
    
    def copy_to_numpy(self, destination: np.ndarray) -> bool:
        """
        复制数据到numpy数组
        """
        try:
            if not self.is_valid:
                normal_logger.error(f"内存块 {self.block_id} 已无效")
                return False
            
            # 检查形状匹配
            if destination.shape != self.data.shape:
                normal_logger.error(
                    f"内存块 {self.block_id} 目标形状不匹配: {destination.shape} vs {self.data.shape}"
                )
                return False
            
            # 复制数据
            np.copyto(destination, self.data)
            self.last_access_time = time.time()
            
            analysis_logger.debug(f"内存块数据输出成功: {self.block_id}")
            return True
            
        except Exception as e:
            exception_logger.exception(f"内存块数据输出失败: {self.block_id}, {str(e)}")
            return False
    
    def set_pool_reference(self, pool_ref: Any, recycle_callback: Callable):
        """
        设置内存池引用和回收回调 - 参考timelinetool的_pool引用
        """
        self._pool_ref = pool_ref
        self._recycle_callback = recycle_callback
    
    def reset(self):
        """
        重置内存块状态 - 用于回收时清理
        """
        with self.ref_lock:
            if self.ref_count > 0:
                normal_logger.warning(f"内存块 {self.block_id} 仍有引用，强制重置")
            
            self.ref_count = 0
            self.is_in_use = False
            self.last_access_time = time.time()
            
            # 清零数据（可选，用于调试）
            if hasattr(self, 'data') and self.data is not None:
                self.data.fill(0)
    
    def invalidate(self):
        """
        使内存块无效 - 用于销毁时
        """
        with self.ref_lock:
            self.is_valid = False
            self.is_in_use = False
            
            # 清理引用
            self._pool_ref = None
            self._recycle_callback = None
    
    def get_stats(self) -> MemoryBlockStats:
        """获取统计信息"""
        with self.ref_lock:
            return MemoryBlockStats(
                block_id=self.block_id,
                width=self.width,
                height=self.height,
                channels=self.channels,
                size_bytes=self.size_bytes,
                ref_count=self.ref_count,
                created_time=self.created_time,
                last_access_time=self.last_access_time,
                total_allocations=self.total_allocations,
                total_releases=self.total_releases
            )
    
    def __str__(self) -> str:
        return (f"MemoryBlock(id={self.block_id}, "
                f"size={self.width}x{self.height}x{self.channels}, "
                f"refs={self.ref_count}, valid={self.is_valid})")
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __del__(self):
        """析构函数 - 确保资源清理"""
        try:
            if hasattr(self, 'ref_count') and self.ref_count > 0:
                normal_logger.warning(f"内存块 {self.block_id} 析构时仍有引用: {self.ref_count}")
            
            self.invalidate()
        except Exception:
            pass  # 析构函数中不抛出异常
