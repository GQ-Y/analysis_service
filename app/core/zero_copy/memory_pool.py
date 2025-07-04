#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: memory_pool.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 零拷贝内存池

基于timelinetool架构实现的内存池，管理FrameBuffer对象的生命周期。

本文件是分析服务项目的一部分。
"""

import threading
import time
import logging
from collections import deque
from typing import Optional, List

from .frame_buffer import FrameBuffer, frame_buffer_stats


class MemoryPool:
    """
    零拷贝内存池
    
    管理预分配的FrameBuffer对象，避免运行时频繁的内存分配和释放。
    实现对象池模式，通过引用计数实现缓冲区的自动回收。
    """
    
    def __init__(
        self,
        pool_size: int = 20,
        height: int = 1080,
        width: int = 1920,
        channels: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化内存池
        
        Args:
            pool_size: 池大小
            height: 帧高度
            width: 帧宽度
            channels: 颜色通道数
            logger: 日志记录器
        """
        self.pool_size = pool_size
        self.height = height
        self.width = width
        self.channels = channels
        self.logger = logger or logging.getLogger(__name__)
        
        # 创建缓冲区池
        self.logger.info(f"🔧 内存池初始化: 创建 {pool_size} 个缓冲区...")
        self._pool: List[FrameBuffer] = []
        self._free_buffers = deque()
        
        # 预分配所有缓冲区
        for i in range(pool_size):
            buffer = FrameBuffer(height, width, channels)
            buffer._pool = self  # 设置池引用
            self._pool.append(buffer)
            self._free_buffers.append(buffer)
            frame_buffer_stats.on_buffer_created(buffer)
        
        # 线程安全
        self._lock = threading.Lock()
        
        # 统计信息
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_recycled": 0,
            "peak_usage": 0,
            "current_usage": 0
        }
        
        total_memory = self.get_total_memory_mb()
        self.logger.info(f"✅ 内存池初始化完成: {pool_size} 个缓冲区, "
                        f"总内存 {total_memory:.1f}MB")
    
    def get_buffer(self, frame_id: int, timestamp: float, stream_id: str = "") -> Optional[FrameBuffer]:
        """
        从池中获取一个空闲的FrameBuffer
        
        Args:
            frame_id: 帧ID
            timestamp: 时间戳
            stream_id: 流ID
            
        Returns:
            Optional[FrameBuffer]: 缓冲区对象，无可用时返回None
        """
        with self._lock:
            self._stats["total_requests"] += 1
            
            if not self._free_buffers:
                self._stats["failed_requests"] += 1
                self.logger.warning(f"⚠️ 内存池: 无可用缓冲区! 当前使用: {self._stats['current_usage']}/{self.pool_size}")
                return None
            
            # 获取空闲缓冲区
            buffer = self._free_buffers.popleft()
            self._stats["successful_requests"] += 1
            self._stats["current_usage"] += 1
            self._stats["peak_usage"] = max(self._stats["peak_usage"], self._stats["current_usage"])
        
        # 在锁外配置缓冲区
        buffer.reset()
        buffer.add_ref()
        buffer.set_metadata(frame_id, timestamp, stream_id)
        
        self._log_usage_debug()
        return buffer
    
    def put_frame(self, frame, frame_id: int, timestamp: float, stream_id: str = "") -> Optional[FrameBuffer]:
        """
        将帧数据放入缓冲区并返回
        
        Args:
            frame: 帧数据 (numpy array)
            frame_id: 帧ID
            timestamp: 时间戳
            stream_id: 流ID
            
        Returns:
            Optional[FrameBuffer]: 包含帧数据的缓冲区，失败返回None
        """
        buffer = self.get_buffer(frame_id, timestamp, stream_id)
        
        if buffer:
            try:
                buffer.copy_frame_data(frame)
                self.logger.debug(f"📥 内存池: 帧 {frame_id} 已存储到缓冲区")
                return buffer
            except Exception as e:
                self.logger.error(f"❌ 内存池: 存储帧 {frame_id} 失败: {e}")
                buffer.release()
                return None
        else:
            self.logger.warning(f"⚠️ 内存池: 丢弃帧 {frame_id} - 无可用缓冲区")
            return None
    
    def _recycle_buffer(self, buffer: FrameBuffer):
        """
        回收缓冲区到池中（由FrameBuffer自动调用）
        
        Args:
            buffer: 要回收的缓冲区
        """
        with self._lock:
            buffer.reset()
            self._free_buffers.append(buffer)
            self._stats["total_recycled"] += 1
            self._stats["current_usage"] -= 1
        
        frame_buffer_stats.on_buffer_recycled(buffer)
        self._log_usage_debug()
    
    def get_stats(self) -> dict:
        """
        获取内存池统计信息
        
        Returns:
            dict: 统计信息
        """
        with self._lock:
            stats = self._stats.copy()
            stats.update({
                "pool_size": self.pool_size,
                "free_buffers": len(self._free_buffers),
                "total_memory_mb": self.get_total_memory_mb(),
                "buffer_size_mb": self.get_buffer_size_mb(),
                "success_rate": (stats["successful_requests"] / max(1, stats["total_requests"])) * 100
            })
        
        return stats
    
    def get_total_memory_mb(self) -> float:
        """获取总内存大小（MB）"""
        return self.pool_size * self.get_buffer_size_mb()
    
    def get_buffer_size_mb(self) -> float:
        """获取单个缓冲区大小（MB）"""
        return (self.height * self.width * self.channels) / (1024 * 1024)
    
    def _log_usage_debug(self):
        """记录使用情况（调试级别）"""
        with self._lock:
            usage = self._stats["current_usage"]
            recycled = self._stats["total_recycled"]
        
        self.logger.debug(f"🔄 内存池使用: {usage}/{self.pool_size} | 已回收: {recycled}")
    
    def cleanup(self):
        """清理内存池"""
        with self._lock:
            # 等待所有缓冲区回收
            active_buffers = self.pool_size - len(self._free_buffers)
            if active_buffers > 0:
                self.logger.warning(f"⚠️ 内存池清理: 仍有 {active_buffers} 个活跃缓冲区")
            
            # 清空池
            self._free_buffers.clear()
            self._pool.clear()
        
        self.logger.info("🧹 内存池已清理")
    
    def force_gc(self):
        """强制垃圾回收（调试用）"""
        import gc
        collected = gc.collect()
        self.logger.debug(f"🗑️ 强制垃圾回收: 回收了 {collected} 个对象")
    
    def __str__(self) -> str:
        stats = self.get_stats()
        return (f"MemoryPool(size={self.pool_size}, "
                f"usage={stats['current_usage']}, "
                f"free={stats['free_buffers']}, "
                f"memory={stats['total_memory_mb']:.1f}MB)")
    
    def __repr__(self) -> str:
        return self.__str__()


class MemoryPoolManager:
    """内存池管理器，支持多个不同规格的内存池"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.pools = {}
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def create_pool(
        self,
        name: str,
        pool_size: int = 20,
        height: int = 1080,
        width: int = 1920,
        channels: int = 3
    ) -> MemoryPool:
        """
        创建命名内存池
        
        Args:
            name: 池名称
            pool_size: 池大小
            height: 帧高度
            width: 帧宽度
            channels: 颜色通道数
            
        Returns:
            MemoryPool: 内存池实例
        """
        with self._lock:
            if name in self.pools:
                self.logger.warning(f"⚠️ 内存池 '{name}' 已存在，将被替换")
            
            pool = MemoryPool(pool_size, height, width, channels, self.logger)
            self.pools[name] = pool
            
            self.logger.info(f"✅ 创建内存池 '{name}': {pool}")
            return pool
    
    def get_pool(self, name: str) -> Optional[MemoryPool]:
        """获取命名内存池"""
        return self.pools.get(name)
    
    def remove_pool(self, name: str) -> bool:
        """移除命名内存池"""
        with self._lock:
            if name in self.pools:
                pool = self.pools.pop(name)
                pool.cleanup()
                self.logger.info(f"🗑️ 移除内存池 '{name}'")
                return True
            return False
    
    def get_all_stats(self) -> dict:
        """获取所有池的统计信息"""
        stats = {}
        for name, pool in self.pools.items():
            stats[name] = pool.get_stats()
        return stats
    
    def cleanup_all(self):
        """清理所有内存池"""
        with self._lock:
            for name, pool in self.pools.items():
                pool.cleanup()
                self.logger.info(f"🧹 清理内存池 '{name}'")
            self.pools.clear()


# 全局内存池管理器
memory_pool_manager = MemoryPoolManager()
