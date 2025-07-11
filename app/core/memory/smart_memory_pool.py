#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: smart_memory_pool.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-11
描述: 智能内存池管理器 - 专为小内存设备优化

针对小内存、小显存设备的智能内存池管理系统，提供：
1. 动态内存池调整
2. 多分辨率缓冲区管理  
3. 内存使用监控和预警
4. 智能垃圾回收机制

本文件是分析服务项目的一部分。
"""

import threading
import time
import logging
import weakref
import psutil
from collections import deque, OrderedDict
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .frame_buffer import FrameBuffer


class MemoryPressureLevel(Enum):
    """内存压力级别"""
    LOW = "low"          # < 60%
    MEDIUM = "medium"    # 60-80%
    HIGH = "high"        # 80-90%
    CRITICAL = "critical" # > 90%


@dataclass
class PoolConfig:
    """内存池配置"""
    max_memory_mb: int = 512        # 最大内存限制（MB）
    base_pool_size: int = 5         # 基础池大小
    max_pool_size: int = 20         # 最大池大小
    cleanup_threshold: float = 0.8  # 清理阈值
    monitor_interval: float = 5.0   # 监控间隔（秒）
    

@dataclass
class MemoryStats:
    """内存统计信息"""
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    usage_percent: float
    pressure_level: MemoryPressureLevel
    timestamp: float


class SmartMemoryPool:
    """
    智能内存池管理器
    
    专为小内存设备优化的内存池管理系统：
    - 动态调整池大小
    - 按分辨率分组管理
    - 实时内存监控
    - 智能清理策略
    """
    
    def __init__(self, config: PoolConfig = None, logger: Optional[logging.Logger] = None):
        """
        初始化智能内存池
        
        Args:
            config: 内存池配置
            logger: 日志记录器
        """
        self.config = config or PoolConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # 多分辨率池管理
        self.pools: Dict[str, 'ResolutionPool'] = {}
        self.pools_lock = threading.RLock()
        
        # 内存监控
        self.memory_stats: List[MemoryStats] = []
        self.current_stats: Optional[MemoryStats] = None
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 全局统计
        self.global_stats = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "allocation_failures": 0,
            "memory_cleanups": 0,
            "pools_created": 0,
            "pools_destroyed": 0
        }
        
        # 清理策略
        self.cleanup_callbacks: List[callable] = []
        
        # 弱引用管理，避免循环引用
        self.active_buffers: weakref.WeakSet = weakref.WeakSet()
        
        self.logger.info(f"🧠 智能内存池初始化: 最大内存 {self.config.max_memory_mb}MB")
        
        # 启动内存监控
        self.start_monitoring()
    
    def get_buffer(self, width: int, height: int, channels: int = 3, 
                   frame_id: int = 0, timestamp: float = 0.0, 
                   stream_id: str = "") -> Optional[FrameBuffer]:
        """
        智能获取缓冲区
        
        Args:
            width: 图像宽度
            height: 图像高度
            channels: 颜色通道数
            frame_id: 帧ID
            timestamp: 时间戳
            stream_id: 流ID
            
        Returns:
            Optional[FrameBuffer]: 缓冲区对象，失败返回None
        """
        resolution_key = f"{width}x{height}x{channels}"
        
        # 检查内存压力
        if not self._check_memory_availability(width, height, channels):
            self.logger.warning(f"⚠️ 内存不足，无法分配 {resolution_key} 缓冲区")
            self.global_stats["allocation_failures"] += 1
            return None
        
        # 获取或创建对应分辨率的池
        pool = self._get_or_create_pool(resolution_key, width, height, channels)
        if not pool:
            self.logger.error(f"❌ 无法创建 {resolution_key} 内存池")
            self.global_stats["allocation_failures"] += 1
            return None
        
        # 从池中获取缓冲区
        buffer = pool.get_buffer(frame_id, timestamp, stream_id)
        if buffer:
            self.global_stats["total_allocations"] += 1
            self.active_buffers.add(buffer)
            self.logger.debug(f"📥 分配 {resolution_key} 缓冲区: {frame_id}")
        else:
            self.global_stats["allocation_failures"] += 1
            self.logger.warning(f"⚠️ 无法从池中获取 {resolution_key} 缓冲区")
        
        return buffer
    
    def _get_or_create_pool(self, resolution_key: str, width: int, height: int, channels: int) -> Optional['ResolutionPool']:
        """
        获取或创建指定分辨率的内存池
        
        Args:
            resolution_key: 分辨率键值
            width: 宽度
            height: 高度
            channels: 通道数
            
        Returns:
            Optional[ResolutionPool]: 内存池对象
        """
        with self.pools_lock:
            # 如果池已存在，直接返回
            if resolution_key in self.pools:
                return self.pools[resolution_key]
            
            # 检查是否可以创建新池
            if not self._can_create_new_pool(width, height, channels):
                # 尝试清理不活跃的池
                self._cleanup_inactive_pools()
                
                # 再次检查
                if not self._can_create_new_pool(width, height, channels):
                    return None
            
            # 创建新池
            pool = ResolutionPool(
                resolution_key=resolution_key,
                width=width,
                height=height,
                channels=channels,
                smart_pool=self,
                logger=self.logger
            )
            
            self.pools[resolution_key] = pool
            self.global_stats["pools_created"] += 1
            
            self.logger.info(f"🆕 创建新内存池: {resolution_key}")
            return pool
    
    def _can_create_new_pool(self, width: int, height: int, channels: int) -> bool:
        """
        检查是否可以创建新的内存池
        
        Args:
            width: 宽度
            height: 高度
            channels: 通道数
            
        Returns:
            bool: 是否可以创建
        """
        # 计算新池的内存需求
        frame_size = width * height * channels * 4  # 假设float32
        pool_memory = frame_size * self.config.base_pool_size
        
        # 检查当前内存使用情况
        current_memory = self._get_current_memory_usage()
        if current_memory + pool_memory > self.config.max_memory_mb * 1024 * 1024:
            return False
        
        # 检查系统内存压力
        if self.current_stats and self.current_stats.pressure_level == MemoryPressureLevel.CRITICAL:
            return False
        
        return True
    
    def _check_memory_availability(self, width: int, height: int, channels: int) -> bool:
        """
        检查内存可用性
        
        Args:
            width: 宽度
            height: 高度  
            channels: 通道数
            
        Returns:
            bool: 内存是否足够
        """
        # 计算单个缓冲区需要的内存
        frame_size = width * height * channels * 4  # 假设float32
        
        # 检查当前内存使用情况
        current_memory = self._get_current_memory_usage()
        if current_memory + frame_size > self.config.max_memory_mb * 1024 * 1024:
            # 尝试清理内存
            self._cleanup_memory()
            
            # 再次检查
            current_memory = self._get_current_memory_usage()
            if current_memory + frame_size > self.config.max_memory_mb * 1024 * 1024:
                return False
        
        # 检查系统内存压力
        if self.current_stats:
            if self.current_stats.pressure_level == MemoryPressureLevel.CRITICAL:
                return False
            elif self.current_stats.pressure_level == MemoryPressureLevel.HIGH:
                # 高压力下，只允许小分辨率
                if width > 1920 or height > 1080:
                    return False
        
        return True
    
    def _get_current_memory_usage(self) -> int:
        """
        获取当前内存使用量（字节）
        
        Returns:
            int: 当前内存使用量
        """
        total_memory = 0
        
        with self.pools_lock:
            for pool in self.pools.values():
                total_memory += pool.get_memory_usage()
        
        return total_memory
    
    def _cleanup_memory(self):
        """
        清理内存
        """
        self.logger.info("🧹 开始内存清理")
        
        # 清理不活跃的池
        cleaned_pools = self._cleanup_inactive_pools()
        
        # 清理过期的缓冲区
        cleaned_buffers = self._cleanup_expired_buffers()
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        self.global_stats["memory_cleanups"] += 1
        
        self.logger.info(f"🧹 内存清理完成: 清理 {cleaned_pools} 个池, {cleaned_buffers} 个缓冲区")
    
    def _cleanup_inactive_pools(self) -> int:
        """
        清理不活跃的内存池
        
        Returns:
            int: 清理的池数量
        """
        cleaned_count = 0
        current_time = time.time()
        
        with self.pools_lock:
            inactive_pools = []
            
            for key, pool in self.pools.items():
                if pool.is_inactive(current_time):
                    inactive_pools.append(key)
            
            for key in inactive_pools:
                pool = self.pools.pop(key)
                pool.cleanup()
                cleaned_count += 1
                self.global_stats["pools_destroyed"] += 1
                self.logger.debug(f"🗑️ 清理不活跃池: {key}")
        
        return cleaned_count
    
    def _cleanup_expired_buffers(self) -> int:
        """
        清理过期的缓冲区
        
        Returns:
            int: 清理的缓冲区数量
        """
        cleaned_count = 0
        
        with self.pools_lock:
            for pool in self.pools.values():
                cleaned_count += pool.cleanup_expired_buffers()
        
        return cleaned_count
    
    def start_monitoring(self):
        """
        启动内存监控
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("📊 内存监控已启动")
    
    def stop_monitoring(self):
        """
        停止内存监控
        """
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        self.logger.info("📊 内存监控已停止")
    
    def _monitor_loop(self):
        """
        内存监控循环
        """
        while self.monitoring:
            try:
                # 获取系统内存信息
                memory_info = psutil.virtual_memory()
                
                # 创建内存统计
                stats = MemoryStats(
                    total_memory_mb=memory_info.total / 1024 / 1024,
                    used_memory_mb=memory_info.used / 1024 / 1024,
                    available_memory_mb=memory_info.available / 1024 / 1024,
                    usage_percent=memory_info.percent,
                    pressure_level=self._get_pressure_level(memory_info.percent),
                    timestamp=time.time()
                )
                
                # 更新当前统计
                self.current_stats = stats
                
                # 保存历史统计
                self.memory_stats.append(stats)
                if len(self.memory_stats) > 100:  # 保持最近100条记录
                    self.memory_stats.pop(0)
                
                # 检查是否需要清理
                if stats.pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                    self._cleanup_memory()
                
                # 记录监控信息
                self.logger.debug(f"📊 内存使用: {stats.usage_percent:.1f}% "
                                f"({stats.used_memory_mb:.1f}MB / {stats.total_memory_mb:.1f}MB)")
                
                time.sleep(self.config.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"❌ 内存监控错误: {e}")
                time.sleep(self.config.monitor_interval)
    
    def _get_pressure_level(self, usage_percent: float) -> MemoryPressureLevel:
        """
        获取内存压力级别
        
        Args:
            usage_percent: 内存使用百分比
            
        Returns:
            MemoryPressureLevel: 压力级别
        """
        if usage_percent < 60:
            return MemoryPressureLevel.LOW
        elif usage_percent < 80:
            return MemoryPressureLevel.MEDIUM
        elif usage_percent < 90:
            return MemoryPressureLevel.HIGH
        else:
            return MemoryPressureLevel.CRITICAL
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取内存池统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.pools_lock:
            pool_stats = {}
            total_buffers = 0
            total_memory = 0
            
            for key, pool in self.pools.items():
                pool_stat = pool.get_stats()
                pool_stats[key] = pool_stat
                total_buffers += pool_stat['total_buffers']
                total_memory += pool_stat['memory_usage']
            
            return {
                "global_stats": self.global_stats.copy(),
                "pool_count": len(self.pools),
                "total_buffers": total_buffers,
                "total_memory_mb": total_memory / 1024 / 1024,
                "active_buffers": len(self.active_buffers),
                "current_memory_stats": self.current_stats.__dict__ if self.current_stats else None,
                "pool_stats": pool_stats
            }
    
    def cleanup(self):
        """
        清理所有资源
        """
        self.logger.info("🧹 开始清理智能内存池")
        
        # 停止监控
        self.stop_monitoring()
        
        # 清理所有池
        with self.pools_lock:
            for pool in self.pools.values():
                pool.cleanup()
            self.pools.clear()
        
        # 清理统计信息
        self.memory_stats.clear()
        self.current_stats = None
        
        self.logger.info("✅ 智能内存池清理完成")


class ResolutionPool:
    """
    分辨率专用内存池
    
    管理特定分辨率的缓冲区，支持动态调整大小
    """
    
    def __init__(self, resolution_key: str, width: int, height: int, channels: int,
                 smart_pool: SmartMemoryPool, logger: logging.Logger):
        """
        初始化分辨率池
        
        Args:
            resolution_key: 分辨率键值
            width: 宽度
            height: 高度
            channels: 通道数
            smart_pool: 智能内存池引用
            logger: 日志记录器
        """
        self.resolution_key = resolution_key
        self.width = width
        self.height = height
        self.channels = channels
        self.smart_pool = smart_pool
        self.logger = logger
        
        # 缓冲区管理
        self.buffers: List[FrameBuffer] = []
        self.free_buffers = deque()
        self.used_buffers: weakref.WeakSet = weakref.WeakSet()
        
        # 线程安全
        self.lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "buffers_created": 0,
            "buffers_destroyed": 0,
            "last_access_time": time.time(),
            "peak_usage": 0,  # 峰值使用量
            "avg_usage": 0.0  # 平均使用量
        }
        
        # 动态调整参数
        self.current_size = 0
        self.max_size = smart_pool.config.max_pool_size
        self.base_size = smart_pool.config.base_pool_size
        
        # 动态调整策略参数
        self.growth_factor = 1.5  # 增长因子
        self.shrink_threshold = 0.3  # 收缩阈值（空闲率）
        self.last_adjust_time = time.time()
        self.adjust_interval = 30  # 调整间隔（秒）
        
        # 使用历史记录（用于预测）
        self.usage_history = deque(maxlen=10)
        
        # 初始化基础缓冲区
        self._expand_pool(self.base_size)
    
    def get_buffer(self, frame_id: int, timestamp: float, stream_id: str) -> Optional[FrameBuffer]:
        """
        获取缓冲区
        
        Args:
            frame_id: 帧ID
            timestamp: 时间戳
            stream_id: 流ID
            
        Returns:
            Optional[FrameBuffer]: 缓冲区对象
        """
        with self.lock:
            self.stats["total_requests"] += 1
            self.stats["last_access_time"] = time.time()
            
            # 记录使用情况
            current_usage = len(self.used_buffers)
            self.usage_history.append(current_usage)
            
            # 更新峰值使用量
            if current_usage > self.stats["peak_usage"]:
                self.stats["peak_usage"] = current_usage
            
            # 检查是否需要动态调整池大小
            self._check_and_adjust_pool_size()
            
            # 如果没有空闲缓冲区，尝试扩展池
            if not self.free_buffers and self.current_size < self.max_size:
                # 智能扩展：基于历史使用预测扩展大小
                expand_size = self._calculate_expand_size()
                if expand_size > 0 and self._expand_pool(expand_size):
                    self.logger.debug(f"🔄 智能扩展 {self.resolution_key} 池: +{expand_size}")
            
            # 获取空闲缓冲区
            if not self.free_buffers:
                self.stats["failed_requests"] += 1
                self.logger.warning(f"⚠️ {self.resolution_key} 池: 无可用缓冲区")
                return None
            
            buffer = self.free_buffers.popleft()
            self.stats["successful_requests"] += 1
            
            # 配置缓冲区
            buffer.reset()
            buffer.add_ref()
            buffer.set_metadata(frame_id, timestamp, stream_id)
            
            # 设置回收回调
            buffer._release_callback = self._recycle_buffer
            
            # 添加到使用中的缓冲区
            self.used_buffers.add(buffer)
            
            return buffer
    
    def _expand_pool(self, count: int) -> bool:
        """
        扩展内存池
        
        Args:
            count: 扩展数量
            
        Returns:
            bool: 是否成功扩展
        """
        try:
            for _ in range(count):
                buffer = FrameBuffer(self.height, self.width, self.channels)
                buffer._pool = self
                self.buffers.append(buffer)
                self.free_buffers.append(buffer)
                self.current_size += 1
                self.stats["buffers_created"] += 1
                
                # 统计信息
                # frame_buffer_stats.on_buffer_created(buffer)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 扩展 {self.resolution_key} 池失败: {e}")
            return False
    
    def _recycle_buffer(self, buffer: FrameBuffer):
        """
        回收缓冲区
        
        Args:
            buffer: 要回收的缓冲区
        """
        with self.lock:
            if buffer in self.buffers:
                buffer.reset()
                self.free_buffers.append(buffer)
                self.logger.debug(f"♻️ 回收 {self.resolution_key} 缓冲区")
    
    def cleanup_expired_buffers(self, max_age: float = 300.0) -> int:
        """
        清理过期缓冲区
        
        Args:
            max_age: 最大存活时间（秒）
            
        Returns:
            int: 清理的缓冲区数量
        """
        cleaned_count = 0
        current_time = time.time()
        
        with self.lock:
            # 检查是否需要收缩池
            if (current_time - self.stats["last_access_time"] > max_age and 
                self.current_size > self.base_size):
                
                # 收缩到基础大小
                shrink_count = self.current_size - self.base_size
                shrink_count = min(shrink_count, len(self.free_buffers))
                
                for _ in range(shrink_count):
                    if self.free_buffers:
                        buffer = self.free_buffers.popleft()
                        self.buffers.remove(buffer)
                        self.current_size -= 1
                        self.stats["buffers_destroyed"] += 1
                        cleaned_count += 1
                        
                        # 统计信息
                        # frame_buffer_stats.on_buffer_destroyed(buffer)
                
                if cleaned_count > 0:
                    self.logger.debug(f"🗑️ 收缩 {self.resolution_key} 池: -{cleaned_count}")
        
        return cleaned_count
    
    def is_inactive(self, current_time: float, inactive_threshold: float = 600.0) -> bool:
        """
        检查池是否不活跃
        
        Args:
            current_time: 当前时间
            inactive_threshold: 不活跃阈值（秒）
            
        Returns:
            bool: 是否不活跃
        """
        with self.lock:
            return (current_time - self.stats["last_access_time"] > inactive_threshold and
                    len(self.used_buffers) == 0)
    
    def get_memory_usage(self) -> int:
        """
        获取内存使用量（字节）
        
        Returns:
            int: 内存使用量
        """
        frame_size = self.width * self.height * self.channels * 4  # 假设float32
        return frame_size * self.current_size
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取池统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.lock:
            return {
                "resolution": self.resolution_key,
                "total_buffers": self.current_size,
                "free_buffers": len(self.free_buffers),
                "used_buffers": len(self.used_buffers),
                "memory_usage": self.get_memory_usage(),
                "stats": self.stats.copy()
            }
    
    def _calculate_expand_size(self) -> int:
        """
        计算扩展大小
        
        基于历史使用情况和当前需求智能计算
        
        Returns:
            int: 扩展大小
        """
        if not self.usage_history:
            return min(2, self.max_size - self.current_size)
        
        # 计算平均使用量和趋势
        avg_usage = sum(self.usage_history) / len(self.usage_history)
        recent_usage = list(self.usage_history)[-3:]  # 最近3次
        recent_avg = sum(recent_usage) / len(recent_usage) if recent_usage else avg_usage
        
        # 如果最近使用量上升，增加扩展量
        if recent_avg > avg_usage * 1.2:  # 上升20%以上
            expand_size = int(self.current_size * (self.growth_factor - 1))
            expand_size = max(2, expand_size)  # 至少扩展2个
        else:
            expand_size = 1  # 保守扩展
        
        # 确保不超过最大限制
        return min(expand_size, self.max_size - self.current_size)
    
    def _check_and_adjust_pool_size(self):
        """
        检查并调整池大小
        
        根据使用情况动态调整池大小
        """
        current_time = time.time()
        
        # 检查调整间隔
        if current_time - self.last_adjust_time < self.adjust_interval:
            return
        
        self.last_adjust_time = current_time
        
        # 计算空闲率
        free_count = len(self.free_buffers)
        free_rate = free_count / self.current_size if self.current_size > 0 else 0
        
        # 如果空闲率过高且池大于基础大小，考虑收缩
        if free_rate > self.shrink_threshold and self.current_size > self.base_size:
            # 计算收缩大小
            shrink_size = int((free_count - self.base_size * self.shrink_threshold) * 0.5)
            shrink_size = min(shrink_size, self.current_size - self.base_size)
            
            if shrink_size > 0:
                self._shrink_pool(shrink_size)
                self.logger.debug(f"📉 收缩 {self.resolution_key} 池: -{shrink_size}")
    
    def _shrink_pool(self, count: int):
        """
        收缩内存池
        
        Args:
            count: 收缩数量
        """
        shrunk = 0
        for _ in range(count):
            if self.free_buffers:
                buffer = self.free_buffers.popleft()
                self.buffers.remove(buffer)
                self.current_size -= 1
                self.stats["buffers_destroyed"] += 1
                shrunk += 1
                
                # 统计信息
                # frame_buffer_stats.on_buffer_destroyed(buffer)
            else:
                break
        
        return shrunk
    
    def cleanup(self):
        """
        清理池资源
        """
        with self.lock:
            # 清理所有缓冲区
            for buffer in self.buffers:
                pass  # frame_buffer_stats.on_buffer_destroyed(buffer)
            
            self.buffers.clear()
            self.free_buffers.clear()
            self.used_buffers.clear()
            self.usage_history.clear()
            self.current_size = 0
            
            self.logger.debug(f"🧹 清理 {self.resolution_key} 池")


# 全局智能内存池实例
_smart_memory_pool: Optional[SmartMemoryPool] = None
_pool_lock = threading.Lock()


def get_smart_memory_pool(config: PoolConfig = None) -> SmartMemoryPool:
    """
    获取全局智能内存池实例
    
    Args:
        config: 内存池配置
        
    Returns:
        SmartMemoryPool: 智能内存池实例
    """
    global _smart_memory_pool
    
    with _pool_lock:
        if _smart_memory_pool is None:
            _smart_memory_pool = SmartMemoryPool(config)
        return _smart_memory_pool


def cleanup_smart_memory_pool():
    """
    清理全局智能内存池
    """
    global _smart_memory_pool
    
    with _pool_lock:
        if _smart_memory_pool:
            _smart_memory_pool.cleanup()
            _smart_memory_pool = None