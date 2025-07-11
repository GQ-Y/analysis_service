#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: enhanced_memory_manager.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-11
描述: 增强内存管理器 - 整合智能内存池

整合现有内存管理功能和新的智能内存池，提供：
1. 统一的内存管理接口
2. 智能内存分配策略
3. 内存使用监控和预警
4. 自动内存优化

本文件是分析服务项目的一部分。
"""

import threading
import time
import logging
import gc
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from .memory_manager import MemoryManager, MemoryStats as BaseMemoryStats
from .smart_memory_pool import SmartMemoryPool, PoolConfig, MemoryPressureLevel
from .frame_buffer import FrameBuffer
from .reference_counter import ReferenceCounter


@dataclass
class EnhancedMemoryStats:
    """增强内存统计信息"""
    # 基础统计
    system_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    usage_percent: float
    
    # 智能池统计
    pool_count: int
    total_buffers: int
    active_buffers: int
    pool_memory_mb: float
    
    # 压力级别
    pressure_level: MemoryPressureLevel
    
    # 性能指标
    allocation_success_rate: float
    cleanup_frequency: float
    
    # 时间戳
    timestamp: float


class EnhancedMemoryManager:
    """
    增强内存管理器
    
    整合现有的内存管理功能和智能内存池，提供统一的内存管理接口。
    专为小内存设备优化，提供智能的内存分配和回收策略。
    """
    
    def __init__(self, 
                 max_memory_mb: int = 512,
                 cleanup_threshold: float = 0.8,
                 monitor_interval: float = 5.0,
                 logger: Optional[logging.Logger] = None):
        """
        初始化增强内存管理器
        
        Args:
            max_memory_mb: 最大内存限制（MB）
            cleanup_threshold: 清理阈值
            monitor_interval: 监控间隔（秒）
            logger: 日志记录器
        """
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold = cleanup_threshold
        self.monitor_interval = monitor_interval
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化组件
        self.base_memory_manager = MemoryManager(
            max_memory_percent=85,  # 基础内存管理器使用85%阈值
            cleanup_threshold=cleanup_threshold,
            monitor_interval=monitor_interval
        )
        
        # 智能内存池配置
        pool_config = PoolConfig(
            max_memory_mb=max_memory_mb,
            base_pool_size=3,  # 小内存设备使用更小的基础池
            max_pool_size=10,  # 限制最大池大小
            cleanup_threshold=cleanup_threshold,
            monitor_interval=monitor_interval
        )
        
        self.smart_pool = SmartMemoryPool(pool_config, logger)
        
        # 统计信息
        self.stats_history: List[EnhancedMemoryStats] = []
        self.current_stats: Optional[EnhancedMemoryStats] = None
        self.max_history_size = 100
        
        # 监控控制
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 回调函数
        self.cleanup_callbacks: List[Callable] = []
        self.warning_callbacks: List[Callable[[EnhancedMemoryStats], None]] = []
        
        # 优化策略
        self.optimization_enabled = True
        self.last_optimization_time = 0
        self.optimization_interval = 60  # 每60秒进行一次优化
        
        self.logger.info(f"🚀 增强内存管理器初始化: 最大内存 {max_memory_mb}MB")
        
        # 启动监控
        self.start_monitoring()
    
    def allocate_buffer(self, width: int, height: int, channels: int = 3,
                       frame_id: int = 0, timestamp: float = 0.0,
                       stream_id: str = "") -> Optional[FrameBuffer]:
        """
        智能分配缓冲区
        
        Args:
            width: 图像宽度
            height: 图像高度
            channels: 颜色通道数
            frame_id: 帧ID
            timestamp: 时间戳
            stream_id: 流ID
            
        Returns:
            Optional[FrameBuffer]: 缓冲区对象
        """
        try:
            # 首先尝试从智能池分配
            buffer = self.smart_pool.get_buffer(
                width, height, channels, frame_id, timestamp, stream_id
            )
            
            if buffer:
                self.logger.debug(f"📥 智能池分配: {width}x{height}x{channels}")
                return buffer
            
            # 如果智能池分配失败，尝试直接分配（应急策略）
            if self._can_emergency_allocate(width, height, channels):
                buffer = self._emergency_allocate(width, height, channels, frame_id, timestamp, stream_id)
                if buffer:
                    self.logger.warning(f"⚠️ 应急分配: {width}x{height}x{channels}")
                    return buffer
            
            # 所有分配方式都失败
            self.logger.error(f"❌ 分配失败: {width}x{height}x{channels}")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 分配异常: {e}")
            return None
    
    def _can_emergency_allocate(self, width: int, height: int, channels: int) -> bool:
        """
        检查是否可以应急分配
        
        Args:
            width: 宽度
            height: 高度
            channels: 通道数
            
        Returns:
            bool: 是否可以应急分配
        """
        # 只有在内存压力不是很高时才允许应急分配
        if self.current_stats and self.current_stats.pressure_level == MemoryPressureLevel.CRITICAL:
            return False
        
        # 计算内存需求
        frame_size = width * height * channels * 4  # 假设float32
        if frame_size > 50 * 1024 * 1024:  # 超过50MB的帧不允许应急分配
            return False
        
        return True
    
    def _emergency_allocate(self, width: int, height: int, channels: int,
                           frame_id: int, timestamp: float, stream_id: str) -> Optional[FrameBuffer]:
        """
        应急分配缓冲区
        
        Args:
            width: 宽度
            height: 高度
            channels: 通道数
            frame_id: 帧ID
            timestamp: 时间戳
            stream_id: 流ID
            
        Returns:
            Optional[FrameBuffer]: 缓冲区对象
        """
        try:
            # 直接创建缓冲区，不使用池
            buffer = FrameBuffer(height, width, channels)
            buffer.set_metadata(frame_id, timestamp, stream_id)
            buffer.add_ref()
            
            # 设置自动释放
            def emergency_cleanup():
                if buffer.get_ref_count() <= 1:
                    buffer.release()
            
            buffer._release_callback = emergency_cleanup
            
            return buffer
            
        except Exception as e:
            self.logger.error(f"❌ 应急分配失败: {e}")
            return None
    
    def start_monitoring(self):
        """
        启动内存监控
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # 启动基础内存管理器监控
        self.base_memory_manager.start_monitoring()
        
        self.logger.info("📊 增强内存监控已启动")
    
    def stop_monitoring(self):
        """
        停止内存监控
        """
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # 停止基础内存管理器监控
        self.base_memory_manager.stop_monitoring()
        
        self.logger.info("📊 增强内存监控已停止")
    
    def _monitor_loop(self):
        """
        内存监控循环
        """
        while self.monitoring:
            try:
                # 收集统计信息
                stats = self._collect_stats()
                
                # 更新当前统计
                self.current_stats = stats
                
                # 保存历史统计
                self.stats_history.append(stats)
                if len(self.stats_history) > self.max_history_size:
                    self.stats_history.pop(0)
                
                # 触发警告回调
                if stats.pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                    for callback in self.warning_callbacks:
                        try:
                            callback(stats)
                        except Exception as e:
                            self.logger.error(f"❌ 警告回调失败: {e}")
                
                # 自动优化
                if self.optimization_enabled:
                    self._auto_optimize()
                
                # 记录监控信息
                self.logger.debug(f"📊 内存使用: {stats.usage_percent:.1f}%, "
                                f"池数量: {stats.pool_count}, "
                                f"活跃缓冲区: {stats.active_buffers}")
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"❌ 监控循环错误: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_stats(self) -> EnhancedMemoryStats:
        """
        收集内存统计信息
        
        Returns:
            EnhancedMemoryStats: 统计信息
        """
        # 获取基础内存统计
        base_stats = self.base_memory_manager.get_stats()
        
        # 获取智能池统计
        pool_stats = self.smart_pool.get_stats()
        
        # 计算性能指标
        total_allocations = pool_stats["global_stats"]["total_allocations"]
        failed_allocations = pool_stats["global_stats"]["allocation_failures"]
        success_rate = (total_allocations - failed_allocations) / max(1, total_allocations) * 100
        
        cleanup_count = pool_stats["global_stats"]["memory_cleanups"]
        uptime = time.time() - (self.stats_history[0].timestamp if self.stats_history else time.time())
        cleanup_frequency = cleanup_count / max(1, uptime / 3600)  # 每小时清理次数
        
        # 获取系统内存信息
        system_stats = self.smart_pool.current_stats
        
        return EnhancedMemoryStats(
            system_memory_mb=system_stats.total_memory_mb if system_stats else 0,
            used_memory_mb=system_stats.used_memory_mb if system_stats else 0,
            available_memory_mb=system_stats.available_memory_mb if system_stats else 0,
            usage_percent=system_stats.usage_percent if system_stats else 0,
            pool_count=pool_stats["pool_count"],
            total_buffers=pool_stats["total_buffers"],
            active_buffers=pool_stats["active_buffers"],
            pool_memory_mb=pool_stats["total_memory_mb"],
            pressure_level=system_stats.pressure_level if system_stats else MemoryPressureLevel.LOW,
            allocation_success_rate=success_rate,
            cleanup_frequency=cleanup_frequency,
            timestamp=time.time()
        )
    
    def _auto_optimize(self):
        """
        自动优化内存使用
        """
        current_time = time.time()
        
        # 检查是否需要优化
        if current_time - self.last_optimization_time < self.optimization_interval:
            return
        
        self.last_optimization_time = current_time
        
        if not self.current_stats:
            return
        
        # 根据内存压力级别执行不同的优化策略
        if self.current_stats.pressure_level == MemoryPressureLevel.CRITICAL:
            self._critical_optimization()
        elif self.current_stats.pressure_level == MemoryPressureLevel.HIGH:
            self._high_pressure_optimization()
        elif self.current_stats.pressure_level == MemoryPressureLevel.MEDIUM:
            self._medium_pressure_optimization()
        else:
            self._low_pressure_optimization()
    
    def _critical_optimization(self):
        """
        严重内存压力优化 - 改进版
        
        使用分阶段清理策略，减少性能影响
        """
        self.logger.warning("🆘 执行严重内存压力优化")
        
        # 阶段1: 快速释放不活跃资源
        self._quick_release_inactive()
        
        # 阶段2: 强制清理（如果需要）
        if self.current_stats and self.current_stats.usage_percent > 95:
            # 清理所有可清理的资源
            self._force_cleanup()
            
            # 阶段3: 深度垃圾回收（最后手段）
            if self.current_stats.usage_percent > 98:
                gc.collect(2)  # 执行完整的垃圾回收
        else:
            # 轻度垃圾回收
            gc.collect(0)  # 只回收第0代
        
        # 触发清理回调
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"❌ 清理回调失败: {e}")
    
    def _quick_release_inactive(self):
        """
        快速释放不活跃资源
        """
        # 清理不活跃的池（更激进）
        with self.smart_pool.pools_lock:
            current_time = time.time()
            pools_to_remove = []
            
            for key, pool in self.smart_pool.pools.items():
                # 降低不活跃阈值，更快释放
                if pool.is_inactive(current_time, inactive_threshold=60.0):  # 1分钟
                    pools_to_remove.append(key)
            
            for key in pools_to_remove:
                pool = self.smart_pool.pools.pop(key)
                pool.cleanup()
                self.logger.debug(f"🗑️ 快速清理池: {key}")
    
    def _high_pressure_optimization(self):
        """
        高内存压力优化 - 改进版
        """
        self.logger.warning("⚠️ 执行高内存压力优化")
        
        # 1. 轻度垃圾回收
        gc.collect(0)  # 只回收第0代，速度快
        
        # 2. 清理不活跃的资源
        self._cleanup_inactive_resources()
        
        # 3. 压缩内存池
        self._compact_memory_pools()
    
    def _medium_pressure_optimization(self):
        """
        中等内存压力优化 - 改进版
        """
        self.logger.info("🔄 执行中等内存压力优化")
        
        # 1. 清理过期资源
        self._cleanup_expired_resources()
        
        # 2. 可选的轻度垃圾回收
        if self.current_stats and self.current_stats.usage_percent > 70:
            gc.collect(0)
    
    def _compact_memory_pools(self):
        """
        压缩内存池，释放碎片空间
        """
        with self.smart_pool.pools_lock:
            for pool in self.smart_pool.pools.values():
                # 触发池的自动调整
                pool._check_and_adjust_pool_size()
    
    def _low_pressure_optimization(self):
        """
        低内存压力优化
        """
        self.logger.debug("✨ 执行低内存压力优化")
        
        # 1. 预防性清理
        self._preventive_cleanup()
    
    def _force_cleanup(self):
        """
        强制清理
        """
        # 清理智能池
        self.smart_pool._cleanup_memory()
        
        # 清理基础内存管理器
        self.base_memory_manager.cleanup()
        
        # 强制Python垃圾回收
        for _ in range(3):
            gc.collect()
    
    def _cleanup_inactive_resources(self):
        """
        清理不活跃资源
        """
        # 清理不活跃的池
        self.smart_pool._cleanup_inactive_pools()
        
        # 清理过期缓冲区
        self.smart_pool._cleanup_expired_buffers()
    
    def _cleanup_expired_resources(self):
        """
        清理过期资源
        """
        # 清理过期缓冲区
        self.smart_pool._cleanup_expired_buffers()
    
    def _preventive_cleanup(self):
        """
        预防性清理
        """
        # 轻度清理
        if len(self.stats_history) > 50:
            self.stats_history = self.stats_history[-50:]
    
    def add_cleanup_callback(self, callback: Callable):
        """
        添加清理回调
        
        Args:
            callback: 清理回调函数
        """
        self.cleanup_callbacks.append(callback)
    
    def add_warning_callback(self, callback: Callable[[EnhancedMemoryStats], None]):
        """
        添加警告回调
        
        Args:
            callback: 警告回调函数
        """
        self.warning_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取完整的内存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        base_stats = self.base_memory_manager.get_stats()
        pool_stats = self.smart_pool.get_stats()
        
        return {
            "enhanced_stats": self.current_stats.__dict__ if self.current_stats else None,
            "base_stats": base_stats,
            "pool_stats": pool_stats,
            "history_count": len(self.stats_history),
            "monitoring": self.monitoring,
            "optimization_enabled": self.optimization_enabled
        }
    
    def get_memory_usage_trend(self, minutes: int = 30) -> List[Dict[str, Any]]:
        """
        获取内存使用趋势
        
        Args:
            minutes: 获取最近多少分钟的趋势
            
        Returns:
            List[Dict[str, Any]]: 趋势数据
        """
        cutoff_time = time.time() - (minutes * 60)
        
        trend_data = []
        for stats in self.stats_history:
            if stats.timestamp >= cutoff_time:
                trend_data.append({
                    "timestamp": stats.timestamp,
                    "usage_percent": stats.usage_percent,
                    "pool_memory_mb": stats.pool_memory_mb,
                    "active_buffers": stats.active_buffers,
                    "pressure_level": stats.pressure_level.value
                })
        
        return trend_data
    
    def force_cleanup(self):
        """
        强制执行内存清理
        """
        self.logger.info("🧹 强制执行内存清理")
        self._force_cleanup()
    
    def set_optimization_enabled(self, enabled: bool):
        """
        设置自动优化开关
        
        Args:
            enabled: 是否启用自动优化
        """
        self.optimization_enabled = enabled
        self.logger.info(f"🔧 自动优化: {'启用' if enabled else '禁用'}")
    
    def cleanup(self):
        """
        清理所有资源
        """
        self.logger.info("🧹 开始清理增强内存管理器")
        
        # 停止监控
        self.stop_monitoring()
        
        # 清理智能池
        self.smart_pool.cleanup()
        
        # 清理基础内存管理器
        self.base_memory_manager.cleanup()
        
        # 清理统计信息
        self.stats_history.clear()
        self.current_stats = None
        
        # 清理回调
        self.cleanup_callbacks.clear()
        self.warning_callbacks.clear()
        
        self.logger.info("✅ 增强内存管理器清理完成")


# 全局增强内存管理器实例
_enhanced_memory_manager: Optional[EnhancedMemoryManager] = None
_manager_lock = threading.Lock()


def get_enhanced_memory_manager(max_memory_mb: int = 512,
                               cleanup_threshold: float = 0.8,
                               monitor_interval: float = 5.0) -> EnhancedMemoryManager:
    """
    获取全局增强内存管理器实例
    
    Args:
        max_memory_mb: 最大内存限制（MB）
        cleanup_threshold: 清理阈值
        monitor_interval: 监控间隔（秒）
        
    Returns:
        EnhancedMemoryManager: 增强内存管理器实例
    """
    global _enhanced_memory_manager
    
    with _manager_lock:
        if _enhanced_memory_manager is None:
            _enhanced_memory_manager = EnhancedMemoryManager(
                max_memory_mb=max_memory_mb,
                cleanup_threshold=cleanup_threshold,
                monitor_interval=monitor_interval
            )
        return _enhanced_memory_manager


def cleanup_enhanced_memory_manager():
    """
    清理全局增强内存管理器
    """
    global _enhanced_memory_manager
    
    with _manager_lock:
        if _enhanced_memory_manager:
            _enhanced_memory_manager.cleanup()
            _enhanced_memory_manager = None