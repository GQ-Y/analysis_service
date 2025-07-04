#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: memory_manager.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 内存管理器

参考timelinetool的内存管理设计，实现智能内存管理和自动回收机制。

本文件是分析服务项目的一部分。
"""

import gc
import psutil
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from .reference_counter import ReferenceCounter
from .frame_buffer import FrameBuffer


@dataclass
class MemoryStats:
    """内存统计信息"""
    total_memory: int  # 总内存（字节）
    available_memory: int  # 可用内存（字节）
    used_memory: int  # 已用内存（字节）
    memory_percent: float  # 内存使用率（百分比）
    process_memory: int  # 进程内存（字节）
    buffer_memory: int  # 缓冲区内存（字节）
    timestamp: datetime  # 统计时间


class MemoryManager:
    """内存管理器"""
    
    def __init__(self, 
                 max_memory_percent: float = 80.0,
                 cleanup_threshold: float = 90.0,
                 monitor_interval: float = 5.0):
        """初始化内存管理器
        
        Args:
            max_memory_percent: 最大内存使用率（百分比）
            cleanup_threshold: 清理阈值（百分比）
            monitor_interval: 监控间隔（秒）
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 配置参数
        self.max_memory_percent = max_memory_percent
        self.cleanup_threshold = cleanup_threshold
        self.monitor_interval = monitor_interval
        
        # 组件
        self.reference_counter = ReferenceCounter()
        self.frame_buffers: Dict[str, FrameBuffer] = {}
        
        # 监控相关
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # 回调函数
        self._cleanup_callbacks: List[Callable] = []
        self._warning_callbacks: List[Callable[[MemoryStats], None]] = []
        
        # 统计信息
        self._stats_history: List[MemoryStats] = []
        self._max_history_size = 100
        
        self.logger.info(f"内存管理器初始化完成，最大内存使用率: {max_memory_percent}%")
    
    def start_monitoring(self):
        """开始内存监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("内存监控已启动")
    
    def stop_monitoring(self):
        """停止内存监控"""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        self.logger.info("内存监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()
                self._update_stats_history(stats)
                
                # 检查内存使用率
                if stats.memory_percent >= self.cleanup_threshold:
                    self.logger.warning(f"内存使用率过高: {stats.memory_percent:.1f}%，开始清理")
                    self.cleanup_memory()
                elif stats.memory_percent >= self.max_memory_percent:
                    self.logger.warning(f"内存使用率警告: {stats.memory_percent:.1f}%")
                    self._trigger_warning_callbacks(stats)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"内存监控异常: {e}")
                time.sleep(self.monitor_interval)
    
    def get_memory_stats(self) -> MemoryStats:
        """获取内存统计信息
        
        Returns:
            MemoryStats: 内存统计信息
        """
        # 系统内存信息
        memory = psutil.virtual_memory()
        
        # 进程内存信息
        process = psutil.Process()
        process_memory = process.memory_info().rss
        
        # 缓冲区内存
        buffer_memory = sum(buffer.get_memory_usage() for buffer in self.frame_buffers.values())
        
        return MemoryStats(
            total_memory=memory.total,
            available_memory=memory.available,
            used_memory=memory.used,
            memory_percent=memory.percent,
            process_memory=process_memory,
            buffer_memory=buffer_memory,
            timestamp=datetime.now()
        )
    
    def _update_stats_history(self, stats: MemoryStats):
        """更新统计历史
        
        Args:
            stats: 内存统计信息
        """
        with self._lock:
            self._stats_history.append(stats)
            if len(self._stats_history) > self._max_history_size:
                self._stats_history.pop(0)
    
    def get_stats_history(self, minutes: int = 10) -> List[MemoryStats]:
        """获取统计历史
        
        Args:
            minutes: 获取最近几分钟的历史
            
        Returns:
            List[MemoryStats]: 统计历史列表
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            return [stats for stats in self._stats_history if stats.timestamp >= cutoff_time]
    
    def create_frame_buffer(self, buffer_id: str, max_size: int = 100, timeout: float = 30.0) -> FrameBuffer:
        """创建帧缓冲区
        
        Args:
            buffer_id: 缓冲区ID
            max_size: 最大缓冲区大小
            timeout: 超时时间（秒）
            
        Returns:
            FrameBuffer: 帧缓冲区实例
        """
        with self._lock:
            if buffer_id in self.frame_buffers:
                self.logger.warning(f"帧缓冲区已存在: {buffer_id}")
                return self.frame_buffers[buffer_id]
            
            buffer = FrameBuffer(
                buffer_id=buffer_id,
                max_size=max_size,
                timeout=timeout,
                reference_counter=self.reference_counter
            )
            
            self.frame_buffers[buffer_id] = buffer
            self.logger.info(f"创建帧缓冲区: {buffer_id}")
            return buffer
    
    def get_frame_buffer(self, buffer_id: str) -> Optional[FrameBuffer]:
        """获取帧缓冲区
        
        Args:
            buffer_id: 缓冲区ID
            
        Returns:
            Optional[FrameBuffer]: 帧缓冲区实例
        """
        return self.frame_buffers.get(buffer_id)
    
    def remove_frame_buffer(self, buffer_id: str):
        """移除帧缓冲区
        
        Args:
            buffer_id: 缓冲区ID
        """
        with self._lock:
            if buffer_id in self.frame_buffers:
                buffer = self.frame_buffers[buffer_id]
                buffer.clear()
                del self.frame_buffers[buffer_id]
                self.logger.info(f"移除帧缓冲区: {buffer_id}")
    
    def cleanup_memory(self):
        """清理内存"""
        self.logger.info("开始内存清理")
        
        # 清理过期的帧缓冲区
        self._cleanup_expired_buffers()
        
        # 清理引用计数器
        self.reference_counter.cleanup_expired()
        
        # 执行自定义清理回调
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"执行清理回调失败: {e}")
        
        # 强制垃圾回收
        collected = gc.collect()
        self.logger.info(f"内存清理完成，回收对象数: {collected}")
    
    def _cleanup_expired_buffers(self):
        """清理过期的帧缓冲区"""
        expired_buffers = []
        
        with self._lock:
            for buffer_id, buffer in self.frame_buffers.items():
                if buffer.is_expired():
                    expired_buffers.append(buffer_id)
        
        for buffer_id in expired_buffers:
            self.remove_frame_buffer(buffer_id)
            self.logger.info(f"清理过期帧缓冲区: {buffer_id}")
    
    def add_cleanup_callback(self, callback: Callable):
        """添加清理回调函数
        
        Args:
            callback: 清理回调函数
        """
        self._cleanup_callbacks.append(callback)
    
    def remove_cleanup_callback(self, callback: Callable):
        """移除清理回调函数
        
        Args:
            callback: 清理回调函数
        """
        if callback in self._cleanup_callbacks:
            self._cleanup_callbacks.remove(callback)
    
    def add_warning_callback(self, callback: Callable[[MemoryStats], None]):
        """添加警告回调函数
        
        Args:
            callback: 警告回调函数
        """
        self._warning_callbacks.append(callback)
    
    def remove_warning_callback(self, callback: Callable[[MemoryStats], None]):
        """移除警告回调函数
        
        Args:
            callback: 警告回调函数
        """
        if callback in self._warning_callbacks:
            self._warning_callbacks.remove(callback)
    
    def _trigger_warning_callbacks(self, stats: MemoryStats):
        """触发警告回调
        
        Args:
            stats: 内存统计信息
        """
        for callback in self._warning_callbacks:
            try:
                callback(stats)
            except Exception as e:
                self.logger.error(f"执行警告回调失败: {e}")
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息
        
        Returns:
            Dict[str, Any]: 缓冲区统计信息
        """
        with self._lock:
            total_buffers = len(self.frame_buffers)
            total_frames = sum(len(buffer) for buffer in self.frame_buffers.values())
            total_memory = sum(buffer.get_memory_usage() for buffer in self.frame_buffers.values())
            
            buffer_details = {}
            for buffer_id, buffer in self.frame_buffers.items():
                buffer_details[buffer_id] = {
                    'frame_count': len(buffer),
                    'memory_usage': buffer.get_memory_usage(),
                    'max_size': buffer.max_size,
                    'is_expired': buffer.is_expired(),
                    'last_access': buffer.last_access_time
                }
            
            return {
                'total_buffers': total_buffers,
                'total_frames': total_frames,
                'total_memory': total_memory,
                'buffer_details': buffer_details
            }
    
    def force_cleanup(self):
        """强制清理所有资源"""
        self.logger.info("强制清理所有资源")
        
        # 清理所有帧缓冲区
        with self._lock:
            for buffer in self.frame_buffers.values():
                buffer.clear()
            self.frame_buffers.clear()
        
        # 清理引用计数器
        self.reference_counter.clear()
        
        # 强制垃圾回收
        collected = gc.collect()
        self.logger.info(f"强制清理完成，回收对象数: {collected}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()
        self.force_cleanup()


# 全局内存管理器实例
_global_memory_manager = None


def get_memory_manager() -> MemoryManager:
    """获取全局内存管理器实例
    
    Returns:
        MemoryManager: 内存管理器实例
    """
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
        _global_memory_manager.start_monitoring()
    return _global_memory_manager


def cleanup_global_memory():
    """清理全局内存"""
    manager = get_memory_manager()
    manager.cleanup_memory()
