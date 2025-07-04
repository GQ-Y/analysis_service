#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: time_axis.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 时间轴管理

基于timelinetool架构实现的时间轴，使用SortedDict确保帧按时间戳排序。

本文件是分析服务项目的一部分。
"""

import time
import threading
import logging
from typing import List, Optional
from sortedcontainers import SortedDict

from .frame_buffer import FrameBuffer


class TimeAxis:
    """
    时间轴管理器
    
    使用SortedDict管理FrameBuffer对象，确保帧按时间戳排序处理。
    提供超时机制防止帧积累，支持单帧和批量获取。
    """
    
    def __init__(
        self,
        timeout_seconds: float = 1.0,
        max_frames: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化时间轴
        
        Args:
            timeout_seconds: 帧超时时间（秒）
            max_frames: 最大帧数限制
            logger: 日志记录器
        """
        self.timeout = timeout_seconds
        self.max_frames = max_frames
        self.logger = logger or logging.getLogger(__name__)
        
        # 使用SortedDict按时间戳排序
        self.slots = SortedDict()
        self.lock = threading.Lock()
        
        # 统计信息
        self._stats = {
            "total_added": 0,
            "total_retrieved": 0,
            "total_timeout": 0,
            "total_dropped": 0,
            "peak_size": 0,
            "current_size": 0
        }
        
        # 日志控制
        self._last_log_time = 0
        self._log_interval = 1.0  # 每秒最多记录一次状态
        
        self.logger.info(f"⏰ 时间轴初始化: 超时={timeout_seconds}s, 最大帧数={max_frames}")
    
    def add_frame(self, frame_buffer: FrameBuffer) -> bool:
        """
        添加帧到时间轴
        
        Args:
            frame_buffer: 帧缓冲区
            
        Returns:
            bool: 是否成功添加
        """
        if not frame_buffer or not frame_buffer.is_valid:
            self.logger.warning("⚠️ 时间轴: 尝试添加无效帧缓冲区")
            return False
        
        with self.lock:
            # 检查容量限制
            if len(self.slots) >= self.max_frames:
                self._stats["total_dropped"] += 1
                self.logger.warning(f"⚠️ 时间轴: 达到最大帧数限制 {self.max_frames}，丢弃帧 {frame_buffer.frame_id}")
                return False
            
            # 添加到时间轴
            timestamp = frame_buffer.timestamp
            self.slots[timestamp] = frame_buffer
            
            # 更新统计
            self._stats["total_added"] += 1
            self._stats["current_size"] = len(self.slots)
            self._stats["peak_size"] = max(self._stats["peak_size"], self._stats["current_size"])
        
        # 定期记录状态
        self._log_status_periodic()
        
        self.logger.debug(f"📥 时间轴: 添加帧 {frame_buffer.frame_id}, 时间戳 {timestamp:.3f}")
        return True
    
    def get_frame(self) -> Optional[FrameBuffer]:
        """
        获取最旧的帧
        
        Returns:
            Optional[FrameBuffer]: 最旧的帧，无可用帧返回None
        """
        with self.lock:
            if not self.slots:
                return None
            
            # 获取最旧的帧（最小时间戳）
            timestamp, frame_buffer = self.slots.popitem(0)
            
            # 更新统计
            self._stats["total_retrieved"] += 1
            self._stats["current_size"] = len(self.slots)
        
        self.logger.debug(f"📤 时间轴: 获取帧 {frame_buffer.frame_id}, 时间戳 {timestamp:.3f}")
        return frame_buffer
    
    def get_batch(self, max_size: int) -> List[FrameBuffer]:
        """
        获取一批最旧的帧
        
        Args:
            max_size: 最大批次大小
            
        Returns:
            List[FrameBuffer]: 帧列表
        """
        batch = []
        
        with self.lock:
            if not self.slots:
                return batch
            
            # 计算实际批次大小
            actual_size = min(len(self.slots), max_size)
            
            # 批量获取最旧的帧
            for _ in range(actual_size):
                timestamp, frame_buffer = self.slots.popitem(0)
                batch.append(frame_buffer)
            
            # 更新统计
            self._stats["total_retrieved"] += len(batch)
            self._stats["current_size"] = len(self.slots)
        
        if batch:
            self.logger.debug(f"📦 时间轴: 获取批次 {len(batch)} 帧")
        
        return batch
    
    def get_due_frames(self, current_time: Optional[float] = None) -> List[FrameBuffer]:
        """
        获取所有超时的帧
        
        Args:
            current_time: 当前时间，None则使用当前时间
            
        Returns:
            List[FrameBuffer]: 超时的帧列表
        """
        if current_time is None:
            current_time = time.time()
        
        due_frames = []
        
        with self.lock:
            if not self.slots:
                return due_frames
            
            # 计算超时阈值
            timeout_threshold = current_time - self.timeout
            
            # 找到所有超时的帧
            due_keys = list(self.slots.irange(maximum=timeout_threshold, inclusive=(True, True)))
            
            if not due_keys:
                return due_frames
            
            # 移除超时的帧
            for key in due_keys:
                frame_buffer = self.slots.pop(key)
                due_frames.append(frame_buffer)
            
            # 更新统计
            self._stats["total_timeout"] += len(due_frames)
            self._stats["current_size"] = len(self.slots)
        
        if due_frames:
            self.logger.debug(f"⏰ 时间轴: 获取 {len(due_frames)} 个超时帧")
        
        return due_frames
    
    def clear(self) -> int:
        """
        清空时间轴
        
        Returns:
            int: 清理的帧数
        """
        with self.lock:
            count = len(self.slots)
            
            # 释放所有帧的引用
            for frame_buffer in self.slots.values():
                frame_buffer.release()
            
            self.slots.clear()
            self._stats["current_size"] = 0
        
        self.logger.info(f"🧹 时间轴: 清理了 {count} 个帧")
        return count
    
    def get_stats(self) -> dict:
        """
        获取时间轴统计信息
        
        Returns:
            dict: 统计信息
        """
        with self.lock:
            stats = self._stats.copy()
            stats.update({
                "timeout_seconds": self.timeout,
                "max_frames": self.max_frames,
                "oldest_timestamp": self._get_oldest_timestamp(),
                "newest_timestamp": self._get_newest_timestamp(),
                "time_span": self._get_time_span()
            })
        
        return stats
    
    def _get_oldest_timestamp(self) -> Optional[float]:
        """获取最旧帧的时间戳"""
        if self.slots:
            return self.slots.peekitem(0)[0]
        return None
    
    def _get_newest_timestamp(self) -> Optional[float]:
        """获取最新帧的时间戳"""
        if self.slots:
            return self.slots.peekitem(-1)[0]
        return None
    
    def _get_time_span(self) -> Optional[float]:
        """获取时间跨度"""
        if len(self.slots) >= 2:
            oldest = self._get_oldest_timestamp()
            newest = self._get_newest_timestamp()
            return newest - oldest
        return None
    
    def _log_status_periodic(self):
        """定期记录状态"""
        now = time.time()
        if now - self._last_log_time >= self._log_interval:
            self._last_log_time = now
            self._log_status(now)
    
    def _log_status(self, current_time: float):
        """记录当前状态"""
        with self.lock:
            pending_count = len(self.slots)
            oldest_ts = self._get_oldest_timestamp()
            newest_ts = self._get_newest_timestamp()
        
        if oldest_ts and newest_ts:
            time_span = newest_ts - oldest_ts
            delay = current_time - newest_ts
            self.logger.info(f"⏰ 时间轴状态: 待处理={pending_count}, "
                           f"时间跨度={time_span:.3f}s, 延迟={delay:.3f}s")
        else:
            self.logger.debug(f"⏰ 时间轴状态: 待处理={pending_count}")
    
    def __len__(self) -> int:
        """返回当前帧数"""
        with self.lock:
            return len(self.slots)
    
    def __str__(self) -> str:
        stats = self.get_stats()
        return (f"TimeAxis(size={stats['current_size']}, "
                f"timeout={self.timeout}s, "
                f"span={stats.get('time_span', 0):.3f}s)")
    
    def __repr__(self) -> str:
        return self.__str__()


class MultiStreamTimeAxis:
    """多流时间轴管理器"""
    
    def __init__(
        self,
        timeout_seconds: float = 1.0,
        max_frames_per_stream: int = 500,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化多流时间轴
        
        Args:
            timeout_seconds: 帧超时时间
            max_frames_per_stream: 每个流的最大帧数
            logger: 日志记录器
        """
        self.timeout = timeout_seconds
        self.max_frames_per_stream = max_frames_per_stream
        self.logger = logger or logging.getLogger(__name__)
        
        self.streams = {}
        self.lock = threading.Lock()
        
        self.logger.info(f"🌊 多流时间轴初始化: 超时={timeout_seconds}s")
    
    def get_or_create_stream(self, stream_id: str) -> TimeAxis:
        """获取或创建流的时间轴"""
        with self.lock:
            if stream_id not in self.streams:
                self.streams[stream_id] = TimeAxis(
                    self.timeout,
                    self.max_frames_per_stream,
                    self.logger
                )
                self.logger.info(f"🆕 创建流时间轴: {stream_id}")
            
            return self.streams[stream_id]
    
    def add_frame(self, frame_buffer: FrameBuffer) -> bool:
        """添加帧到对应流的时间轴"""
        stream_id = frame_buffer.stream_id or "default"
        time_axis = self.get_or_create_stream(stream_id)
        return time_axis.add_frame(frame_buffer)
    
    def get_all_stats(self) -> dict:
        """获取所有流的统计信息"""
        stats = {}
        with self.lock:
            for stream_id, time_axis in self.streams.items():
                stats[stream_id] = time_axis.get_stats()
        return stats
    
    def cleanup_stream(self, stream_id: str) -> bool:
        """清理指定流"""
        with self.lock:
            if stream_id in self.streams:
                self.streams[stream_id].clear()
                del self.streams[stream_id]
                self.logger.info(f"🧹 清理流时间轴: {stream_id}")
                return True
            return False
    
    def cleanup_all(self):
        """清理所有流"""
        with self.lock:
            for stream_id, time_axis in self.streams.items():
                time_axis.clear()
                self.logger.info(f"🧹 清理流时间轴: {stream_id}")
            self.streams.clear()
