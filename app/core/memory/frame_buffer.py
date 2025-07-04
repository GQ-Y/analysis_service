#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: frame_buffer.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 帧缓冲区

参考timelinetool的设计，实现高效的帧缓冲和管理机制。

本文件是分析服务项目的一部分。
"""

import threading
import logging
import sys
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .reference_counter import ReferenceCounter


@dataclass
class FrameInfo:
    """帧信息"""
    frame_id: str  # 帧ID
    timestamp: datetime  # 时间戳
    frame_data: np.ndarray  # 帧数据
    metadata: Dict[str, Any]  # 元数据
    reference_id: str  # 引用ID
    size_bytes: int  # 大小（字节）


class FrameBuffer:
    """帧缓冲区"""
    
    def __init__(self, 
                 buffer_id: str,
                 max_size: int = 100,
                 timeout: float = 30.0,
                 reference_counter: Optional[ReferenceCounter] = None):
        """初始化帧缓冲区
        
        Args:
            buffer_id: 缓冲区ID
            max_size: 最大缓冲区大小
            timeout: 超时时间（秒）
            reference_counter: 引用计数器
        """
        self.buffer_id = buffer_id
        self.max_size = max_size
        self.timeout = timeout
        self.reference_counter = reference_counter or ReferenceCounter()
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{buffer_id}]")
        
        # 帧存储（使用OrderedDict保持插入顺序）
        self._frames: OrderedDict[str, FrameInfo] = OrderedDict()
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 统计信息
        self._total_frames_added = 0
        self._total_frames_removed = 0
        self._total_bytes_processed = 0
        
        # 时间戳
        self.created_at = datetime.now()
        self.last_access_time = datetime.now()
        
        self.logger.info(f"帧缓冲区初始化完成，最大大小: {max_size}")
    
    def add_frame(self, frame_data: np.ndarray, frame_id: str = None, metadata: Dict[str, Any] = None) -> str:
        """添加帧到缓冲区
        
        Args:
            frame_data: 帧数据
            frame_id: 帧ID，如果不提供则自动生成
            metadata: 元数据
            
        Returns:
            str: 帧ID
        """
        if frame_id is None:
            frame_id = f"{self.buffer_id}_frame_{len(self._frames)}_{datetime.now().timestamp()}"
        
        if metadata is None:
            metadata = {}
        
        # 计算帧大小
        frame_size = frame_data.nbytes
        
        with self._lock:
            # 检查缓冲区是否已满
            if len(self._frames) >= self.max_size:
                self._remove_oldest_frame()
            
            # 添加引用
            reference_id = self.reference_counter.add_reference(
                frame_data, 
                f"{frame_id}_ref",
                cleanup_callback=lambda: self._cleanup_frame_reference(frame_id)
            )
            
            # 创建帧信息
            frame_info = FrameInfo(
                frame_id=frame_id,
                timestamp=datetime.now(),
                frame_data=frame_data,
                metadata=metadata,
                reference_id=reference_id,
                size_bytes=frame_size
            )
            
            # 添加到缓冲区
            self._frames[frame_id] = frame_info
            
            # 更新统计信息
            self._total_frames_added += 1
            self._total_bytes_processed += frame_size
            self.last_access_time = datetime.now()
            
            self.logger.debug(f"添加帧: {frame_id}, 大小: {frame_size} bytes")
            return frame_id
    
    def get_frame(self, frame_id: str) -> Optional[np.ndarray]:
        """获取帧数据
        
        Args:
            frame_id: 帧ID
            
        Returns:
            Optional[np.ndarray]: 帧数据
        """
        with self._lock:
            if frame_id not in self._frames:
                return None
            
            frame_info = self._frames[frame_id]
            self.last_access_time = datetime.now()
            
            # 检查帧是否过期
            if self._is_frame_expired(frame_info):
                self._remove_frame(frame_id)
                return None
            
            return frame_info.frame_data
    
    def get_frame_info(self, frame_id: str) -> Optional[FrameInfo]:
        """获取帧信息
        
        Args:
            frame_id: 帧ID
            
        Returns:
            Optional[FrameInfo]: 帧信息
        """
        with self._lock:
            if frame_id not in self._frames:
                return None
            
            frame_info = self._frames[frame_id]
            self.last_access_time = datetime.now()
            
            # 检查帧是否过期
            if self._is_frame_expired(frame_info):
                self._remove_frame(frame_id)
                return None
            
            return frame_info
    
    def get_latest_frame(self) -> Optional[Tuple[str, np.ndarray]]:
        """获取最新帧
        
        Returns:
            Optional[Tuple[str, np.ndarray]]: (帧ID, 帧数据)
        """
        with self._lock:
            if not self._frames:
                return None
            
            # 获取最后添加的帧
            frame_id = next(reversed(self._frames))
            frame_data = self.get_frame(frame_id)
            
            if frame_data is not None:
                return frame_id, frame_data
            
            return None
    
    def get_frames_in_range(self, start_time: datetime, end_time: datetime) -> List[Tuple[str, np.ndarray]]:
        """获取时间范围内的帧
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[Tuple[str, np.ndarray]]: 帧列表
        """
        frames = []
        
        with self._lock:
            for frame_id, frame_info in self._frames.items():
                if start_time <= frame_info.timestamp <= end_time:
                    if not self._is_frame_expired(frame_info):
                        frames.append((frame_id, frame_info.frame_data))
        
        return frames
    
    def remove_frame(self, frame_id: str) -> bool:
        """移除指定帧
        
        Args:
            frame_id: 帧ID
            
        Returns:
            bool: 是否成功移除
        """
        with self._lock:
            return self._remove_frame(frame_id)
    
    def _remove_frame(self, frame_id: str) -> bool:
        """内部移除帧方法
        
        Args:
            frame_id: 帧ID
            
        Returns:
            bool: 是否成功移除
        """
        if frame_id not in self._frames:
            return False
        
        frame_info = self._frames[frame_id]
        
        # 移除引用
        self.reference_counter.remove_reference(frame_info.reference_id)
        
        # 从缓冲区移除
        del self._frames[frame_id]
        
        # 更新统计信息
        self._total_frames_removed += 1
        
        self.logger.debug(f"移除帧: {frame_id}")
        return True
    
    def _remove_oldest_frame(self):
        """移除最旧的帧"""
        if self._frames:
            oldest_frame_id = next(iter(self._frames))
            self._remove_frame(oldest_frame_id)
            self.logger.debug(f"移除最旧帧: {oldest_frame_id}")
    
    def _cleanup_frame_reference(self, frame_id: str):
        """清理帧引用回调
        
        Args:
            frame_id: 帧ID
        """
        self.logger.debug(f"清理帧引用: {frame_id}")
    
    def _is_frame_expired(self, frame_info: FrameInfo) -> bool:
        """检查帧是否过期
        
        Args:
            frame_info: 帧信息
            
        Returns:
            bool: 是否过期
        """
        if self.timeout <= 0:
            return False
        
        age = (datetime.now() - frame_info.timestamp).total_seconds()
        return age > self.timeout
    
    def cleanup_expired_frames(self) -> int:
        """清理过期帧
        
        Returns:
            int: 清理的帧数量
        """
        expired_frames = []
        
        with self._lock:
            for frame_id, frame_info in self._frames.items():
                if self._is_frame_expired(frame_info):
                    expired_frames.append(frame_id)
        
        for frame_id in expired_frames:
            self._remove_frame(frame_id)
        
        if expired_frames:
            self.logger.info(f"清理过期帧: {len(expired_frames)} 个")
        
        return len(expired_frames)
    
    def clear(self):
        """清空缓冲区"""
        with self._lock:
            # 移除所有引用
            for frame_info in self._frames.values():
                self.reference_counter.remove_reference(frame_info.reference_id)
            
            # 清空缓冲区
            frame_count = len(self._frames)
            self._frames.clear()
            
            self.logger.info(f"清空缓冲区，移除 {frame_count} 个帧")
    
    def get_memory_usage(self) -> int:
        """获取内存使用量
        
        Returns:
            int: 内存使用量（字节）
        """
        with self._lock:
            return sum(frame_info.size_bytes for frame_info in self._frames.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            current_frames = len(self._frames)
            memory_usage = self.get_memory_usage()
            
            # 计算平均帧大小
            avg_frame_size = memory_usage / current_frames if current_frames > 0 else 0
            
            # 计算缓冲区年龄
            buffer_age = (datetime.now() - self.created_at).total_seconds()
            
            # 计算最后访问时间差
            last_access_age = (datetime.now() - self.last_access_time).total_seconds()
            
            return {
                'buffer_id': self.buffer_id,
                'current_frames': current_frames,
                'max_size': self.max_size,
                'memory_usage_bytes': memory_usage,
                'average_frame_size_bytes': avg_frame_size,
                'total_frames_added': self._total_frames_added,
                'total_frames_removed': self._total_frames_removed,
                'total_bytes_processed': self._total_bytes_processed,
                'buffer_age_seconds': buffer_age,
                'last_access_age_seconds': last_access_age,
                'timeout_seconds': self.timeout,
                'is_expired': self.is_expired()
            }
    
    def is_expired(self) -> bool:
        """检查缓冲区是否过期
        
        Returns:
            bool: 是否过期
        """
        if self.timeout <= 0:
            return False
        
        age = (datetime.now() - self.last_access_time).total_seconds()
        return age > self.timeout
    
    def __len__(self) -> int:
        """返回缓冲区中的帧数量"""
        return len(self._frames)
    
    def __contains__(self, frame_id: str) -> bool:
        """检查帧是否在缓冲区中"""
        return frame_id in self._frames
    
    def __iter__(self):
        """迭代帧ID"""
        with self._lock:
            return iter(list(self._frames.keys()))
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"FrameBuffer({self.buffer_id}, {len(self._frames)}/{self.max_size} frames)"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"FrameBuffer(buffer_id='{self.buffer_id}', "
                f"frames={len(self._frames)}, max_size={self.max_size}, "
                f"timeout={self.timeout}, memory={self.get_memory_usage()} bytes)")
