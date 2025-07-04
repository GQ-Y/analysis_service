#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: frame_buffer.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 零拷贝帧缓冲区

基于timelinetool架构实现的零拷贝帧缓冲区，支持引用计数和自动内存回收。

本文件是分析服务项目的一部分。
"""

import numpy as np
import threading
import time
from typing import Optional, Any


class FrameBuffer:
    """
    零拷贝帧缓冲区
    
    封装视频帧数据和元数据，实现引用计数机制，
    确保在多个组件间传递时无需复制数据。
    """
    
    def __init__(self, height: int = 1080, width: int = 1920, channels: int = 3):
        """
        初始化帧缓冲区
        
        Args:
            height: 帧高度
            width: 帧宽度  
            channels: 颜色通道数
        """
        self.height = height
        self.width = width
        self.channels = channels
        
        # 预分配帧数据内存
        self.frame_data = np.zeros((height, width, channels), dtype=np.uint8)
        
        # 元数据
        self.frame_id: int = -1
        self.timestamp: float = -1.0
        self.stream_id: str = ""
        self.analysis_results: dict = {}
        
        # 引用计数机制
        self._ref_count = 0
        self._ref_lock = threading.Lock()
        self._pool: Optional['MemoryPool'] = None
        
        # 状态标记
        self._is_valid = True
        self._created_at = time.time()
    
    def add_ref(self) -> 'FrameBuffer':
        """
        增加引用计数
        
        Returns:
            FrameBuffer: 返回自身，支持链式调用
        """
        with self._ref_lock:
            if not self._is_valid:
                raise RuntimeError("尝试引用已失效的FrameBuffer")
            self._ref_count += 1
        return self
    
    def release(self):
        """
        释放引用，当引用计数为0时自动回收到内存池
        """
        with self._ref_lock:
            if not self._is_valid:
                return
            
            self._ref_count -= 1
            
            if self._ref_count <= 0:
                self._is_valid = False
                if self._pool:
                    self._pool._recycle_buffer(self)
    
    def copy_frame_data(self, frame: np.ndarray):
        """
        复制帧数据到缓冲区
        
        Args:
            frame: 输入帧数据
        """
        if not self._is_valid:
            raise RuntimeError("尝试写入已失效的FrameBuffer")
        
        # 如果尺寸不匹配，进行resize
        if frame.shape[:2] != (self.height, self.width):
            import cv2
            frame = cv2.resize(frame, (self.width, self.height))
        
        # 零拷贝写入
        np.copyto(self.frame_data, frame)
    
    def get_frame_copy(self) -> np.ndarray:
        """
        获取帧数据的副本
        
        Returns:
            np.ndarray: 帧数据副本
        """
        if not self._is_valid:
            raise RuntimeError("尝试读取已失效的FrameBuffer")
        
        return self.frame_data.copy()
    
    def get_frame_view(self) -> np.ndarray:
        """
        获取帧数据的视图（零拷贝）
        
        Returns:
            np.ndarray: 帧数据视图
        """
        if not self._is_valid:
            raise RuntimeError("尝试读取已失效的FrameBuffer")
        
        return self.frame_data
    
    def set_metadata(self, frame_id: int, timestamp: float, stream_id: str = ""):
        """
        设置帧元数据
        
        Args:
            frame_id: 帧ID
            timestamp: 时间戳
            stream_id: 流ID
        """
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.stream_id = stream_id
    
    def add_analysis_result(self, analyzer_name: str, result: Any):
        """
        添加分析结果
        
        Args:
            analyzer_name: 分析器名称
            result: 分析结果
        """
        self.analysis_results[analyzer_name] = result
    
    def get_analysis_result(self, analyzer_name: str) -> Optional[Any]:
        """
        获取分析结果
        
        Args:
            analyzer_name: 分析器名称
            
        Returns:
            Optional[Any]: 分析结果，不存在返回None
        """
        return self.analysis_results.get(analyzer_name)
    
    def clear_analysis_results(self):
        """清空分析结果"""
        self.analysis_results.clear()
    
    @property
    def ref_count(self) -> int:
        """获取当前引用计数"""
        with self._ref_lock:
            return self._ref_count
    
    @property
    def is_valid(self) -> bool:
        """检查缓冲区是否有效"""
        return self._is_valid
    
    @property
    def size_mb(self) -> float:
        """获取缓冲区大小（MB）"""
        return self.frame_data.nbytes / (1024 * 1024)
    
    def reset(self):
        """重置缓冲区状态（内存池回收时调用）"""
        self.frame_id = -1
        self.timestamp = -1.0
        self.stream_id = ""
        self.analysis_results.clear()
        self._ref_count = 0
        self._is_valid = True
    
    def __str__(self) -> str:
        return (f"FrameBuffer(id={self.frame_id}, "
                f"timestamp={self.timestamp:.3f}, "
                f"refs={self._ref_count}, "
                f"valid={self._is_valid})")
    
    def __repr__(self) -> str:
        return self.__str__()


class FrameBufferStats:
    """帧缓冲区统计信息"""
    
    def __init__(self):
        self.total_created = 0
        self.total_recycled = 0
        self.current_active = 0
        self.peak_active = 0
        self.total_memory_mb = 0.0
        self.lock = threading.Lock()
    
    def on_buffer_created(self, buffer: FrameBuffer):
        """缓冲区创建时调用"""
        with self.lock:
            self.total_created += 1
            self.current_active += 1
            self.peak_active = max(self.peak_active, self.current_active)
            self.total_memory_mb += buffer.size_mb
    
    def on_buffer_recycled(self, buffer: FrameBuffer):
        """缓冲区回收时调用"""
        with self.lock:
            self.total_recycled += 1
            self.current_active -= 1
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self.lock:
            return {
                "total_created": self.total_created,
                "total_recycled": self.total_recycled,
                "current_active": self.current_active,
                "peak_active": self.peak_active,
                "total_memory_mb": round(self.total_memory_mb, 2),
                "recycle_rate": (self.total_recycled / max(1, self.total_created)) * 100
            }


# 全局统计实例
frame_buffer_stats = FrameBufferStats()
