#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
时间轴管理器
管理多个时间轴的创建、配置和生命周期
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.core.zero_copy.time_axis import TimeAxis


class TimeAxisManager:
    """时间轴管理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化时间轴管理器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.time_axes: Dict[str, TimeAxis] = {}
        
    def create_time_axis(self, 
                        axis_id: str,
                        max_frames: int = 1000,
                        timeout_seconds: float = 30.0,
                        **kwargs) -> TimeAxis:
        """
        创建时间轴
        
        Args:
            axis_id: 时间轴ID
            max_frames: 最大帧数
            timeout_seconds: 超时时间（秒）
            **kwargs: 其他参数
            
        Returns:
            TimeAxis: 时间轴实例
        """
        if axis_id in self.time_axes:
            self.logger.warning(f"时间轴已存在: {axis_id}")
            return self.time_axes[axis_id]
        
        time_axis = TimeAxis(
            max_frames=max_frames,
            timeout_seconds=timeout_seconds,
            logger=self.logger
        )
        
        self.time_axes[axis_id] = time_axis
        self.logger.info(f"✅ 时间轴已创建: {axis_id}")
        
        return time_axis
    
    def get_time_axis(self, axis_id: str) -> Optional[TimeAxis]:
        """
        获取时间轴
        
        Args:
            axis_id: 时间轴ID
            
        Returns:
            Optional[TimeAxis]: 时间轴实例
        """
        return self.time_axes.get(axis_id)
    
    def remove_time_axis(self, axis_id: str) -> bool:
        """
        移除时间轴
        
        Args:
            axis_id: 时间轴ID
            
        Returns:
            bool: 是否移除成功
        """
        if axis_id in self.time_axes:
            time_axis = self.time_axes[axis_id]
            time_axis.cleanup()
            del self.time_axes[axis_id]
            self.logger.info(f"🗑️ 时间轴已移除: {axis_id}")
            return True
        return False
    
    def cleanup_all(self):
        """清理所有时间轴"""
        for axis_id in list(self.time_axes.keys()):
            self.remove_time_axis(axis_id)
        
        self.logger.info("🧹 所有时间轴已清理")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total_axes": len(self.time_axes),
            "axes": {}
        }
        
        for axis_id, time_axis in self.time_axes.items():
            if hasattr(time_axis, 'get_stats'):
                stats["axes"][axis_id] = time_axis.get_stats()
            else:
                stats["axes"][axis_id] = {
                    "size": len(time_axis),
                    "status": "active" if hasattr(time_axis, 'running') and time_axis.running else "inactive"
                }
        
        return stats
    
    def get_frames_for_time_range(
        self,
        start_timestamp: float,
        end_timestamp: float,
        axis_id: Optional[str] = None
    ) -> List:
        """
        获取指定时间范围内的帧序列
        
        Args:
            start_timestamp: 开始时间戳
            end_timestamp: 结束时间戳
            axis_id: 时间轴ID，None则从第一个可用的时间轴获取
            
        Returns:
            List: 指定时间范围内的帧列表
        """
        # 如果没有指定时间轴ID，使用第一个可用的时间轴
        if axis_id is None:
            if not self.time_axes:
                self.logger.warning("⚠️ 时间轴管理器: 没有可用的时间轴")
                return []
            axis_id = next(iter(self.time_axes))
        
        time_axis = self.get_time_axis(axis_id)
        if not time_axis:
            self.logger.warning(f"⚠️ 时间轴管理器: 时间轴不存在 {axis_id}")
            return []
        
        return time_axis.get_frames_for_time_range(start_timestamp, end_timestamp) 