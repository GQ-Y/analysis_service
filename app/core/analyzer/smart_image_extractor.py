#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: smart_image_extractor.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-11
描述: 智能图像数据提取器

优化图像数据提取策略，减少内存拷贝和提高性能：
1. 智能选择零拷贝或拷贝策略
2. 内存压力感知的提取模式
3. 自适应的数据处理方式
4. 内存使用统计和优化

本文件是分析服务项目的一部分。
"""

import logging
import time
import threading
from typing import Union, Optional, Dict, Any, List
from enum import Enum
import numpy as np

from ..memory.enhanced_memory_manager import get_enhanced_memory_manager, MemoryPressureLevel


class ExtractionStrategy(Enum):
    """图像提取策略"""
    ZERO_COPY = "zero_copy"      # 零拷贝（使用视图）
    COPY = "copy"               # 拷贝（创建副本）
    ADAPTIVE = "adaptive"       # 自适应（根据内存压力选择）


class SmartImageExtractor:
    """
    智能图像数据提取器
    
    根据内存压力和数据特性智能选择最优的图像提取策略，
    专为小内存设备优化，平衡性能和内存使用。
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化智能图像提取器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # 获取增强内存管理器
        self.memory_manager = get_enhanced_memory_manager()
        
        # 默认策略
        self.default_strategy = ExtractionStrategy.ADAPTIVE
        
        # 统计信息
        self.stats = {
            "total_extractions": 0,
            "zero_copy_count": 0,
            "copy_count": 0,
            "adaptive_count": 0,
            "extraction_failures": 0,
            "total_processing_time": 0.0,
            "memory_pressure_switches": 0
        }
        
        # 线程安全
        self.stats_lock = threading.Lock()
        
        # 性能阈值
        self.large_image_threshold = 2 * 1024 * 1024  # 2MB
        self.copy_time_threshold = 0.01  # 10ms
        
        # 策略缓存（优化性能）
        self._strategy_cache = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = 100
        self._cache_hits = 0
        self._cache_misses = 0
        
        self.logger.info("🧠 智能图像提取器初始化完成")
    
    def extract_image_data(self, frame_data: Union[np.ndarray, Any], 
                          strategy: Optional[ExtractionStrategy] = None,
                          analyzer_context: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
        """
        智能提取图像数据
        
        Args:
            frame_data: 帧数据（可能是numpy数组或帧缓冲区）
            strategy: 提取策略（可选）
            analyzer_context: 分析器上下文信息（可选）
            
        Returns:
            Optional[np.ndarray]: 提取的图像数组
        """
        start_time = time.time()
        
        try:
            with self.stats_lock:
                self.stats["total_extractions"] += 1
            
            # 如果已经是numpy数组，直接返回
            if isinstance(frame_data, np.ndarray):
                self.logger.debug("📥 直接返回numpy数组")
                return frame_data
            
            # 选择提取策略
            chosen_strategy = strategy or self.default_strategy
            if chosen_strategy == ExtractionStrategy.ADAPTIVE:
                chosen_strategy = self._choose_adaptive_strategy(frame_data, analyzer_context)
            
            # 执行提取
            result = self._extract_with_strategy(frame_data, chosen_strategy)
            
            # 更新统计
            processing_time = time.time() - start_time
            with self.stats_lock:
                self.stats["total_processing_time"] += processing_time
                
                if chosen_strategy == ExtractionStrategy.ZERO_COPY:
                    self.stats["zero_copy_count"] += 1
                elif chosen_strategy == ExtractionStrategy.COPY:
                    self.stats["copy_count"] += 1
                else:
                    self.stats["adaptive_count"] += 1
            
            self.logger.debug(f"📥 提取完成: {chosen_strategy.value}, 耗时: {processing_time:.3f}s")
            return result
            
        except Exception as e:
            with self.stats_lock:
                self.stats["extraction_failures"] += 1
            
            self.logger.error(f"❌ 图像提取失败: {e}")
            return None
    
    def _choose_adaptive_strategy(self, frame_data: Any, 
                                 analyzer_context: Optional[Dict[str, Any]] = None) -> ExtractionStrategy:
        """
        选择自适应策略 - 优化版本
        
        使用策略决策表减少条件判断，提高性能
        
        Args:
            frame_data: 帧数据
            analyzer_context: 分析器上下文
            
        Returns:
            ExtractionStrategy: 选择的策略
        """
        # 快速路径：检查基本条件
        supports_zero_copy = self._supports_zero_copy(frame_data)
        is_large = self._is_large_image(frame_data)
        is_safe = self._is_safe_for_copy(frame_data, analyzer_context)
        
        # 获取当前内存统计
        current_stats = self.memory_manager.current_stats
        if not current_stats:
            # 无统计信息时的默认策略
            return ExtractionStrategy.ZERO_COPY if (supports_zero_copy and is_large) else ExtractionStrategy.COPY
        
        pressure_level = current_stats.pressure_level
        
        # 使用策略决策表（优化性能）
        # 决策优先级：内存压力 > 图像大小 > 安全性
        if pressure_level == MemoryPressureLevel.CRITICAL:
            # 严重压力：始终尝试零拷贝
            if supports_zero_copy:
                self.logger.debug("🆘 严重内存压力，使用零拷贝")
                return ExtractionStrategy.ZERO_COPY
            else:
                self.logger.warning("🆘 严重内存压力，但不支持零拷贝")
                return ExtractionStrategy.COPY
        
        elif pressure_level == MemoryPressureLevel.HIGH:
            # 高压力：大图像或不安全时使用零拷贝
            if supports_zero_copy and (is_large or not is_safe):
                self.logger.debug("⚠️ 高内存压力，使用零拷贝")
                return ExtractionStrategy.ZERO_COPY
            else:
                return ExtractionStrategy.COPY
        
        elif pressure_level == MemoryPressureLevel.MEDIUM:
            # 中等压力：平衡性能和安全性
            if is_large and supports_zero_copy:
                return ExtractionStrategy.ZERO_COPY
            elif is_safe:
                return ExtractionStrategy.COPY
            elif supports_zero_copy:
                return ExtractionStrategy.ZERO_COPY
            else:
                return ExtractionStrategy.COPY
        
        else:  # LOW pressure
            # 低压力：优先安全性
            if is_safe:
                return ExtractionStrategy.COPY
            elif supports_zero_copy:
                return ExtractionStrategy.ZERO_COPY
            else:
                return ExtractionStrategy.COPY
    
    def _extract_with_strategy(self, frame_data: Any, strategy: ExtractionStrategy) -> Optional[np.ndarray]:
        """
        使用指定策略提取图像数据
        
        Args:
            frame_data: 帧数据
            strategy: 提取策略
            
        Returns:
            Optional[np.ndarray]: 提取的图像数组
        """
        try:
            if strategy == ExtractionStrategy.ZERO_COPY:
                return self._extract_zero_copy(frame_data)
            else:
                return self._extract_copy(frame_data)
        except Exception as e:
            self.logger.error(f"❌ 策略 {strategy.value} 提取失败: {e}")
            
            # 回退策略
            if strategy == ExtractionStrategy.ZERO_COPY:
                self.logger.warning("🔄 回退到拷贝策略")
                return self._extract_copy(frame_data)
            else:
                self.logger.warning("🔄 回退到零拷贝策略")
                return self._extract_zero_copy(frame_data)
    
    def _extract_zero_copy(self, frame_data: Any) -> Optional[np.ndarray]:
        """
        零拷贝提取
        
        Args:
            frame_data: 帧数据
            
        Returns:
            Optional[np.ndarray]: 图像数组视图
        """
        # 优先级1: 零拷贝视图
        if hasattr(frame_data, 'get_frame_view'):
            return frame_data.get_frame_view()
        
        # 优先级2: 直接访问属性
        if hasattr(frame_data, 'frame_data'):
            return frame_data.frame_data
        
        # 不支持零拷贝
        raise ValueError("不支持零拷贝提取")
    
    def _extract_copy(self, frame_data: Any) -> Optional[np.ndarray]:
        """
        拷贝提取
        
        Args:
            frame_data: 帧数据
            
        Returns:
            Optional[np.ndarray]: 图像数组副本
        """
        # 优先级1: 显式拷贝方法
        if hasattr(frame_data, 'get_frame_copy'):
            return frame_data.get_frame_copy()
        
        # 优先级2: 零拷贝后拷贝
        if hasattr(frame_data, 'get_frame_view'):
            view = frame_data.get_frame_view()
            return view.copy()
        
        # 优先级3: 直接访问后拷贝
        if hasattr(frame_data, 'frame_data'):
            return frame_data.frame_data.copy()
        
        # 不支持任何提取方式
        raise ValueError("不支持任何提取方式")
    
    def _supports_zero_copy(self, frame_data: Any) -> bool:
        """
        检查是否支持零拷贝
        
        Args:
            frame_data: 帧数据
            
        Returns:
            bool: 是否支持零拷贝
        """
        return (hasattr(frame_data, 'get_frame_view') or 
                hasattr(frame_data, 'frame_data'))
    
    def _is_large_image(self, frame_data: Any) -> bool:
        """
        检查是否是大图像
        
        Args:
            frame_data: 帧数据
            
        Returns:
            bool: 是否是大图像
        """
        try:
            # 尝试获取图像尺寸
            if hasattr(frame_data, 'get_frame_size'):
                size = frame_data.get_frame_size()
                return size > self.large_image_threshold
            
            # 尝试从元数据获取尺寸
            if hasattr(frame_data, 'width') and hasattr(frame_data, 'height'):
                channels = getattr(frame_data, 'channels', 3)
                size = frame_data.width * frame_data.height * channels * 4  # 假设float32
                return size > self.large_image_threshold
            
            # 尝试从形状获取尺寸
            if hasattr(frame_data, 'shape'):
                shape = frame_data.shape
                size = np.prod(shape) * 4  # 假设float32
                return size > self.large_image_threshold
            
            # 默认认为是大图像
            return True
            
        except Exception:
            # 获取尺寸失败，默认认为是大图像
            return True
    
    def _is_safe_for_copy(self, frame_data: Any, analyzer_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        检查是否可以安全拷贝
        
        Args:
            frame_data: 帧数据
            analyzer_context: 分析器上下文
            
        Returns:
            bool: 是否可以安全拷贝
        """
        # 如果分析器声明会修改数据，则不能使用零拷贝
        if analyzer_context and analyzer_context.get('modifies_input', False):
            return True
        
        # 如果是并发分析，拷贝更安全
        if analyzer_context and analyzer_context.get('concurrent_analysis', False):
            return True
        
        # 如果图像较小，拷贝开销不大
        if not self._is_large_image(frame_data):
            return True
        
        # 默认认为可以拷贝
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.stats_lock:
            stats = self.stats.copy()
        
        # 计算衍生统计
        total_extractions = stats["total_extractions"]
        if total_extractions > 0:
            stats["zero_copy_rate"] = stats["zero_copy_count"] / total_extractions * 100
            stats["copy_rate"] = stats["copy_count"] / total_extractions * 100
            stats["adaptive_rate"] = stats["adaptive_count"] / total_extractions * 100
            stats["failure_rate"] = stats["extraction_failures"] / total_extractions * 100
            stats["avg_processing_time"] = stats["total_processing_time"] / total_extractions
        else:
            stats["zero_copy_rate"] = 0
            stats["copy_rate"] = 0
            stats["adaptive_rate"] = 0
            stats["failure_rate"] = 0
            stats["avg_processing_time"] = 0
        
        return stats
    
    def reset_stats(self):
        """
        重置统计信息
        """
        with self.stats_lock:
            self.stats = {
                "total_extractions": 0,
                "zero_copy_count": 0,
                "copy_count": 0,
                "adaptive_count": 0,
                "extraction_failures": 0,
                "total_processing_time": 0.0,
                "memory_pressure_switches": 0
            }
        
        self.logger.info("📊 统计信息已重置")
    
    def set_default_strategy(self, strategy: ExtractionStrategy):
        """
        设置默认策略
        
        Args:
            strategy: 默认策略
        """
        self.default_strategy = strategy
        self.logger.info(f"🔧 默认策略设置为: {strategy.value}")
    
    def set_large_image_threshold(self, threshold_mb: float):
        """
        设置大图像阈值
        
        Args:
            threshold_mb: 阈值（MB）
        """
        self.large_image_threshold = threshold_mb * 1024 * 1024
        self.logger.info(f"🔧 大图像阈值设置为: {threshold_mb}MB")


# 全局智能图像提取器实例
_smart_image_extractor: Optional[SmartImageExtractor] = None
_extractor_lock = threading.Lock()


def get_smart_image_extractor() -> SmartImageExtractor:
    """
    获取全局智能图像提取器实例
    
    Returns:
        SmartImageExtractor: 智能图像提取器实例
    """
    global _smart_image_extractor
    
    with _extractor_lock:
        if _smart_image_extractor is None:
            _smart_image_extractor = SmartImageExtractor()
        return _smart_image_extractor


def cleanup_smart_image_extractor():
    """
    清理全局智能图像提取器
    """
    global _smart_image_extractor
    
    with _extractor_lock:
        if _smart_image_extractor:
            _smart_image_extractor.reset_stats()
            _smart_image_extractor = None