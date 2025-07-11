#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: __init__.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 内存管理包

包含内存管理相关的组件，参考timelinetool的内存管理设计。

本文件是分析服务项目的一部分。
"""

from .memory_manager import MemoryManager, get_memory_manager
from .frame_buffer import FrameBuffer
from .reference_counter import ReferenceCounter
from .smart_memory_pool import SmartMemoryPool, PoolConfig, get_smart_memory_pool, cleanup_smart_memory_pool
from .enhanced_memory_manager import EnhancedMemoryManager, get_enhanced_memory_manager, cleanup_enhanced_memory_manager
from .performance_monitor import PerformanceMonitor, get_performance_monitor, cleanup_performance_monitor

__all__ = [
    'MemoryManager',
    'FrameBuffer',
    'ReferenceCounter',
    'SmartMemoryPool',
    'PoolConfig',
    'EnhancedMemoryManager',
    'PerformanceMonitor',
    'get_memory_manager',
    'get_smart_memory_pool',
    'cleanup_smart_memory_pool',
    'get_enhanced_memory_manager',
    'cleanup_enhanced_memory_manager',
    'get_performance_monitor',
    'cleanup_performance_monitor'
]
