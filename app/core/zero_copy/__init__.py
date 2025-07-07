#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: __init__.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 零拷贝核心包

基于timelinetool架构实现的零拷贝AI分析引擎。

本文件是分析服务项目的一部分。
"""

from .frame_buffer import FrameBuffer, FrameBufferStats, frame_buffer_stats
from .memory_pool import MemoryPool, MemoryPoolManager, memory_pool_manager
from .time_axis import TimeAxis, MultiStreamTimeAxis
from .stream_capture import StreamCapture, MultiStreamCapture
from .video_file_processor import VideoFileProcessor
from .image_processor import ImageProcessor
from .ai_analyzer import BaseAnalyzer, MockAnalyzer, AnalysisWorker, AnalysisEngine
from .result_processor import ResultProcessor, BatchResultProcessor
from .video_player import VideoPlayer, MultiWindowPlayer, multi_window_player

__all__ = [
    # 帧缓冲区
    'FrameBuffer',
    'FrameBufferStats',
    'frame_buffer_stats',

    # 内存池
    'MemoryPool',
    'MemoryPoolManager',
    'memory_pool_manager',

    # 时间轴
    'TimeAxis',
    'MultiStreamTimeAxis',

    # 流捕获
    'StreamCapture',
    'MultiStreamCapture',

    # 专用处理器
    'VideoFileProcessor',
    'ImageProcessor',

    # AI分析器
    'BaseAnalyzer',
    'MockAnalyzer',
    'AnalysisWorker',
    'AnalysisEngine',

    # 结果处理
    'ResultProcessor',
    'BatchResultProcessor',

    # 视频播放器
    'VideoPlayer',
    'MultiWindowPlayer',
    'multi_window_player'
]
