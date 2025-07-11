#!/usr/bin/env python3
"""
结果处理模块
包含检测过滤器和三个结果处理器
"""

from .detection_filter import DetectionFilter, FilterPipeline
from .callback_processor import CallbackProcessor  
from .storage_processor import StorageProcessor
# from .video_processor import VideoProcessor  # 已删除，由VideoPlaybackService替代
from .result_pipeline import ResultProcessingPipeline

__all__ = [
    "DetectionFilter",
    "FilterPipeline", 
    "CallbackProcessor",
    "StorageProcessor",
    # "VideoProcessor",  # 已删除，由VideoPlaybackService替代
    "ResultProcessingPipeline"
] 