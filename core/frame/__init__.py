"""
帧处理模块
提供零拷贝帧处理的核心功能
"""

from .frame_metadata import (
    FrameMetadata,
    FrameAnalysisStatus
)
from .frame_index import FrameIndex
from .frame_reference import (
    FrameReference,
    FrameReferenceManager
)

__all__ = [
    # 帧元数据
    "FrameMetadata",
    "FrameAnalysisStatus",
    
    # 帧索引
    "FrameIndex",
    
    # 帧引用
    "FrameReference",
    "FrameReferenceManager",
]
