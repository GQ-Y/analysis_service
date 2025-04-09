"""
分析类型定义
"""
from enum import Enum

class AnalysisType(str, Enum):
    """分析类型"""
    DETECTION = "detection"  # 目标检测
    SEGMENTATION = "segmentation"  # 实例分割
    TRACKING = "tracking"  # 目标跟踪
    CROSS_CAMERA = "cross_camera"  # 跨摄像头跟踪 