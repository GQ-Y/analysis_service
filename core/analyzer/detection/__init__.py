"""
目标检测子模块 - 重构版
提供基于YOLO的对象检测功能
"""

from .yolo_detector import YOLODetector
from .yolo_detection_analyzer import YOLODetectionAnalyzer

# 导入YOLOE检测器(如果已实现)
try:
    from .yoloe_detector import YOLOEDetector
    __all__ = ["YOLODetector", "YOLOEDetector", "YOLODetectionAnalyzer"]
except ImportError:
    __all__ = ["YOLODetector", "YOLODetectionAnalyzer"]