"""
目标检测子模块
提供基于YOLO和YOLOE的对象检测功能
"""

from .yolo_detector import YOLODetector

# 导入YOLOE检测器(如果已实现)
try:
    from .yoloe_detector import YOLOEDetector
    __all__ = ["YOLODetector", "YOLOEDetector"]
except ImportError:
    __all__ = ["YOLODetector"]