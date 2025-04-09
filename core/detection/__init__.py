"""
检测模块
提供基于YOLO的对象检测功能和其他对象检测功能
"""

from .yolo_detector import YOLODetector
# 将来导入其他检测器
# from core.detection.object_detector import ObjectDetector

__all__ = ["YOLODetector"]
# 将来扩展__all__列表
# __all__ = ["YOLODetector", "ObjectDetector"]

"""
包含图像对象检测相关功能
"""

# 将来导入对象检测器
# from core.detection.object_detector import ObjectDetector

# __all__ = ['ObjectDetector'] 