"""
图像分割子模块
提供基于YOLO和YOLOE的图像分割功能
"""

from .yolo_segmentor import YOLOSegmentor

# 导入YOLOE分割器(如果已实现)
try:
    from .yoloe_segmentor import YOLOESegmentor
    __all__ = ["YOLOSegmentor", "YOLOESegmentor"]
except ImportError:
    __all__ = ["YOLOSegmentor"]