"""
跨摄像头跟踪子模块
提供跨摄像头目标跟踪功能
"""

# 导入跨摄像头跟踪器(如果已实现)
try:
    from .cross_camera_tracker import CrossCameraTracker
    from .feature_extractor import FeatureExtractor
    __all__ = ["CrossCameraTracker", "FeatureExtractor"]
except ImportError:
    __all__ = []
