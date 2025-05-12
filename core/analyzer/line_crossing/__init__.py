"""
越界检测子模块
提供目标越界检测功能
"""

# 导入越界检测器(如果已实现)
try:
    from .line_crossing_detector import LineCrossingDetector
    __all__ = ["LineCrossingDetector"]
except ImportError:
    __all__ = []
