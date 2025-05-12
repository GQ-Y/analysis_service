"""
目标跟踪模块
提供目标跟踪功能
"""
from .tracker import Tracker

# 导入YOLO跟踪器
try:
    from .yolo_tracker import YOLOTracker
    __all__ = ["Tracker", "YOLOTracker"]
except ImportError:
    __all__ = ["Tracker"]

# 导入YOLOE跟踪器
try:
    from .yoloe_tracker import YOLOETracker
    if "YOLOETracker" not in __all__:
        __all__.append("YOLOETracker")
except ImportError:
    pass
