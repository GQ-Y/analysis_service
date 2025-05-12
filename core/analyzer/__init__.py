"""
分析器模块
包含目标检测、实例分割和目标跟踪功能
"""
from .base import BaseAnalyzer

# 导入分析器实现
from .detection import YOLODetector
from .segmentation import YOLOSegmentor

__all__ = [
    "BaseAnalyzer",
    "YOLODetector",
    "YOLOSegmentor",
    "create_analyzer"
]

def create_analyzer(analysis_type: str, **kwargs):
    """
    工厂函数，根据分析类型创建相应的分析器实例

    Args:
        analysis_type: 分析类型，如 "detection", "segmentation", "tracking"
        **kwargs: 传递给分析器的参数

    Returns:
        BaseAnalyzer: 分析器实例
    """
    if analysis_type == "detection":
        return YOLODetector(**kwargs)
    elif analysis_type == "segmentation":
        return YOLOSegmentor(**kwargs)
    elif analysis_type == "tracking":
        # 暂时使用检测器作为跟踪器的基础
        return YOLODetector(**kwargs)
    else:
        raise ValueError(f"不支持的分析类型: {analysis_type}")