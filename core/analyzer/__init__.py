"""
分析器模块
包含目标检测、实例分割、目标跟踪、跨摄像头跟踪和越界检测功能
"""
# 导入基础分析器类
from .base_analyzer import (
    BaseAnalyzer,
    DetectionAnalyzer,
    TrackingAnalyzer,
    SegmentationAnalyzer,
    CrossCameraTrackingAnalyzer,
    LineCrossingAnalyzer
)

# 导入分析器工厂和模型加载器
from .analyzer_factory import AnalyzerFactory
from .model_loader import ModelLoader

# 导入具体分析器实现
from .detection import YOLODetector
from .segmentation import YOLOSegmentor

# 为了向后兼容，保留旧的create_analyzer函数
def create_analyzer(analysis_type: str, **kwargs):
    """
    工厂函数，根据分析类型创建相应的分析器实例

    注意: 此函数保留用于向后兼容，新代码应使用AnalyzerFactory

    Args:
        analysis_type: 分析类型，如 "detection", "segmentation", "tracking"
        **kwargs: 传递给分析器的参数

    Returns:
        BaseAnalyzer: 分析器实例
    """
    # 将字符串分析类型转换为整数类型
    analysis_type_id = AnalyzerFactory.get_analysis_type_id(analysis_type)

    # 使用AnalyzerFactory创建分析器
    return AnalyzerFactory.create_analyzer(
        analysis_type=analysis_type_id,
        model_code=kwargs.get("model_code"),
        engine_type=kwargs.get("engine_type", 0),
        yolo_version=kwargs.get("yolo_version", 0),
        **kwargs
    )

__all__ = [
    # 基础类
    "BaseAnalyzer",
    "DetectionAnalyzer",
    "TrackingAnalyzer",
    "SegmentationAnalyzer",
    "CrossCameraTrackingAnalyzer",
    "LineCrossingAnalyzer",

    # 工厂和加载器
    "AnalyzerFactory",
    "ModelLoader",

    # 具体实现
    "YOLODetector",
    "YOLOSegmentor",

    # 向后兼容函数
    "create_analyzer"
]