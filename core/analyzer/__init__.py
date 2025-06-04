"""
分析器模块 - 重构版
目前仅包含目标检测功能，其他功能将在后续版本中添加
"""
# 导入基础分析器类
from .base_analyzer import (
    BaseAnalyzer,
    DetectionAnalyzer
    # 以下类型已移除，将在后续版本中添加
    # TrackingAnalyzer,
    # SegmentationAnalyzer,
    # CrossCameraTrackingAnalyzer,
    # LineCrossingAnalyzer
)

# 导入分析器工厂、注册表和模型加载器
from .analyzer_factory import AnalyzerFactory, analyzer_factory
from .registry import AnalyzerRegistry, register_analyzer
from .model_loader import ModelLoader

# 导入具体分析器实现
from .detection import YOLODetector

# 为了向后兼容，保留旧的create_analyzer函数
def create_analyzer(analysis_type: str, **kwargs):
    """
    工厂函数，根据分析类型创建相应的分析器实例

    注意: 此函数保留用于向后兼容，新代码应使用AnalyzerFactory

    Args:
        analysis_type: 分析类型，目前仅支持 "detection"
        **kwargs: 传递给分析器的参数

    Returns:
        BaseAnalyzer: 分析器实例
    """
    # 直接使用AnalyzerFactory创建分析器
    # 获取分析器名称，如果没有指定则使用默认名称
    analyzer_name = kwargs.get("analyzer_name", "default")
    
    return analyzer_factory.create_analyzer(
        analysis_type,  # 第一个参数：analyzer_type
        analyzer_name,  # 第二个参数：name  
        kwargs          # 第三个参数：config
    )

__all__ = [
    # 基础类
    "BaseAnalyzer",
    "DetectionAnalyzer",
    
    # 工厂、注册表和加载器
    "AnalyzerFactory",
    "AnalyzerRegistry",
    "register_analyzer",
    "ModelLoader",

    # 具体实现
    "YOLODetector",

    # 向后兼容函数
    "create_analyzer"
]