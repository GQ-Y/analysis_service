"""
分析器工厂模块
负责创建不同类型的分析器实例
"""
from typing import Dict, Any, Optional, Union, Type

from shared.utils.logger import setup_logger
from core.analyzer.base_analyzer import (
    BaseAnalyzer,
    DetectionAnalyzer,
    TrackingAnalyzer,
    SegmentationAnalyzer,
    CrossCameraTrackingAnalyzer,
    LineCrossingAnalyzer
)
from core.analyzer.detection import YOLODetectionAnalyzer

logger = setup_logger(__name__)

class AnalyzerFactory:
    """分析器工厂类"""

    # 分析类型映射
    ANALYSIS_TYPE_MAP = {
        "detection": 1,
        "tracking": 2,
        "segmentation": 3,
        "cross_camera_tracking": 4,
        "line_crossing": 5
    }

    # 分析类型名称映射
    ANALYSIS_TYPE_NAME_MAP = {
        1: "detection",
        2: "tracking",
        3: "segmentation",
        4: "cross_camera_tracking",
        5: "line_crossing"
    }

    # 分析器类映射
    ANALYZER_CLASS_MAP = {
        1: YOLODetectionAnalyzer,  # 使用具体的YOLO检测分析器实现
        2: TrackingAnalyzer,
        3: SegmentationAnalyzer,
        4: CrossCameraTrackingAnalyzer,
        5: LineCrossingAnalyzer
    }

    @classmethod
    def create_analyzer(cls, analysis_type: Union[int, str], model_code: Optional[str] = None,
                       engine_type: int = 0, yolo_version: int = 0, **kwargs) -> BaseAnalyzer:
        """
        创建分析器实例

        Args:
            analysis_type: 分析类型 (1=检测, 2=跟踪, 3=分割, 4=跨摄像头跟踪, 5=越界检测)
                          或者对应的字符串名称
            model_code: 模型代码
            engine_type: 推理引擎类型 (0=PyTorch, 1=ONNX, 2=TensorRT, 3=OpenVINO, 4=Pytron)
            yolo_version: YOLO版本 (0=v8n, 1=v8s, 2=v8l, 3=v8x, 4=11s, 5=11m, 6=11l)
            **kwargs: 其他参数

        Returns:
            BaseAnalyzer: 分析器实例

        Raises:
            ValueError: 当分析类型不支持时
        """
        # 如果分析类型是字符串，转换为整数
        if isinstance(analysis_type, str):
            analysis_type = cls.ANALYSIS_TYPE_MAP.get(analysis_type.lower())
            if analysis_type is None:
                raise ValueError(f"不支持的分析类型名称: {analysis_type}")

        # 获取分析器类
        analyzer_class = cls.ANALYZER_CLASS_MAP.get(analysis_type)
        if analyzer_class is None:
            raise ValueError(f"不支持的分析类型: {analysis_type}")

        # 检查是否需要特殊处理YOLOE模型
        if model_code and "yoloe" in model_code.lower():
            return cls._create_yoloe_analyzer(analysis_type, model_code, engine_type, yolo_version, **kwargs)

        # 创建分析器实例
        logger.info(f"创建分析器: 类型={cls.ANALYSIS_TYPE_NAME_MAP.get(analysis_type, analysis_type)}, "
                   f"模型={model_code}, 引擎={engine_type}, YOLO版本={yolo_version}")

        return analyzer_class(model_code, engine_type, yolo_version, **kwargs)

    @classmethod
    def _create_yoloe_analyzer(cls, analysis_type: int, model_code: str,
                              engine_type: int, yolo_version: int, **kwargs) -> BaseAnalyzer:
        """
        创建YOLOE分析器实例

        Args:
            analysis_type: 分析类型
            model_code: 模型代码
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            **kwargs: 其他参数

        Returns:
            BaseAnalyzer: YOLOE分析器实例
        """
        # 导入YOLOE分析器
        try:
            from core.analyzer.yoloe.yoloe_analyzer import (
                YOLOEDetectionAnalyzer,
                YOLOESegmentationAnalyzer,
                YOLOETrackingAnalyzer
            )

            # 根据分析类型创建YOLOE分析器
            if analysis_type == 1:  # 检测
                return YOLOEDetectionAnalyzer(model_code, engine_type, yolo_version, **kwargs)
            elif analysis_type == 2:  # 跟踪
                return YOLOETrackingAnalyzer(model_code, engine_type, yolo_version, **kwargs)
            elif analysis_type == 3:  # 分割
                return YOLOESegmentationAnalyzer(model_code, engine_type, yolo_version, **kwargs)
            else:
                # 对于其他分析类型，使用标准分析器
                logger.warning(f"YOLOE模型不支持分析类型 {analysis_type}，使用标准分析器")
                analyzer_class = cls.ANALYZER_CLASS_MAP.get(analysis_type)
                return analyzer_class(model_code, engine_type, yolo_version, **kwargs)

        except ImportError:
            logger.warning("YOLOE分析器模块未找到，使用标准分析器")
            analyzer_class = cls.ANALYZER_CLASS_MAP.get(analysis_type)
            return analyzer_class(model_code, engine_type, yolo_version, **kwargs)

    @classmethod
    def get_analysis_type_name(cls, analysis_type: int) -> str:
        """
        获取分析类型名称

        Args:
            analysis_type: 分析类型ID

        Returns:
            str: 分析类型名称
        """
        return cls.ANALYSIS_TYPE_NAME_MAP.get(analysis_type, f"未知类型({analysis_type})")

    @classmethod
    def get_analysis_type_id(cls, analysis_type_name: str) -> int:
        """
        获取分析类型ID

        Args:
            analysis_type_name: 分析类型名称

        Returns:
            int: 分析类型ID

        Raises:
            ValueError: 当分析类型名称不支持时
        """
        analysis_type = cls.ANALYSIS_TYPE_MAP.get(analysis_type_name.lower())
        if analysis_type is None:
            raise ValueError(f"不支持的分析类型名称: {analysis_type_name}")
        return analysis_type
