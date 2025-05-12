"""
基础分析器模块
定义所有分析器的基类和通用接口
"""
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from datetime import datetime

from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseAnalyzer(ABC):
    """基础分析器抽象类"""

    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0,
                 yolo_version: int = 0, device: str = "auto"):
        """
        初始化基础分析器

        Args:
            model_code: 模型代码，如果提供则立即加载模型
            engine_type: 推理引擎类型 (0=PyTorch, 1=ONNX, 2=TensorRT, 3=OpenVINO, 4=Pytron)
            yolo_version: YOLO版本 (0=v8n, 1=v8s, 2=v8l, 3=v8x, 4=11s, 5=11m, 6=11l)
            device: 推理设备 ("cpu", "cuda", "auto")
        """
        self.model = None
        self.current_model_code = None
        self.engine_type = engine_type
        self.yolo_version = yolo_version
        self.device = device

        # 记录初始化信息
        logger.info(f"初始化分析器: 引擎类型={self._get_engine_name()}, YOLO版本={self._get_yolo_version_name()}, 设备={device}")

        # 如果提供了model_code，立即加载模型
        if model_code:
            # 注意：这里不能直接使用await，因为__init__不是异步方法
            # 创建一个异步任务来加载模型
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环已经在运行，创建一个任务
                    asyncio.create_task(self.load_model(model_code))
                else:
                    # 如果事件循环没有运行，直接运行直到完成
                    loop.run_until_complete(self.load_model(model_code))
            except Exception as e:
                logger.error(f"初始化时加载模型失败: {str(e)}")

    def _get_engine_name(self) -> str:
        """获取引擎名称"""
        engine_names = {
            0: "PyTorch",
            1: "ONNX",
            2: "TensorRT",
            3: "OpenVINO",
            4: "Pytron"
        }
        return engine_names.get(self.engine_type, f"未知引擎({self.engine_type})")

    def _get_yolo_version_name(self) -> str:
        """获取YOLO版本名称"""
        version_names = {
            0: "YOLOv8n",
            1: "YOLOv8s",
            2: "YOLOv8l",
            3: "YOLOv8x",
            4: "YOLO11s",
            5: "YOLO11m",
            6: "YOLO11l"
        }
        return version_names.get(self.yolo_version, f"未知版本({self.yolo_version})")

    @abstractmethod
    async def load_model(self, model_code: str) -> bool:
        """
        加载模型

        Args:
            model_code: 模型代码

        Returns:
            bool: 是否成功加载模型
        """
        pass

    @abstractmethod
    async def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        分析图像

        Args:
            image: 输入图像
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 分析结果
        """
        pass

    @abstractmethod
    async def process_video_frame(self, frame: np.ndarray, frame_index: int, **kwargs) -> Dict[str, Any]:
        """
        处理视频帧

        Args:
            frame: 视频帧
            frame_index: 帧索引
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        pass

    @property
    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            Dict[str, Any]: 模型信息
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """释放资源"""
        pass

    def get_analysis_type(self) -> str:
        """
        获取分析类型

        Returns:
            str: 分析类型名称
        """
        return "base"


class DetectionAnalyzer(BaseAnalyzer):
    """检测分析器基类"""

    def get_analysis_type(self) -> str:
        return "detection"


class TrackingAnalyzer(BaseAnalyzer):
    """跟踪分析器基类"""

    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0,
                 yolo_version: int = 0, device: str = "auto",
                 tracker_type: int = 0, **kwargs):
        """
        初始化跟踪分析器

        Args:
            model_code: 模型代码
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            device: 推理设备
            tracker_type: 跟踪器类型 (0=SORT, 1=ByteTrack, 2=DeepSORT)
            **kwargs: 其他参数
        """
        super().__init__(model_code, engine_type, yolo_version, device)
        self.tracker_type = tracker_type
        self.tracker = None

        # 初始化跟踪器
        self._init_tracker(**kwargs)

    def _init_tracker(self, **kwargs):
        """初始化跟踪器"""
        from core.analyzer.tracking import Tracker

        # 获取跟踪器参数
        max_age = kwargs.get("max_age", 30)
        min_hits = kwargs.get("min_hits", 3)
        iou_threshold = kwargs.get("iou_threshold", 0.3)

        # 创建跟踪器
        self.tracker = Tracker(
            tracker_type=self.tracker_type,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )

        # 设置计数线（如果有）
        if "counting_line" in kwargs and kwargs.get("counting_enabled", False):
            self.tracker.set_counting_line(
                kwargs["counting_line"],
                enabled=kwargs.get("counting_enabled", False)
            )

        # 设置速度估计（如果启用）
        if kwargs.get("speed_estimation", False):
            self.tracker.enable_speed_estimation(
                enabled=True,
                pixels_per_meter=kwargs.get("pixels_per_meter", 100),
                fps=kwargs.get("fps", 25)
            )

    def get_analysis_type(self) -> str:
        return "tracking"


class SegmentationAnalyzer(BaseAnalyzer):
    """分割分析器基类"""

    def get_analysis_type(self) -> str:
        return "segmentation"


class CrossCameraTrackingAnalyzer(TrackingAnalyzer):
    """跨摄像头跟踪分析器基类"""

    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0,
                 yolo_version: int = 0, device: str = "auto",
                 tracker_type: int = 0, **kwargs):
        """
        初始化跨摄像头跟踪分析器

        Args:
            model_code: 模型代码
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            device: 推理设备
            tracker_type: 跟踪器类型
            **kwargs: 其他参数
        """
        super().__init__(model_code, engine_type, yolo_version, device, tracker_type, **kwargs)

        # 跨摄像头相关参数
        self.camera_id = kwargs.get("camera_id", "")
        self.related_cameras = kwargs.get("related_cameras", [])
        self.feature_extractor = None

        # 初始化特征提取器
        self._init_feature_extractor(kwargs.get("feature_type", 0))

    def _init_feature_extractor(self, feature_type: int):
        """初始化特征提取器"""
        # TODO: 实现特征提取器初始化
        pass

    def get_analysis_type(self) -> str:
        return "cross_camera_tracking"


class LineCrossingAnalyzer(TrackingAnalyzer):
    """越界检测分析器基类"""

    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0,
                 yolo_version: int = 0, device: str = "auto",
                 tracker_type: int = 0, **kwargs):
        """
        初始化越界检测分析器

        Args:
            model_code: 模型代码
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            device: 推理设备
            tracker_type: 跟踪器类型
            **kwargs: 其他参数
        """
        # 确保启用计数功能
        kwargs["counting_enabled"] = True

        # 如果没有提供计数线，使用默认值
        if "counting_line" not in kwargs:
            # 默认使用水平中线
            kwargs["counting_line"] = [(0.1, 0.5), (0.9, 0.5)]

        super().__init__(model_code, engine_type, yolo_version, device, tracker_type, **kwargs)

    def get_analysis_type(self) -> str:
        return "line_crossing"
