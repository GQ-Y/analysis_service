"""
基础分析器模块 - 重构版
定义所有分析器的基类和通用接口，支持更灵活的参数传递
"""
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from datetime import datetime

# 使用新的日志记录器
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class BaseAnalyzer(ABC):
    """分析器基类"""

    def __init__(self, model_code: Optional[str] = None, device: str = "auto", **kwargs):
        """
        初始化分析器
        
        Args:
            model_code: 模型代码，如果提供则立即加载模型
            device: 推理设备 ("cpu", "cuda", "auto")
            **kwargs: 其他参数
        """
        self.analyzer_type = self.get_analysis_type()
        normal_logger.info(f"初始化分析器: 类型={self.analyzer_type}, 设备={device}")
        
        # 基本属性
        self.model_code = model_code
        self.device = device
        self.half_precision = kwargs.get("half_precision", False)
        self.custom_weights_path = kwargs.get("custom_weights_path")
        self.loaded = False  # 模型是否已加载
        
        # 加载模型（如果提供了模型代码）
        if model_code:
            import asyncio
            try:
                # 尝试获取现有事件循环
                loop = asyncio.get_running_loop()
            except RuntimeError:  # 如果没有正在运行的事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            try:
                if loop.is_running():
                    asyncio.create_task(self.load_model(model_code))
                    normal_logger.info(f"已为模型 {model_code} 创建异步加载任务。")
                else:
                    normal_logger.info(f"同步加载模型 {model_code}...")
                    loop.run_until_complete(self.load_model(model_code))
                    normal_logger.info(f"模型 {model_code} 同步加载完成。")
            except Exception as e:
                exception_logger.exception(f"在初始化过程中加载模型 {model_code} 失败: {e}")

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
    """目标检测分析器基类"""
    
    def __init__(self, model_code: Optional[str] = None, device: str = "auto", **kwargs):
        """
        初始化目标检测分析器
        
        Args:
            model_code: 模型代码，如果提供则立即加载模型
            device: 推理设备 ("cpu", "cuda", "auto")
            **kwargs: 其他参数，包括：
                - confidence: 置信度阈值
                - iou_threshold: IoU阈值
                - max_detections: 最大检测目标数
                - nested_detection: 是否启用嵌套检测
        """
        # 检测参数
        self.confidence = kwargs.get("confidence", 0.5)
        self.iou_threshold = kwargs.get("iou_threshold", 0.45)
        self.max_detections = kwargs.get("max_detections", 100)
        self.nested_detection = kwargs.get("nested_detection", False)
        
        # 调用父类初始化
        super().__init__(model_code, device, **kwargs)
        
    def get_analysis_type(self) -> str:
        """获取分析类型"""
        return "detection"

    async def detect(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        检测图像中的目标
        
        Args:
            frame: 输入图像
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 检测结果
        """
        raise NotImplementedError("子类必须实现detect方法")
        
    @property
    def model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        info = {
            "analyzer_type": self.get_analysis_type(),
            "analyzer_name": self.__class__.__name__,
            "model_code": self.model_code,
            "device": self.device,
            "half_precision": self.half_precision,
            "custom_weights_path": self.custom_weights_path,
            "confidence": self.confidence,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "nested_detection": self.nested_detection,
            "loaded": self.loaded
        }
        return info
