"""
YOLOE跟踪器模块
结合YOLOE检测器和跟踪器实现目标跟踪功能
支持文本提示、图像提示和无提示推理
"""
import os
import time
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

from shared.utils.logger import setup_logger
from core.analyzer.yoloe.yoloe_analyzer import YOLOETrackingAnalyzer

logger = setup_logger(__name__)

class YOLOETracker(YOLOETrackingAnalyzer):
    """YOLOE跟踪器实现"""
    
    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0, 
                 yolo_version: int = 0, device: str = "auto", 
                 tracker_type: int = 0, **kwargs):
        """
        初始化YOLOE跟踪器
        
        Args:
            model_code: 模型代码
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            device: 推理设备
            tracker_type: 跟踪器类型
            **kwargs: 其他参数
        """
        super().__init__(model_code, engine_type, yolo_version, device, tracker_type, **kwargs)
        
        # 设置默认参数
        self.confidence = kwargs.get("confidence", 0.25)
        self.iou_threshold = kwargs.get("iou_threshold", 0.45)
        self.max_detections = kwargs.get("max_detections", 100)
        
        logger.info(f"初始化YOLOE跟踪器: 置信度={self.confidence}, IoU阈值={self.iou_threshold}")
    
    async def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        对输入图像进行目标检测和跟踪
        
        Args:
            image: BGR格式的输入图像
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 跟踪结果
        """
        # 更新参数
        confidence = kwargs.get("confidence", self.confidence)
        iou_threshold = kwargs.get("iou_threshold", self.iou_threshold)
        max_detections = kwargs.get("max_detections", self.max_detections)
        
        # 调用父类的detect方法
        result = await super().detect(image, 
                                     confidence=confidence,
                                     iou_threshold=iou_threshold,
                                     max_detections=max_detections,
                                     **kwargs)
        
        return result
