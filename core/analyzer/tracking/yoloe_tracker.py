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

# 使用新的日志记录器
from shared.utils.logger import get_normal_logger, get_exception_logger, get_test_logger
from core.analyzer.yoloe.yoloe_analyzer import YOLOETrackingAnalyzer
from core.analyzer.tracking.tracker import TrackerType

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()

class YOLOETracker(YOLOETrackingAnalyzer):
    """YOLOE跟踪器实现"""
    
    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0, 
                 yolo_version: int = 0, device: str = "auto", 
                 tracker_type_name: str = "sort",
                 **kwargs):
        """
        初始化YOLOE跟踪器
        
        Args:
            model_code: 模型代码
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            device: 推理设备
            tracker_type_name: 跟踪器类型名称
            **kwargs: 其他参数
        """
        try:
            tracker_type_enum = TrackerType[tracker_type_name.upper()]
        except KeyError:
            exception_logger.error(f"无效的YOLOE跟踪器类型名称: '{tracker_type_name}'. 将使用默认的SORT。")
            tracker_type_enum = TrackerType.SORT
        
        super().__init__(model_code, engine_type, yolo_version, device, 
                         tracker_type=tracker_type_enum.value, **kwargs)
        
        self.confidence = kwargs.get("confidence", 0.25)
        self.iou_threshold = kwargs.get("iou_threshold", 0.45)
        self.max_detections = kwargs.get("max_detections", 100)
        
        normal_logger.info(f"YOLOE跟踪器已初始化: 模型={model_code}, 置信度={self.confidence}, IoU阈值={self.iou_threshold}, 跟踪器类型={tracker_type_enum.name}")
        test_logger.info(f"[初始化] YOLOE跟踪器 (YOLOETracker): 模型={model_code}, 跟踪器={tracker_type_enum.name}, 默认置信度={self.confidence}")
    
    async def track(self, image: np.ndarray, task_name: Optional[str] = "未命名YOLOE跟踪任务", **kwargs) -> Dict[str, Any]:
        """
        对输入图像进行目标检测和跟踪
        
        Args:
            image: BGR格式的输入图像
            task_name: 任务名称
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 跟踪结果
        """
        log_prefix = f"[YOLOE跟踪] 任务={task_name}, 模型={self.current_model_code or '未知'}"
        test_logger.info(f"{log_prefix} | YOLOETracker 开始处理图像")
        
        current_confidence = kwargs.pop("confidence", self.confidence)
        current_iou_threshold = kwargs.pop("iou_threshold", self.iou_threshold)
        current_max_detections = kwargs.pop("max_detections", self.max_detections)
        
        test_logger.info(f"{log_prefix} | 使用参数: 置信度={current_confidence}, IoU={current_iou_threshold}, 最大检测数={current_max_detections}")
        
        try:
            result = await super().track(image, 
                                         confidence=current_confidence,
                                         iou_threshold=current_iou_threshold,
                                         max_detections=current_max_detections,
                                         task_name=task_name,
                                         **kwargs)
            test_logger.info(f"{log_prefix} | YOLOETracker 跟踪完成, 跟踪ID数: {len(result.get('tracking_results', []))}")
            return result
        except Exception as e:
            exception_logger.exception(f"YOLOE跟踪器在任务 {task_name} 的跟踪过程中失败")
            test_logger.info(f"{log_prefix} | YOLOETracker 跟踪失败: {str(e)}")
            return {
                "detections": [], "tracking_results": [], "pre_process_time": 0, "inference_time": 0,
                "post_process_time": 0, "annotated_image_bytes": None
            }
    
    async def detect(self, image: np.ndarray, task_name: Optional[str] = "未命名YOLOE检测(跟踪)任务", **kwargs) -> Dict[str, Any]:
        """
        对输入图像进行目标检测和跟踪
        
        Args:
            image: BGR格式的输入图像
            task_name: 任务名称
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 跟踪结果
        """
        return await self.track(image, task_name=task_name, **kwargs)
