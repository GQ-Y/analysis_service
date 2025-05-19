"""
YOLOE检测器模块
实现基于YOLOE的目标检测功能
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
from core.analyzer.yoloe.yoloe_analyzer import YOLOEDetectionAnalyzer

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()

class YOLOEDetector(YOLOEDetectionAnalyzer):
    """YOLOE检测器实现"""
    
    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0, 
                 yolo_version: int = 0, device: str = "auto", **kwargs):
        """
        初始化YOLOE检测器
        
        Args:
            model_code: 模型代码
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            device: 推理设备
            **kwargs: 其他参数
        """
        super().__init__(model_code, engine_type, yolo_version, device, **kwargs)
        
        # 设置默认参数
        self.confidence = kwargs.get("confidence", 0.25)
        self.iou_threshold = kwargs.get("iou_threshold", 0.45)
        self.max_detections = kwargs.get("max_detections", 100)
        
        normal_logger.info(f"初始化YOLOE检测器: 置信度={self.confidence}, IoU阈值={self.iou_threshold}")
        test_logger.info(f"[初始化] YOLOE检测器 (YOLOEDetector) 使用模型: {model_code}, 默认置信度: {self.confidence}")
    
    async def detect(self, image: np.ndarray, task_name: Optional[str] = "未命名YOLOE检测任务", **kwargs) -> Dict[str, Any]:
        """
        对输入图像进行目标检测
        
        Args:
            image: BGR格式的输入图像
            task_name: 任务名称
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 检测结果
        """
        log_prefix = f"[YOLOE检测] 任务={task_name}, 模型={self.current_model_code or '未知'}"
        test_logger.info(f"{log_prefix} | YOLOEDetector 开始检测图像")
        
        # 更新参数，如果kwargs中没有，则使用实例的默认值
        confidence = kwargs.pop("confidence", self.confidence)
        iou_threshold = kwargs.pop("iou_threshold", self.iou_threshold)
        max_detections = kwargs.pop("max_detections", self.max_detections)
        
        test_logger.info(f"{log_prefix} | 使用参数: 置信度={confidence}, IoU={iou_threshold}, 最大检测数={max_detections}")
        
        try:
            # 调用父类的detect方法，传递更新后的参数和task_name
            result = await super().detect(image, 
                                         confidence=confidence,
                                         iou_threshold=iou_threshold,
                                         max_detections=max_detections,
                                         task_name=task_name, # 确保task_name传递给父类
                                         **kwargs)
            test_logger.info(f"{log_prefix} | YOLOEDetector 检测完成, 目标数: {len(result.get('detections', []))}")
            return result
        except Exception as e:
            exception_logger.exception(f"YOLOE检测器在任务 {task_name} 的检测过程中失败")
            test_logger.info(f"{log_prefix} | YOLOEDetector 检测失败: {str(e)}")
            # 返回一个与父类错误时一致的结构
            return {
                "detections": [],
                "pre_process_time": 0,
                "inference_time": 0,
                "post_process_time": 0,
                "annotated_image_bytes": None
            }
