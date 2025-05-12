"""
YOLO跟踪器模块
结合YOLO检测器和跟踪器实现目标跟踪功能
"""
import os
import time
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

from shared.utils.logger import setup_logger
from core.analyzer.base_analyzer import TrackingAnalyzer
from core.analyzer.detection import YOLODetector
from core.analyzer.tracking.tracker import Tracker

logger = setup_logger(__name__)

class YOLOTracker(TrackingAnalyzer):
    """YOLO跟踪器实现"""
    
    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0, 
                 yolo_version: int = 0, device: str = "auto", 
                 tracker_type: int = 0, **kwargs):
        """
        初始化YOLO跟踪器
        
        Args:
            model_code: 模型代码
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            device: 推理设备
            tracker_type: 跟踪器类型
            **kwargs: 其他参数
        """
        super().__init__(model_code, engine_type, yolo_version, device, tracker_type, **kwargs)
        
        # 创建YOLO检测器
        self.detector = YOLODetector(model_code, engine_type, yolo_version, device)
        
        # 设置默认参数
        self.confidence = kwargs.get("confidence", 0.25)
        self.iou_threshold = kwargs.get("iou_threshold", 0.45)
        self.max_detections = kwargs.get("max_detections", 100)
        
        logger.info(f"初始化YOLO跟踪器: 置信度={self.confidence}, IoU阈值={self.iou_threshold}")
    
    async def load_model(self, model_code: str) -> bool:
        """
        加载模型
        
        Args:
            model_code: 模型代码
            
        Returns:
            bool: 是否成功加载模型
        """
        # 加载检测器模型
        result = await self.detector.load_model(model_code)
        
        # 更新当前模型代码
        if result:
            self.current_model_code = model_code
            self.model = self.detector.model
        
        return result
    
    async def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        对输入图像进行目标检测和跟踪
        
        Args:
            image: BGR格式的输入图像
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 跟踪结果
        """
        start_time = time.time()
        
        try:
            # 更新参数
            confidence = kwargs.get("confidence", self.confidence)
            iou_threshold = kwargs.get("iou_threshold", self.iou_threshold)
            max_detections = kwargs.get("max_detections", self.max_detections)
            
            # 执行检测
            detection_result = await self.detector.detect(
                image, 
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                **kwargs
            )
            
            # 获取检测结果
            detections = detection_result.get("detections", [])
            
            # 使用跟踪器更新跟踪结果
            if self.tracker:
                tracked_objects = self.tracker.update(detections)
            else:
                tracked_objects = []
            
            # 计算总时间
            total_time = (time.time() - start_time) * 1000
            
            # 构建返回结果
            result = {
                "detections": detections,
                "tracked_objects": tracked_objects,
                "pre_process_time": detection_result.get("pre_process_time", 0),
                "inference_time": detection_result.get("inference_time", 0),
                "post_process_time": detection_result.get("post_process_time", 0),
                "tracking_time": total_time - detection_result.get("pre_process_time", 0) - detection_result.get("inference_time", 0) - detection_result.get("post_process_time", 0),
                "annotated_image_bytes": detection_result.get("annotated_image_bytes")
            }
            
            return result
            
        except Exception as e:
            logger.error(f"YOLO跟踪失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "detections": [],
                "tracked_objects": [],
                "pre_process_time": 0,
                "inference_time": 0,
                "post_process_time": 0,
                "tracking_time": 0,
                "annotated_image_bytes": None
            }
    
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
        # 调用detect方法处理帧
        result = await self.detect(frame, **kwargs)
        
        # 添加帧索引
        result["frame_index"] = frame_index
        
        return result
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        if not hasattr(self, "detector") or not self.detector or not self.detector.model:
            return {
                "loaded": False,
                "model_code": None
            }
            
        detector_info = self.detector.model_info
        
        return {
            "loaded": detector_info.get("loaded", False),
            "model_code": self.current_model_code,
            "engine_type": self.engine_type,
            "yolo_version": self.yolo_version,
            "device": self.device,
            "tracker_type": self.tracker_type if hasattr(self, "tracker_type") else None
        }
    
    def release(self) -> None:
        """释放资源"""
        if hasattr(self, "detector") and self.detector:
            self.detector.release()
            self.detector = None
            
        if hasattr(self, "tracker") and self.tracker:
            self.tracker = None
            
        self.model = None
        logger.info("YOLO跟踪器资源已释放")
