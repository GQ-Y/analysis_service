"""
YOLO检测分析器模块 - 重构版
基于Ultralytics YOLO的对象检测分析器
"""
import os
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from core.analyzer.base_analyzer import DetectionAnalyzer
from core.analyzer.detection.yolo_detector import YOLODetector
from core.analyzer.registry import register_analyzer
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

@register_analyzer("detection")
class YOLODetectionAnalyzer(DetectionAnalyzer):
    """YOLO检测分析器 - 重构版，支持插件化加载"""

    def __init__(self, model_code: Optional[str] = None, device: str = "auto", **kwargs):
        """
        初始化YOLO检测分析器
        
        Args:
            model_code: 模型代码
            device: 推理设备 ("cpu", "cuda", "auto")
            **kwargs: 其他参数，包括：
                - custom_weights_path: 自定义权重路径
                - half_precision: 是否使用半精度
                - confidence: 置信度阈值
                - iou_threshold: IoU阈值
                - max_detections: 最大检测目标数
                - classes: 类别列表
                - nested_detection: 是否启用嵌套检测
        """
        super().__init__(model_code, device, **kwargs)
        
        # 初始化检测器
        self.detector = YOLODetector(model_code, device, **kwargs)
        
        # 检测参数
        self.confidence = kwargs.get("confidence", 0.25)
        self.iou_threshold = kwargs.get("iou_threshold", 0.45)
        self.max_detections = kwargs.get("max_detections", 100)
        self.classes = kwargs.get("classes")
        self.nested_detection = kwargs.get("nested_detection", False)
        
        # 检测统计
        self._detection_count = 0
        self._total_detection_time = 0
        self._frame_count = 0
        
        normal_logger.info(f"YOLO检测分析器初始化: 置信度={self.confidence}, IoU阈值={self.iou_threshold}")
        if self.nested_detection:
            normal_logger.info("启用嵌套检测")

    async def load_model(self, model_code: str) -> bool:
        """
        加载模型
        
        Args:
            model_code: 模型代码
            
        Returns:
            bool: 是否成功加载模型
        """
        try:
            # 加载检测器模型
            success = await self.detector.load_model(model_code)
            if success:
                self.current_model_code = model_code
                normal_logger.info(f"YOLO检测分析器成功加载模型: {model_code}")
            return success
        except Exception as e:
            exception_logger.exception(f"YOLO检测分析器加载模型失败: {e}")
            return False

    async def detect(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        检测图像中的目标

        Args:
            frame: 输入图像
            **kwargs: 其他参数，包括：
                - confidence: 置信度阈值，覆盖初始化时设置的值
                - iou_threshold: IoU阈值，覆盖初始化时设置的值
                - max_detections: 最大检测目标数，覆盖初始化时设置的值
                - classes: 类别列表，限制只检测特定类别
                - return_image: 是否返回标注后的图像

        Returns:
            Dict[str, Any]: 检测结果
        """
        # 记录开始时间，用于性能统计
        start_time = time.time()
        frame_count = self._frame_count + 1
        
        # 输出更详细的日志
        normal_logger.info(f"开始检测第{frame_count}帧，帧大小: {frame.shape}")
        
        # 获取参数
        confidence = kwargs.get("confidence", self.confidence)
        iou_threshold = kwargs.get("iou_threshold", self.iou_threshold)
        max_detections = kwargs.get("max_detections", self.max_detections)
        
        # 检查模型是否已加载
        if not self.loaded:
            normal_logger.warning(f"模型未加载，无法执行检测")
            return {
                "success": False,
                "error": "模型未加载",
                "detections": [],
                "stats": {
                    "detection_time": time.time() - start_time,
                    "average_time": time.time() - start_time,
                    "detection_count": frame_count
                }
            }
        
        # 使用检测器进行检测
        try:
            # 调用YOLO检测器的detect方法
            detect_result = await self.detector.detect(
                frame,
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                **kwargs
            )
            
            # 检查检测结果
            if not detect_result["success"]:
                normal_logger.warning(f"检测失败: {detect_result.get('error', '未知错误')}")
                return detect_result
                
            # 获取检测结果
            detections = detect_result["detections"]
            
            # 添加检测结果到统计信息
            self._detection_count = frame_count
            self._total_detection_time += time.time() - start_time
            
            # 记录检测统计信息
            detect_result["stats"]["detection_time"] = time.time() - start_time
            detect_result["stats"]["average_time"] = self._total_detection_time / self._detection_count
            detect_result["stats"]["detection_count"] = self._detection_count
            
            # 输出检测结果
            normal_logger.info(f"成功检测第{frame_count}帧，耗时: {time.time() - start_time:.4f}秒，检测到{len(detections)}个目标")
            if len(detections) > 0:
                normal_logger.info(f"检测结果: {detections[:3] if len(detections) > 3 else detections}")
            
            return detect_result
            
        except Exception as e:
            exception_logger.exception(f"YOLO检测分析器执行检测时出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "detections": [],
                "stats": {
                    "detection_time": time.time() - start_time,
                    "average_time": self._total_detection_time / (self._detection_count or 1),
                    "detection_count": self._detection_count
                }
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
        # 更新帧计数
        self._frame_count += 1
        
        # 执行检测
        result = await self.detect(frame, **kwargs)
        
        # 添加帧信息
        result["frame_index"] = frame_index
        result["frame_count"] = self._frame_count
        
        return result

    def _process_nested_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理嵌套检测关系
        
        Args:
            detections: 检测结果列表
            
        Returns:
            List[Dict[str, Any]]: 处理后的检测结果
        """
        # 遍历所有检测结果，计算IoU并确定嵌套关系
        for i, det_i in enumerate(detections):
            det_i["contains"] = []
            det_i["contained_by"] = []
            
            x1_i, y1_i, x2_i, y2_i = det_i["bbox"]
            area_i = (x2_i - x1_i) * (y2_i - y1_i)
            
            for j, det_j in enumerate(detections):
                if i == j:
                    continue
                    
                x1_j, y1_j, x2_j, y2_j = det_j["bbox"]
                
                # 计算重叠区域
                x1_inter = max(x1_i, x1_j)
                y1_inter = max(y1_i, y1_j)
                x2_inter = min(x2_i, x2_j)
                y2_inter = min(y2_i, y2_j)
                
                if x1_inter < x2_inter and y1_inter < y2_inter:
                    area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                    area_j = (x2_j - x1_j) * (y2_j - y1_j)
                    
                    # 判断嵌套关系
                    if area_inter / area_j > 0.95:  # j基本被i包含
                        det_i["contains"].append(j)
                        det_j["contained_by"].append(i)
                        
        return detections

    @property
    def model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        base_info = {
            "analyzer_type": "detection",
            "analyzer_name": self.__class__.__name__,
            "model_code": self.current_model_code,
            "device": self.device,
            "half_precision": self.half_precision,
            "custom_weights_path": self.custom_weights_path,
            "confidence": self.confidence,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "nested_detection": self.nested_detection
        }
        
        # 添加检测器信息
        if hasattr(self.detector, "model_info"):
            detector_info = self.detector.model_info
            base_info.update(detector_info)
            
        return base_info

    def release(self) -> None:
        """释放资源"""
        if self.detector:
            self.detector.release()
            normal_logger.info("YOLO检测分析器已释放资源")
