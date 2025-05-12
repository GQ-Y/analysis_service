"""
YOLO检测分析器模块
实现基于YOLOv8的目标检测分析器
"""
import os
import time
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

from shared.utils.logger import setup_logger
from core.analyzer.base_analyzer import DetectionAnalyzer
from core.analyzer.detection.yolo_detector import YOLODetector

logger = setup_logger(__name__)

class YOLODetectionAnalyzer(DetectionAnalyzer):
    """YOLO检测分析器实现"""
    
    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0, 
                 yolo_version: int = 0, device: str = "auto", **kwargs):
        """
        初始化YOLO检测分析器
        
        Args:
            model_code: 模型代码，如果提供则立即加载模型
            engine_type: 推理引擎类型 (0=PyTorch, 1=ONNX, 2=TensorRT, 3=OpenVINO, 4=Pytron)
            yolo_version: YOLO版本 (0=v8n, 1=v8s, 2=v8l, 3=v8x, 4=11s, 5=11m, 6=11l)
            device: 推理设备 ("cpu", "cuda", "auto")
            **kwargs: 其他参数
        """
        super().__init__(model_code, engine_type, yolo_version, device)
        
        # 创建YOLO检测器
        self.detector = YOLODetector(model_code)
        
        # 记录初始化信息
        logger.info(f"初始化YOLO检测分析器: 引擎类型={self._get_engine_name()}, YOLO版本={self._get_yolo_version_name()}, 设备={device}")
    
    async def load_model(self, model_code: str) -> bool:
        """
        加载模型
        
        Args:
            model_code: 模型代码
            
        Returns:
            bool: 是否成功加载模型
        """
        try:
            # 使用检测器加载模型
            await self.detector.load_model(model_code)
            
            # 更新当前模型代码
            self.current_model_code = model_code
            
            logger.info(f"YOLO检测分析器成功加载模型: {model_code}")
            return True
            
        except Exception as e:
            logger.error(f"YOLO检测分析器加载模型失败: {str(e)}")
            return False
    
    async def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        分析图像
        
        Args:
            image: 输入图像
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            # 使用检测器进行检测
            results = await self.detector.detect(image, **kwargs)
            return results
            
        except Exception as e:
            logger.error(f"YOLO检测分析器检测失败: {str(e)}")
            return {
                "detections": [],
                "pre_process_time": 0,
                "inference_time": 0,
                "post_process_time": 0
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
        # 对于检测分析器，直接调用detect方法
        results = await self.detect(frame, **kwargs)
        
        # 添加帧索引
        results["frame_index"] = frame_index
        
        return results
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "model_code": self.current_model_code,
            "engine_type": self.engine_type,
            "engine_name": self._get_engine_name(),
            "yolo_version": self.yolo_version,
            "yolo_version_name": self._get_yolo_version_name(),
            "device": self.device
        }
    
    def release(self) -> None:
        """释放资源"""
        # 释放检测器资源
        if hasattr(self.detector, "model") and self.detector.model is not None:
            try:
                # 释放CUDA内存
                if hasattr(self.detector.model, "to"):
                    self.detector.model.to("cpu")
                
                # 删除模型
                self.detector.model = None
                
                # 强制垃圾回收
                import gc
                gc.collect()
                
                logger.info("YOLO检测分析器资源已释放")
            except Exception as e:
                logger.error(f"释放YOLO检测分析器资源时出错: {str(e)}")
