"""
YOLO检测分析器模块
实现基于YOLOv8的目标检测分析器
"""
import os
import time
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

# 使用新的日志记录器
from shared.utils.logger import get_normal_logger, get_exception_logger, get_test_logger
from core.analyzer.base_analyzer import DetectionAnalyzer
from core.analyzer.detection.yolo_detector import YOLODetector

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()

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
        self.detector = YOLODetector(model_code, engine_type, yolo_version, device)
        
        # 记录初始化信息
        normal_logger.info(f"YOLO检测分析器已初始化，使用内部YOLO检测器实例。模型代码: {model_code}")
        test_logger.info(f"[初始化] YOLO检测分析器 (YOLODetectionAnalyzer) 使用模型: {model_code}")
    
    async def load_model(self, model_code: str) -> bool:
        """
        加载模型
        
        Args:
            model_code: 模型代码
            
        Returns:
            bool: 是否成功加载模型
        """
        test_logger.info(f"[模型加载] YOLODetectionAnalyzer 开始加载模型: {model_code}")
        try:
            # 使用检测器加载模型
            await self.detector.load_model(model_code)
            
            # 更新当前模型代码
            self.current_model_code = self.detector.current_model_code
            
            normal_logger.info(f"YOLO检测分析器成功加载模型: {self.current_model_code}")
            test_logger.info(f"[模型加载] YOLODetectionAnalyzer 模型 {self.current_model_code} 加载成功。")
            return True
            
        except Exception as e:
            exception_logger.exception(f"YOLO检测分析器加载模型 {model_code} 失败")
            test_logger.info(f"[模型加载] YOLODetectionAnalyzer 模型 {model_code} 加载失败: {str(e)}")
            return False
    
    async def detect(self, image: np.ndarray, task_name: Optional[str] = "未命名分析任务", **kwargs) -> Dict[str, Any]:
        """
        分析图像
        
        Args:
            image: 输入图像
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        log_prefix = f"[分析器检测] 任务={task_name}, 模型={self.current_model_code or '未知'}"
        test_logger.info(f"{log_prefix} | YOLODetectionAnalyzer 开始检测图像")
        try:
            # 确保将task_name传递给底层的detector
            kwargs['task_name'] = task_name 
            results = await self.detector.detect(image, **kwargs)
            test_logger.info(f"{log_prefix} | YOLODetectionAnalyzer 检测完成, 结果目标数: {len(results.get('detections', []))}")
            return results
            
        except Exception as e:
            exception_logger.exception(f"YOLO检测分析器在任务 {task_name} 检测过程中失败")
            test_logger.info(f"{log_prefix} | YOLODetectionAnalyzer 检测失败: {str(e)}")
            return {
                "detections": [],
                "pre_process_time": 0,
                "inference_time": 0,
                "post_process_time": 0,
                "annotated_image_bytes": None
            }
    
    async def process_video_frame(self, frame: np.ndarray, frame_index: int, task_name: Optional[str] = "视频帧处理任务", **kwargs) -> Dict[str, Any]:
        """
        处理视频帧
        
        Args:
            frame: 视频帧
            frame_index: 帧索引
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        log_prefix = f"[分析器帧处理] 任务={task_name}, 模型={self.current_model_code or '未知'}, 帧={frame_index}"
        test_logger.info(f"{log_prefix} | YOLODetectionAnalyzer 开始处理视频帧")
        # 确保将task_name传递下去
        kwargs['task_name'] = task_name
        results = await self.detect(frame, **kwargs)
        results["frame_index"] = frame_index
        test_logger.info(f"{log_prefix} | YOLODetectionAnalyzer 处理视频帧完成, 检测到目标数: {len(results.get('detections', []))}")
        return results
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        # 从底层的detector获取更详细和准确的模型信息
        if hasattr(self.detector, 'model_info') and callable(self.detector.model_info):
            info = self.detector.model_info
        else: # 备用，如果detector没有model_info属性
            info = {
                "loaded": self.detector.model is not None if self.detector else False,
                "model_code": self.current_model_code,
                "engine_type": self.engine_type,
                "engine_name": self._get_engine_name(),
                "yolo_version": self.yolo_version,
                "yolo_version_name": self._get_yolo_version_name(),
                "device": self.device
            }
        test_logger.info(f"[模型信息] YOLODetectionAnalyzer 当前模型信息: {info}")
        return info
    
    def release(self) -> None:
        """释放资源"""
        normal_logger.info(f"开始释放YOLO检测分析器资源 (模型: {self.current_model_code or '无'})...")
        test_logger.info(f"[资源释放] YOLODetectionAnalyzer 开始释放资源 (模型: {self.current_model_code or '无'})")
        if hasattr(self.detector, 'release') and callable(self.detector.release):
            self.detector.release()
            normal_logger.info("内部YOLO检测器资源已调用释放。")
        else:
            normal_logger.warning("内部YOLO检测器没有可调用的release方法。")
        normal_logger.info("YOLO检测分析器资源释放完毕。")
        test_logger.info("[资源释放] YOLODetectionAnalyzer 资源释放完成。")
