"""
分析服务
提供图像和视频分析相关的业务逻辑
"""
from typing import Dict, Any, List, Optional
import asyncio
import numpy as np
import cv2
import os
import base64
from datetime import datetime

from core.analyzer import create_analyzer
from core.config import settings
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class AnalysisService:
    """分析服务"""
    
    def __init__(self):
        """初始化"""
        self.analyzers = {}
        
    async def get_analyzer(self, analysis_type: str, model_code: str) -> Any:
        """
        获取分析器实例
        
        Args:
            analysis_type: 分析类型
            model_code: 模型代码
            
        Returns:
            Any: 分析器实例
        """
        key = f"{analysis_type}_{model_code}"
        
        if key not in self.analyzers:
            # 创建新的分析器实例
            analyzer = create_analyzer(analysis_type)
            await analyzer.load_model(model_code)
            self.analyzers[key] = analyzer
            
        return self.analyzers[key]
        
    async def analyze_image(self, image_path: str, analysis_type: str, model_code: str, **kwargs) -> Dict[str, Any]:
        """
        分析图像
        
        Args:
            image_path: 图像路径
            analysis_type: 分析类型
            model_code: 模型代码
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
                
            # 获取分析器
            analyzer = await self.get_analyzer(analysis_type, model_code)
            
            # 执行分析
            result = await analyzer.detect(image, **kwargs)
            
            # 处理结果
            processed_result = self._process_result(result, image)
            
            return {
                "success": True,
                "result": processed_result
            }
            
        except Exception as e:
            exception_logger.exception(f"分析图像失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def _process_result(self, result: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """
        处理分析结果
        
        Args:
            result: 原始分析结果
            image: 原始图像
            
        Returns:
            Dict[str, Any]: 处理后的结果
        """
        # 提取检测结果
        detections = result.get("detections", [])
        
        # 获取图像尺寸
        height, width = image.shape[:2]
        
        # 处理检测结果
        processed_detections = []
        for det in detections:
            # 提取边界框
            bbox = det.get("bbox", {})
            
            # 计算中心点和尺寸
            x1 = bbox.get("x1", 0)
            y1 = bbox.get("y1", 0)
            x2 = bbox.get("x2", 0)
            y2 = bbox.get("y2", 0)
            
            # 计算中心点和尺寸
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            box_width = x2 - x1
            box_height = y2 - y1
            
            # 构建处理后的检测结果
            processed_detection = {
                "class_id": det.get("class_id", 0),
                "class_name": det.get("class_name", "unknown"),
                "confidence": det.get("confidence", 0),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": center_x,
                    "center_y": center_y,
                    "width": box_width,
                    "height": box_height
                }
            }
            
            # 添加跟踪ID（如果有）
            if "track_id" in det:
                processed_detection["track_id"] = det["track_id"]
                
            processed_detections.append(processed_detection)
            
        # 构建处理后的结果
        processed_result = {
            "detections": processed_detections,
            "frame_info": {
                "width": width,
                "height": height,
                "timestamp": datetime.now().isoformat()
            },
            "analysis_info": {
                "inference_time": result.get("inference_time", 0),
                "pre_process_time": result.get("pre_process_time", 0),
                "post_process_time": result.get("post_process_time", 0)
            }
        }
        
        # 添加标注图像（如果有）
        if "annotated_image_bytes" in result and result["annotated_image_bytes"]:
            processed_result["image_results"] = {
                "annotated": {
                    "format": "jpg",
                    "base64": base64.b64encode(result["annotated_image_bytes"]).decode('utf-8')
                }
            }
            
        return processed_result
