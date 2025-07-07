#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析结果模型
定义分析结果的数据结构
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Detection:
    """检测结果"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    area: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "area": self.area
        }


@dataclass
class AnalysisResult:
    """分析结果"""
    frame_id: int
    timestamp: float
    model_name: str
    image_shape: tuple
    detections: List[Detection]
    inference_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "image_shape": self.image_shape,
            "detections": [det.to_dict() for det in self.detections],
            "inference_time": self.inference_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """从字典创建实例"""
        detections = [
            Detection(**det_data) for det_data in data.get("detections", [])
        ]
        
        return cls(
            frame_id=data["frame_id"],
            timestamp=data["timestamp"],
            model_name=data["model_name"],
            image_shape=tuple(data["image_shape"]),
            detections=detections,
            inference_time=data.get("inference_time")
        ) 