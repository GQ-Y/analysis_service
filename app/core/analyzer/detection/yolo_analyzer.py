#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO目标检测分析器 - 纯粹的模型推理，专注输入输出
"""

import cv2
import numpy as np
import time
import logging
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ultralytics import YOLO

from app.core.storage.model_manager import ModelManager, ModelInfo
from app.models.base_model import AnalysisTypeEnum
from app.core.analyzer.analyzer_registry import register_analyzer
from app.core.analyzer.base_analyzer import BaseAnalyzer


@register_analyzer(AnalysisTypeEnum.DETECTION)
class YoloDetectionAnalyzer(BaseAnalyzer):
    """YOLO目标检测分析器 - 专注于模型推理的输入输出"""
    
    def __init__(self, model_code: str, model_manager: ModelManager = None, **kwargs):
        """
        初始化YOLO检测分析器
        
        Args:
            model_code: 模型代码
            model_manager: 模型管理器
            **kwargs: 其他参数
        """
        # 提取特定参数
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        self.iou_threshold = kwargs.get('iou_threshold', 0.4)
        self.max_detections = kwargs.get('max_detections', 100)
        self.input_size = kwargs.get('input_size', (640, 640))
        self.use_custom_nms = kwargs.get('use_custom_nms', False)
        self.class_specific_nms = kwargs.get('class_specific_nms', False)
        self.half_precision = kwargs.get('half_precision', False)
        
        
        # 清理kwargs中的特定参数，避免传递给父类
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in [
            'confidence_threshold', 'iou_threshold', 'max_detections', 'input_size',
            'use_custom_nms', 'class_specific_nms', 'half_precision'
        ]}
        
        # 调用父类初始化
        super().__init__(model_code, **filtered_kwargs)
        
        self.model_manager = model_manager or ModelManager("storage")
        self.model = None
        self.model_info: Optional[ModelInfo] = None
        self.class_names: Dict[int, str] = {}
        self.input_size: Tuple[int, int] = (640, 640)
        
        # 只有在提供了model_code时才初始化模型
        if model_code:
            self._load_model()
    
    def _load_model(self) -> bool:
        """加载YOLO模型"""
        try:
            # 获取模型信息
            self.model_info = self.model_manager.get_model_info(self.model_code)
            if not self.model_info:
                self.logger.error(f"❌ 未找到模型: {self.model_code}")
                return False
            
            # 获取类别信息
            self.class_names = self.model_info.classes
            self.input_size = self.model_info.input_size
            
            # 获取模型路径
            model_path = self.model_manager.get_model_path(self.model_code)
            if not model_path:
                self.logger.error(f"❌ 无法获取模型路径: {self.model_code}")
                return False
            
            # 加载YOLO模型
            self.logger.info(f"🔄 加载YOLO模型: {model_path}")
            self.model = YOLO(model_path)
            
            # 设置设备
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to('cuda')
                self.logger.info(f"🚀 YOLO模型已加载到GPU: {self.model_code}")
            else:
                self.logger.info(f"🖥️ YOLO模型已加载到CPU: {self.model_code}")
            
            # 预热模型
            self._warmup_model()
            
            self.logger.info(f"✅ YOLO分析器初始化成功: {self.model_code}")
            self.loaded = True  # 设置加载标志
            return True
            
        except Exception as e:
            self.logger.error(f"❌ YOLO分析器初始化失败 {self.model_code}: {e}")
            return False
    
    def _warmup_model(self):
        """预热模型"""
        if not self.model:
            return
            
        try:
            warmup_img = np.random.randint(0, 255, (*self.input_size, 3), dtype=np.uint8)
            self.logger.info("🔥 预热YOLO模型...")
            start_time = time.time()
            _ = self.model(warmup_img, verbose=False)
            warmup_time = time.time() - start_time
            self.logger.info(f"🔥 模型预热完成，耗时: {warmup_time:.3f}s")
        except Exception as e:
            self.logger.warning(f"⚠️ 模型预热失败: {e}")
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        对图像进行目标检测
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            Dict[str, Any]: 检测结果
        """
        start_time = time.time()
        
        try:
            if not self.model:
                raise ValueError(f"模型未加载: {self.model_code}")
            
            # 使用真实YOLO模型进行推理
            if self.use_custom_nms:
                # 使用自定义NMS时，禁用YOLO内置NMS（设置iou=1.0）
                results = self.model(image, 
                                   conf=self.confidence_threshold,
                                   iou=1.0,  # 禁用内置NMS
                                   verbose=False)
            else:
                # 使用YOLO内置NMS
                results = self.model(image, 
                                   conf=self.confidence_threshold,
                                   iou=self.iou_threshold,
                                   verbose=False)
            
            # 提取检测结果
            detections = self._extract_detections(results[0], image.shape)
            
            # 应用自定义NMS（如果启用）
            if self.use_custom_nms and detections:
                if self.class_specific_nms:
                    detections = self.apply_class_specific_nms(detections, self.iou_threshold)
                else:
                    detections = self.apply_nms(detections, self.iou_threshold)
            
            inference_time = time.time() - start_time
            
            # 【新增】YOLO检测日志
            nms_info = ""
            if self.use_custom_nms:
                nms_type = "按类别NMS" if self.class_specific_nms else "全局NMS"
                nms_info = f", {nms_type}"
            
            self.logger.debug(f"🤖 [{self.model_code}] YOLO推理完成: "
                           f"检测{len(detections)}个目标, "
                           f"推理耗时{inference_time*1000:.1f}ms, "
                           f"设备:{self.device}, "
                           f"置信度阈值:{self.confidence_threshold}{nms_info}")
            
            return {
                "detections": detections,
                "inference_time": inference_time,
                "model_code": self.model_code,
                "image_shape": image.shape,
                "device": self.device,
                "nms_info": {
                    "use_custom_nms": self.use_custom_nms,
                    "class_specific_nms": self.class_specific_nms,
                    "iou_threshold": self.iou_threshold
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ YOLO检测失败: {e}")
            return {
                "detections": [],
                "inference_time": time.time() - start_time,
                "model_code": self.model_code,
                "image_shape": image.shape,
                "error": str(e)
            }
    
    def _extract_detections(self, result, image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """提取YOLO检测结果"""
        detections = []
        
        if result.boxes is None:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        height, width = image_shape[:2]
        
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            detection = {
                "class_id": int(class_id),
                "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                "confidence": float(conf),
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1), 
                    "x2": float(x2),
                    "y2": float(y2),
                    "width": float(bbox_width),
                    "height": float(bbox_height)
                },
                "area": float(bbox_width * bbox_height)
            }
            
            detections.append(detection)
        
        return detections
    
    
    def batch_detect(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        批量检测 - 真正的批处理实现
        
        Args:
            images: 图像列表
            
        Returns:
            List[Dict[str, Any]]: 检测结果列表
        """
        if not images:
            return []
            
        start_time = time.time()
        
        try:
            if not self.model:
                raise ValueError(f"模型未加载: {self.model_code}")
            
            # 使用真实YOLO模型进行批量推理
            if self.use_custom_nms:
                # 使用自定义NMS时，禁用YOLO内置NMS（设置iou=1.0）
                results = self.model(images, 
                                   conf=self.confidence_threshold,
                                   iou=1.0,  # 禁用内置NMS
                                   verbose=False)
            else:
                # 使用YOLO内置NMS
                results = self.model(images, 
                                   conf=self.confidence_threshold,
                                   iou=self.iou_threshold,
                                   verbose=False)
            
            # 提取每张图像的检测结果
            batch_results = []
            for i, (result, image) in enumerate(zip(results, images)):
                detections = self._extract_detections(result, image.shape)
                
                # 应用自定义NMS（如果启用）
                if self.use_custom_nms and detections:
                    if self.class_specific_nms:
                        detections = self.apply_class_specific_nms(detections, self.iou_threshold)
                    else:
                        detections = self.apply_nms(detections, self.iou_threshold)
                
                batch_result = {
                    "detections": detections,
                    "batch_index": i,
                    "model_code": self.model_code,
                    "image_shape": image.shape,
                    "device": self.device,
                        "nms_info": {
                        "use_custom_nms": self.use_custom_nms,
                        "class_specific_nms": self.class_specific_nms,
                        "iou_threshold": self.iou_threshold
                    }
                }
                batch_results.append(batch_result)
            
            # 计算批处理总时间
            total_inference_time = time.time() - start_time
            
            # 为每个结果添加推理时间（平均分配）
            avg_inference_time = total_inference_time / len(images)
            for result in batch_results:
                result["inference_time"] = avg_inference_time
                result["batch_total_time"] = total_inference_time
            
            # 【新增】批处理日志
            total_detections = sum(len(result["detections"]) for result in batch_results)
            nms_info = ""
            if self.use_custom_nms:
                nms_type = "按类别NMS" if self.class_specific_nms else "全局NMS"
                nms_info = f", {nms_type}"
            
            self.logger.debug(f"🚀 [{self.model_code}] 批量YOLO推理完成: "
                           f"处理{len(images)}张图像, "
                           f"检测{total_detections}个目标, "
                           f"批处理耗时{total_inference_time*1000:.1f}ms, "
                           f"平均{avg_inference_time*1000:.1f}ms/张{nms_info}")
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"❌ YOLO批量检测失败: {e}")
            # 返回错误结果
            error_results = []
            for i in range(len(images)):
                error_results.append({
                    "detections": [],
                    "batch_index": i,
                    "error": str(e),
                    "model_code": self.model_code,
                    "image_shape": images[i].shape if i < len(images) else (0, 0, 0),
                    "inference_time": 0.0
                })
            return error_results
    
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_code": self.model_code,
            "analysis_type": "detection",
            "class_names": self.class_names,
            "input_size": self.input_size,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "use_custom_nms": self.use_custom_nms,
            "class_specific_nms": self.class_specific_nms
        }
    
    def cleanup(self):
        """清理资源"""
        if self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.logger.info(f"🧹 YOLO检测分析器资源已清理: {self.model_code}")

    async def analyze_batch_async(self, frames: List[np.ndarray], confidence_threshold: float = None, iou_threshold: float = None) -> List[Dict[str, Any]]:
        """异步批量分析（简单包装同步版本）"""
        return self.analyze_batch(frames, confidence_threshold, iou_threshold)

    # === 适配器方法：兼容AnalysisWorker期望的BaseAnalyzer接口 ===
    
    @property
    def name(self):
        """分析器名称（兼容BaseAnalyzer接口）"""
        return f"yolo_{self.model_code}"
    
    def process_frame(self, frame_buffer) -> Dict[str, Any]:
        """
        处理单帧（AnalysisWorker期望的接口）
        
        Args:
            frame_buffer: 帧缓冲区对象
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            # 从帧缓冲区提取图像数据
            if hasattr(frame_buffer, 'get_frame_view'):
                image = frame_buffer.get_frame_view()
            elif hasattr(frame_buffer, 'frame_data'):
                image = frame_buffer.frame_data
            elif hasattr(frame_buffer, 'get_frame_copy'):
                image = frame_buffer.get_frame_copy()
            else:
                raise ValueError(f"不支持的帧缓冲区类型: {type(frame_buffer)}")
            
            # 调用原有的检测方法
            return self.detect(image)
            
        except Exception as e:
            self.logger.error(f"❌ YOLO处理帧失败: {e}")
            return {
                "detections": [],
                "error": str(e),
                "frame_id": getattr(frame_buffer, 'frame_id', -1),
                "analyzer": self.name
            }
    
    def process_batch(self, frame_buffers: List) -> List[Dict[str, Any]]:
        """
        批量处理（AnalysisWorker期望的接口）
        
        Args:
            frame_buffers: 帧缓冲区列表
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        results = []
        for frame_buffer in frame_buffers:
            result = self.process_frame(frame_buffer)
            results.append(result)
        return results
    
    def reset_stats(self):
        """重置统计信息（兼容BaseAnalyzer接口）"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "model_code": self.model_code,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "input_size": self.input_size,
            "class_count": len(self.class_names)
        }

    def get_analysis_type(self) -> AnalysisTypeEnum:
        """获取分析类型"""
        return AnalysisTypeEnum.DETECTION
    
    async def load_model(self, model_code: str) -> bool:
        """加载模型 - 兼容新的接口"""
        self.model_code = model_code
        return self._load_model()
    
    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表"""
        # 这里可以返回所有支持的YOLO模型
        return ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]
    
    async def analyze_frame(self, frame: np.ndarray, config=None) -> Dict[str, Any]:
        """分析单帧 - 符合BaseAnalyzer接口"""
        return self.detect(frame)
    
    async def analyze_batch(self, frames: List[np.ndarray], config=None) -> List[Dict[str, Any]]:
        """批量分析 - 符合BaseAnalyzer接口"""
        return self.batch_detect(frames)


class YoloDetectionAnalyzerFactory:
    """YOLO检测分析器工厂"""
    
    @staticmethod
    def create(model_code: str, model_manager: ModelManager, **kwargs) -> Optional[YoloDetectionAnalyzer]:
        """
        创建YOLO检测分析器
        
        Args:
            model_code: 模型代码
            model_manager: 模型管理器
            **kwargs: 其他参数
            
        Returns:
            Optional[YoloDetectionAnalyzer]: 分析器实例
        """
        try:
            return YoloDetectionAnalyzer(model_code, model_manager, **kwargs)
        except Exception as e:
            logging.error(f"❌ 创建YOLO检测分析器失败 {model_code}: {e}")
            return None 