#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: base_analyzer.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 基础分析器

定义所有分析器的基类和通用接口，提供统一的分析器框架。

本文件是分析服务项目的一部分。
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

import numpy as np

from app.models.base_model import AnalysisConfig, DeviceTypeEnum, AnalysisTypeEnum


class BaseAnalyzer(ABC):
    """分析器基类"""
    
    def __init__(self, model_code: Optional[str] = None, device: str = "auto", **kwargs):
        """初始化分析器
        
        Args:
            model_code: 模型代码，如果提供则立即加载模型
            device: 推理设备 ("cpu", "cuda", "auto")
            **kwargs: 其他参数
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.analyzer_type = self.get_analysis_type()
        
        # 基本属性
        self.model_code = model_code
        self.device = self._auto_select_device(device)
        self.half_precision = kwargs.get("half_precision", False)
        self.custom_weights_path = kwargs.get("custom_weights_path")
        self.loaded = False
        
        # 性能统计
        self.stats = {
            'total_frames': 0,
            'total_time': 0.0,
            'avg_fps': 0.0,
            'last_inference_time': 0.0
        }
        
        self.logger.info(f"初始化分析器: 类型={self.analyzer_type}, 设备={self.device}")
    
    @abstractmethod
    def get_analysis_type(self) -> AnalysisTypeEnum:
        """获取分析类型
        
        Returns:
            AnalysisTypeEnum: 分析类型
        """
        pass
    
    @abstractmethod
    async def load_model(self, model_code: str) -> bool:
        """加载模型
        
        Args:
            model_code: 模型代码
            
        Returns:
            bool: 是否加载成功
        """
        pass
    
    @abstractmethod
    async def analyze_frame(self, frame: np.ndarray, config: AnalysisConfig = None) -> Dict[str, Any]:
        """分析单帧
        
        Args:
            frame: 输入帧
            config: 分析配置
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        pass
    
    @abstractmethod
    async def analyze_batch(self, frames: List[np.ndarray], config: AnalysisConfig = None) -> List[Dict[str, Any]]:
        """批量分析
        
        Args:
            frames: 输入帧列表
            config: 分析配置
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表
        
        Returns:
            List[str]: 支持的模型代码列表
        """
        pass
    
    def _auto_select_device(self, device: str) -> str:
        """自动选择设备
        
        Args:
            device: 设备类型
            
        Returns:
            str: 选择的设备
        """
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载
        
        Returns:
            bool: 是否已加载
        """
        return self.loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            'analyzer_type': self.analyzer_type,
            'model_code': self.model_code,
            'device': self.device,
            'loaded': self.loaded,
            'half_precision': self.half_precision,
            'custom_weights_path': self.custom_weights_path
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计
        
        Returns:
            Dict[str, Any]: 性能统计信息
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_frames': 0,
            'total_time': 0.0,
            'avg_fps': 0.0,
            'last_inference_time': 0.0
        }
    
    def _update_stats(self, inference_time: float):
        """更新统计信息
        
        Args:
            inference_time: 推理时间
        """
        self.stats['total_frames'] += 1
        self.stats['total_time'] += inference_time
        self.stats['last_inference_time'] = inference_time
        
        if self.stats['total_time'] > 0:
            self.stats['avg_fps'] = self.stats['total_frames'] / self.stats['total_time']
    
    async def warmup(self, warmup_frames: int = 5):
        """模型预热
        
        Args:
            warmup_frames: 预热帧数
        """
        if not self.loaded:
            self.logger.warning("模型未加载，跳过预热")
            return
        
        self.logger.info(f"开始模型预热，预热帧数: {warmup_frames}")
        
        # 🔧 动态获取输入尺寸，而不是硬编码
        input_size = self._get_input_size()
        self.logger.info(f"使用输入尺寸进行预热: {input_size}")
        
        # 创建虚拟帧进行预热
        dummy_frame = np.random.randint(0, 255, (*input_size, 3), dtype=np.uint8)
        
        for i in range(warmup_frames):
            try:
                await self.analyze_frame(dummy_frame)
                self.logger.debug(f"预热进度: {i+1}/{warmup_frames}")
            except Exception as e:
                self.logger.warning(f"预热失败: {e}")
                break
        
        # 重置统计信息，不计入预热时间
        self.reset_stats()
        self.logger.info("模型预热完成")
    
    def _get_input_size(self) -> Tuple[int, int]:
        """获取模型输入尺寸
        
        Returns:
            Tuple[int, int]: 输入尺寸 (width, height)
        """
        # 默认尺寸
        default_size = (640, 640)
        
        try:
            # 1. 如果有 input_size 属性，直接使用
            if hasattr(self, 'input_size') and self.input_size:
                size = self.input_size
                if isinstance(size, (tuple, list)) and len(size) >= 2:
                    return tuple(size[:2])  # 只取前两个维度
            
            # 2. 如果有 model_info 属性，从中获取
            if hasattr(self, 'model_info') and self.model_info:
                if hasattr(self.model_info, 'input_size') and self.model_info.input_size:
                    size = self.model_info.input_size
                    if isinstance(size, (tuple, list)) and len(size) >= 2:
                        return tuple(size[:2])
            
            # 3. 尝试从模型管理器获取
            if hasattr(self, 'model_manager') and hasattr(self, 'model_code'):
                model_info = self.model_manager.get_model_info(self.model_code)
                if model_info and hasattr(model_info, 'input_size') and model_info.input_size:
                    size = model_info.input_size
                    if isinstance(size, (tuple, list)) and len(size) >= 2:
                        return tuple(size[:2])
            
            # 4. 如果都没有，使用默认尺寸
            self.logger.warning(f"无法获取输入尺寸，使用默认尺寸: {default_size}")
            return default_size
            
        except Exception as e:
            self.logger.warning(f"获取输入尺寸失败，使用默认尺寸: {e}")
            return default_size
    
    def validate_config(self, config: AnalysisConfig) -> bool:
        """验证配置
        
        Args:
            config: 分析配置
            
        Returns:
            bool: 配置是否有效
        """
        if not config:
            return True
        
        # 验证置信度阈值
        if not (0.0 <= config.confidence <= 1.0):
            self.logger.error(f"无效的置信度阈值: {config.confidence}")
            return False
        
        # 验证IoU阈值
        if not (0.0 <= config.iou_threshold <= 1.0):
            self.logger.error(f"无效的IoU阈值: {config.iou_threshold}")
            return False
        
        # 验证批处理大小
        if config.batch_size < 1:
            self.logger.error(f"无效的批处理大小: {config.batch_size}")
            return False
        
        return True
    
    def preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """预处理帧
        
        Args:
            frame: 输入帧
            target_size: 目标尺寸 (width, height)
            
        Returns:
            np.ndarray: 预处理后的帧
        """
        if target_size:
            import cv2
            frame = cv2.resize(frame, target_size)
        
        # 确保帧格式正确
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def postprocess_results(self, results: Dict[str, Any], config: AnalysisConfig = None) -> Dict[str, Any]:
        """后处理结果
        
        Args:
            results: 原始结果
            config: 分析配置
            
        Returns:
            Dict[str, Any]: 后处理后的结果
        """
        # 添加时间戳
        results['timestamp'] = datetime.now().isoformat()
        
        # 添加分析器信息
        results['analyzer_info'] = {
            'type': self.analyzer_type,
            'model_code': self.model_code,
            'device': self.device
        }
        
        # 如果有配置，添加配置信息
        if config:
            results['config'] = {
                'confidence': config.confidence,
                'iou_threshold': config.iou_threshold,
                'classes': config.classes
            }
        
        return results
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("清理分析器资源")
        self.loaded = False
        self.reset_stats()
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(type={self.analyzer_type}, model={self.model_code}, device={self.device})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"{self.__class__.__name__}("
                f"analyzer_type='{self.analyzer_type}', "
                f"model_code='{self.model_code}', "
                f"device='{self.device}', "
                f"loaded={self.loaded})")


class DetectionAnalyzer(BaseAnalyzer):
    """目标检测分析器基类"""
    
    def get_analysis_type(self) -> AnalysisTypeEnum:
        """获取分析类型"""
        return AnalysisTypeEnum.DETECTION
    
    def filter_detections_by_confidence(self, detections: List[Dict], confidence_threshold: float) -> List[Dict]:
        """根据置信度过滤检测结果
        
        Args:
            detections: 检测结果列表
            confidence_threshold: 置信度阈值
            
        Returns:
            List[Dict]: 过滤后的检测结果
        """
        return [det for det in detections if det.get('confidence', 0) >= confidence_threshold]
    
    def filter_detections_by_classes(self, detections: List[Dict], target_classes: List[str]) -> List[Dict]:
        """根据类别过滤检测结果
        
        Args:
            detections: 检测结果列表
            target_classes: 目标类别列表
            
        Returns:
            List[Dict]: 过滤后的检测结果
        """
        if not target_classes:
            return detections
        
        return [det for det in detections if det.get('class_name') in target_classes]
    
    def apply_nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """应用非极大值抑制
        
        Args:
            detections: 检测结果列表
            iou_threshold: IoU阈值
            
        Returns:
            List[Dict]: NMS后的检测结果
        """
        # 这里应该实现实际的NMS算法
        # 暂时返回原始结果
        return detections


class ClassificationAnalyzer(BaseAnalyzer):
    """图像分类分析器基类"""
    
    def get_analysis_type(self) -> AnalysisTypeEnum:
        """获取分析类型"""
        return AnalysisTypeEnum.CLASSIFICATION
    
    def get_top_k_predictions(self, predictions: List[Dict], k: int = 5) -> List[Dict]:
        """获取Top-K预测结果
        
        Args:
            predictions: 预测结果列表
            k: 返回的数量
            
        Returns:
            List[Dict]: Top-K预测结果
        """
        sorted_predictions = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)
        return sorted_predictions[:k]
