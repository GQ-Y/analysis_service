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

    def apply_nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """应用非极大值抑制
        
        Args:
            detections: 检测结果列表，每个检测包含bbox和confidence
            iou_threshold: IoU阈值
            
        Returns:
            List[Dict]: NMS后的检测结果
        """
        if not detections:
            return []
        
        # 按置信度排序（降序）
        sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # 存储保留的检测结果
        keep_detections = []
        
        while sorted_detections:
            # 取出置信度最高的检测
            current_detection = sorted_detections.pop(0)
            keep_detections.append(current_detection)
            
            # 计算当前检测与剩余检测的IoU
            remaining_detections = []
            for detection in sorted_detections:
                iou = self._calculate_iou(current_detection, detection)
                
                # 如果IoU小于阈值，则保留该检测
                if iou < iou_threshold:
                    remaining_detections.append(detection)
                # 否则抑制该检测（不添加到remaining_detections中）
            
            sorted_detections = remaining_detections
        
        return keep_detections
    
    def _calculate_iou(self, det1: Dict, det2: Dict) -> float:
        """计算两个检测框的IoU
        
        Args:
            det1: 第一个检测结果
            det2: 第二个检测结果
            
        Returns:
            float: IoU值
        """
        try:
            # 提取边界框坐标
            box1 = det1.get('bbox', {})
            box2 = det2.get('bbox', {})
            
            x1_1, y1_1, x2_1, y2_1 = box1.get('x1', 0), box1.get('y1', 0), box1.get('x2', 0), box1.get('y2', 0)
            x1_2, y1_2, x2_2, y2_2 = box2.get('x1', 0), box2.get('y1', 0), box2.get('x2', 0), box2.get('y2', 0)
            
            # 计算交集区域
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            # 如果没有交集，返回0
            if x1_inter >= x2_inter or y1_inter >= y2_inter:
                return 0.0
            
            # 计算交集面积
            intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # 计算两个框的面积
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            
            # 计算并集面积
            union_area = area1 + area2 - intersection_area
            
            # 避免除零
            if union_area <= 0:
                return 0.0
            
            # 计算IoU
            iou = intersection_area / union_area
            return iou
            
        except Exception as e:
            self.logger.warning(f"计算IoU时出错: {e}")
            return 0.0
    
    def apply_class_specific_nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """应用按类别分组的NMS
        
        Args:
            detections: 检测结果列表
            iou_threshold: IoU阈值
            
        Returns:
            List[Dict]: 按类别应用NMS后的检测结果
        """
        if not detections:
            return []
        
        # 按类别分组
        class_groups = {}
        for detection in detections:
            class_id = detection.get('class_id', 0)
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(detection)
        
        # 对每个类别单独应用NMS
        final_detections = []
        for class_id, class_detections in class_groups.items():
            nms_detections = self.apply_nms(class_detections, iou_threshold)
            final_detections.extend(nms_detections)
        
        # 按置信度重新排序
        final_detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return final_detections

    def __len__(self) -> int:
        """返回注册的分析器数量"""
        return len(self._analyzers)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            'total_frames': self.stats['total_frames'],
            'total_time': self.stats['total_time'],
            'avg_fps': self.stats['avg_fps'],
            'last_inference_time': self.stats['last_inference_time'],
            'analyzer_type': self.analyzer_type.value if hasattr(self.analyzer_type, 'value') else str(self.analyzer_type),
            'device': self.device,
            'loaded': self.loaded
        }
    
    def update_statistics(self, inference_time: float):
        """更新性能统计信息"""
        self.stats['total_frames'] += 1
        self.stats['total_time'] += inference_time
        self.stats['last_inference_time'] = inference_time
        if self.stats['total_time'] > 0:
            self.stats['avg_fps'] = self.stats['total_frames'] / self.stats['total_time']
    
    # ===== 统一接口调用方法 =====
    
    def unified_analyze_frame(self, frame_data: Union[np.ndarray, Any], config: AnalysisConfig = None) -> Dict[str, Any]:
        """
        统一的单帧分析接口
        
        自动适配不同的分析器实现方式，按优先级尝试调用：
        1. analyze_frame() - 标准抽象方法
        2. detect() - YOLO等检测器的传统方法
        3. process_frame() - 兼容性方法
        
        Args:
            frame_data: 输入帧数据（可能是numpy数组或帧缓冲区）
            config: 分析配置
            
        Returns:
            Dict[str, Any]: 统一格式的分析结果
        """
        start_time = time.time()
        
        try:
            result = None
            method_used = None
            
            # 优先级1: 标准的analyze_frame方法
            if hasattr(self, 'analyze_frame') and callable(getattr(self, 'analyze_frame')):
                try:
                    # 检查是否是协程函数
                    import asyncio
                    if asyncio.iscoroutinefunction(self.analyze_frame):
                        # 协程函数处理
                        result = asyncio.run(self.analyze_frame(frame_data, config))
                    else:
                        # 同步方法
                        result = self.analyze_frame(frame_data, config)
                    method_used = "analyze_frame"
                except Exception as e:
                    self.logger.debug(f"analyze_frame方法调用失败: {e}")
            
            # 优先级2: 检测器的detect方法
            if result is None and hasattr(self, 'detect') and callable(getattr(self, 'detect')):
                try:
                    # 如果frame_data是帧缓冲区，提取图像数据
                    image_data = self._extract_image_from_frame_data(frame_data)
                    result = self.detect(image_data)
                    method_used = "detect"
                except Exception as e:
                    self.logger.debug(f"detect方法调用失败: {e}")
            
            # 优先级3: 兼容性的process_frame方法
            if result is None and hasattr(self, 'process_frame') and callable(getattr(self, 'process_frame')):
                try:
                    result = self.process_frame(frame_data)
                    method_used = "process_frame"
                except Exception as e:
                    self.logger.debug(f"process_frame方法调用失败: {e}")
            
            # 如果所有方法都失败，返回错误结果
            if result is None:
                result = {
                    "error": "没有可用的分析方法",
                    "detections": [],
                    "inference_time": 0.0,
                    "analyzer_type": self.analyzer_type.value if hasattr(self.analyzer_type, 'value') else str(self.analyzer_type)
                }
                method_used = "none"
            
            # 标准化结果格式
            result = self._standardize_result(result)
            
            # 更新统计信息
            inference_time = time.time() - start_time
            self.update_statistics(inference_time)
            
            # 添加元数据
            result.update({
                "method_used": method_used,
                "analyzer_type": self.analyzer_type.value if hasattr(self.analyzer_type, 'value') else str(self.analyzer_type),
                "device": self.device,
                "total_inference_time": inference_time
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"统一分析接口调用失败: {e}")
            return {
                "error": str(e),
                "detections": [],
                "inference_time": 0.0,
                "method_used": "error",
                "analyzer_type": self.analyzer_type.value if hasattr(self.analyzer_type, 'value') else str(self.analyzer_type)
            }
    
    def unified_analyze_batch(self, frame_data_list: List[Union[np.ndarray, Any]], config: AnalysisConfig = None) -> List[Dict[str, Any]]:
        """
        统一的批量分析接口
        
        自动适配不同的分析器实现方式，按优先级尝试调用：
        1. analyze_batch() - 标准抽象方法
        2. batch_detect() - YOLO等检测器的批处理方法
        3. process_batch() - 兼容性方法
        4. 逐帧调用unified_analyze_frame() - 回退方案
        
        Args:
            frame_data_list: 输入帧数据列表
            config: 分析配置
            
        Returns:
            List[Dict[str, Any]]: 统一格式的分析结果列表
        """
        start_time = time.time()
        
        try:
            results = None
            method_used = None
            
            # 优先级1: 标准的analyze_batch方法
            if hasattr(self, 'analyze_batch') and callable(getattr(self, 'analyze_batch')):
                try:
                    # 检查是否是协程函数
                    import asyncio
                    if asyncio.iscoroutinefunction(self.analyze_batch):
                        # 协程函数处理
                        results = asyncio.run(self.analyze_batch(frame_data_list, config))
                    else:
                        # 同步方法
                        results = self.analyze_batch(frame_data_list, config)
                    method_used = "analyze_batch"
                except Exception as e:
                    self.logger.debug(f"analyze_batch方法调用失败: {e}")
            
            # 优先级2: 检测器的batch_detect方法
            if results is None and hasattr(self, 'batch_detect') and callable(getattr(self, 'batch_detect')):
                try:
                    # 如果frame_data_list包含帧缓冲区，提取图像数据
                    image_data_list = [self._extract_image_from_frame_data(frame_data) for frame_data in frame_data_list]
                    results = self.batch_detect(image_data_list)
                    method_used = "batch_detect"
                except Exception as e:
                    self.logger.debug(f"batch_detect方法调用失败: {e}")
            
            # 优先级3: 兼容性的process_batch方法
            if results is None and hasattr(self, 'process_batch') and callable(getattr(self, 'process_batch')):
                try:
                    results = self.process_batch(frame_data_list)
                    method_used = "process_batch"
                except Exception as e:
                    self.logger.debug(f"process_batch方法调用失败: {e}")
            
            # 优先级4: 回退到逐帧处理
            if results is None:
                try:
                    results = []
                    for frame_data in frame_data_list:
                        result = self.unified_analyze_frame(frame_data, config)
                        results.append(result)
                    method_used = "fallback_frame_by_frame"
                except Exception as e:
                    self.logger.debug(f"逐帧处理回退失败: {e}")
            
            # 如果所有方法都失败，返回错误结果
            if results is None:
                results = []
                for i, frame_data in enumerate(frame_data_list):
                    results.append({
                        "error": "没有可用的批处理方法",
                        "detections": [],
                        "inference_time": 0.0,
                        "batch_index": i,
                        "analyzer_type": self.analyzer_type.value if hasattr(self.analyzer_type, 'value') else str(self.analyzer_type)
                    })
                method_used = "none"
            
            # 标准化结果格式
            results = [self._standardize_result(result) for result in results]
            
            # 更新统计信息
            total_inference_time = time.time() - start_time
            self.update_statistics(total_inference_time)
            
            # 添加批处理元数据
            for i, result in enumerate(results):
                result.update({
                    "method_used": method_used,
                    "analyzer_type": self.analyzer_type.value if hasattr(self.analyzer_type, 'value') else str(self.analyzer_type),
                    "device": self.device,
                    "batch_index": i,
                    "batch_size": len(frame_data_list),
                    "batch_total_time": total_inference_time
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"统一批处理接口调用失败: {e}")
            return [{
                "error": str(e),
                "detections": [],
                "inference_time": 0.0,
                "method_used": "error",
                "batch_index": i,
                "analyzer_type": self.analyzer_type.value if hasattr(self.analyzer_type, 'value') else str(self.analyzer_type)
            } for i in range(len(frame_data_list))]
    
    def _extract_image_from_frame_data(self, frame_data: Union[np.ndarray, Any]) -> np.ndarray:
        """
        从帧数据中提取图像数组
        
        Args:
            frame_data: 帧数据（可能是numpy数组或帧缓冲区）
            
        Returns:
            np.ndarray: 图像数组
        """
        if isinstance(frame_data, np.ndarray):
            return frame_data
        
        # 尝试从帧缓冲区提取图像
        if hasattr(frame_data, 'get_frame_view'):
            return frame_data.get_frame_view()
        elif hasattr(frame_data, 'frame_data'):
            return frame_data.frame_data
        elif hasattr(frame_data, 'get_frame_copy'):
            return frame_data.get_frame_copy()
        else:
            raise ValueError(f"无法从帧数据中提取图像: {type(frame_data)}")
    
    def _standardize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化分析结果格式
        
        Args:
            result: 原始分析结果
            
        Returns:
            Dict[str, Any]: 标准化后的结果
        """
        if not isinstance(result, dict):
            return {
                "error": "分析结果格式无效",
                "detections": [],
                "inference_time": 0.0
            }
        
        # 确保必要字段存在
        standardized = {
            "detections": result.get("detections", []),
            "inference_time": result.get("inference_time", 0.0),
            "confidence": result.get("confidence", 0.0),
            "model_name": result.get("model_name", "unknown"),
            "image_shape": result.get("image_shape", None)
        }
        
        # 保留其他字段
        for key, value in result.items():
            if key not in standardized:
                standardized[key] = value
        
        return standardized


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
