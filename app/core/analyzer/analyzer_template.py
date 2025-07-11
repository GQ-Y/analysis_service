#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析器模板
提供通用的分析器实现模板，帮助开发者快速创建新的分析器
"""

import logging
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod

from .base_analyzer import BaseAnalyzer
from .analyzer_registry import register_analyzer
from app.models.base_model import AnalysisTypeEnum, AnalysisConfig


class AnalyzerTemplate(BaseAnalyzer):
    """分析器模板基类
    
    提供常用的分析器功能实现，子类只需要实现核心的分析逻辑
    """
    
    def __init__(self, model_code: str = None, **kwargs):
        """初始化分析器模板
        
        Args:
            model_code: 模型代码
            **kwargs: 其他参数
        """
        super().__init__(model_code, **kwargs)
        
        # 模板特定配置
        self.template_version = "1.0.0"
        self.supports_batch = kwargs.get('supports_batch', True)
        self.supports_async = kwargs.get('supports_async', True)
        self.max_batch_size = kwargs.get('max_batch_size', 32)
        
        # 预处理和后处理配置
        self.preprocess_config = kwargs.get('preprocess_config', {})
        self.postprocess_config = kwargs.get('postprocess_config', {})
        
        # 性能监控
        self.enable_profiling = kwargs.get('enable_profiling', False)
        self.profiling_data = {}
        
        self.logger.info(f"分析器模板初始化完成: {self.__class__.__name__}")
    
    @abstractmethod
    def get_analysis_type(self) -> AnalysisTypeEnum:
        """获取分析类型（子类必须实现）"""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表（子类必须实现）"""
        pass
    
    @abstractmethod
    def _load_model_impl(self, model_code: str) -> bool:
        """加载模型的具体实现（子类必须实现）
        
        Args:
            model_code: 模型代码
            
        Returns:
            bool: 是否加载成功
        """
        pass
    
    @abstractmethod
    def _analyze_impl(self, data: Any, config: Optional[AnalysisConfig] = None) -> Dict[str, Any]:
        """分析的具体实现（子类必须实现）
        
        Args:
            data: 输入数据
            config: 分析配置
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        pass
    
    def load_model(self, model_code: str) -> bool:
        """加载模型（模板实现）
        
        Args:
            model_code: 模型代码
            
        Returns:
            bool: 是否加载成功
        """
        try:
            self.logger.info(f"开始加载模型: {model_code}")
            
            # 验证模型代码
            if not self._validate_model_code(model_code):
                return False
            
            # 调用子类实现
            success = self._load_model_impl(model_code)
            
            if success:
                self.model_code = model_code
                self.loaded = True
                self.logger.info(f"模型加载成功: {model_code}")
            else:
                self.logger.error(f"模型加载失败: {model_code}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"加载模型异常 {model_code}: {e}")
            return False
    
    def _validate_model_code(self, model_code: str) -> bool:
        """验证模型代码
        
        Args:
            model_code: 模型代码
            
        Returns:
            bool: 是否有效
        """
        if not model_code:
            self.logger.error("模型代码为空")
            return False
        
        supported_models = self.get_supported_models()
        if supported_models and model_code not in supported_models:
            self.logger.error(f"不支持的模型: {model_code}, 支持的模型: {supported_models}")
            return False
        
        return True
    
    def preprocess(self, data: Any, config: Optional[AnalysisConfig] = None) -> Any:
        """预处理数据
        
        Args:
            data: 原始数据
            config: 分析配置
            
        Returns:
            Any: 预处理后的数据
        """
        # 子类可以重写此方法实现自定义预处理
        return data
    
    def postprocess(self, result: Dict[str, Any], config: Optional[AnalysisConfig] = None) -> Dict[str, Any]:
        """后处理结果
        
        Args:
            result: 原始结果
            config: 分析配置
            
        Returns:
            Dict[str, Any]: 后处理后的结果
        """
        # 子类可以重写此方法实现自定义后处理
        return result
    
    def analyze_frame(self, frame_data: Any, config: Optional[AnalysisConfig] = None) -> Dict[str, Any]:
        """分析单帧（模板实现）
        
        Args:
            frame_data: 帧数据
            config: 分析配置
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        if not self.loaded:
            return self._create_error_result("模型未加载")
        
        try:
            # 开始性能监控
            if self.enable_profiling:
                import time
                start_time = time.time()
            
            # 预处理
            processed_data = self.preprocess(frame_data, config)
            
            # 分析
            result = self._analyze_impl(processed_data, config)
            
            # 后处理
            final_result = self.postprocess(result, config)
            
            # 结束性能监控
            if self.enable_profiling:
                end_time = time.time()
                self._update_profiling_data('analyze_frame', end_time - start_time)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"分析单帧异常: {e}")
            return self._create_error_result(f"分析异常: {e}")
    
    def analyze_batch(self, frame_data_list: List[Any], config: Optional[AnalysisConfig] = None) -> List[Dict[str, Any]]:
        """分析批量数据（模板实现）
        
        Args:
            frame_data_list: 帧数据列表
            config: 分析配置
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        if not self.loaded:
            return [self._create_error_result("模型未加载") for _ in frame_data_list]
        
        if not self.supports_batch:
            # 如果不支持批处理，逐个处理
            return [self.analyze_frame(frame_data, config) for frame_data in frame_data_list]
        
        try:
            # 开始性能监控
            if self.enable_profiling:
                import time
                start_time = time.time()
            
            # 批量预处理
            processed_data_list = [self.preprocess(data, config) for data in frame_data_list]
            
            # 批量分析（子类可以重写实现真正的批量处理）
            results = self._analyze_batch_impl(processed_data_list, config)
            
            # 批量后处理
            final_results = [self.postprocess(result, config) for result in results]
            
            # 结束性能监控
            if self.enable_profiling:
                end_time = time.time()
                self._update_profiling_data('analyze_batch', end_time - start_time)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"批量分析异常: {e}")
            return [self._create_error_result(f"分析异常: {e}") for _ in frame_data_list]
    
    def _analyze_batch_impl(self, data_list: List[Any], config: Optional[AnalysisConfig] = None) -> List[Dict[str, Any]]:
        """批量分析的具体实现（子类可以重写）
        
        Args:
            data_list: 数据列表
            config: 分析配置
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        # 默认实现：逐个调用单个分析
        return [self._analyze_impl(data, config) for data in data_list]
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """创建错误结果
        
        Args:
            error_message: 错误消息
            
        Returns:
            Dict[str, Any]: 错误结果
        """
        return {
            'success': False,
            'error': error_message,
            'timestamp': self._get_current_timestamp(),
            'analyzer_type': self.get_analysis_type().value,
            'model_code': self.model_code
        }
    
    def _update_profiling_data(self, operation: str, duration: float):
        """更新性能监控数据
        
        Args:
            operation: 操作名称
            duration: 持续时间
        """
        if operation not in self.profiling_data:
            self.profiling_data[operation] = {
                'count': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0
            }
        
        data = self.profiling_data[operation]
        data['count'] += 1
        data['total_time'] += duration
        data['avg_time'] = data['total_time'] / data['count']
        data['min_time'] = min(data['min_time'], duration)
        data['max_time'] = max(data['max_time'], duration)
    
    def get_profiling_data(self) -> Dict[str, Any]:
        """获取性能监控数据
        
        Returns:
            Dict[str, Any]: 性能监控数据
        """
        return self.profiling_data.copy()
    
    def reset_profiling_data(self):
        """重置性能监控数据"""
        self.profiling_data.clear()
    
    def _get_current_timestamp(self) -> str:
        """获取当前时间戳
        
        Returns:
            str: 时间戳字符串
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_template_info(self) -> Dict[str, Any]:
        """获取模板信息
        
        Returns:
            Dict[str, Any]: 模板信息
        """
        return {
            'template_version': self.template_version,
            'analyzer_class': self.__class__.__name__,
            'analysis_type': self.get_analysis_type().value,
            'supports_batch': self.supports_batch,
            'supports_async': self.supports_async,
            'max_batch_size': self.max_batch_size,
            'model_code': self.model_code,
            'loaded': self.loaded,
            'profiling_enabled': self.enable_profiling
        }


class SimpleAnalyzerTemplate(AnalyzerTemplate):
    """简单分析器模板
    
    提供最基本的分析器实现，适合快速原型开发
    """
    
    def __init__(self, analysis_type: AnalysisTypeEnum, supported_models: List[str] = None, **kwargs):
        """初始化简单分析器模板
        
        Args:
            analysis_type: 分析类型
            supported_models: 支持的模型列表
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.analysis_type_enum = analysis_type
        self.supported_models_list = supported_models or []
        
        # 简单模式配置
        self.mock_mode = kwargs.get('mock_mode', False)
        self.mock_delay = kwargs.get('mock_delay', 0.1)
    
    def get_analysis_type(self) -> AnalysisTypeEnum:
        """获取分析类型"""
        return self.analysis_type_enum
    
    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表"""
        return self.supported_models_list
    
    def _load_model_impl(self, model_code: str) -> bool:
        """加载模型的具体实现"""
        if self.mock_mode:
            self.logger.info(f"模拟模式：模型加载成功 {model_code}")
            return True
        
        # 子类需要实现实际的模型加载逻辑
        self.logger.warning(f"简单模板未实现模型加载逻辑: {model_code}")
        return False
    
    def _analyze_impl(self, data: Any, config: Optional[AnalysisConfig] = None) -> Dict[str, Any]:
        """分析的具体实现"""
        if self.mock_mode:
            import time
            time.sleep(self.mock_delay)
            
            return {
                'success': True,
                'mock_result': True,
                'analysis_type': self.analysis_type_enum.value,
                'model_code': self.model_code,
                'timestamp': self._get_current_timestamp()
            }
        
        # 子类需要实现实际的分析逻辑
        return self._create_error_result("简单模板未实现分析逻辑")


# 装饰器：快速创建分析器
def quick_analyzer(analysis_type: AnalysisTypeEnum, supported_models: List[str] = None):
    """快速创建分析器的装饰器
    
    Args:
        analysis_type: 分析类型
        supported_models: 支持的模型列表
    
    Returns:
        装饰器函数
    """
    def decorator(cls):
        # 确保类继承自AnalyzerTemplate
        if not issubclass(cls, AnalyzerTemplate):
            raise ValueError(f"类 {cls.__name__} 必须继承自 AnalyzerTemplate")
        
        # 自动注册分析器
        register_analyzer(analysis_type)(cls)
        
        # 添加默认的get_analysis_type方法
        if not hasattr(cls, 'get_analysis_type'):
            cls.get_analysis_type = lambda self: analysis_type
        
        # 添加默认的get_supported_models方法
        if not hasattr(cls, 'get_supported_models'):
            cls.get_supported_models = lambda self: supported_models or []
        
        return cls
    
    return decorator


# 示例：使用模板创建分析器
@quick_analyzer(AnalysisTypeEnum.DETECTION, ['example_model'])
class ExampleAnalyzer(AnalyzerTemplate):
    """示例分析器"""
    
    def _load_model_impl(self, model_code: str) -> bool:
        """加载模型"""
        # 实现模型加载逻辑
        self.logger.info(f"加载示例模型: {model_code}")
        return True
    
    def _analyze_impl(self, data: Any, config: Optional[AnalysisConfig] = None) -> Dict[str, Any]:
        """分析实现"""
        # 实现分析逻辑
        return {
            'success': True,
            'detections': [],
            'timestamp': self._get_current_timestamp()
        } 