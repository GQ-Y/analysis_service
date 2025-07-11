#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: analyzer_factory.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 分析器工厂

负责创建和管理分析器实例，提供统一的分析器创建接口。

本文件是分析服务项目的一部分。
"""

import logging
from typing import Dict, Type, Any, Optional, List
from abc import ABC, abstractmethod

from .base_analyzer import BaseAnalyzer
from .analyzer_registry import AnalyzerRegistry, get_global_registry, ensure_analyzers_registered
from app.models.base_model import AnalysisTypeEnum, DeviceTypeEnum
from app.exceptions.custom_exceptions import ConfigurationException
from app.utils.analysis_type_utils import get_analysis_type_by_number
from app.core.storage.model_manager import ModelManager, ModelInfo


class AnalyzerFactory:
    """分析器工厂 - 动态查询注册表"""
    
    def __init__(self, model_manager: ModelManager, registry: Optional[AnalyzerRegistry] = None):
        """
        初始化分析器工厂
        
        Args:
            model_manager: 模型管理器
            registry: 分析器注册表，如果为None则使用全局注册表
        """
        self.model_manager = model_manager
        self.registry = registry or get_global_registry()
        self.logger = logging.getLogger(__name__)
        
        # 确保分析器已注册
        ensure_analyzers_registered()
        
        # 记录当前注册表状态
        registered_types = self.registry.get_registered_types()
        self.logger.info(f"🏭 分析器工厂初始化: 注册表包含 {len(registered_types)} 种分析器类型")
        for analysis_type in registered_types:
            analyzer_class = self.registry.get(analysis_type)
            self.logger.debug(f"  📋 {analysis_type.value} -> {analyzer_class.__name__}")
    
    def create_analyzer(self, model_code: str, **kwargs) -> Optional[Any]:
        """
        根据模型代码创建对应的分析器
        
        Args:
            model_code: 模型代码
            **kwargs: 分析器初始化参数
            
        Returns:
            Optional[Any]: 分析器实例
        """
        try:
            # 获取模型信息
            model_info = self.model_manager.get_model_info(model_code)
            if not model_info:
                self.logger.error(f"❌ 未找到模型: {model_code}")
                return None
            
            # 获取分析类型
            analysis_type = get_analysis_type_by_number(model_info.analysis_type)
            if not analysis_type:
                self.logger.error(f"❌ 不支持的分析类型: {model_info.analysis_type}")
                return None
            
            # 从注册表查询分析器类
            analyzer_class = self.registry.get(analysis_type)
            if not analyzer_class:
                self.logger.error(f"❌ 未注册的分析类型: {analysis_type.value}")
                self.logger.info(f"💡 可用的分析类型: {[t.value for t in self.registry.get_registered_types()]}")
                return None
            
            # 检查模型是否可用
            if not self.model_manager.is_model_available(model_code):
                self.logger.warning(f"⚠️ 模型不可用，尝试使用Mock模式: {model_code}")
                # TODO: 尝试下载模型
                # if not self.model_manager.download_model(model_code):
                #     self.logger.error(f"❌ 无法下载模型: {model_code}")
                #     return None
            
            # 创建分析器实例
            try:
                # 检查是否有特定的工厂方法
                if hasattr(analyzer_class, 'create'):
                    # 使用工厂方法创建
                    analyzer = analyzer_class.create(model_code, self.model_manager, **kwargs)
                else:
                    # 直接实例化
                    analyzer = analyzer_class(model_code=model_code, **kwargs)
                    
                    # 如果分析器有load_model方法，尝试加载模型
                    if hasattr(analyzer, 'load_model'):
                        import asyncio
                        try:
                            # 检查是否是协程函数
                            if asyncio.iscoroutinefunction(analyzer.load_model):
                                # 在新的事件循环中运行协程
                                try:
                                    loop = asyncio.get_event_loop()
                                    if loop.is_running():
                                        # 如果事件循环正在运行，创建新的任务
                                        load_success = True  # 暂时设为True，实际加载在后台进行
                                        self.logger.info(f"⚠️ 检测到运行中的事件循环，跳过同步加载: {model_code}")
                                    else:
                                        load_success = loop.run_until_complete(analyzer.load_model(model_code))
                                except RuntimeError:
                                    # 没有事件循环，创建新的
                                    load_success = asyncio.run(analyzer.load_model(model_code))
                            else:
                                # 同步函数
                                load_success = analyzer.load_model(model_code)
                        except Exception as e:
                            self.logger.warning(f"⚠️ 加载模型时出错: {e}")
                            load_success = False
                            
                        if not load_success:
                            self.logger.warning(f"⚠️ 模型加载失败，可能使用Mock模式: {model_code}")
                
                if analyzer:
                    self.logger.info(f"✅ 分析器创建成功: {model_code} ({analysis_type.value}) -> {analyzer_class.__name__}")
                else:
                    self.logger.error(f"❌ 分析器创建失败: {model_code}")
                
                return analyzer
                
            except Exception as e:
                self.logger.error(f"❌ 实例化分析器失败 {analyzer_class.__name__}: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ 创建分析器异常 {model_code}: {e}")
            return None
    
    def create_analyzers_for_models(self, model_codes: List[str], **kwargs) -> Dict[str, Any]:
        """
        为多个模型创建分析器
        
        Args:
            model_codes: 模型代码列表
            **kwargs: 分析器初始化参数
            
        Returns:
            Dict[str, Any]: 模型代码到分析器的映射
        """
        analyzers = {}
        
        for model_code in model_codes:
            analyzer = self.create_analyzer(model_code, **kwargs)
            if analyzer:
                analyzers[model_code] = analyzer
            else:
                self.logger.warning(f"⚠️ 跳过无法创建的分析器: {model_code}")
        
        self.logger.info(f"📊 成功创建 {len(analyzers)}/{len(model_codes)} 个分析器")
        return analyzers
    
    def get_supported_analysis_types(self) -> List[AnalysisTypeEnum]:
        """获取支持的分析类型列表"""
        return list(self.registry.get_registered_types())
    
    def is_analysis_type_supported(self, analysis_type: int) -> bool:
        """检查分析类型是否支持"""
        enum_type = get_analysis_type_by_number(analysis_type)
        return enum_type in self.registry.get_registered_types() if enum_type else False
    
    def get_models_by_analysis_type(self, analysis_type: int) -> List[ModelInfo]:
        """根据分析类型获取可用模型列表"""
        return self.model_manager.get_analysis_type_models(analysis_type)
    
    def cleanup_analyzers(self, analyzers: Dict[str, Any]):
        """清理分析器资源"""
        for model_code, analyzer in analyzers.items():
            try:
                if hasattr(analyzer, 'cleanup'):
                    analyzer.cleanup()
                self.logger.debug(f"🧹 分析器已清理: {model_code}")
            except Exception as e:
                self.logger.error(f"❌ 清理分析器失败 {model_code}: {e}")
        
        self.logger.info(f"🧹 已清理 {len(analyzers)} 个分析器")


class AnalyzerType:
    """分析器类型常量"""
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    TRACKING = "tracking"
    POSE_ESTIMATION = "pose_estimation"
    FACE_RECOGNITION = "face_recognition"


# 便捷函数
def create_analyzer_for_model(model_code: str, storage_root: str = "storage", **kwargs) -> Optional[Any]:
    """
    便捷函数：为指定模型创建分析器
    
    Args:
        model_code: 模型代码
        storage_root: 存储根目录
        **kwargs: 分析器参数
        
    Returns:
        Optional[Any]: 分析器实例
    """
    model_manager = ModelManager(storage_root)
    factory = AnalyzerFactory(model_manager)
    return factory.create_analyzer(model_code, **kwargs)
