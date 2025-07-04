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
from .analyzer_registry import AnalyzerRegistry
from app.models.base_model import AnalysisTypeEnum, DeviceTypeEnum
from app.exceptions.custom_exceptions import ConfigurationException


class AnalyzerFactory:
    """分析器工厂"""
    
    def __init__(self):
        """初始化分析器工厂"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry = AnalyzerRegistry()
        self.analyzer_instances: Dict[str, BaseAnalyzer] = {}
        self._initialized = False
    
    def initialize(self):
        """初始化工厂"""
        if self._initialized:
            return
        
        # 注册默认分析器
        self._register_default_analyzers()
        self._initialized = True
        self.logger.info("分析器工厂初始化完成")
    
    def _register_default_analyzers(self):
        """注册默认分析器"""
        # 这里可以注册默认的分析器实现
        # 由于我们没有具体的分析器实现，暂时跳过
        pass
    
    def register_analyzer(self, analysis_type: AnalysisTypeEnum, analyzer_class: Type[BaseAnalyzer]):
        """注册分析器类
        
        Args:
            analysis_type: 分析类型
            analyzer_class: 分析器类
        """
        self.registry.register(analysis_type, analyzer_class)
        self.logger.info(f"注册分析器: {analysis_type} -> {analyzer_class.__name__}")
    
    def unregister_analyzer(self, analysis_type: AnalysisTypeEnum):
        """注销分析器类
        
        Args:
            analysis_type: 分析类型
        """
        self.registry.unregister(analysis_type)
        self.logger.info(f"注销分析器: {analysis_type}")
    
    async def create_analyzer(
        self,
        analysis_type: AnalysisTypeEnum,
        model_code: str,
        device: DeviceTypeEnum = DeviceTypeEnum.AUTO,
        **kwargs
    ) -> BaseAnalyzer:
        """创建分析器实例
        
        Args:
            analysis_type: 分析类型
            model_code: 模型代码
            device: 设备类型
            **kwargs: 其他参数
            
        Returns:
            BaseAnalyzer: 分析器实例
        """
        # 确保工厂已初始化
        if not self._initialized:
            self.initialize()
        
        # 生成实例键
        instance_key = f"{analysis_type}_{model_code}_{device}"
        
        # 检查是否已有实例
        if instance_key in self.analyzer_instances:
            analyzer = self.analyzer_instances[instance_key]
            if analyzer.is_loaded():
                self.logger.info(f"复用已有分析器实例: {instance_key}")
                return analyzer
        
        # 获取分析器类
        analyzer_class = self.registry.get(analysis_type)
        if not analyzer_class:
            raise ConfigurationException(f"未找到分析器类型: {analysis_type}")
        
        # 创建分析器实例
        try:
            analyzer = analyzer_class(
                model_code=model_code,
                device=device.value,
                **kwargs
            )
            
            # 加载模型
            success = await analyzer.load_model(model_code)
            if not success:
                raise ConfigurationException(f"加载模型失败: {model_code}")
            
            # 缓存实例
            self.analyzer_instances[instance_key] = analyzer
            
            self.logger.info(f"创建分析器实例成功: {instance_key}")
            return analyzer
            
        except Exception as e:
            self.logger.error(f"创建分析器实例失败: {e}")
            raise ConfigurationException(f"创建分析器失败: {str(e)}")
    
    def get_analyzer(self, analysis_type: AnalysisTypeEnum, model_code: str, device: DeviceTypeEnum = DeviceTypeEnum.AUTO) -> Optional[BaseAnalyzer]:
        """获取已创建的分析器实例
        
        Args:
            analysis_type: 分析类型
            model_code: 模型代码
            device: 设备类型
            
        Returns:
            Optional[BaseAnalyzer]: 分析器实例
        """
        instance_key = f"{analysis_type}_{model_code}_{device}"
        return self.analyzer_instances.get(instance_key)
    
    def remove_analyzer(self, analysis_type: AnalysisTypeEnum, model_code: str, device: DeviceTypeEnum = DeviceTypeEnum.AUTO):
        """移除分析器实例
        
        Args:
            analysis_type: 分析类型
            model_code: 模型代码
            device: 设备类型
        """
        instance_key = f"{analysis_type}_{model_code}_{device}"
        
        if instance_key in self.analyzer_instances:
            analyzer = self.analyzer_instances[instance_key]
            # 清理资源
            try:
                analyzer.cleanup()
            except Exception as e:
                self.logger.warning(f"清理分析器资源失败: {e}")
            
            del self.analyzer_instances[instance_key]
            self.logger.info(f"移除分析器实例: {instance_key}")
    
    def get_supported_types(self) -> List[AnalysisTypeEnum]:
        """获取支持的分析类型
        
        Returns:
            List[AnalysisTypeEnum]: 支持的分析类型列表
        """
        return self.registry.get_registered_types()
    
    def get_analyzer_info(self, analysis_type: AnalysisTypeEnum) -> Optional[Dict[str, Any]]:
        """获取分析器信息
        
        Args:
            analysis_type: 分析类型
            
        Returns:
            Optional[Dict[str, Any]]: 分析器信息
        """
        analyzer_class = self.registry.get(analysis_type)
        if not analyzer_class:
            return None
        
        return {
            'analysis_type': analysis_type,
            'class_name': analyzer_class.__name__,
            'module': analyzer_class.__module__,
            'supported_models': []  # 这里可以添加支持的模型列表
        }
    
    def get_all_analyzer_info(self) -> List[Dict[str, Any]]:
        """获取所有分析器信息
        
        Returns:
            List[Dict[str, Any]]: 所有分析器信息列表
        """
        info_list = []
        for analysis_type in self.get_supported_types():
            info = self.get_analyzer_info(analysis_type)
            if info:
                info_list.append(info)
        return info_list
    
    def get_instance_stats(self) -> Dict[str, Any]:
        """获取实例统计信息
        
        Returns:
            Dict[str, Any]: 实例统计信息
        """
        total_instances = len(self.analyzer_instances)
        loaded_instances = sum(1 for analyzer in self.analyzer_instances.values() if analyzer.is_loaded())
        
        # 按类型统计
        type_stats = {}
        for instance_key, analyzer in self.analyzer_instances.items():
            analysis_type = analyzer.analyzer_type
            if analysis_type not in type_stats:
                type_stats[analysis_type] = 0
            type_stats[analysis_type] += 1
        
        # 按设备统计
        device_stats = {}
        for analyzer in self.analyzer_instances.values():
            device = analyzer.device
            if device not in device_stats:
                device_stats[device] = 0
            device_stats[device] += 1
        
        return {
            'total_instances': total_instances,
            'loaded_instances': loaded_instances,
            'type_distribution': type_stats,
            'device_distribution': device_stats,
            'instance_keys': list(self.analyzer_instances.keys())
        }
    
    async def cleanup_all(self):
        """清理所有分析器实例"""
        self.logger.info("开始清理所有分析器实例")
        
        for instance_key, analyzer in list(self.analyzer_instances.items()):
            try:
                await analyzer.cleanup()
                self.logger.info(f"清理分析器实例: {instance_key}")
            except Exception as e:
                self.logger.error(f"清理分析器实例失败 {instance_key}: {e}")
        
        self.analyzer_instances.clear()
        self.logger.info("所有分析器实例清理完成")
    
    async def warmup_all(self, warmup_frames: int = 5):
        """预热所有分析器实例
        
        Args:
            warmup_frames: 预热帧数
        """
        self.logger.info(f"开始预热所有分析器实例，预热帧数: {warmup_frames}")
        
        for instance_key, analyzer in self.analyzer_instances.items():
            try:
                if analyzer.is_loaded():
                    await analyzer.warmup(warmup_frames)
                    self.logger.info(f"预热分析器实例: {instance_key}")
            except Exception as e:
                self.logger.error(f"预热分析器实例失败 {instance_key}: {e}")
        
        self.logger.info("所有分析器实例预热完成")


# 全局分析器工厂实例
_analyzer_factory = None


def get_analyzer_factory() -> AnalyzerFactory:
    """获取全局分析器工厂实例
    
    Returns:
        AnalyzerFactory: 分析器工厂实例
    """
    global _analyzer_factory
    if _analyzer_factory is None:
        _analyzer_factory = AnalyzerFactory()
        _analyzer_factory.initialize()
    return _analyzer_factory


async def create_analyzer(
    analysis_type: AnalysisTypeEnum,
    model_code: str,
    device: DeviceTypeEnum = DeviceTypeEnum.AUTO,
    **kwargs
) -> BaseAnalyzer:
    """创建分析器的便捷函数
    
    Args:
        analysis_type: 分析类型
        model_code: 模型代码
        device: 设备类型
        **kwargs: 其他参数
        
    Returns:
        BaseAnalyzer: 分析器实例
    """
    factory = get_analyzer_factory()
    return await factory.create_analyzer(analysis_type, model_code, device, **kwargs)


def register_analyzer(analysis_type: AnalysisTypeEnum, analyzer_class: Type[BaseAnalyzer]):
    """注册分析器的便捷函数
    
    Args:
        analysis_type: 分析类型
        analyzer_class: 分析器类
    """
    factory = get_analyzer_factory()
    factory.register_analyzer(analysis_type, analyzer_class)
