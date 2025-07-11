#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: analyzer_registry.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 分析器注册表

管理分析器类的注册和查找，提供分析器类型到实现类的映射。

本文件是分析服务项目的一部分。
"""

import logging
from typing import Dict, Type, Optional, List

from .base_analyzer import BaseAnalyzer
from app.models.base_model import AnalysisTypeEnum


class AnalyzerRegistry:
    """分析器注册表"""
    
    def __init__(self):
        """初始化分析器注册表"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._analyzers: Dict[AnalysisTypeEnum, Type[BaseAnalyzer]] = {}
    
    def register(self, analysis_type: AnalysisTypeEnum, analyzer_class: Type[BaseAnalyzer]):
        """注册分析器类
        
        Args:
            analysis_type: 分析类型
            analyzer_class: 分析器类
            
        Raises:
            ValueError: 如果分析器类无效
        """
        # 验证分析器类
        if not issubclass(analyzer_class, BaseAnalyzer):
            raise ValueError(f"分析器类必须继承自BaseAnalyzer: {analyzer_class}")
        
        # 检查是否已注册
        if analysis_type in self._analyzers:
            old_class = self._analyzers[analysis_type]
            self.logger.warning(f"覆盖已注册的分析器: {analysis_type} {old_class.__name__} -> {analyzer_class.__name__}")
        
        self._analyzers[analysis_type] = analyzer_class
        self.logger.info(f"注册分析器: {analysis_type} -> {analyzer_class.__name__}")
    
    def unregister(self, analysis_type: AnalysisTypeEnum):
        """注销分析器类
        
        Args:
            analysis_type: 分析类型
        """
        if analysis_type in self._analyzers:
            analyzer_class = self._analyzers[analysis_type]
            del self._analyzers[analysis_type]
            self.logger.info(f"注销分析器: {analysis_type} -> {analyzer_class.__name__}")
        else:
            self.logger.warning(f"尝试注销未注册的分析器: {analysis_type}")
    
    def get(self, analysis_type: AnalysisTypeEnum) -> Optional[Type[BaseAnalyzer]]:
        """获取分析器类
        
        Args:
            analysis_type: 分析类型
            
        Returns:
            Optional[Type[BaseAnalyzer]]: 分析器类，如果未找到则返回None
        """
        return self._analyzers.get(analysis_type)
    
    def is_registered(self, analysis_type: AnalysisTypeEnum) -> bool:
        """检查分析器类型是否已注册
        
        Args:
            analysis_type: 分析类型
            
        Returns:
            bool: 是否已注册
        """
        return analysis_type in self._analyzers
    
    def get_registered_types(self) -> List[AnalysisTypeEnum]:
        """获取所有已注册的分析器类型
        
        Returns:
            List[AnalysisTypeEnum]: 已注册的分析器类型列表
        """
        return list(self._analyzers.keys())
    
    def get_all_analyzers(self) -> Dict[AnalysisTypeEnum, Type[BaseAnalyzer]]:
        """获取所有已注册的分析器
        
        Returns:
            Dict[AnalysisTypeEnum, Type[BaseAnalyzer]]: 分析器类型到类的映射
        """
        return self._analyzers.copy()
    
    def clear(self):
        """清空所有注册的分析器"""
        count = len(self._analyzers)
        self._analyzers.clear()
        self.logger.info(f"清空所有注册的分析器，共{count}个")
    
    def get_registry_info(self) -> Dict[str, any]:
        """获取注册表信息
        
        Returns:
            Dict[str, any]: 注册表信息
        """
        analyzer_info = {}
        for analysis_type, analyzer_class in self._analyzers.items():
            analyzer_info[analysis_type.value] = {
                'class_name': analyzer_class.__name__,
                'module': analyzer_class.__module__,
                'doc': analyzer_class.__doc__
            }
        
        return {
            'total_registered': len(self._analyzers),
            'registered_types': [t.value for t in self._analyzers.keys()],
            'analyzers': analyzer_info
        }
    
    def validate_registry(self) -> List[str]:
        """验证注册表中的分析器类
        
        Returns:
            List[str]: 验证错误列表，如果为空则表示所有分析器都有效
        """
        errors = []
        
        for analysis_type, analyzer_class in self._analyzers.items():
            try:
                # 检查是否继承自BaseAnalyzer
                if not issubclass(analyzer_class, BaseAnalyzer):
                    errors.append(f"{analysis_type}: 分析器类{analyzer_class.__name__}未继承自BaseAnalyzer")
                
                # 检查是否实现了必需的抽象方法
                required_methods = ['get_analysis_type', 'load_model', 'analyze_frame', 'analyze_batch', 'get_supported_models']
                for method_name in required_methods:
                    if not hasattr(analyzer_class, method_name):
                        errors.append(f"{analysis_type}: 分析器类{analyzer_class.__name__}缺少方法{method_name}")
                    elif getattr(analyzer_class, method_name) is None:
                        errors.append(f"{analysis_type}: 分析器类{analyzer_class.__name__}的方法{method_name}未实现")
                
                # 尝试创建实例（不加载模型）
                try:
                    instance = analyzer_class()
                    # 检查分析类型是否匹配
                    if hasattr(instance, 'get_analysis_type'):
                        instance_type = instance.get_analysis_type()
                        if instance_type != analysis_type:
                            errors.append(f"{analysis_type}: 分析器实例返回的类型{instance_type}与注册类型不匹配")
                except Exception as e:
                    errors.append(f"{analysis_type}: 创建分析器实例失败: {str(e)}")
                
            except Exception as e:
                errors.append(f"{analysis_type}: 验证分析器类失败: {str(e)}")
        
        return errors
    
    def __len__(self) -> int:
        """返回注册的分析器数量"""
        return len(self._analyzers)
    
    def __contains__(self, analysis_type: AnalysisTypeEnum) -> bool:
        """检查分析器类型是否在注册表中"""
        return analysis_type in self._analyzers
    
    def __iter__(self):
        """迭代注册的分析器类型"""
        return iter(self._analyzers.keys())
    
    def __str__(self) -> str:
        """字符串表示"""
        types = [t.value for t in self._analyzers.keys()]
        return f"AnalyzerRegistry({len(self._analyzers)} types: {types})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"AnalyzerRegistry(analyzers={dict((k.value, v.__name__) for k, v in self._analyzers.items())})"


# 全局分析器注册表实例
_global_registry = AnalyzerRegistry()


def get_global_registry() -> AnalyzerRegistry:
    """获取全局分析器注册表
    
    Returns:
        AnalyzerRegistry: 全局分析器注册表实例
    """
    return _global_registry


def register_analyzer_globally(analysis_type: AnalysisTypeEnum, analyzer_class: Type[BaseAnalyzer]):
    """在全局注册表中注册分析器
    
    Args:
        analysis_type: 分析类型
        analyzer_class: 分析器类
    """
    _global_registry.register(analysis_type, analyzer_class)


def get_analyzer_class(analysis_type: AnalysisTypeEnum) -> Optional[Type[BaseAnalyzer]]:
    """从全局注册表获取分析器类
    
    Args:
        analysis_type: 分析类型
        
    Returns:
        Optional[Type[BaseAnalyzer]]: 分析器类
    """
    return _global_registry.get(analysis_type)


def is_analyzer_registered(analysis_type: AnalysisTypeEnum) -> bool:
    """检查分析器是否在全局注册表中注册
    
    Args:
        analysis_type: 分析类型
        
    Returns:
        bool: 是否已注册
    """
    return _global_registry.is_registered(analysis_type)


def get_all_registered_types() -> List[AnalysisTypeEnum]:
    """获取所有已注册的分析器类型
    
    Returns:
        List[AnalysisTypeEnum]: 已注册的分析器类型列表
    """
    return _global_registry.get_registered_types()


def register_analyzer(analysis_type: AnalysisTypeEnum):
    """分析器注册装饰器
    
    Args:
        analysis_type: 分析类型
        
    Returns:
        装饰器函数
    """
    def decorator(analyzer_class: Type[BaseAnalyzer]):
        """装饰器实现"""
        # 自动注册到全局注册表
        _global_registry.register(analysis_type, analyzer_class)
        return analyzer_class
    return decorator


def auto_discover_analyzers(package_path: str = "app.core.analyzer"):
    """自动发现并注册分析器
    
    Args:
        package_path: 要搜索的包路径
    """
    import importlib
    import pkgutil
    import inspect
    
    logger = logging.getLogger(__name__)
    
    try:
        # 导入包
        package = importlib.import_module(package_path)
        
        # 遍历包中的所有模块
        for importer, modname, ispkg in pkgutil.walk_packages(
            package.__path__, 
            package.__name__ + "."
        ):
            try:
                # 导入模块
                module = importlib.import_module(modname)
                
                # 检查模块中的所有类
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # 检查是否是BaseAnalyzer的子类且不是BaseAnalyzer本身
                    if (issubclass(obj, BaseAnalyzer) and 
                        obj != BaseAnalyzer and 
                        hasattr(obj, 'get_analysis_type')):
                        
                        try:
                            # 尝试获取分析类型
                            instance = obj()
                            analysis_type = instance.get_analysis_type()
                            
                            # 如果还没有注册，则注册
                            if not _global_registry.is_registered(analysis_type):
                                _global_registry.register(analysis_type, obj)
                                logger.info(f"🔍 自动发现并注册分析器: {analysis_type.value} -> {obj.__name__}")
                            else:
                                logger.debug(f"📋 分析器已注册: {analysis_type.value} -> {obj.__name__}")
                                
                        except Exception as e:
                            logger.debug(f"⚠️ 无法自动注册分析器 {obj.__name__}: {e}")
                            
            except Exception as e:
                logger.debug(f"⚠️ 导入模块失败 {modname}: {e}")
                
    except Exception as e:
        logger.error(f"❌ 自动发现分析器失败: {e}")


def ensure_analyzers_registered():
    """确保分析器已注册 - 在需要时调用"""
    if len(_global_registry) == 0:
        logging.getLogger(__name__).info("🔍 开始自动发现分析器...")
        auto_discover_analyzers()
        
        # 如果仍然没有注册任何分析器，手动注册已知的分析器
        if len(_global_registry) == 0:
            _register_known_analyzers()


def _register_known_analyzers():
    """手动注册已知的分析器"""
    logger = logging.getLogger(__name__)
    
    try:
        # 注册YOLO检测分析器
        from app.core.analyzer.detection.yolo_analyzer import YoloDetectionAnalyzer
        if not _global_registry.is_registered(AnalysisTypeEnum.DETECTION):
            _global_registry.register(AnalysisTypeEnum.DETECTION, YoloDetectionAnalyzer)
            logger.info("🔧 手动注册YOLO检测分析器")
            
    except ImportError as e:
        logger.warning(f"⚠️ 无法导入YOLO检测分析器: {e}")
    
    # TODO: 注册其他已知的分析器
    # try:
    #     from app.core.analyzer.classification.classification_analyzer import ClassificationAnalyzer
    #     if not _global_registry.is_registered(AnalysisTypeEnum.CLASSIFICATION):
    #         _global_registry.register(AnalysisTypeEnum.CLASSIFICATION, ClassificationAnalyzer)
    # except ImportError:
    #     pass
