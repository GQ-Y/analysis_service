"""
分析器注册表模块
管理所有已注册的分析器
"""
from typing import Dict, Any, Type, Optional, List
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

# 将在初始化时导入
BaseAnalyzer = None

class AnalyzerRegistry:
    """分析器注册表，管理所有已注册的分析器"""
    
    # 注册表存储结构: {analysis_type: {analyzer_name: analyzer_class}}
    _registry = {}
    
    @classmethod
    def register(cls, analysis_type: str, analyzer_class):
        """
        注册分析器类
        
        Args:
            analysis_type: 分析类型，如 "detection"
            analyzer_class: 分析器类
            
        Returns:
            analyzer_class: 返回原始类，方便装饰器使用
        """
        # 防止循环导入
        global BaseAnalyzer
        if BaseAnalyzer is None:
            from core.analyzer.base_analyzer import BaseAnalyzer as BA
            BaseAnalyzer = BA
            
        # 检查是否是BaseAnalyzer的子类
        if not issubclass(analyzer_class, BaseAnalyzer):
            raise TypeError(f"只有BaseAnalyzer的子类才能被注册为分析器: {analyzer_class.__name__}")
            
        if analysis_type not in cls._registry:
            cls._registry[analysis_type] = {}
            
        # 使用类名作为分析器名称
        analyzer_name = analyzer_class.__name__
        cls._registry[analysis_type][analyzer_name] = analyzer_class
        normal_logger.info(f"已注册分析器: {analyzer_name} -> {analysis_type}")
        
        return analyzer_class
    
    @classmethod
    def get_analyzer(cls, analysis_type: str, analyzer_name: Optional[str] = None):
        """
        获取分析器类
        
        Args:
            analysis_type: 分析类型
            analyzer_name: 特定分析器名称，如果提供则使用指定的分析器实现
            
        Returns:
            分析器类
            
        Raises:
            ValueError: 当找不到匹配的分析器时
        """
        if analysis_type not in cls._registry:
            raise ValueError(f"不支持的分析类型: {analysis_type}")
            
        analyzers = cls._registry[analysis_type]
        if not analyzers:
            raise ValueError(f"分析类型 {analysis_type} 没有注册的分析器")
            
        # 如果指定了分析器名称，返回指定的分析器
        if analyzer_name and analyzer_name in analyzers:
            return analyzers[analyzer_name]
            
        # 否则返回第一个注册的分析器
        # 这里可以根据需要添加优先级逻辑
        return next(iter(analyzers.values()))
    
    @classmethod
    def list_analyzers(cls) -> Dict[str, List[str]]:
        """
        列出所有已注册的分析器
        
        Returns:
            Dict[str, List[str]]: {分析类型: [分析器名称列表]}
        """
        result = {}
        for analysis_type, analyzers in cls._registry.items():
            result[analysis_type] = list(analyzers.keys())
        return result


def register_analyzer(analysis_type: str):
    """
    分析器注册装饰器
    
    用法:
    @register_analyzer("detection")
    class MyDetectionAnalyzer(BaseAnalyzer):
        ...
        
    Args:
        analysis_type: 分析类型，如 "detection"
        
    Returns:
        装饰器函数
    """
    def decorator(cls):
        return AnalyzerRegistry.register(analysis_type, cls)
    return decorator 