"""
分析器工厂模块
负责创建和管理分析器实例
"""
import threading
from typing import Dict, Any, Optional, Type
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class AnalyzerFactory:
    """分析器工厂类"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AnalyzerFactory, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化分析器工厂"""
        if self._initialized:
            return
            
        self._analyzers: Dict[str, Dict[str, Type]] = {}
        self._initialized = False
    
    def initialize(self) -> None:
        """初始化分析器工厂"""
        if self._initialized:
            return
            
        # 扫描并注册分析器
        from .discovery import discover_analyzers
        from .registry import AnalyzerRegistry
        
        # 先扫描和注册所有分析器
        discover_analyzers()
        
        # 从注册表中获取分析器
        registry_analyzers = AnalyzerRegistry.list_analyzers()
        for analysis_type, analyzer_names in registry_analyzers.items():
            if analysis_type not in self._analyzers:
                self._analyzers[analysis_type] = {}
            for analyzer_name in analyzer_names:
                analyzer_class = AnalyzerRegistry.get_analyzer(analysis_type, analyzer_name)
                self._analyzers[analysis_type][analyzer_name] = analyzer_class
        
        self._initialized = True
    
    def register_analyzer(self, name: str, analyzer_type: str, analyzer_class: Type) -> None:
        """注册分析器类型
        
        Args:
            name: 分析器名称
            analyzer_type: 分析器类型
            analyzer_class: 分析器类
        """
        if analyzer_type not in self._analyzers:
            self._analyzers[analyzer_type] = {}
        self._analyzers[analyzer_type][name] = analyzer_class
    
    def create_analyzer(self, analyzer_type: str, name: str, config: Dict[str, Any]) -> Optional[Any]:
        """创建分析器实例
        
        Args:
            analyzer_type: 分析器类型
            name: 分析器名称
            config: 配置参数
            
        Returns:
            Optional[Any]: 分析器实例，如果创建失败则返回None
        """
        try:
            if not self._initialized:
                self.initialize()
                
            if analyzer_type not in self._analyzers:
                exception_logger.error(f"未知的分析器类型: {analyzer_type}")
                return None
                
            if name not in self._analyzers[analyzer_type]:
                exception_logger.error(f"未知的分析器名称: {name}")
                return None
                
            analyzer_class = self._analyzers[analyzer_type][name]
            return analyzer_class(config)
        except Exception as e:
            exception_logger.exception(f"创建分析器失败: {str(e)}")
            return None
    
    def get_analyzer_types(self) -> Dict[str, Dict[str, str]]:
        """获取所有已注册的分析器类型
        
        Returns:
            Dict[str, Dict[str, str]]: 分析器类型信息
        """
        if not self._initialized:
            self.initialize()
            
        return {
            analyzer_type: {
                name: analyzer_class.__name__ 
                for name, analyzer_class in analyzers.items()
            }
            for analyzer_type, analyzers in self._analyzers.items()
        }

# 工厂实例
analyzer_factory = AnalyzerFactory()
