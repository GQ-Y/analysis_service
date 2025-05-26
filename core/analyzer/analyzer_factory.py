"""
分析器工厂模块 - 重构版
负责创建不同类型的分析器实例，使用插件化的自动发现和注册机制
"""
from typing import Dict, Any, Optional, Union, Type

# 使用新的日志记录器
from shared.utils.logger import get_normal_logger, get_exception_logger
from core.analyzer.base_analyzer import BaseAnalyzer
from core.analyzer.registry import AnalyzerRegistry
from core.analyzer.discovery import discover_analyzers

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class AnalyzerFactory:
    """分析器工厂类 - 重构版，使用注册表"""

    # 是否已初始化
    _initialized = False

    # 分析类型名称映射表
    ANALYSIS_TYPE_MAP = {
        "detection": "detection"
    }
    
    @classmethod
    def initialize(cls):
        """初始化工厂，扫描并加载所有可用的分析器"""
        if not cls._initialized:
            discover_analyzers()
            cls._initialized = True
            
            # 输出可用的分析器列表
            analyzers = AnalyzerRegistry.list_analyzers()
            normal_logger.info(f"可用分析器: {analyzers}")

    @classmethod
    def create_analyzer(cls, analysis_type: Union[int, str], model_code: Optional[str] = None, **kwargs) -> BaseAnalyzer:
        """
        创建分析器实例
        
        Args:
            analysis_type: 分析类型，只支持"detection"
            model_code: 模型代码
            **kwargs: 其他参数，将直接传递给分析器构造函数
                可选参数包括：
                - analyzer_name: 特定分析器名称，如果提供则使用指定的分析器实现
                - device: 推理设备
                - custom_weights_path: 自定义权重路径
                - half_precision: 是否使用半精度
                - 以及其他分析器特定参数
        
        Returns:
            BaseAnalyzer: 分析器实例
            
        Raises:
            ValueError: 当分析类型不支持或创建分析器失败时
        """
        # 确保工厂已初始化
        if not cls._initialized:
            cls.initialize()
            
        # 如果分析类型是整数ID，转换为字符串(向后兼容)
        if isinstance(analysis_type, int):
            if analysis_type != 1:  # 只支持detection (ID=1)
                raise ValueError(f"不支持的分析类型ID: {analysis_type}，只支持detection(ID=1)")
            analysis_type = "detection"
        
        # 将分析类型转换为标准格式
        analysis_type = cls.ANALYSIS_TYPE_MAP.get(analysis_type, analysis_type)
        
        # 检查分析类型是否为detection
        if analysis_type != "detection":
            raise ValueError(f"不支持的分析类型: {analysis_type}，只支持detection")
        
        # 获取特定分析器名称（如果提供）
        analyzer_name = kwargs.pop("analyzer_name", None)
        
        try:
            # 从注册表获取分析器类
            analyzer_class = AnalyzerRegistry.get_analyzer(analysis_type, analyzer_name)
            
            # 创建分析器实例
            normal_logger.info(f"创建分析器: 类型={analysis_type}, 实现={analyzer_class.__name__}, 模型={model_code}")
            analyzer = analyzer_class(model_code=model_code, **kwargs)
            return analyzer
            
        except ValueError as e:
            exception_logger.error(f"创建分析器失败: {e}")
            raise
            
        except Exception as e:
            exception_logger.exception(f"创建分析器出错: {e}")
            raise ValueError(f"创建分析器出错: {e}")
            
    @classmethod
    def list_available_analyzers(cls):
        """
        列出所有可用的分析器
        
        Returns:
            Dict[str, List[str]]: {分析类型: [分析器名称列表]}
        """
        # 确保工厂已初始化
        if not cls._initialized:
            cls.initialize()
            
        return AnalyzerRegistry.list_analyzers()

# 创建工厂实例
analyzer_factory = AnalyzerFactory()

# 初始化工厂 - 在导入时自动运行
AnalyzerFactory.initialize()
