"""
分析器发现模块
自动扫描和加载所有分析器
"""
import os
import importlib
import inspect
import pkgutil
from pathlib import Path
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

# 排除的目录和模块
EXCLUDED_MODULES = ['discovery', 'registry']
EXCLUDED_PACKAGES = ['yoloe', 'tracking', 'segmentation']  # 已移除的功能包

def discover_analyzers():
    """
    自动发现并导入所有分析器
    
    扫描analyzer目录下所有模块，寻找并注册BaseAnalyzer的子类
    """
    # 避免循环导入
    from core.analyzer.base_analyzer import BaseAnalyzer
    
    analyzer_pkg = 'core.analyzer'
    # 获取当前模块所在目录
    analyzer_path = Path(__file__).parent
    
    normal_logger.info(f"开始扫描分析器: 路径={analyzer_path}")
    
    # 递归导入所有模块
    for _, name, is_pkg in pkgutil.iter_modules([str(analyzer_path)]):
        # 跳过自身和排除的模块/包
        if name in EXCLUDED_MODULES or name in EXCLUDED_PACKAGES:
            normal_logger.debug(f"跳过排除的模块/包: {name}")
            continue
            
        # 导入模块
        module_name = f"{analyzer_pkg}.{name}"
        try:
            normal_logger.debug(f"尝试导入模块: {module_name}")
            module = importlib.import_module(module_name)
            
            # 如果是包，递归导入
            if is_pkg:
                _discover_in_package(module_name, BaseAnalyzer)
                
            # 扫描模块中的所有类
            _scan_module(module, BaseAnalyzer)
                
        except ImportError as e:
            exception_logger.exception(f"导入模块 {module_name} 失败: {e}")
    
    normal_logger.info("分析器扫描完成")

def _discover_in_package(package_name, base_class):
    """
    递归导入包中的所有模块
    
    Args:
        package_name: 包名
        base_class: 基类，用于识别分析器
    """
    # 检查包名是否包含排除的包
    package_short_name = package_name.split('.')[-1]
    if package_short_name in EXCLUDED_PACKAGES:
        normal_logger.debug(f"跳过排除的包: {package_name}")
        return
        
    try:
        package = importlib.import_module(package_name)
        package_path = Path(package.__file__).parent
        
        normal_logger.debug(f"扫描包: {package_name}, 路径={package_path}")
        
        for _, name, is_pkg in pkgutil.iter_modules([str(package_path)]):
            # 跳过排除的模块/包
            if name in EXCLUDED_MODULES or name in EXCLUDED_PACKAGES:
                normal_logger.debug(f"跳过排除的模块/包: {name}")
                continue
                
            module_name = f"{package_name}.{name}"
            try:
                normal_logger.debug(f"尝试导入模块: {module_name}")
                module = importlib.import_module(module_name)
                
                # 如果是包，递归导入
                if is_pkg:
                    _discover_in_package(module_name, base_class)
                    
                # 扫描模块中的所有类
                _scan_module(module, base_class)
                    
            except ImportError as e:
                exception_logger.warning(f"导入模块 {module_name} 失败: {e}")
    except Exception as e:
        exception_logger.warning(f"处理包 {package_name} 时出错: {e}")

def _scan_module(module, base_class):
    """
    扫描模块中的所有类，找出BaseAnalyzer的子类
    
    这些类如果没有使用@register_analyzer装饰器，则不会被自动注册。
    
    Args:
        module: 模块对象
        base_class: 基类，用于识别分析器
    """
    from core.analyzer.registry import AnalyzerRegistry
    
    for name, obj in inspect.getmembers(module):
        # 只处理类对象
        if inspect.isclass(obj) and issubclass(obj, base_class) and obj != base_class:
            # 检查类是否已在注册表中
            registered = False
            for analysis_type, analyzers in AnalyzerRegistry._registry.items():
                if obj.__name__ in analyzers:
                    registered = True
                    break
                    
            # 如果类尚未注册，尝试自动注册
            if not registered:
                # 尝试获取分析类型
                try:
                    instance = obj()
                    analysis_type = instance.get_analysis_type()
                    # 只有基本类型或抽象类会返回这些值，我们不自动注册它们
                    if analysis_type not in ['base', 'detection', 'tracking', 'segmentation']:
                        normal_logger.info(f"自动注册分析器: {name} -> {analysis_type}")
                        AnalyzerRegistry.register(analysis_type, obj)
                except Exception as e:
                    # 某些抽象类可能无法实例化，跳过
                    if "abstract" not in str(e).lower():
                        normal_logger.debug(f"无法自动注册分析器 {name}: {e}") 