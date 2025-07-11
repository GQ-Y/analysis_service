#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析器插件系统
提供通用的分析器插件加载和管理机制
"""

import os
import json
import yaml
import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Union
from abc import ABC, abstractmethod

from .base_analyzer import BaseAnalyzer
from .analyzer_registry import get_global_registry, register_analyzer_globally
from app.models.base_model import AnalysisTypeEnum


class AnalyzerPlugin(ABC):
    """分析器插件基类"""
    
    @abstractmethod
    def get_plugin_info(self) -> Dict[str, Any]:
        """获取插件信息
        
        Returns:
            Dict[str, Any]: 插件信息
        """
        pass
    
    @abstractmethod
    def get_analyzer_classes(self) -> List[Type[BaseAnalyzer]]:
        """获取插件提供的分析器类列表
        
        Returns:
            List[Type[BaseAnalyzer]]: 分析器类列表
        """
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件
        
        Args:
            config: 插件配置
            
        Returns:
            bool: 是否初始化成功
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """清理插件资源"""
        pass


class AnalyzerPluginManager:
    """分析器插件管理器"""
    
    def __init__(self, plugin_dirs: List[str] = None, config_file: str = None):
        """初始化插件管理器
        
        Args:
            plugin_dirs: 插件目录列表
            config_file: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.registry = get_global_registry()
        
        # 默认插件目录
        self.plugin_dirs = plugin_dirs or [
            "app/core/analyzer/plugins",
            "plugins/analyzers",
            "external_plugins"
        ]
        
        # 配置文件
        self.config_file = config_file or "config/analyzer_plugins.yaml"
        
        # 已加载的插件
        self.loaded_plugins: Dict[str, AnalyzerPlugin] = {}
        
        # 插件配置
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        
        # 加载配置
        self._load_config()
    
    def _load_config(self):
        """加载插件配置"""
        if not os.path.exists(self.config_file):
            self.logger.info(f"插件配置文件不存在，使用默认配置: {self.config_file}")
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            self.plugin_configs = config.get('plugins', {})
            self.logger.info(f"已加载插件配置: {len(self.plugin_configs)} 个插件")
            
        except Exception as e:
            self.logger.error(f"加载插件配置失败: {e}")
    
    def discover_plugins(self) -> List[str]:
        """发现可用的插件
        
        Returns:
            List[str]: 插件模块路径列表
        """
        plugins = []
        
        for plugin_dir in self.plugin_dirs:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                continue
            
            # 查找Python文件
            for py_file in plugin_path.rglob("*.py"):
                if py_file.name.startswith('_'):
                    continue
                
                # 构建模块路径
                relative_path = py_file.relative_to(Path.cwd())
                module_path = str(relative_path.with_suffix('')).replace(os.sep, '.')
                plugins.append(module_path)
        
        self.logger.info(f"发现 {len(plugins)} 个插件文件")
        return plugins
    
    def load_plugin(self, module_path: str) -> bool:
        """加载单个插件
        
        Args:
            module_path: 插件模块路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            # 导入模块
            module = importlib.import_module(module_path)
            
            # 查找插件类
            plugin_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, AnalyzerPlugin) and 
                    obj != AnalyzerPlugin):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                self.logger.debug(f"模块中未找到插件类: {module_path}")
                return False
            
            # 创建插件实例
            plugin = plugin_class()
            plugin_info = plugin.get_plugin_info()
            plugin_name = plugin_info.get('name', module_path)
            
            # 获取插件配置
            plugin_config = self.plugin_configs.get(plugin_name, {})
            
            # 初始化插件
            if not plugin.initialize(plugin_config):
                self.logger.error(f"插件初始化失败: {plugin_name}")
                return False
            
            # 注册分析器类
            analyzer_classes = plugin.get_analyzer_classes()
            for analyzer_class in analyzer_classes:
                if hasattr(analyzer_class, 'get_analysis_type'):
                    try:
                        instance = analyzer_class()
                        analysis_type = instance.get_analysis_type()
                        register_analyzer_globally(analysis_type, analyzer_class)
                        self.logger.info(f"注册插件分析器: {analysis_type.value} -> {analyzer_class.__name__}")
                    except Exception as e:
                        self.logger.error(f"注册插件分析器失败: {analyzer_class.__name__}: {e}")
            
            # 保存插件
            self.loaded_plugins[plugin_name] = plugin
            self.logger.info(f"插件加载成功: {plugin_name} ({len(analyzer_classes)} 个分析器)")
            return True
            
        except Exception as e:
            self.logger.error(f"加载插件失败 {module_path}: {e}")
            return False
    
    def load_all_plugins(self) -> int:
        """加载所有插件
        
        Returns:
            int: 成功加载的插件数量
        """
        plugins = self.discover_plugins()
        loaded_count = 0
        
        for plugin_path in plugins:
            if self.load_plugin(plugin_path):
                loaded_count += 1
        
        self.logger.info(f"插件加载完成: {loaded_count}/{len(plugins)} 个插件成功加载")
        return loaded_count
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 是否卸载成功
        """
        if plugin_name not in self.loaded_plugins:
            self.logger.warning(f"插件未加载: {plugin_name}")
            return False
        
        try:
            plugin = self.loaded_plugins[plugin_name]
            plugin.cleanup()
            del self.loaded_plugins[plugin_name]
            self.logger.info(f"插件卸载成功: {plugin_name}")
            return True
        except Exception as e:
            self.logger.error(f"卸载插件失败 {plugin_name}: {e}")
            return False
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """获取插件信息
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            Optional[Dict[str, Any]]: 插件信息
        """
        if plugin_name not in self.loaded_plugins:
            return None
        
        return self.loaded_plugins[plugin_name].get_plugin_info()
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出所有已加载的插件
        
        Returns:
            List[Dict[str, Any]]: 插件信息列表
        """
        plugins = []
        for plugin_name, plugin in self.loaded_plugins.items():
            info = plugin.get_plugin_info()
            info['name'] = plugin_name
            plugins.append(info)
        
        return plugins
    
    def cleanup_all(self):
        """清理所有插件"""
        for plugin_name in list(self.loaded_plugins.keys()):
            self.unload_plugin(plugin_name)
        
        self.logger.info("所有插件已清理")


class ConfigDrivenAnalyzerFactory:
    """配置驱动的分析器工厂"""
    
    def __init__(self, config_file: str = "config/analyzer_configs.yaml"):
        """初始化配置驱动工厂
        
        Args:
            config_file: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        self.analyzer_configs: Dict[str, Dict[str, Any]] = {}
        self.registry = get_global_registry()
        
        # 加载配置
        self._load_config()
    
    def _load_config(self):
        """加载分析器配置"""
        if not os.path.exists(self.config_file):
            self.logger.info(f"分析器配置文件不存在: {self.config_file}")
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            self.analyzer_configs = config.get('analyzers', {})
            self.logger.info(f"已加载分析器配置: {len(self.analyzer_configs)} 个配置")
            
        except Exception as e:
            self.logger.error(f"加载分析器配置失败: {e}")
    
    def create_analyzer_from_config(self, config_name: str, **override_params) -> Optional[BaseAnalyzer]:
        """根据配置创建分析器
        
        Args:
            config_name: 配置名称
            **override_params: 覆盖参数
            
        Returns:
            Optional[BaseAnalyzer]: 分析器实例
        """
        if config_name not in self.analyzer_configs:
            self.logger.error(f"未找到分析器配置: {config_name}")
            return None
        
        config = self.analyzer_configs[config_name].copy()
        
        # 应用覆盖参数
        config.update(override_params)
        
        # 获取分析类型
        analysis_type_name = config.get('analysis_type')
        if not analysis_type_name:
            self.logger.error(f"配置中缺少analysis_type: {config_name}")
            return None
        
        try:
            analysis_type = AnalysisTypeEnum(analysis_type_name)
        except ValueError:
            self.logger.error(f"无效的分析类型: {analysis_type_name}")
            return None
        
        # 获取分析器类
        analyzer_class = self.registry.get(analysis_type)
        if not analyzer_class:
            self.logger.error(f"未注册的分析器类型: {analysis_type}")
            return None
        
        # 创建分析器实例
        try:
            # 移除非构造函数参数
            constructor_params = config.copy()
            constructor_params.pop('analysis_type', None)
            
            analyzer = analyzer_class(**constructor_params)
            self.logger.info(f"根据配置创建分析器成功: {config_name} -> {analyzer_class.__name__}")
            return analyzer
            
        except Exception as e:
            self.logger.error(f"创建分析器失败 {config_name}: {e}")
            return None
    
    def get_available_configs(self) -> List[str]:
        """获取可用的配置名称列表
        
        Returns:
            List[str]: 配置名称列表
        """
        return list(self.analyzer_configs.keys())
    
    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """获取指定配置
        
        Args:
            config_name: 配置名称
            
        Returns:
            Optional[Dict[str, Any]]: 配置信息
        """
        return self.analyzer_configs.get(config_name)


# 全局插件管理器实例
_plugin_manager = None


def get_plugin_manager() -> AnalyzerPluginManager:
    """获取全局插件管理器
    
    Returns:
        AnalyzerPluginManager: 插件管理器实例
    """
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = AnalyzerPluginManager()
    return _plugin_manager


def initialize_plugin_system():
    """初始化插件系统"""
    plugin_manager = get_plugin_manager()
    plugin_manager.load_all_plugins()


def cleanup_plugin_system():
    """清理插件系统"""
    global _plugin_manager
    if _plugin_manager:
        _plugin_manager.cleanup_all()
        _plugin_manager = None 