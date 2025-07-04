#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: plugin_loader.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 插件加载器

负责动态加载和发现插件。

本文件是分析服务项目的一部分。
"""

import os
import sys
import importlib
import importlib.util
import inspect
import logging
from typing import List, Dict, Any, Type, Optional
from pathlib import Path

from .base_plugin import BasePlugin
from .plugin_registry import PluginRegistry


class PluginLoader:
    """插件加载器"""
    
    def __init__(self, registry: PluginRegistry):
        """初始化插件加载器
        
        Args:
            registry: 插件注册表
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry = registry
        self._loaded_modules: Dict[str, Any] = {}
        
        self.logger.info("插件加载器初始化完成")
    
    def discover_plugins(self, plugin_dirs: List[str]) -> List[str]:
        """发现插件
        
        Args:
            plugin_dirs: 插件目录列表
            
        Returns:
            List[str]: 发现的插件名称列表
        """
        discovered_plugins = []
        
        for plugin_dir in plugin_dirs:
            if not os.path.exists(plugin_dir):
                self.logger.warning(f"插件目录不存在: {plugin_dir}")
                continue
            
            self.logger.info(f"扫描插件目录: {plugin_dir}")
            plugins = self._scan_directory(plugin_dir)
            discovered_plugins.extend(plugins)
        
        self.logger.info(f"发现 {len(discovered_plugins)} 个插件")
        return discovered_plugins
    
    def _scan_directory(self, plugin_dir: str) -> List[str]:
        """扫描目录中的插件
        
        Args:
            plugin_dir: 插件目录
            
        Returns:
            List[str]: 插件名称列表
        """
        plugins = []
        plugin_path = Path(plugin_dir)
        
        # 扫描Python文件
        for py_file in plugin_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            
            try:
                plugin_name = self._load_plugin_from_file(str(py_file))
                if plugin_name:
                    plugins.append(plugin_name)
            except Exception as e:
                self.logger.error(f"加载插件文件失败 {py_file}: {e}")
        
        # 扫描插件包
        for plugin_package in plugin_path.iterdir():
            if plugin_package.is_dir() and not plugin_package.name.startswith("__"):
                init_file = plugin_package / "__init__.py"
                if init_file.exists():
                    try:
                        plugin_name = self._load_plugin_from_package(str(plugin_package))
                        if plugin_name:
                            plugins.append(plugin_name)
                    except Exception as e:
                        self.logger.error(f"加载插件包失败 {plugin_package}: {e}")
        
        return plugins
    
    def _load_plugin_from_file(self, file_path: str) -> Optional[str]:
        """从文件加载插件
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[str]: 插件名称
        """
        file_name = Path(file_path).stem
        module_name = f"plugin_{file_name}"
        
        # 加载模块
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            self.logger.error(f"无法创建模块规范: {file_path}")
            return None
        
        module = importlib.util.module_from_spec(spec)
        self._loaded_modules[module_name] = module
        
        # 执行模块
        spec.loader.exec_module(module)
        
        # 查找插件类
        plugin_classes = self._find_plugin_classes(module)
        
        if not plugin_classes:
            self.logger.warning(f"文件中未找到插件类: {file_path}")
            return None
        
        # 注册插件类
        for plugin_class in plugin_classes:
            success = self.registry.register_plugin_class(plugin_class)
            if success:
                # 获取插件名称
                temp_instance = plugin_class()
                plugin_name = temp_instance.get_plugin_info().name
                self.logger.info(f"从文件加载插件: {plugin_name} ({file_path})")
                return plugin_name
        
        return None
    
    def _load_plugin_from_package(self, package_path: str) -> Optional[str]:
        """从包加载插件
        
        Args:
            package_path: 包路径
            
        Returns:
            Optional[str]: 插件名称
        """
        package_name = Path(package_path).name
        module_name = f"plugin_package_{package_name}"
        
        # 添加包路径到sys.path
        parent_dir = str(Path(package_path).parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        try:
            # 导入包
            module = importlib.import_module(package_name)
            self._loaded_modules[module_name] = module
            
            # 查找插件类
            plugin_classes = self._find_plugin_classes(module)
            
            if not plugin_classes:
                self.logger.warning(f"包中未找到插件类: {package_path}")
                return None
            
            # 注册插件类
            for plugin_class in plugin_classes:
                success = self.registry.register_plugin_class(plugin_class)
                if success:
                    # 获取插件名称
                    temp_instance = plugin_class()
                    plugin_name = temp_instance.get_plugin_info().name
                    self.logger.info(f"从包加载插件: {plugin_name} ({package_path})")
                    return plugin_name
            
            return None
            
        except ImportError as e:
            self.logger.error(f"导入插件包失败 {package_path}: {e}")
            return None
        finally:
            # 移除添加的路径
            if parent_dir in sys.path:
                sys.path.remove(parent_dir)
    
    def _find_plugin_classes(self, module) -> List[Type[BasePlugin]]:
        """在模块中查找插件类
        
        Args:
            module: 模块对象
            
        Returns:
            List[Type[BasePlugin]]: 插件类列表
        """
        plugin_classes = []
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # 检查是否是BasePlugin的子类
            if (issubclass(obj, BasePlugin) and 
                obj != BasePlugin and 
                obj.__module__ == module.__name__):
                plugin_classes.append(obj)
        
        return plugin_classes
    
    def load_plugin_by_name(self, plugin_name: str, plugin_dirs: List[str]) -> bool:
        """按名称加载插件
        
        Args:
            plugin_name: 插件名称
            plugin_dirs: 插件目录列表
            
        Returns:
            bool: 是否加载成功
        """
        for plugin_dir in plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue
            
            plugin_path = Path(plugin_dir)
            
            # 查找插件文件
            plugin_file = plugin_path / f"{plugin_name}.py"
            if plugin_file.exists():
                try:
                    loaded_name = self._load_plugin_from_file(str(plugin_file))
                    return loaded_name == plugin_name
                except Exception as e:
                    self.logger.error(f"加载插件失败 {plugin_name}: {e}")
                    return False
            
            # 查找插件包
            plugin_package = plugin_path / plugin_name
            if plugin_package.is_dir() and (plugin_package / "__init__.py").exists():
                try:
                    loaded_name = self._load_plugin_from_package(str(plugin_package))
                    return loaded_name == plugin_name
                except Exception as e:
                    self.logger.error(f"加载插件包失败 {plugin_name}: {e}")
                    return False
        
        self.logger.error(f"未找到插件: {plugin_name}")
        return False
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """重新加载插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 是否重新加载成功
        """
        # 查找对应的模块
        module_to_reload = None
        for module_name, module in self._loaded_modules.items():
            plugin_classes = self._find_plugin_classes(module)
            for plugin_class in plugin_classes:
                temp_instance = plugin_class()
                if temp_instance.get_plugin_info().name == plugin_name:
                    module_to_reload = module
                    break
            if module_to_reload:
                break
        
        if not module_to_reload:
            self.logger.error(f"未找到插件对应的模块: {plugin_name}")
            return False
        
        try:
            # 注销旧的插件类
            self.registry.unregister_plugin_class(plugin_name)
            
            # 重新加载模块
            importlib.reload(module_to_reload)
            
            # 重新注册插件类
            plugin_classes = self._find_plugin_classes(module_to_reload)
            for plugin_class in plugin_classes:
                temp_instance = plugin_class()
                if temp_instance.get_plugin_info().name == plugin_name:
                    success = self.registry.register_plugin_class(plugin_class)
                    if success:
                        self.logger.info(f"重新加载插件成功: {plugin_name}")
                        return True
            
            self.logger.error(f"重新加载插件失败，未找到插件类: {plugin_name}")
            return False
            
        except Exception as e:
            self.logger.error(f"重新加载插件失败 {plugin_name}: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 是否卸载成功
        """
        try:
            # 注销插件类和实例
            self.registry.unregister_plugin_instance(plugin_name)
            self.registry.unregister_plugin_class(plugin_name)
            
            # 查找并移除对应的模块
            module_to_remove = None
            for module_name, module in self._loaded_modules.items():
                plugin_classes = self._find_plugin_classes(module)
                for plugin_class in plugin_classes:
                    temp_instance = plugin_class()
                    if temp_instance.get_plugin_info().name == plugin_name:
                        module_to_remove = module_name
                        break
                if module_to_remove:
                    break
            
            if module_to_remove:
                del self._loaded_modules[module_to_remove]
            
            self.logger.info(f"卸载插件成功: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"卸载插件失败 {plugin_name}: {e}")
            return False
    
    def get_loaded_modules(self) -> Dict[str, Any]:
        """获取已加载的模块
        
        Returns:
            Dict[str, Any]: 模块字典
        """
        return self._loaded_modules.copy()
    
    def clear_loaded_modules(self):
        """清空已加载的模块"""
        self._loaded_modules.clear()
        self.logger.info("已清空加载的模块")
    
    def get_loader_stats(self) -> Dict[str, Any]:
        """获取加载器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'loaded_modules': len(self._loaded_modules),
            'module_names': list(self._loaded_modules.keys()),
            'registry_stats': self.registry.get_registry_stats()
        }
