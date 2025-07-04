#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: plugin_registry.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 插件注册表

管理插件的注册、查找和依赖关系。

本文件是分析服务项目的一部分。
"""

import logging
from typing import Dict, List, Optional, Type, Set
from collections import defaultdict, deque

from .base_plugin import BasePlugin, PluginInfo, PluginStatus


class PluginRegistry:
    """插件注册表"""
    
    def __init__(self):
        """初始化插件注册表"""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 插件类注册表
        self._plugin_classes: Dict[str, Type[BasePlugin]] = {}
        
        # 插件实例注册表
        self._plugin_instances: Dict[str, BasePlugin] = {}
        
        # 插件信息缓存
        self._plugin_info_cache: Dict[str, PluginInfo] = {}
        
        # 依赖关系图
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._dependents: Dict[str, Set[str]] = defaultdict(set)
        
        # 分类索引
        self._categories: Dict[str, Set[str]] = defaultdict(set)
        
        self.logger.info("插件注册表初始化完成")
    
    def register_plugin_class(self, plugin_class: Type[BasePlugin]) -> bool:
        """注册插件类
        
        Args:
            plugin_class: 插件类
            
        Returns:
            bool: 是否注册成功
        """
        try:
            # 创建临时实例获取插件信息
            temp_instance = plugin_class()
            plugin_info = temp_instance.get_plugin_info()
            plugin_name = plugin_info.name
            
            # 检查是否已注册
            if plugin_name in self._plugin_classes:
                self.logger.warning(f"插件类已存在，将被覆盖: {plugin_name}")
            
            # 注册插件类
            self._plugin_classes[plugin_name] = plugin_class
            self._plugin_info_cache[plugin_name] = plugin_info
            
            # 更新依赖关系
            self._update_dependencies(plugin_name, plugin_info.dependencies)
            
            # 更新分类索引
            self._categories[plugin_info.category].add(plugin_name)
            
            self.logger.info(f"注册插件类: {plugin_name} v{plugin_info.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"注册插件类失败: {e}")
            return False
    
    def unregister_plugin_class(self, plugin_name: str) -> bool:
        """注销插件类
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 是否注销成功
        """
        if plugin_name not in self._plugin_classes:
            self.logger.warning(f"插件类不存在: {plugin_name}")
            return False
        
        # 检查是否有实例在运行
        if plugin_name in self._plugin_instances:
            instance = self._plugin_instances[plugin_name]
            if instance.is_active():
                self.logger.error(f"无法注销活跃的插件: {plugin_name}")
                return False
        
        # 检查是否有其他插件依赖此插件
        if plugin_name in self._dependents and self._dependents[plugin_name]:
            dependent_plugins = list(self._dependents[plugin_name])
            self.logger.error(f"无法注销被依赖的插件 {plugin_name}，依赖者: {dependent_plugins}")
            return False
        
        # 移除插件类
        del self._plugin_classes[plugin_name]
        
        # 清理缓存
        if plugin_name in self._plugin_info_cache:
            plugin_info = self._plugin_info_cache[plugin_name]
            self._categories[plugin_info.category].discard(plugin_name)
            del self._plugin_info_cache[plugin_name]
        
        # 清理依赖关系
        self._clear_dependencies(plugin_name)
        
        # 移除实例
        if plugin_name in self._plugin_instances:
            del self._plugin_instances[plugin_name]
        
        self.logger.info(f"注销插件类: {plugin_name}")
        return True
    
    def register_plugin_instance(self, plugin_instance: BasePlugin) -> bool:
        """注册插件实例
        
        Args:
            plugin_instance: 插件实例
            
        Returns:
            bool: 是否注册成功
        """
        plugin_info = plugin_instance.get_plugin_info()
        plugin_name = plugin_info.name
        
        # 检查插件类是否已注册
        if plugin_name not in self._plugin_classes:
            self.logger.error(f"插件类未注册: {plugin_name}")
            return False
        
        # 检查是否已有实例
        if plugin_name in self._plugin_instances:
            self.logger.warning(f"插件实例已存在，将被覆盖: {plugin_name}")
        
        # 注册实例
        self._plugin_instances[plugin_name] = plugin_instance
        
        self.logger.info(f"注册插件实例: {plugin_name}")
        return True
    
    def unregister_plugin_instance(self, plugin_name: str) -> bool:
        """注销插件实例
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 是否注销成功
        """
        if plugin_name not in self._plugin_instances:
            self.logger.warning(f"插件实例不存在: {plugin_name}")
            return False
        
        instance = self._plugin_instances[plugin_name]
        
        # 检查插件状态
        if instance.is_active():
            self.logger.error(f"无法注销活跃的插件实例: {plugin_name}")
            return False
        
        # 移除实例
        del self._plugin_instances[plugin_name]
        
        self.logger.info(f"注销插件实例: {plugin_name}")
        return True
    
    def get_plugin_class(self, plugin_name: str) -> Optional[Type[BasePlugin]]:
        """获取插件类
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            Optional[Type[BasePlugin]]: 插件类
        """
        return self._plugin_classes.get(plugin_name)
    
    def get_plugin_instance(self, plugin_name: str) -> Optional[BasePlugin]:
        """获取插件实例
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            Optional[BasePlugin]: 插件实例
        """
        return self._plugin_instances.get(plugin_name)
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """获取插件信息
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            Optional[PluginInfo]: 插件信息
        """
        return self._plugin_info_cache.get(plugin_name)
    
    def list_plugin_classes(self) -> List[str]:
        """列出所有已注册的插件类
        
        Returns:
            List[str]: 插件名称列表
        """
        return list(self._plugin_classes.keys())
    
    def list_plugin_instances(self) -> List[str]:
        """列出所有已注册的插件实例
        
        Returns:
            List[str]: 插件名称列表
        """
        return list(self._plugin_instances.keys())
    
    def list_plugins_by_category(self, category: str) -> List[str]:
        """按分类列出插件
        
        Args:
            category: 插件分类
            
        Returns:
            List[str]: 插件名称列表
        """
        return list(self._categories.get(category, set()))
    
    def list_categories(self) -> List[str]:
        """列出所有插件分类
        
        Returns:
            List[str]: 分类列表
        """
        return list(self._categories.keys())
    
    def get_dependencies(self, plugin_name: str) -> Set[str]:
        """获取插件依赖
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            Set[str]: 依赖插件集合
        """
        return self._dependencies.get(plugin_name, set()).copy()
    
    def get_dependents(self, plugin_name: str) -> Set[str]:
        """获取依赖此插件的插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            Set[str]: 依赖者插件集合
        """
        return self._dependents.get(plugin_name, set()).copy()
    
    def resolve_load_order(self, plugin_names: List[str] = None) -> List[str]:
        """解析插件加载顺序
        
        Args:
            plugin_names: 要加载的插件列表，如果为None则包含所有插件
            
        Returns:
            List[str]: 按依赖关系排序的插件列表
        """
        if plugin_names is None:
            plugin_names = self.list_plugin_classes()
        
        # 拓扑排序
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        # 构建图和入度
        for plugin_name in plugin_names:
            dependencies = self.get_dependencies(plugin_name)
            in_degree[plugin_name] = len(dependencies)
            
            for dep in dependencies:
                if dep in plugin_names:
                    graph[dep].append(plugin_name)
        
        # 拓扑排序
        queue = deque([plugin for plugin in plugin_names if in_degree[plugin] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否有循环依赖
        if len(result) != len(plugin_names):
            remaining = set(plugin_names) - set(result)
            self.logger.error(f"检测到循环依赖，无法解析的插件: {remaining}")
            raise ValueError(f"循环依赖: {remaining}")
        
        return result
    
    def validate_dependencies(self, plugin_names: List[str] = None) -> Dict[str, List[str]]:
        """验证插件依赖
        
        Args:
            plugin_names: 要验证的插件列表，如果为None则验证所有插件
            
        Returns:
            Dict[str, List[str]]: 插件名称到缺失依赖的映射
        """
        if plugin_names is None:
            plugin_names = self.list_plugin_classes()
        
        missing_deps = {}
        available_plugins = set(self.list_plugin_classes())
        
        for plugin_name in plugin_names:
            dependencies = self.get_dependencies(plugin_name)
            missing = [dep for dep in dependencies if dep not in available_plugins]
            if missing:
                missing_deps[plugin_name] = missing
        
        return missing_deps
    
    def _update_dependencies(self, plugin_name: str, dependencies: List[str]):
        """更新依赖关系
        
        Args:
            plugin_name: 插件名称
            dependencies: 依赖列表
        """
        # 清理旧的依赖关系
        old_deps = self._dependencies.get(plugin_name, set())
        for dep in old_deps:
            self._dependents[dep].discard(plugin_name)
        
        # 设置新的依赖关系
        self._dependencies[plugin_name] = set(dependencies)
        for dep in dependencies:
            self._dependents[dep].add(plugin_name)
    
    def _clear_dependencies(self, plugin_name: str):
        """清理插件的依赖关系
        
        Args:
            plugin_name: 插件名称
        """
        # 清理作为依赖者的关系
        dependencies = self._dependencies.get(plugin_name, set())
        for dep in dependencies:
            self._dependents[dep].discard(plugin_name)
        
        # 清理作为被依赖者的关系
        dependents = self._dependents.get(plugin_name, set())
        for dependent in dependents:
            self._dependencies[dependent].discard(plugin_name)
        
        # 移除记录
        if plugin_name in self._dependencies:
            del self._dependencies[plugin_name]
        if plugin_name in self._dependents:
            del self._dependents[plugin_name]
    
    def get_registry_stats(self) -> Dict[str, any]:
        """获取注册表统计信息
        
        Returns:
            Dict[str, any]: 统计信息
        """
        active_instances = sum(1 for instance in self._plugin_instances.values() if instance.is_active())
        loaded_instances = sum(1 for instance in self._plugin_instances.values() if instance.is_loaded())
        
        category_stats = {cat: len(plugins) for cat, plugins in self._categories.items()}
        
        return {
            'total_plugin_classes': len(self._plugin_classes),
            'total_plugin_instances': len(self._plugin_instances),
            'active_instances': active_instances,
            'loaded_instances': loaded_instances,
            'categories': category_stats,
            'total_dependencies': sum(len(deps) for deps in self._dependencies.values()),
            'plugins_with_dependencies': len([p for p in self._dependencies.values() if p])
        }
    
    def clear(self):
        """清空注册表"""
        self._plugin_classes.clear()
        self._plugin_instances.clear()
        self._plugin_info_cache.clear()
        self._dependencies.clear()
        self._dependents.clear()
        self._categories.clear()
        
        self.logger.info("插件注册表已清空")
    
    def __len__(self) -> int:
        """返回已注册的插件类数量"""
        return len(self._plugin_classes)
    
    def __contains__(self, plugin_name: str) -> bool:
        """检查插件是否已注册"""
        return plugin_name in self._plugin_classes
    
    def __iter__(self):
        """迭代插件名称"""
        return iter(self._plugin_classes.keys())
