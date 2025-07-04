#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: plugin_manager.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 插件管理器

统一管理插件的生命周期，包括加载、初始化、启动、停止等。

本文件是分析服务项目的一部分。
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_plugin import BasePlugin, PluginStatus
from .plugin_registry import PluginRegistry
from .plugin_loader import PluginLoader
from app.exceptions.custom_exceptions import ConfigurationException


class PluginManager:
    """插件管理器"""
    
    def __init__(self, plugin_dirs: List[str] = None):
        """初始化插件管理器
        
        Args:
            plugin_dirs: 插件目录列表
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 组件
        self.registry = PluginRegistry()
        self.loader = PluginLoader(self.registry)
        
        # 配置
        self.plugin_dirs = plugin_dirs or []
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        
        # 状态
        self._initialized = False
        self._startup_order: List[str] = []
        
        self.logger.info("插件管理器初始化完成")
    
    async def initialize(self, auto_discover: bool = True) -> bool:
        """初始化插件管理器
        
        Args:
            auto_discover: 是否自动发现插件
            
        Returns:
            bool: 是否初始化成功
        """
        if self._initialized:
            self.logger.warning("插件管理器已初始化")
            return True
        
        try:
            # 自动发现插件
            if auto_discover and self.plugin_dirs:
                discovered_plugins = self.loader.discover_plugins(self.plugin_dirs)
                self.logger.info(f"自动发现 {len(discovered_plugins)} 个插件")
            
            # 验证依赖关系
            missing_deps = self.registry.validate_dependencies()
            if missing_deps:
                self.logger.error(f"插件依赖验证失败: {missing_deps}")
                return False
            
            # 解析启动顺序
            self._startup_order = self.registry.resolve_load_order()
            self.logger.info(f"插件启动顺序: {self._startup_order}")
            
            self._initialized = True
            self.logger.info("插件管理器初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"插件管理器初始化失败: {e}")
            return False
    
    async def start_all_plugins(self) -> Dict[str, bool]:
        """启动所有插件
        
        Returns:
            Dict[str, bool]: 插件名称到启动结果的映射
        """
        if not self._initialized:
            raise ConfigurationException("插件管理器未初始化")
        
        results = {}
        
        for plugin_name in self._startup_order:
            try:
                success = await self.start_plugin(plugin_name)
                results[plugin_name] = success
                
                if not success:
                    self.logger.error(f"插件启动失败: {plugin_name}")
                
            except Exception as e:
                self.logger.error(f"启动插件异常 {plugin_name}: {e}")
                results[plugin_name] = False
        
        successful_count = sum(1 for success in results.values() if success)
        self.logger.info(f"插件启动完成，成功: {successful_count}/{len(results)}")
        
        return results
    
    async def start_plugin(self, plugin_name: str) -> bool:
        """启动单个插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 是否启动成功
        """
        # 检查插件是否存在
        plugin_class = self.registry.get_plugin_class(plugin_name)
        if not plugin_class:
            self.logger.error(f"插件类不存在: {plugin_name}")
            return False
        
        # 检查是否已有实例
        plugin_instance = self.registry.get_plugin_instance(plugin_name)
        if plugin_instance:
            if plugin_instance.is_active():
                self.logger.warning(f"插件已在运行: {plugin_name}")
                return True
        else:
            # 创建插件实例
            try:
                plugin_instance = plugin_class()
                self.registry.register_plugin_instance(plugin_instance)
            except Exception as e:
                self.logger.error(f"创建插件实例失败 {plugin_name}: {e}")
                return False
        
        # 检查依赖
        dependencies = self.registry.get_dependencies(plugin_name)
        for dep in dependencies:
            dep_instance = self.registry.get_plugin_instance(dep)
            if not dep_instance or not dep_instance.is_active():
                self.logger.error(f"插件依赖未满足 {plugin_name} -> {dep}")
                return False
        
        try:
            # 设置状态
            plugin_instance.set_status(PluginStatus.INITIALIZING)
            
            # 初始化插件
            config = self.plugin_configs.get(plugin_name, {})
            success = await plugin_instance.initialize(config)
            if not success:
                plugin_instance.set_status(PluginStatus.ERROR, "初始化失败")
                return False
            
            plugin_instance.set_status(PluginStatus.LOADED)
            
            # 启动插件
            success = await plugin_instance.start()
            if not success:
                plugin_instance.set_status(PluginStatus.ERROR, "启动失败")
                return False
            
            plugin_instance.set_status(PluginStatus.ACTIVE)
            self.logger.info(f"插件启动成功: {plugin_name}")
            return True
            
        except Exception as e:
            plugin_instance.set_status(PluginStatus.ERROR, str(e))
            self.logger.error(f"启动插件失败 {plugin_name}: {e}")
            return False
    
    async def stop_plugin(self, plugin_name: str) -> bool:
        """停止单个插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 是否停止成功
        """
        plugin_instance = self.registry.get_plugin_instance(plugin_name)
        if not plugin_instance:
            self.logger.warning(f"插件实例不存在: {plugin_name}")
            return True
        
        if not plugin_instance.is_active():
            self.logger.warning(f"插件未在运行: {plugin_name}")
            return True
        
        # 检查是否有其他插件依赖此插件
        dependents = self.registry.get_dependents(plugin_name)
        active_dependents = []
        for dep in dependents:
            dep_instance = self.registry.get_plugin_instance(dep)
            if dep_instance and dep_instance.is_active():
                active_dependents.append(dep)
        
        if active_dependents:
            self.logger.error(f"无法停止插件，有活跃的依赖者 {plugin_name}: {active_dependents}")
            return False
        
        try:
            # 停止插件
            success = await plugin_instance.stop()
            if success:
                plugin_instance.set_status(PluginStatus.INACTIVE)
                self.logger.info(f"插件停止成功: {plugin_name}")
            else:
                plugin_instance.set_status(PluginStatus.ERROR, "停止失败")
                self.logger.error(f"插件停止失败: {plugin_name}")
            
            return success
            
        except Exception as e:
            plugin_instance.set_status(PluginStatus.ERROR, str(e))
            self.logger.error(f"停止插件异常 {plugin_name}: {e}")
            return False
    
    async def stop_all_plugins(self) -> Dict[str, bool]:
        """停止所有插件
        
        Returns:
            Dict[str, bool]: 插件名称到停止结果的映射
        """
        results = {}
        
        # 按启动顺序的逆序停止
        stop_order = list(reversed(self._startup_order))
        
        for plugin_name in stop_order:
            plugin_instance = self.registry.get_plugin_instance(plugin_name)
            if plugin_instance and plugin_instance.is_active():
                try:
                    success = await self.stop_plugin(plugin_name)
                    results[plugin_name] = success
                except Exception as e:
                    self.logger.error(f"停止插件异常 {plugin_name}: {e}")
                    results[plugin_name] = False
        
        successful_count = sum(1 for success in results.values() if success)
        self.logger.info(f"插件停止完成，成功: {successful_count}/{len(results)}")
        
        return results
    
    async def restart_plugin(self, plugin_name: str) -> bool:
        """重启插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 是否重启成功
        """
        self.logger.info(f"重启插件: {plugin_name}")
        
        # 停止插件
        stop_success = await self.stop_plugin(plugin_name)
        if not stop_success:
            return False
        
        # 等待一小段时间
        await asyncio.sleep(0.1)
        
        # 启动插件
        start_success = await self.start_plugin(plugin_name)
        return start_success
    
    def set_plugin_config(self, plugin_name: str, config: Dict[str, Any]):
        """设置插件配置
        
        Args:
            plugin_name: 插件名称
            config: 插件配置
        """
        self.plugin_configs[plugin_name] = config.copy() if config else {}
        
        # 如果插件已加载，更新其配置
        plugin_instance = self.registry.get_plugin_instance(plugin_name)
        if plugin_instance:
            plugin_instance.set_config(config)
        
        self.logger.info(f"设置插件配置: {plugin_name}")
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """获取插件配置
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            Dict[str, Any]: 插件配置
        """
        return self.plugin_configs.get(plugin_name, {}).copy()
    
    def get_plugin_status(self, plugin_name: str) -> Optional[PluginStatus]:
        """获取插件状态
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            Optional[PluginStatus]: 插件状态
        """
        plugin_instance = self.registry.get_plugin_instance(plugin_name)
        if plugin_instance:
            return plugin_instance.get_status()
        return None
    
    def get_all_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有插件状态
        
        Returns:
            Dict[str, Dict[str, Any]]: 插件状态信息
        """
        status_info = {}
        
        for plugin_name in self.registry.list_plugin_classes():
            plugin_instance = self.registry.get_plugin_instance(plugin_name)
            if plugin_instance:
                status_info[plugin_name] = plugin_instance.get_runtime_info()
            else:
                plugin_info = self.registry.get_plugin_info(plugin_name)
                status_info[plugin_name] = {
                    'name': plugin_info.name if plugin_info else plugin_name,
                    'version': plugin_info.version if plugin_info else 'unknown',
                    'status': PluginStatus.UNLOADED,
                    'loaded_at': None,
                    'runtime_seconds': 0,
                    'error_message': None,
                    'is_active': False,
                    'is_loaded': False
                }
        
        return status_info
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        all_status = self.get_all_plugin_status()
        
        active_count = sum(1 for info in all_status.values() if info['is_active'])
        loaded_count = sum(1 for info in all_status.values() if info['is_loaded'])
        error_count = sum(1 for info in all_status.values() if info.get('status') == PluginStatus.ERROR)
        
        return {
            'initialized': self._initialized,
            'total_plugins': len(all_status),
            'active_plugins': active_count,
            'loaded_plugins': loaded_count,
            'error_plugins': error_count,
            'startup_order': self._startup_order,
            'plugin_dirs': self.plugin_dirs,
            'registry_stats': self.registry.get_registry_stats(),
            'loader_stats': self.loader.get_loader_stats()
        }
    
    async def cleanup(self):
        """清理插件管理器"""
        self.logger.info("开始清理插件管理器")
        
        # 停止所有插件
        await self.stop_all_plugins()
        
        # 清理所有插件实例
        for plugin_name in self.registry.list_plugin_instances():
            plugin_instance = self.registry.get_plugin_instance(plugin_name)
            if plugin_instance:
                try:
                    await plugin_instance.cleanup()
                except Exception as e:
                    self.logger.error(f"清理插件失败 {plugin_name}: {e}")
        
        # 清理注册表和加载器
        self.registry.clear()
        self.loader.clear_loaded_modules()
        
        # 重置状态
        self._initialized = False
        self._startup_order.clear()
        self.plugin_configs.clear()
        
        self.logger.info("插件管理器清理完成")


# 全局插件管理器实例
_global_plugin_manager = None


def get_plugin_manager() -> PluginManager:
    """获取全局插件管理器实例
    
    Returns:
        PluginManager: 插件管理器实例
    """
    global _global_plugin_manager
    if _global_plugin_manager is None:
        _global_plugin_manager = PluginManager()
    return _global_plugin_manager


async def initialize_plugin_system(plugin_dirs: List[str] = None) -> bool:
    """初始化插件系统
    
    Args:
        plugin_dirs: 插件目录列表
        
    Returns:
        bool: 是否初始化成功
    """
    manager = get_plugin_manager()
    if plugin_dirs:
        manager.plugin_dirs = plugin_dirs
    
    return await manager.initialize()
