#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: service_factory.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 服务工厂

负责创建和管理所有业务服务实例，提供统一的服务创建接口。

本文件是分析服务项目的一部分。
"""

import inspect
import importlib
from typing import Dict, Type, Any, Optional
from pathlib import Path

from .base_service import BaseService
from config.dependencies import get_container


class ServiceFactory:
    """服务工厂"""
    
    def __init__(self):
        """初始化服务工厂"""
        self.container = get_container()
        self.service_classes: Dict[str, Type[BaseService]] = {}
        self.service_instances: Dict[str, BaseService] = {}
        self._discovered = False
    
    def discover_services(self, package_path: str = "app.services") -> Dict[str, Type[BaseService]]:
        """发现服务类
        
        Args:
            package_path: 服务包路径
            
        Returns:
            Dict[str, Type[BaseService]]: 服务类字典
        """
        if self._discovered:
            return self.service_classes
        
        services = {}
        
        # 获取服务目录
        services_dir = Path(__file__).parent
        
        # 遍历所有Python文件
        for file_path in services_dir.glob("*_service.py"):
            module_name = file_path.stem
            
            # 跳过基础服务文件
            if module_name == 'base_service':
                continue
            
            try:
                # 导入模块
                module = importlib.import_module(f"{package_path}.{module_name}")
                
                # 查找服务类
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseService) and 
                        obj != BaseService and 
                        name.endswith('Service')):
                        service_name = self._get_service_name(name)
                        services[service_name] = obj
                        
            except ImportError as e:
                print(f"导入服务模块失败 {module_name}: {e}")
        
        self.service_classes.update(services)
        self._discovered = True
        return services
    
    def create_service(self, service_name: str, **kwargs) -> BaseService:
        """创建服务实例
        
        Args:
            service_name: 服务名称
            **kwargs: 构造参数
            
        Returns:
            BaseService: 服务实例
        """
        # 确保已发现服务
        if not self._discovered:
            self.discover_services()
        
        # 检查是否已有实例
        if service_name in self.service_instances:
            return self.service_instances[service_name]
        
        # 获取服务类
        service_class = self.service_classes.get(service_name)
        if not service_class:
            raise ValueError(f"未找到服务: {service_name}")
        
        # 创建服务实例
        try:
            service_instance = service_class(**kwargs)
            self.service_instances[service_name] = service_instance
            return service_instance
        except Exception as e:
            raise RuntimeError(f"创建服务实例失败 {service_name}: {e}")
    
    def get_service(self, service_name: str) -> Optional[BaseService]:
        """获取服务实例
        
        Args:
            service_name: 服务名称
            
        Returns:
            Optional[BaseService]: 服务实例
        """
        return self.service_instances.get(service_name)
    
    def register_service(self, service_name: str, service_instance: BaseService):
        """注册服务实例
        
        Args:
            service_name: 服务名称
            service_instance: 服务实例
        """
        self.service_instances[service_name] = service_instance
    
    def register_service_class(self, service_name: str, service_class: Type[BaseService]):
        """注册服务类
        
        Args:
            service_name: 服务名称
            service_class: 服务类
        """
        self.service_classes[service_name] = service_class
    
    async def initialize_all_services(self):
        """初始化所有服务"""
        for service_name, service_instance in self.service_instances.items():
            try:
                await service_instance.initialize()
                print(f"服务初始化成功: {service_name}")
            except Exception as e:
                print(f"服务初始化失败 {service_name}: {e}")
    
    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息
        
        Returns:
            Dict[str, Any]: 服务信息
        """
        return {
            'discovered_classes': list(self.service_classes.keys()),
            'created_instances': list(self.service_instances.keys()),
            'total_classes': len(self.service_classes),
            'total_instances': len(self.service_instances)
        }
    
    def _get_service_name(self, class_name: str) -> str:
        """从类名获取服务名称
        
        Args:
            class_name: 类名
            
        Returns:
            str: 服务名称
        """
        # 移除Service后缀并转换为snake_case
        name = class_name.replace('Service', '')
        
        # 转换为snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        
        return name


class ServiceManager:
    """服务管理器"""
    
    def __init__(self):
        """初始化服务管理器"""
        self.factory = ServiceFactory()
        self.container = get_container()
        self._initialized = False
    
    async def initialize(self):
        """初始化服务管理器"""
        if self._initialized:
            return
        
        # 发现所有服务
        self.factory.discover_services()
        
        # 创建核心服务实例
        await self._create_core_services()
        
        # 注册服务到容器
        self._register_services_to_container()
        
        # 初始化所有服务
        await self.factory.initialize_all_services()
        
        self._initialized = True
        print("服务管理器初始化完成")
    
    async def _create_core_services(self):
        """创建核心服务实例"""
        core_services = [
            'task_service',
            'stream_service', 
            'discovery_service'
        ]
        
        for service_name in core_services:
            try:
                service_instance = self.factory.create_service(service_name)
                print(f"核心服务创建成功: {service_name}")
            except Exception as e:
                print(f"核心服务创建失败 {service_name}: {e}")
    
    def _register_services_to_container(self):
        """将服务注册到依赖注入容器"""
        for service_name, service_instance in self.factory.service_instances.items():
            self.container.instance(service_name, service_instance)
            print(f"服务已注册到容器: {service_name}")
    
    def get_service(self, service_name: str) -> Optional[BaseService]:
        """获取服务实例
        
        Args:
            service_name: 服务名称
            
        Returns:
            Optional[BaseService]: 服务实例
        """
        return self.factory.get_service(service_name)
    
    def create_service(self, service_name: str, **kwargs) -> BaseService:
        """创建服务实例
        
        Args:
            service_name: 服务名称
            **kwargs: 构造参数
            
        Returns:
            BaseService: 服务实例
        """
        return self.factory.create_service(service_name, **kwargs)
    
    def get_all_services(self) -> Dict[str, BaseService]:
        """获取所有服务实例
        
        Returns:
            Dict[str, BaseService]: 服务实例字典
        """
        return self.factory.service_instances.copy()
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态
        
        Returns:
            Dict[str, Any]: 服务状态信息
        """
        status = {
            'initialized': self._initialized,
            'services': {}
        }
        
        for service_name, service_instance in self.factory.service_instances.items():
            status['services'][service_name] = {
                'class': service_instance.__class__.__name__,
                'initialized': getattr(service_instance, '_initialized', False),
                'status': 'active'
            }
        
        return status
    
    async def shutdown(self):
        """关闭服务管理器"""
        print("正在关闭服务管理器...")
        
        # 这里可以添加服务清理逻辑
        for service_name, service_instance in self.factory.service_instances.items():
            try:
                # 如果服务有cleanup方法，调用它
                if hasattr(service_instance, 'cleanup'):
                    await service_instance.cleanup()
                print(f"服务清理完成: {service_name}")
            except Exception as e:
                print(f"服务清理失败 {service_name}: {e}")
        
        self._initialized = False
        print("服务管理器已关闭")


# 全局服务管理器实例
_service_manager = None


def get_service_manager() -> ServiceManager:
    """获取全局服务管理器实例
    
    Returns:
        ServiceManager: 服务管理器实例
    """
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager


async def initialize_services():
    """初始化所有服务"""
    manager = get_service_manager()
    await manager.initialize()


async def shutdown_services():
    """关闭所有服务"""
    manager = get_service_manager()
    await manager.shutdown()
