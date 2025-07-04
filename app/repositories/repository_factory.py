#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: repository_factory.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 仓储工厂

负责创建和管理所有仓储实例，提供统一的仓储创建接口。

本文件是分析服务项目的一部分。
"""

import inspect
import importlib
from typing import Dict, Type, Any, Optional
from pathlib import Path

from .base_repository import BaseRepository
from config.dependencies import get_container


class RepositoryFactory:
    """仓储工厂"""
    
    def __init__(self):
        """初始化仓储工厂"""
        self.container = get_container()
        self.repository_classes: Dict[str, Type[BaseRepository]] = {}
        self.repository_instances: Dict[str, BaseRepository] = {}
        self._discovered = False
    
    def discover_repositories(self, package_path: str = "app.repositories") -> Dict[str, Type[BaseRepository]]:
        """发现仓储类
        
        Args:
            package_path: 仓储包路径
            
        Returns:
            Dict[str, Type[BaseRepository]]: 仓储类字典
        """
        if self._discovered:
            return self.repository_classes
        
        repositories = {}
        
        # 获取仓储目录
        repositories_dir = Path(__file__).parent
        
        # 遍历所有Python文件
        for file_path in repositories_dir.glob("*_repository.py"):
            module_name = file_path.stem
            
            # 跳过基础仓储文件
            if module_name == 'base_repository':
                continue
            
            try:
                # 导入模块
                module = importlib.import_module(f"{package_path}.{module_name}")
                
                # 查找仓储类
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseRepository) and 
                        obj != BaseRepository and 
                        name.endswith('Repository')):
                        repository_name = self._get_repository_name(name)
                        repositories[repository_name] = obj
                        
            except ImportError as e:
                print(f"导入仓储模块失败 {module_name}: {e}")
        
        self.repository_classes.update(repositories)
        self._discovered = True
        return repositories
    
    def create_repository(self, repository_name: str, **kwargs) -> BaseRepository:
        """创建仓储实例
        
        Args:
            repository_name: 仓储名称
            **kwargs: 构造参数
            
        Returns:
            BaseRepository: 仓储实例
        """
        # 确保已发现仓储
        if not self._discovered:
            self.discover_repositories()
        
        # 检查是否已有实例
        if repository_name in self.repository_instances:
            return self.repository_instances[repository_name]
        
        # 获取仓储类
        repository_class = self.repository_classes.get(repository_name)
        if not repository_class:
            raise ValueError(f"未找到仓储: {repository_name}")
        
        # 创建仓储实例
        try:
            repository_instance = repository_class(**kwargs)
            self.repository_instances[repository_name] = repository_instance
            return repository_instance
        except Exception as e:
            raise RuntimeError(f"创建仓储实例失败 {repository_name}: {e}")
    
    def get_repository(self, repository_name: str) -> Optional[BaseRepository]:
        """获取仓储实例
        
        Args:
            repository_name: 仓储名称
            
        Returns:
            Optional[BaseRepository]: 仓储实例
        """
        return self.repository_instances.get(repository_name)
    
    def register_repository(self, repository_name: str, repository_instance: BaseRepository):
        """注册仓储实例
        
        Args:
            repository_name: 仓储名称
            repository_instance: 仓储实例
        """
        self.repository_instances[repository_name] = repository_instance
    
    def register_repository_class(self, repository_name: str, repository_class: Type[BaseRepository]):
        """注册仓储类
        
        Args:
            repository_name: 仓储名称
            repository_class: 仓储类
        """
        self.repository_classes[repository_name] = repository_class
    
    def get_repository_info(self) -> Dict[str, Any]:
        """获取仓储信息
        
        Returns:
            Dict[str, Any]: 仓储信息
        """
        return {
            'discovered_classes': list(self.repository_classes.keys()),
            'created_instances': list(self.repository_instances.keys()),
            'total_classes': len(self.repository_classes),
            'total_instances': len(self.repository_instances)
        }
    
    def _get_repository_name(self, class_name: str) -> str:
        """从类名获取仓储名称
        
        Args:
            class_name: 类名
            
        Returns:
            str: 仓储名称
        """
        # 移除Repository后缀并转换为snake_case
        name = class_name.replace('Repository', '')
        
        # 转换为snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        
        return name


class RepositoryManager:
    """仓储管理器"""
    
    def __init__(self):
        """初始化仓储管理器"""
        self.factory = RepositoryFactory()
        self.container = get_container()
        self._initialized = False
    
    def initialize(self):
        """初始化仓储管理器"""
        if self._initialized:
            return
        
        # 发现所有仓储
        self.factory.discover_repositories()
        
        # 创建核心仓储实例
        self._create_core_repositories()
        
        # 注册仓储到容器
        self._register_repositories_to_container()
        
        self._initialized = True
        print("仓储管理器初始化完成")
    
    def _create_core_repositories(self):
        """创建核心仓储实例"""
        core_repositories = [
            'task',
            'task_queue',
            'task_result',
            'task_statistics',
            'stream',
            'stream_statistics',
            'stream_health',
            'user',
            'user_session',
            'user_preferences',
            'user_activity'
        ]
        
        for repository_name in core_repositories:
            try:
                repository_instance = self.factory.create_repository(repository_name)
                print(f"核心仓储创建成功: {repository_name}")
            except Exception as e:
                print(f"核心仓储创建失败 {repository_name}: {e}")
    
    def _register_repositories_to_container(self):
        """将仓储注册到依赖注入容器"""
        for repository_name, repository_instance in self.factory.repository_instances.items():
            # 注册为单例
            self.container.instance(f"{repository_name}_repository", repository_instance)
            print(f"仓储已注册到容器: {repository_name}_repository")
    
    def get_repository(self, repository_name: str) -> Optional[BaseRepository]:
        """获取仓储实例
        
        Args:
            repository_name: 仓储名称
            
        Returns:
            Optional[BaseRepository]: 仓储实例
        """
        return self.factory.get_repository(repository_name)
    
    def create_repository(self, repository_name: str, **kwargs) -> BaseRepository:
        """创建仓储实例
        
        Args:
            repository_name: 仓储名称
            **kwargs: 构造参数
            
        Returns:
            BaseRepository: 仓储实例
        """
        return self.factory.create_repository(repository_name, **kwargs)
    
    def get_all_repositories(self) -> Dict[str, BaseRepository]:
        """获取所有仓储实例
        
        Returns:
            Dict[str, BaseRepository]: 仓储实例字典
        """
        return self.factory.repository_instances.copy()
    
    def get_repository_status(self) -> Dict[str, Any]:
        """获取仓储状态
        
        Returns:
            Dict[str, Any]: 仓储状态信息
        """
        status = {
            'initialized': self._initialized,
            'repositories': {}
        }
        
        for repository_name, repository_instance in self.factory.repository_instances.items():
            status['repositories'][repository_name] = {
                'class': repository_instance.__class__.__name__,
                'model_class': getattr(repository_instance, 'model_class', None).__name__ if hasattr(repository_instance, 'model_class') else None,
                'status': 'active'
            }
        
        return status
    
    def cleanup_repositories(self):
        """清理仓储资源"""
        print("正在清理仓储资源...")
        
        # 这里可以添加仓储清理逻辑
        for repository_name, repository_instance in self.factory.repository_instances.items():
            try:
                # 如果仓储有cleanup方法，调用它
                if hasattr(repository_instance, 'cleanup'):
                    repository_instance.cleanup()
                print(f"仓储清理完成: {repository_name}")
            except Exception as e:
                print(f"仓储清理失败 {repository_name}: {e}")
        
        self._initialized = False
        print("仓储管理器已清理")


class UnitOfWork:
    """工作单元模式实现"""
    
    def __init__(self, repository_manager: RepositoryManager):
        """初始化工作单元
        
        Args:
            repository_manager: 仓储管理器
        """
        self.repository_manager = repository_manager
        self._repositories: Dict[str, BaseRepository] = {}
        self._transaction_started = False
    
    def __enter__(self):
        """进入上下文"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        if exc_type is not None:
            # 发生异常，回滚事务
            self.rollback()
        else:
            # 正常退出，提交事务
            self.commit()
    
    async def __aenter__(self):
        """异步进入上下文"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步退出上下文"""
        if exc_type is not None:
            # 发生异常，回滚事务
            await self.rollback_async()
        else:
            # 正常退出，提交事务
            await self.commit_async()
    
    def get_repository(self, repository_name: str) -> BaseRepository:
        """获取仓储实例
        
        Args:
            repository_name: 仓储名称
            
        Returns:
            BaseRepository: 仓储实例
        """
        if repository_name not in self._repositories:
            repository = self.repository_manager.get_repository(repository_name)
            if not repository:
                raise ValueError(f"仓储不存在: {repository_name}")
            self._repositories[repository_name] = repository
        
        return self._repositories[repository_name]
    
    async def begin_transaction(self):
        """开始事务"""
        if self._transaction_started:
            return
        
        for repository in self._repositories.values():
            if hasattr(repository, 'begin_transaction'):
                await repository.begin_transaction()
        
        self._transaction_started = True
    
    def commit(self):
        """提交事务（同步）"""
        if not self._transaction_started:
            return
        
        for repository in self._repositories.values():
            if hasattr(repository, 'commit_transaction'):
                # 注意：这里应该是同步版本
                pass
        
        self._transaction_started = False
    
    async def commit_async(self):
        """提交事务（异步）"""
        if not self._transaction_started:
            return
        
        for repository in self._repositories.values():
            if hasattr(repository, 'commit_transaction'):
                await repository.commit_transaction()
        
        self._transaction_started = False
    
    def rollback(self):
        """回滚事务（同步）"""
        if not self._transaction_started:
            return
        
        for repository in self._repositories.values():
            if hasattr(repository, 'rollback_transaction'):
                # 注意：这里应该是同步版本
                pass
        
        self._transaction_started = False
    
    async def rollback_async(self):
        """回滚事务（异步）"""
        if not self._transaction_started:
            return
        
        for repository in self._repositories.values():
            if hasattr(repository, 'rollback_transaction'):
                await repository.rollback_transaction()
        
        self._transaction_started = False


# 全局仓储管理器实例
_repository_manager = None


def get_repository_manager() -> RepositoryManager:
    """获取全局仓储管理器实例
    
    Returns:
        RepositoryManager: 仓储管理器实例
    """
    global _repository_manager
    if _repository_manager is None:
        _repository_manager = RepositoryManager()
    return _repository_manager


def initialize_repositories():
    """初始化所有仓储"""
    manager = get_repository_manager()
    manager.initialize()


def cleanup_repositories():
    """清理所有仓储"""
    manager = get_repository_manager()
    manager.cleanup_repositories()


def create_unit_of_work() -> UnitOfWork:
    """创建工作单元
    
    Returns:
        UnitOfWork: 工作单元实例
    """
    manager = get_repository_manager()
    return UnitOfWork(manager)
