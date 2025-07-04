#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: __init__.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 仓储包初始化

统一导出所有仓储类，提供便捷的导入接口。

本文件是分析服务项目的一部分。
"""

# 基础仓储
from .base_repository import (
    BaseRepository,
    InMemoryRepository
)

# 任务仓储
from .task_repository import (
    TaskRepository,
    TaskQueueRepository,
    TaskResultRepository,
    TaskStatisticsRepository
)

# 流仓储
from .stream_repository import (
    StreamRepository,
    StreamStatisticsRepository,
    StreamHealthRepository
)

# 用户仓储
from .user_repository import (
    UserRepository,
    UserSessionRepository,
    UserPreferencesRepository,
    UserActivityRepository
)

# 仓储工厂
from .repository_factory import (
    RepositoryFactory,
    RepositoryManager,
    UnitOfWork,
    get_repository_manager,
    initialize_repositories,
    cleanup_repositories,
    create_unit_of_work
)

# 导出所有仓储类
__all__ = [
    # 基础仓储
    'BaseRepository',
    'InMemoryRepository',

    # 任务仓储
    'TaskRepository',
    'TaskQueueRepository',
    'TaskResultRepository',
    'TaskStatisticsRepository',

    # 流仓储
    'StreamRepository',
    'StreamStatisticsRepository',
    'StreamHealthRepository',

    # 用户仓储
    'UserRepository',
    'UserSessionRepository',
    'UserPreferencesRepository',
    'UserActivityRepository',

    # 仓储工厂
    'RepositoryFactory',
    'RepositoryManager',
    'UnitOfWork',
    'get_repository_manager',
    'initialize_repositories',
    'cleanup_repositories',
    'create_unit_of_work'
]


def get_repository(repository_name: str):
    """获取仓储实例的便捷函数

    Args:
        repository_name: 仓储名称

    Returns:
        仓储实例或None
    """
    manager = get_repository_manager()
    return manager.get_repository(repository_name)


def create_repository(repository_name: str, **kwargs):
    """创建仓储实例的便捷函数

    Args:
        repository_name: 仓储名称
        **kwargs: 构造参数

    Returns:
        仓储实例
    """
    manager = get_repository_manager()
    return manager.create_repository(repository_name, **kwargs)
