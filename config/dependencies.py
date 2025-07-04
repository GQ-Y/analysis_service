#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: dependencies.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 依赖注入配置文件

定义依赖注入容器的配置，包括接口绑定、单例绑定、工厂绑定等。
参考Hyperf的依赖注入配置方式，使用Python的dependency-injector。

本文件是分析服务项目的一部分。
"""

from typing import Dict, Any
from dependency_injector import containers, providers

# 依赖注入绑定配置
DEPENDENCY_BINDINGS = {
    # 接口绑定
    "interfaces": {
        "core.analyzers.analyzer_interface.AnalyzerInterface": "core.analyzers.yolo_analyzer.YoloAnalyzer",
        "core.streams.stream_interface.StreamInterface": "core.streams.rtsp_stream.RtspStream",
        "app.repositories.task_repository.TaskRepositoryInterface": "app.repositories.task_repository.TaskRepository",
        "app.repositories.stream_repository.StreamRepositoryInterface": "app.repositories.stream_repository.StreamRepository",
    },
    
    # 单例绑定
    "singletons": [
        "core.memory.memory_manager.MemoryManager",
        "core.container.container.Container",
        "app.services.discovery_service.DiscoveryService",
        "shared.utils.logger.Logger",
        "core.redis_manager.RedisManager",
    ],
    
    # 工厂绑定
    "factories": {
        "core.analyzers.factory.analyzer_factory.AnalyzerFactory": {
            "config": "config.plugins.analysis.ANALYSIS_PLUGIN.config.models"
        },
        "app.services.task_service.TaskService": {
            "repository": "app.repositories.task_repository.TaskRepository",
            "analyzer_factory": "core.analyzers.factory.analyzer_factory.AnalyzerFactory",
        },
        "app.services.stream_service.StreamService": {
            "repository": "app.repositories.stream_repository.StreamRepository",
            "stream_manager": "core.streams.stream_manager.StreamManager",
        },
    },
}

# 服务提供者配置
SERVICE_PROVIDERS = [
    "core.container.service_provider.CoreServiceProvider",
    "app.providers.app_service_provider.AppServiceProvider", 
    "plugins.analysis.providers.analysis_service_provider.AnalysisServiceProvider",
    "plugins.streaming.providers.streaming_service_provider.StreamingServiceProvider",
    "plugins.monitoring.providers.monitoring_service_provider.MonitoringServiceProvider",
]

# 依赖注入容器配置类
class DependencyContainer(containers.DeclarativeContainer):
    """依赖注入容器配置"""
    
    # 配置提供者
    config = providers.Configuration()
    
    # Redis连接（暂时跳过，避免配置复杂性）
    # redis_client = providers.Singleton("redis.Redis")
    
    # 内存管理器
    memory_manager = providers.Singleton(
        "app.core.memory.memory_manager.MemoryManager"
    )
    
    # 分析器工厂
    analyzer_factory = providers.Factory(
        "app.core.analyzer.analyzer_factory.AnalyzerFactory"
    )

    # 存储管理器
    storage_manager = providers.Singleton(
        "app.core.storage.storage_manager.StorageManager"
    )

    # 存储服务
    storage_service = providers.Singleton(
        "app.services.storage_service.StorageService"
    )
    
    # 注意：其他服务和仓储通过工厂函数获取，避免循环导入

# FastAPI依赖注入函数
def get_container() -> DependencyContainer:
    """获取依赖注入容器"""
    return DependencyContainer()

def get_memory_manager():
    """获取内存管理器"""
    container = get_container()
    return container.memory_manager()

def get_analyzer_factory():
    """获取分析器工厂"""
    container = get_container()
    return container.analyzer_factory()


def get_storage_manager():
    """获取存储管理器"""
    from app.core.storage import get_storage_manager
    return get_storage_manager()


def get_storage_service():
    """获取存储服务"""
    from app.services.storage_service import get_storage_service
    return get_storage_service()

# 依赖注入装饰器配置
DEPENDENCY_DECORATORS = {
    "memory_manager": get_memory_manager,
    "analyzer_factory": get_analyzer_factory,
    "storage_manager": get_storage_manager,
    "storage_service": get_storage_service,
}


async def setup_dependencies(container):
    """设置依赖注入

    Args:
        container: 依赖注入容器
    """
    # 这里可以添加依赖注入设置逻辑
    # 由于我们还没有实现具体的服务，暂时跳过
    pass
