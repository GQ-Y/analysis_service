#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: __init__.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 模型包初始化

统一导出所有数据模型，提供便捷的导入接口。

本文件是分析服务项目的一部分。
"""

# 基础模型
from .base_model import (
    BaseEntity,
    BaseConfig,
    BaseResponse,
    SuccessResponse,
    ErrorResponse,
    PaginatedResponse,
    PaginationMeta,
    ROIConfig,
    AnalysisConfig,
    # 枚举
    TaskStatusEnum,
    StreamStatusEnum,
    AnalysisTypeEnum,
    StreamTypeEnum,
    DeviceTypeEnum,
    PriorityEnum,
    StatusEnum
)

# 任务模型
from .task_model import (
    TaskModel,
    TaskQueue,
    TaskResult,
    TaskStatistics
)

# 流模型
from .stream_model import (
    StreamModel,
    StreamConfig,
    StreamStatistics,
    StreamHealth
)

# 用户模型
from .user_model import (
    UserModel,
    UserSession,
    UserPreferences,
    UserActivity,
    UserStatusEnum,
    RoleEnum
)

# 系统模型
from .system_model import (
    SystemConfig,
    SystemInfo,
    SystemMetrics,
    SystemHealth,
    SystemLog,
    LogLevelEnum,
    ServiceStatusEnum
)

# 响应模型
from .response_model import (
    ResponseModel,
    ErrorResponseModel,
    PaginationResponseModel,
    create_success_response,
    create_error_response,
    create_pagination_response
)

# 模型注册表
from .model_registry import (
    ModelRegistry,
    ModelValidator,
    ModelSerializer,
    get_model_registry,
    get_model_validator,
    get_model_serializer
)

# 导出所有模型类
__all__ = [
    # 基础模型
    'BaseEntity',
    'BaseConfig',
    'BaseResponse',
    'SuccessResponse',
    'ErrorResponse',
    'PaginatedResponse',
    'PaginationMeta',
    'ROIConfig',
    'AnalysisConfig',

    # 枚举
    'TaskStatusEnum',
    'StreamStatusEnum',
    'AnalysisTypeEnum',
    'StreamTypeEnum',
    'DeviceTypeEnum',
    'PriorityEnum',
    'StatusEnum',
    'UserStatusEnum',
    'RoleEnum',
    'LogLevelEnum',
    'ServiceStatusEnum',

    # 任务模型
    'TaskModel',
    'TaskQueue',
    'TaskResult',
    'TaskStatistics',

    # 流模型
    'StreamModel',
    'StreamConfig',
    'StreamStatistics',
    'StreamHealth',

    # 用户模型
    'UserModel',
    'UserSession',
    'UserPreferences',
    'UserActivity',

    # 系统模型
    'SystemConfig',
    'SystemInfo',
    'SystemMetrics',
    'SystemHealth',
    'SystemLog',

    # 响应模型
    'ResponseModel',
    'ErrorResponseModel',
    'PaginationResponseModel',
    'create_success_response',
    'create_error_response',
    'create_pagination_response',

    # 模型注册表
    'ModelRegistry',
    'ModelValidator',
    'ModelSerializer',
    'get_model_registry',
    'get_model_validator',
    'get_model_serializer'
]


def initialize_models():
    """初始化模型系统"""
    registry = get_model_registry()
    registry.discover_models()
    print(f"模型系统初始化完成，发现 {len(registry.models)} 个模型")


def get_all_model_schemas():
    """获取所有模型的JSON Schema"""
    registry = get_model_registry()
    schemas = {}

    for name, model_class in registry.get_all_models().items():
        try:
            schemas[name] = model_class.model_json_schema()
        except Exception as e:
            print(f"获取模型Schema失败 {name}: {e}")

    return schemas
