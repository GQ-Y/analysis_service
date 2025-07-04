#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: __init__.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 异常包初始化

统一导出所有异常类和异常处理组件，提供便捷的导入接口。

本文件是分析服务项目的一部分。
"""

# 基础异常
from .business_exception import BusinessException

# 异常处理器
from .exception_handler import (
    BaseExceptionHandler,
    BusinessExceptionHandler,
    HTTPExceptionHandler,
    ValidationExceptionHandler,
    DatabaseExceptionHandler,
    TimeoutExceptionHandler,
    PermissionExceptionHandler,
    RateLimitExceptionHandler,
    DefaultExceptionHandler,
    ExceptionHandlerManager,
    get_exception_manager,
    global_exception_handler
)

# 自定义异常
from .custom_exceptions import (
    # 任务异常
    TaskException,
    TaskNotFoundError,
    TaskAlreadyRunningError,
    TaskNotRunningError,
    TaskConfigurationError,
    TaskQuotaExceededError,

    # 流异常
    StreamException,
    StreamNotFoundError,
    StreamConnectionError,
    StreamOfflineError,
    StreamConfigurationError,

    # 用户异常
    UserException,
    UserNotFoundError,
    UserAlreadyExistsError,
    UserInactiveError,

    # 认证异常
    AuthenticationError,
    InvalidCredentialsError,
    TokenExpiredError,
    InvalidTokenError,

    # 授权异常
    AuthorizationError,
    InsufficientPermissionError,

    # 资源异常
    ResourceException,
    ResourceNotFoundError,
    ResourceConflictError,

    # 验证异常
    ValidationException,
    RequiredFieldError,
    InvalidFieldValueError,

    # 配置异常
    ConfigurationException,
    MissingConfigurationError,
    InvalidConfigurationError,

    # 外部服务异常
    ExternalServiceException,
    ServiceUnavailableError,
    ServiceTimeoutError,

    # 限流异常
    RateLimitException
)

# 异常中间件
from .exception_middleware import (
    ExceptionMiddleware,
    RequestTrackingMiddleware,
    PerformanceMonitoringMiddleware,
    create_exception_middleware,
    create_request_tracking_middleware,
    create_performance_monitoring_middleware
)

# 导出所有异常类和组件
__all__ = [
    # 基础异常
    'BusinessException',

    # 异常处理器
    'BaseExceptionHandler',
    'BusinessExceptionHandler',
    'HTTPExceptionHandler',
    'ValidationExceptionHandler',
    'DatabaseExceptionHandler',
    'TimeoutExceptionHandler',
    'PermissionExceptionHandler',
    'RateLimitExceptionHandler',
    'DefaultExceptionHandler',
    'ExceptionHandlerManager',
    'get_exception_manager',
    'global_exception_handler',

    # 任务异常
    'TaskException',
    'TaskNotFoundError',
    'TaskAlreadyRunningError',
    'TaskNotRunningError',
    'TaskConfigurationError',
    'TaskQuotaExceededError',

    # 流异常
    'StreamException',
    'StreamNotFoundError',
    'StreamConnectionError',
    'StreamOfflineError',
    'StreamConfigurationError',

    # 用户异常
    'UserException',
    'UserNotFoundError',
    'UserAlreadyExistsError',
    'UserInactiveError',

    # 认证异常
    'AuthenticationError',
    'InvalidCredentialsError',
    'TokenExpiredError',
    'InvalidTokenError',

    # 授权异常
    'AuthorizationError',
    'InsufficientPermissionError',

    # 资源异常
    'ResourceException',
    'ResourceNotFoundError',
    'ResourceConflictError',

    # 验证异常
    'ValidationException',
    'RequiredFieldError',
    'InvalidFieldValueError',

    # 配置异常
    'ConfigurationException',
    'MissingConfigurationError',
    'InvalidConfigurationError',

    # 外部服务异常
    'ExternalServiceException',
    'ServiceUnavailableError',
    'ServiceTimeoutError',

    # 限流异常
    'RateLimitException',

    # 异常中间件
    'ExceptionMiddleware',
    'RequestTrackingMiddleware',
    'PerformanceMonitoringMiddleware',
    'create_exception_middleware',
    'create_request_tracking_middleware',
    'create_performance_monitoring_middleware'
]
