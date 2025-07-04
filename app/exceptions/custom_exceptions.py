#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: custom_exceptions.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 自定义异常类

定义应用程序特定的异常类，提供更精确的错误分类和处理。

本文件是分析服务项目的一部分。
"""

from typing import Any, Dict, Optional
from .business_exception import BusinessException


class TaskException(BusinessException):
    """任务相关异常"""
    
    def __init__(self, message: str, task_id: str = None, details: Dict[str, Any] = None):
        """初始化任务异常
        
        Args:
            message: 错误消息
            task_id: 任务ID
            details: 错误详情
        """
        super().__init__(message, status_code=400, details=details)
        self.task_id = task_id


class TaskNotFoundError(TaskException):
    """任务不存在异常"""
    
    def __init__(self, task_id: str):
        """初始化任务不存在异常
        
        Args:
            task_id: 任务ID
        """
        super().__init__(f"任务不存在: {task_id}", task_id=task_id)
        self.status_code = 404


class TaskAlreadyRunningError(TaskException):
    """任务已在运行异常"""
    
    def __init__(self, task_id: str):
        """初始化任务已在运行异常
        
        Args:
            task_id: 任务ID
        """
        super().__init__(f"任务已在运行: {task_id}", task_id=task_id)
        self.status_code = 409


class TaskNotRunningError(TaskException):
    """任务未运行异常"""
    
    def __init__(self, task_id: str, current_status: str):
        """初始化任务未运行异常
        
        Args:
            task_id: 任务ID
            current_status: 当前状态
        """
        super().__init__(
            f"任务未运行，当前状态: {current_status}",
            task_id=task_id,
            details={'current_status': current_status}
        )


class TaskConfigurationError(TaskException):
    """任务配置错误异常"""
    
    def __init__(self, message: str, config_field: str = None, task_id: str = None):
        """初始化任务配置错误异常
        
        Args:
            message: 错误消息
            config_field: 配置字段
            task_id: 任务ID
        """
        details = {'config_field': config_field} if config_field else None
        super().__init__(f"任务配置错误: {message}", task_id=task_id, details=details)


class TaskQuotaExceededError(TaskException):
    """任务配额超限异常"""
    
    def __init__(self, user_id: str, current_count: int, max_count: int):
        """初始化任务配额超限异常
        
        Args:
            user_id: 用户ID
            current_count: 当前任务数
            max_count: 最大任务数
        """
        super().__init__(
            f"任务数量超限，当前: {current_count}，最大: {max_count}",
            details={
                'user_id': user_id,
                'current_count': current_count,
                'max_count': max_count
            }
        )
        self.status_code = 429


class StreamException(BusinessException):
    """流相关异常"""
    
    def __init__(self, message: str, stream_id: str = None, details: Dict[str, Any] = None):
        """初始化流异常
        
        Args:
            message: 错误消息
            stream_id: 流ID
            details: 错误详情
        """
        super().__init__(message, status_code=400, details=details)
        self.stream_id = stream_id


class StreamNotFoundError(StreamException):
    """流不存在异常"""
    
    def __init__(self, stream_id: str):
        """初始化流不存在异常
        
        Args:
            stream_id: 流ID
        """
        super().__init__(f"流不存在: {stream_id}", stream_id=stream_id)
        self.status_code = 404


class StreamConnectionError(StreamException):
    """流连接异常"""
    
    def __init__(self, stream_id: str, reason: str):
        """初始化流连接异常
        
        Args:
            stream_id: 流ID
            reason: 连接失败原因
        """
        super().__init__(
            f"流连接失败: {reason}",
            stream_id=stream_id,
            details={'reason': reason}
        )


class StreamOfflineError(StreamException):
    """流离线异常"""
    
    def __init__(self, stream_id: str):
        """初始化流离线异常
        
        Args:
            stream_id: 流ID
        """
        super().__init__(f"流已离线: {stream_id}", stream_id=stream_id)
        self.status_code = 503


class StreamConfigurationError(StreamException):
    """流配置错误异常"""
    
    def __init__(self, message: str, config_field: str = None, stream_id: str = None):
        """初始化流配置错误异常
        
        Args:
            message: 错误消息
            config_field: 配置字段
            stream_id: 流ID
        """
        details = {'config_field': config_field} if config_field else None
        super().__init__(f"流配置错误: {message}", stream_id=stream_id, details=details)


class UserException(BusinessException):
    """用户相关异常"""
    
    def __init__(self, message: str, user_id: str = None, details: Dict[str, Any] = None):
        """初始化用户异常
        
        Args:
            message: 错误消息
            user_id: 用户ID
            details: 错误详情
        """
        super().__init__(message, status_code=400, details=details)
        self.user_id = user_id


class UserNotFoundError(UserException):
    """用户不存在异常"""
    
    def __init__(self, user_id: str = None, username: str = None, email: str = None):
        """初始化用户不存在异常
        
        Args:
            user_id: 用户ID
            username: 用户名
            email: 邮箱
        """
        if user_id:
            message = f"用户不存在: {user_id}"
        elif username:
            message = f"用户不存在: {username}"
        elif email:
            message = f"用户不存在: {email}"
        else:
            message = "用户不存在"
        
        super().__init__(message, user_id=user_id)
        self.status_code = 404


class UserAlreadyExistsError(UserException):
    """用户已存在异常"""
    
    def __init__(self, field: str, value: str):
        """初始化用户已存在异常
        
        Args:
            field: 字段名
            value: 字段值
        """
        super().__init__(
            f"用户{field}已存在: {value}",
            details={'field': field, 'value': value}
        )
        self.status_code = 409


class UserInactiveError(UserException):
    """用户未激活异常"""
    
    def __init__(self, user_id: str):
        """初始化用户未激活异常
        
        Args:
            user_id: 用户ID
        """
        super().__init__(f"用户未激活: {user_id}", user_id=user_id)
        self.status_code = 403


class AuthenticationError(BusinessException):
    """认证异常"""
    
    def __init__(self, message: str = "认证失败", details: Dict[str, Any] = None):
        """初始化认证异常
        
        Args:
            message: 错误消息
            details: 错误详情
        """
        super().__init__(message, status_code=401, details=details)


class InvalidCredentialsError(AuthenticationError):
    """无效凭据异常"""
    
    def __init__(self):
        """初始化无效凭据异常"""
        super().__init__("用户名或密码错误")


class TokenExpiredError(AuthenticationError):
    """令牌过期异常"""
    
    def __init__(self):
        """初始化令牌过期异常"""
        super().__init__("令牌已过期")


class InvalidTokenError(AuthenticationError):
    """无效令牌异常"""
    
    def __init__(self):
        """初始化无效令牌异常"""
        super().__init__("无效的令牌")


class AuthorizationError(BusinessException):
    """授权异常"""
    
    def __init__(self, message: str = "权限不足", required_permission: str = None):
        """初始化授权异常
        
        Args:
            message: 错误消息
            required_permission: 所需权限
        """
        details = {'required_permission': required_permission} if required_permission else None
        super().__init__(message, status_code=403, details=details)


class InsufficientPermissionError(AuthorizationError):
    """权限不足异常"""
    
    def __init__(self, required_permission: str):
        """初始化权限不足异常
        
        Args:
            required_permission: 所需权限
        """
        super().__init__(
            f"权限不足，需要权限: {required_permission}",
            required_permission=required_permission
        )


class ResourceException(BusinessException):
    """资源相关异常"""
    
    def __init__(self, message: str, resource_type: str = None, resource_id: str = None):
        """初始化资源异常
        
        Args:
            message: 错误消息
            resource_type: 资源类型
            resource_id: 资源ID
        """
        details = {}
        if resource_type:
            details['resource_type'] = resource_type
        if resource_id:
            details['resource_id'] = resource_id
        
        super().__init__(message, details=details if details else None)


class ResourceNotFoundError(ResourceException):
    """资源不存在异常"""
    
    def __init__(self, resource_type: str, resource_id: str):
        """初始化资源不存在异常
        
        Args:
            resource_type: 资源类型
            resource_id: 资源ID
        """
        super().__init__(
            f"{resource_type}不存在: {resource_id}",
            resource_type=resource_type,
            resource_id=resource_id
        )
        self.status_code = 404


class ResourceConflictError(ResourceException):
    """资源冲突异常"""
    
    def __init__(self, resource_type: str, resource_id: str, reason: str):
        """初始化资源冲突异常
        
        Args:
            resource_type: 资源类型
            resource_id: 资源ID
            reason: 冲突原因
        """
        super().__init__(
            f"{resource_type}冲突: {reason}",
            resource_type=resource_type,
            resource_id=resource_id
        )
        self.status_code = 409


class ValidationException(BusinessException):
    """验证异常"""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        """初始化验证异常
        
        Args:
            message: 错误消息
            field: 字段名
            value: 字段值
        """
        details = {}
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = value
        
        super().__init__(message, status_code=422, details=details if details else None)


class RequiredFieldError(ValidationException):
    """必需字段异常"""
    
    def __init__(self, field: str):
        """初始化必需字段异常
        
        Args:
            field: 字段名
        """
        super().__init__(f"字段 {field} 是必需的", field=field)


class InvalidFieldValueError(ValidationException):
    """无效字段值异常"""
    
    def __init__(self, field: str, value: Any, reason: str = None):
        """初始化无效字段值异常
        
        Args:
            field: 字段名
            value: 字段值
            reason: 无效原因
        """
        message = f"字段 {field} 的值无效: {value}"
        if reason:
            message += f" ({reason})"
        
        super().__init__(message, field=field, value=value)


class ConfigurationException(BusinessException):
    """配置异常"""
    
    def __init__(self, message: str, config_key: str = None):
        """初始化配置异常
        
        Args:
            message: 错误消息
            config_key: 配置键
        """
        details = {'config_key': config_key} if config_key else None
        super().__init__(message, status_code=500, details=details)


class MissingConfigurationError(ConfigurationException):
    """缺少配置异常"""
    
    def __init__(self, config_key: str):
        """初始化缺少配置异常
        
        Args:
            config_key: 配置键
        """
        super().__init__(f"缺少配置: {config_key}", config_key=config_key)


class InvalidConfigurationError(ConfigurationException):
    """无效配置异常"""
    
    def __init__(self, config_key: str, reason: str):
        """初始化无效配置异常
        
        Args:
            config_key: 配置键
            reason: 无效原因
        """
        super().__init__(f"配置无效: {config_key} - {reason}", config_key=config_key)


class ExternalServiceException(BusinessException):
    """外部服务异常"""
    
    def __init__(self, service_name: str, message: str, status_code: int = 502):
        """初始化外部服务异常
        
        Args:
            service_name: 服务名称
            message: 错误消息
            status_code: HTTP状态码
        """
        super().__init__(
            f"外部服务 {service_name} 错误: {message}",
            status_code=status_code,
            details={'service_name': service_name}
        )


class ServiceUnavailableError(ExternalServiceException):
    """服务不可用异常"""
    
    def __init__(self, service_name: str):
        """初始化服务不可用异常
        
        Args:
            service_name: 服务名称
        """
        super().__init__(service_name, "服务不可用", status_code=503)


class ServiceTimeoutError(ExternalServiceException):
    """服务超时异常"""
    
    def __init__(self, service_name: str, timeout_seconds: float):
        """初始化服务超时异常
        
        Args:
            service_name: 服务名称
            timeout_seconds: 超时时间
        """
        super().__init__(
            service_name,
            f"服务超时 ({timeout_seconds}秒)",
            status_code=504
        )


class RateLimitException(BusinessException):
    """限流异常"""
    
    def __init__(self, limit: int, window: int, retry_after: int = None):
        """初始化限流异常
        
        Args:
            limit: 限制次数
            window: 时间窗口（秒）
            retry_after: 重试等待时间（秒）
        """
        message = f"请求频率超限，限制: {limit}次/{window}秒"
        if retry_after:
            message += f"，请在 {retry_after} 秒后重试"
        
        details = {
            'limit': limit,
            'window': window,
            'retry_after': retry_after
        }
        
        super().__init__(message, status_code=429, details=details)
