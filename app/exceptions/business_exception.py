#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: business_exception.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 业务异常

定义业务逻辑相关的异常类。

本文件是分析服务项目的一部分。
"""

from typing import Any, Dict, Optional


class BusinessException(Exception):
    """业务异常基类"""
    
    def __init__(
        self,
        message: str = "业务异常",
        error_code: str = "BUSINESS_ERROR",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 400
    ):
        """初始化业务异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            details: 错误详情
            status_code: HTTP状态码
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "status_code": self.status_code
        }
    
    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"
    
    def __repr__(self) -> str:
        return f"BusinessException(error_code='{self.error_code}', message='{self.message}')"


class ValidationException(BusinessException):
    """验证异常"""
    
    def __init__(self, message: str = "验证失败", field: str = None, **kwargs):
        super().__init__(message, "VALIDATION_ERROR", **kwargs)
        if field:
            self.details["field"] = field


class AuthenticationException(BusinessException):
    """认证异常"""
    
    def __init__(self, message: str = "认证失败", **kwargs):
        super().__init__(message, "AUTHENTICATION_ERROR", status_code=401, **kwargs)


class AuthorizationException(BusinessException):
    """授权异常"""
    
    def __init__(self, message: str = "权限不足", **kwargs):
        super().__init__(message, "AUTHORIZATION_ERROR", status_code=403, **kwargs)


class ResourceNotFoundException(BusinessException):
    """资源未找到异常"""
    
    def __init__(self, message: str = "资源未找到", resource_type: str = None, **kwargs):
        super().__init__(message, "RESOURCE_NOT_FOUND", status_code=404, **kwargs)
        if resource_type:
            self.details["resource_type"] = resource_type


class ResourceConflictException(BusinessException):
    """资源冲突异常"""
    
    def __init__(self, message: str = "资源冲突", **kwargs):
        super().__init__(message, "RESOURCE_CONFLICT", status_code=409, **kwargs)


class RateLimitException(BusinessException):
    """限流异常"""
    
    def __init__(self, message: str = "请求过于频繁", **kwargs):
        super().__init__(message, "RATE_LIMIT_EXCEEDED", status_code=429, **kwargs)


class ServiceUnavailableException(BusinessException):
    """服务不可用异常"""
    
    def __init__(self, message: str = "服务不可用", **kwargs):
        super().__init__(message, "SERVICE_UNAVAILABLE", status_code=503, **kwargs)
