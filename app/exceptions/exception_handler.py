#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: exception_handler.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 异常处理器

参考Hyperf的异常处理器设计，实现分层异常处理机制。
提供统一的异常处理和响应格式化。

本文件是分析服务项目的一部分。
"""

import logging
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union
from datetime import datetime

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from .business_exception import BusinessException
from app.models.base_model import ErrorResponse


class BaseExceptionHandler(ABC):
    """异常处理器基类"""
    
    def __init__(self):
        """初始化异常处理器"""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def can_handle(self, exception: Exception) -> bool:
        """检查是否可以处理该异常
        
        Args:
            exception: 异常对象
            
        Returns:
            bool: 是否可以处理
        """
        pass
    
    @abstractmethod
    async def handle(self, request: Request, exception: Exception) -> JSONResponse:
        """处理异常
        
        Args:
            request: 请求对象
            exception: 异常对象
            
        Returns:
            JSONResponse: 响应对象
        """
        pass
    
    def get_request_id(self, request: Request) -> Optional[str]:
        """获取请求ID
        
        Args:
            request: 请求对象
            
        Returns:
            Optional[str]: 请求ID
        """
        return getattr(request.state, 'request_id', None)
    
    def log_exception(self, request: Request, exception: Exception, level: str = 'error'):
        """记录异常日志
        
        Args:
            request: 请求对象
            exception: 异常对象
            level: 日志级别
        """
        extra_data = {
            'request_id': self.get_request_id(request),
            'method': request.method,
            'url': str(request.url),
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'user_agent': request.headers.get('user-agent'),
            'client_ip': self._get_client_ip(request)
        }
        
        if level == 'error':
            self.logger.error(
                f"异常处理: {type(exception).__name__}: {str(exception)}",
                exc_info=True,
                extra=extra_data
            )
        elif level == 'warning':
            self.logger.warning(
                f"异常处理: {type(exception).__name__}: {str(exception)}",
                extra=extra_data
            )
        elif level == 'info':
            self.logger.info(
                f"异常处理: {type(exception).__name__}: {str(exception)}",
                extra=extra_data
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP
        
        Args:
            request: 请求对象
            
        Returns:
            str: 客户端IP
        """
        # 检查代理头
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # 返回直接连接的IP
        return request.client.host if request.client else 'unknown'
    
    def create_error_response(
        self,
        message: str,
        code: int = HTTP_500_INTERNAL_SERVER_ERROR,
        error_type: str = None,
        details: Any = None,
        request_id: str = None
    ) -> JSONResponse:
        """创建错误响应
        
        Args:
            message: 错误消息
            code: HTTP状态码
            error_type: 错误类型
            details: 错误详情
            request_id: 请求ID
            
        Returns:
            JSONResponse: 错误响应
        """
        error_response = ErrorResponse(
            code=code,
            message=message,
            error_type=error_type,
            details=details,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=code,
            content=error_response.model_dump()
        )


class BusinessExceptionHandler(BaseExceptionHandler):
    """业务异常处理器"""
    
    def can_handle(self, exception: Exception) -> bool:
        """检查是否可以处理业务异常"""
        return isinstance(exception, BusinessException)
    
    async def handle(self, request: Request, exception: BusinessException) -> JSONResponse:
        """处理业务异常"""
        # 记录警告级别日志
        self.log_exception(request, exception, 'warning')
        
        return self.create_error_response(
            message=str(exception),
            code=exception.status_code,
            error_type='BusinessError',
            details=exception.details,
            request_id=self.get_request_id(request)
        )


class HTTPExceptionHandler(BaseExceptionHandler):
    """HTTP异常处理器"""
    
    def can_handle(self, exception: Exception) -> bool:
        """检查是否可以处理HTTP异常"""
        return isinstance(exception, HTTPException)
    
    async def handle(self, request: Request, exception: HTTPException) -> JSONResponse:
        """处理HTTP异常"""
        # 根据状态码决定日志级别
        level = 'warning' if exception.status_code < 500 else 'error'
        self.log_exception(request, exception, level)
        
        return self.create_error_response(
            message=exception.detail,
            code=exception.status_code,
            error_type='HTTPError',
            request_id=self.get_request_id(request)
        )


class ValidationExceptionHandler(BaseExceptionHandler):
    """验证异常处理器"""
    
    def can_handle(self, exception: Exception) -> bool:
        """检查是否可以处理验证异常"""
        from pydantic import ValidationError
        return isinstance(exception, ValidationError)
    
    async def handle(self, request: Request, exception) -> JSONResponse:
        """处理验证异常"""
        from pydantic import ValidationError
        
        # 记录警告级别日志
        self.log_exception(request, exception, 'warning')
        
        # 格式化验证错误
        errors = []
        for error in exception.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            errors.append({
                'field': field,
                'message': error['msg'],
                'type': error['type']
            })
        
        return self.create_error_response(
            message="数据验证失败",
            code=422,
            error_type='ValidationError',
            details={'errors': errors},
            request_id=self.get_request_id(request)
        )


class DatabaseExceptionHandler(BaseExceptionHandler):
    """数据库异常处理器"""
    
    def can_handle(self, exception: Exception) -> bool:
        """检查是否可以处理数据库异常"""
        # 这里可以根据实际使用的数据库驱动来判断
        exception_name = type(exception).__name__
        database_exceptions = [
            'IntegrityError',
            'OperationalError',
            'DatabaseError',
            'DataError',
            'InterfaceError',
            'InternalError',
            'NotSupportedError',
            'ProgrammingError'
        ]
        return exception_name in database_exceptions
    
    async def handle(self, request: Request, exception: Exception) -> JSONResponse:
        """处理数据库异常"""
        # 记录错误级别日志
        self.log_exception(request, exception, 'error')
        
        # 不暴露具体的数据库错误信息
        return self.create_error_response(
            message="数据库操作失败",
            code=500,
            error_type='DatabaseError',
            request_id=self.get_request_id(request)
        )


class TimeoutExceptionHandler(BaseExceptionHandler):
    """超时异常处理器"""
    
    def can_handle(self, exception: Exception) -> bool:
        """检查是否可以处理超时异常"""
        import asyncio
        return isinstance(exception, (asyncio.TimeoutError, TimeoutError))
    
    async def handle(self, request: Request, exception: Exception) -> JSONResponse:
        """处理超时异常"""
        # 记录警告级别日志
        self.log_exception(request, exception, 'warning')
        
        return self.create_error_response(
            message="请求超时",
            code=408,
            error_type='TimeoutError',
            request_id=self.get_request_id(request)
        )


class PermissionExceptionHandler(BaseExceptionHandler):
    """权限异常处理器"""
    
    def can_handle(self, exception: Exception) -> bool:
        """检查是否可以处理权限异常"""
        exception_name = type(exception).__name__
        return exception_name in ['PermissionError', 'Forbidden', 'Unauthorized']
    
    async def handle(self, request: Request, exception: Exception) -> JSONResponse:
        """处理权限异常"""
        # 记录警告级别日志
        self.log_exception(request, exception, 'warning')
        
        # 根据异常类型确定状态码
        if type(exception).__name__ == 'Unauthorized':
            status_code = 401
            message = "未授权访问"
        else:
            status_code = 403
            message = "权限不足"
        
        return self.create_error_response(
            message=message,
            code=status_code,
            error_type='PermissionError',
            request_id=self.get_request_id(request)
        )


class RateLimitExceptionHandler(BaseExceptionHandler):
    """限流异常处理器"""
    
    def can_handle(self, exception: Exception) -> bool:
        """检查是否可以处理限流异常"""
        exception_name = type(exception).__name__
        return 'RateLimit' in exception_name or 'TooManyRequests' in exception_name
    
    async def handle(self, request: Request, exception: Exception) -> JSONResponse:
        """处理限流异常"""
        # 记录信息级别日志
        self.log_exception(request, exception, 'info')
        
        return self.create_error_response(
            message="请求过于频繁，请稍后再试",
            code=429,
            error_type='RateLimitError',
            request_id=self.get_request_id(request)
        )


class DefaultExceptionHandler(BaseExceptionHandler):
    """默认异常处理器"""
    
    def can_handle(self, exception: Exception) -> bool:
        """默认处理器可以处理所有异常"""
        return True
    
    async def handle(self, request: Request, exception: Exception) -> JSONResponse:
        """处理未知异常"""
        # 记录错误级别日志
        self.log_exception(request, exception, 'error')
        
        # 在开发环境下可以返回详细错误信息
        import os
        debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
        
        if debug_mode:
            details = {
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            }
        else:
            details = None
        
        return self.create_error_response(
            message="服务器内部错误",
            code=500,
            error_type='InternalError',
            details=details,
            request_id=self.get_request_id(request)
        )


class ExceptionHandlerManager:
    """异常处理器管理器"""
    
    def __init__(self):
        """初始化异常处理器管理器"""
        self.handlers: list[BaseExceptionHandler] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 注册默认处理器
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """注册默认异常处理器"""
        # 按优先级顺序注册处理器
        self.register_handler(BusinessExceptionHandler())
        self.register_handler(HTTPExceptionHandler())
        self.register_handler(ValidationExceptionHandler())
        self.register_handler(DatabaseExceptionHandler())
        self.register_handler(TimeoutExceptionHandler())
        self.register_handler(PermissionExceptionHandler())
        self.register_handler(RateLimitExceptionHandler())
        
        # 默认处理器必须最后注册
        self.register_handler(DefaultExceptionHandler())
    
    def register_handler(self, handler: BaseExceptionHandler):
        """注册异常处理器
        
        Args:
            handler: 异常处理器实例
        """
        self.handlers.append(handler)
        self.logger.info(f"注册异常处理器: {handler.__class__.__name__}")
    
    def unregister_handler(self, handler_class: Type[BaseExceptionHandler]):
        """注销异常处理器
        
        Args:
            handler_class: 异常处理器类
        """
        self.handlers = [h for h in self.handlers if not isinstance(h, handler_class)]
        self.logger.info(f"注销异常处理器: {handler_class.__name__}")
    
    async def handle_exception(self, request: Request, exception: Exception) -> JSONResponse:
        """处理异常
        
        Args:
            request: 请求对象
            exception: 异常对象
            
        Returns:
            JSONResponse: 响应对象
        """
        # 查找合适的处理器
        for handler in self.handlers:
            if handler.can_handle(exception):
                try:
                    return await handler.handle(request, exception)
                except Exception as handler_exception:
                    # 处理器本身出错，记录日志并继续查找下一个处理器
                    self.logger.error(
                        f"异常处理器 {handler.__class__.__name__} 处理异常时出错: {handler_exception}",
                        exc_info=True
                    )
                    continue
        
        # 如果所有处理器都失败，返回默认错误响应
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'code': 500,
                'message': '服务器内部错误',
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def get_handlers_info(self) -> list[Dict[str, str]]:
        """获取处理器信息
        
        Returns:
            list[Dict[str, str]]: 处理器信息列表
        """
        return [
            {
                'name': handler.__class__.__name__,
                'module': handler.__class__.__module__
            }
            for handler in self.handlers
        ]


# 全局异常处理器管理器实例
_exception_manager = ExceptionHandlerManager()


def get_exception_manager() -> ExceptionHandlerManager:
    """获取全局异常处理器管理器
    
    Returns:
        ExceptionHandlerManager: 异常处理器管理器
    """
    return _exception_manager


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """全局异常处理函数
    
    Args:
        request: 请求对象
        exc: 异常对象
        
    Returns:
        JSONResponse: 响应对象
    """
    return await _exception_manager.handle_exception(request, exc)
