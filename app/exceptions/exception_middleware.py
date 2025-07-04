#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: exception_middleware.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 异常处理中间件

提供全局异常捕获和处理，集成请求追踪和监控。

本文件是分析服务项目的一部分。
"""

import time
import uuid
import logging
from typing import Callable, Any
from datetime import datetime

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .exception_handler import get_exception_manager
from app.models.base_model import ErrorResponse


class ExceptionMiddleware(BaseHTTPMiddleware):
    """异常处理中间件"""
    
    def __init__(self, app: ASGIApp, debug: bool = False):
        """初始化异常处理中间件
        
        Args:
            app: ASGI应用
            debug: 是否调试模式
        """
        super().__init__(app)
        self.debug = debug
        self.logger = logging.getLogger(self.__class__.__name__)
        self.exception_manager = get_exception_manager()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求
        
        Args:
            request: 请求对象
            call_next: 下一个处理函数
            
        Returns:
            Response: 响应对象
        """
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 记录请求开始时间
        start_time = time.time()
        request.state.start_time = start_time
        
        # 记录请求信息
        self._log_request_start(request, request_id)
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 记录请求完成
            self._log_request_complete(request, response, start_time)
            
            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{(time.time() - start_time) * 1000:.2f}ms"
            
            return response
            
        except Exception as exc:
            # 处理异常
            return await self._handle_exception(request, exc, start_time)
    
    async def _handle_exception(self, request: Request, exc: Exception, start_time: float) -> JSONResponse:
        """处理异常
        
        Args:
            request: 请求对象
            exc: 异常对象
            start_time: 请求开始时间
            
        Returns:
            JSONResponse: 错误响应
        """
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录异常信息
        self._log_exception(request, exc, process_time)
        
        # 使用异常管理器处理异常
        response = await self.exception_manager.handle_exception(request, exc)
        
        # 添加响应头
        request_id = getattr(request.state, 'request_id', 'unknown')
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{process_time * 1000:.2f}ms"
        response.headers["X-Error-Handled"] = "true"
        
        return response
    
    def _log_request_start(self, request: Request, request_id: str):
        """记录请求开始
        
        Args:
            request: 请求对象
            request_id: 请求ID
        """
        self.logger.info(
            f"请求开始: {request.method} {request.url}",
            extra={
                'request_id': request_id,
                'method': request.method,
                'url': str(request.url),
                'user_agent': request.headers.get('user-agent'),
                'client_ip': self._get_client_ip(request),
                'content_type': request.headers.get('content-type'),
                'content_length': request.headers.get('content-length')
            }
        )
    
    def _log_request_complete(self, request: Request, response: Response, start_time: float):
        """记录请求完成
        
        Args:
            request: 请求对象
            response: 响应对象
            start_time: 请求开始时间
        """
        process_time = time.time() - start_time
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        self.logger.info(
            f"请求完成: {request.method} {request.url} - {response.status_code}",
            extra={
                'request_id': request_id,
                'method': request.method,
                'url': str(request.url),
                'status_code': response.status_code,
                'process_time': process_time,
                'response_size': len(response.body) if hasattr(response, 'body') else None
            }
        )
    
    def _log_exception(self, request: Request, exc: Exception, process_time: float):
        """记录异常信息
        
        Args:
            request: 请求对象
            exc: 异常对象
            process_time: 处理时间
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        self.logger.error(
            f"请求异常: {request.method} {request.url} - {type(exc).__name__}: {str(exc)}",
            exc_info=True,
            extra={
                'request_id': request_id,
                'method': request.method,
                'url': str(request.url),
                'exception_type': type(exc).__name__,
                'exception_message': str(exc),
                'process_time': process_time,
                'client_ip': self._get_client_ip(request),
                'user_agent': request.headers.get('user-agent')
            }
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


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """请求追踪中间件"""
    
    def __init__(self, app: ASGIApp):
        """初始化请求追踪中间件
        
        Args:
            app: ASGI应用
        """
        super().__init__(app)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求
        
        Args:
            request: 请求对象
            call_next: 下一个处理函数
            
        Returns:
            Response: 响应对象
        """
        # 如果没有请求ID，生成一个
        if not hasattr(request.state, 'request_id'):
            request.state.request_id = str(uuid.uuid4())
        
        # 记录请求开始时间
        if not hasattr(request.state, 'start_time'):
            request.state.start_time = time.time()
        
        # 添加追踪信息到请求状态
        request.state.trace_info = {
            'request_id': request.state.request_id,
            'start_time': request.state.start_time,
            'method': request.method,
            'url': str(request.url),
            'client_ip': self._get_client_ip(request),
            'user_agent': request.headers.get('user-agent'),
            'stages': []
        }
        
        # 添加请求开始阶段
        self._add_trace_stage(request, 'request_start', 'Request received')
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 添加请求完成阶段
            self._add_trace_stage(request, 'request_complete', f'Response: {response.status_code}')
            
            # 添加追踪头到响应
            self._add_trace_headers(request, response)
            
            return response
            
        except Exception as exc:
            # 添加异常阶段
            self._add_trace_stage(request, 'exception', f'Exception: {type(exc).__name__}')
            raise
    
    def _add_trace_stage(self, request: Request, stage: str, description: str):
        """添加追踪阶段
        
        Args:
            request: 请求对象
            stage: 阶段名称
            description: 阶段描述
        """
        if hasattr(request.state, 'trace_info'):
            current_time = time.time()
            start_time = request.state.trace_info['start_time']
            
            request.state.trace_info['stages'].append({
                'stage': stage,
                'description': description,
                'timestamp': current_time,
                'elapsed_ms': (current_time - start_time) * 1000
            })
    
    def _add_trace_headers(self, request: Request, response: Response):
        """添加追踪头到响应
        
        Args:
            request: 请求对象
            response: 响应对象
        """
        if hasattr(request.state, 'trace_info'):
            trace_info = request.state.trace_info
            
            # 添加基本追踪信息
            response.headers["X-Request-ID"] = trace_info['request_id']
            
            # 计算总处理时间
            total_time = (time.time() - trace_info['start_time']) * 1000
            response.headers["X-Response-Time"] = f"{total_time:.2f}ms"
            
            # 添加阶段数量
            response.headers["X-Trace-Stages"] = str(len(trace_info['stages']))
    
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


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """性能监控中间件"""
    
    def __init__(self, app: ASGIApp, slow_request_threshold: float = 1.0):
        """初始化性能监控中间件
        
        Args:
            app: ASGI应用
            slow_request_threshold: 慢请求阈值（秒）
        """
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self.request_stats = {
            'total_requests': 0,
            'slow_requests': 0,
            'error_requests': 0,
            'total_time': 0.0
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求
        
        Args:
            request: 请求对象
            call_next: 下一个处理函数
            
        Returns:
            Response: 响应对象
        """
        start_time = time.time()
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 更新统计信息
            self._update_stats(process_time, False)
            
            # 检查是否为慢请求
            if process_time > self.slow_request_threshold:
                self._log_slow_request(request, process_time)
            
            # 添加性能头
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            
            return response
            
        except Exception as exc:
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 更新统计信息（错误请求）
            self._update_stats(process_time, True)
            
            raise
    
    def _update_stats(self, process_time: float, is_error: bool):
        """更新统计信息
        
        Args:
            process_time: 处理时间
            is_error: 是否为错误请求
        """
        self.request_stats['total_requests'] += 1
        self.request_stats['total_time'] += process_time
        
        if process_time > self.slow_request_threshold:
            self.request_stats['slow_requests'] += 1
        
        if is_error:
            self.request_stats['error_requests'] += 1
    
    def _log_slow_request(self, request: Request, process_time: float):
        """记录慢请求
        
        Args:
            request: 请求对象
            process_time: 处理时间
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        self.logger.warning(
            f"慢请求检测: {request.method} {request.url} - {process_time:.4f}s",
            extra={
                'request_id': request_id,
                'method': request.method,
                'url': str(request.url),
                'process_time': process_time,
                'threshold': self.slow_request_threshold,
                'client_ip': self._get_client_ip(request)
            }
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
    
    def get_stats(self) -> dict:
        """获取统计信息
        
        Returns:
            dict: 统计信息
        """
        total_requests = self.request_stats['total_requests']
        
        if total_requests == 0:
            return {
                'total_requests': 0,
                'slow_requests': 0,
                'error_requests': 0,
                'average_time': 0.0,
                'slow_request_rate': 0.0,
                'error_rate': 0.0
            }
        
        return {
            'total_requests': total_requests,
            'slow_requests': self.request_stats['slow_requests'],
            'error_requests': self.request_stats['error_requests'],
            'average_time': self.request_stats['total_time'] / total_requests,
            'slow_request_rate': self.request_stats['slow_requests'] / total_requests,
            'error_rate': self.request_stats['error_requests'] / total_requests
        }


def create_exception_middleware(debug: bool = False) -> ExceptionMiddleware:
    """创建异常处理中间件
    
    Args:
        debug: 是否调试模式
        
    Returns:
        ExceptionMiddleware: 异常处理中间件
    """
    return ExceptionMiddleware(debug=debug)


def create_request_tracking_middleware() -> RequestTrackingMiddleware:
    """创建请求追踪中间件
    
    Returns:
        RequestTrackingMiddleware: 请求追踪中间件
    """
    return RequestTrackingMiddleware


def create_performance_monitoring_middleware(slow_request_threshold: float = 1.0) -> PerformanceMonitoringMiddleware:
    """创建性能监控中间件
    
    Args:
        slow_request_threshold: 慢请求阈值（秒）
        
    Returns:
        PerformanceMonitoringMiddleware: 性能监控中间件
    """
    return PerformanceMonitoringMiddleware(slow_request_threshold=slow_request_threshold)
