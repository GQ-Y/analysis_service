#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: logging_middleware.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 日志中间件

记录HTTP请求和响应的详细信息，包括请求时间、响应时间、状态码等。

本文件是分析服务项目的一部分。
"""

import time
import json
import logging
from typing import Callable
from fastapi import Request, Response
from .base_middleware import BaseMiddleware


class LoggingMiddleware(BaseMiddleware):
    """日志中间件"""
    
    def __init__(self, **kwargs):
        """初始化日志中间件
        
        Args:
            log_requests: 是否记录请求
            log_responses: 是否记录响应
            log_body: 是否记录请求体
            exclude_paths: 排除的路径列表
            logger_name: 日志器名称
        """
        super().__init__(**kwargs)
        
        self.log_requests = kwargs.get('log_requests', True)
        self.log_responses = kwargs.get('log_responses', True)
        self.log_body = kwargs.get('log_body', False)
        self.exclude_paths = kwargs.get('exclude_paths', [])
        
        # 初始化日志器
        logger_name = kwargs.get('logger_name', 'middleware.logging')
        self.logger = logging.getLogger(logger_name)
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """处理请求日志
        
        Args:
            request: HTTP请求对象
            call_next: 下一个中间件或处理器
            
        Returns:
            Response: HTTP响应对象
        """
        if not self.should_process(request):
            return await call_next(request)
        
        # 记录开始时间
        start_time = time.time()
        request_id = self.get_request_id(request)
        
        # 记录请求信息
        if self.log_requests:
            await self._log_request(request, request_id)
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录响应信息
            if self.log_responses:
                await self._log_response(request, response, process_time, request_id)
            
            # 添加响应时间头
            response.headers['X-Response-Time'] = f"{process_time:.3f}s"
            response.headers['X-Request-ID'] = request_id
            
            return response
            
        except Exception as e:
            # 记录异常
            process_time = time.time() - start_time
            await self._log_exception(request, e, process_time, request_id)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """记录请求信息
        
        Args:
            request: HTTP请求对象
            request_id: 请求ID
        """
        # 基本请求信息
        log_data = {
            'request_id': request_id,
            'method': request.method,
            'url': str(request.url),
            'path': request.url.path,
            'query_params': dict(request.query_params),
            'client_ip': self.get_client_ip(request),
            'user_agent': request.headers.get('user-agent', ''),
            'content_type': request.headers.get('content-type', ''),
            'content_length': request.headers.get('content-length', '0'),
        }
        
        # 记录请求头
        if self.params.get('log_headers', False):
            log_data['headers'] = dict(request.headers)
        
        # 记录请求体
        if self.log_body and request.method in ['POST', 'PUT', 'PATCH']:
            try:
                body = await self._get_request_body(request)
                if body:
                    log_data['body'] = body
            except Exception as e:
                log_data['body_error'] = str(e)
        
        self.logger.info(f"请求开始: {request.method} {request.url.path}", extra=log_data)
    
    async def _log_response(self, request: Request, response: Response, process_time: float, request_id: str):
        """记录响应信息
        
        Args:
            request: HTTP请求对象
            response: HTTP响应对象
            process_time: 处理时间
            request_id: 请求ID
        """
        log_data = {
            'request_id': request_id,
            'method': request.method,
            'path': request.url.path,
            'status_code': response.status_code,
            'process_time': f"{process_time:.3f}s",
            'response_size': response.headers.get('content-length', '0'),
        }
        
        # 记录响应头
        if self.params.get('log_headers', False):
            log_data['response_headers'] = dict(response.headers)
        
        # 根据状态码选择日志级别
        if response.status_code >= 500:
            log_level = logging.ERROR
            message = f"请求完成(服务器错误): {request.method} {request.url.path}"
        elif response.status_code >= 400:
            log_level = logging.WARNING
            message = f"请求完成(客户端错误): {request.method} {request.url.path}"
        else:
            log_level = logging.INFO
            message = f"请求完成: {request.method} {request.url.path}"
        
        self.logger.log(log_level, message, extra=log_data)
    
    async def _log_exception(self, request: Request, exception: Exception, process_time: float, request_id: str):
        """记录异常信息
        
        Args:
            request: HTTP请求对象
            exception: 异常对象
            process_time: 处理时间
            request_id: 请求ID
        """
        log_data = {
            'request_id': request_id,
            'method': request.method,
            'path': request.url.path,
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'process_time': f"{process_time:.3f}s",
        }
        
        self.logger.error(
            f"请求异常: {request.method} {request.url.path} - {type(exception).__name__}: {exception}",
            extra=log_data,
            exc_info=True
        )
    
    async def _get_request_body(self, request: Request) -> str:
        """获取请求体内容
        
        Args:
            request: HTTP请求对象
            
        Returns:
            str: 请求体内容
        """
        try:
            # 读取请求体
            body = await request.body()
            if not body:
                return ""
            
            # 尝试解析为JSON
            content_type = request.headers.get('content-type', '')
            if 'application/json' in content_type:
                try:
                    json_data = json.loads(body.decode('utf-8'))
                    # 过滤敏感信息
                    return self._filter_sensitive_data(json_data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            
            # 对于非JSON内容，只记录前1000个字符
            body_str = body.decode('utf-8', errors='ignore')
            if len(body_str) > 1000:
                body_str = body_str[:1000] + "...(truncated)"
            
            return body_str
            
        except Exception:
            return "[无法读取请求体]"
    
    def _filter_sensitive_data(self, data) -> str:
        """过滤敏感数据
        
        Args:
            data: 要过滤的数据
            
        Returns:
            str: 过滤后的JSON字符串
        """
        if isinstance(data, dict):
            filtered_data = {}
            sensitive_keys = ['password', 'token', 'secret', 'key', 'auth']
            
            for key, value in data.items():
                if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                    filtered_data[key] = "[已过滤]"
                elif isinstance(value, (dict, list)):
                    filtered_data[key] = self._filter_sensitive_data(value)
                else:
                    filtered_data[key] = value
            
            return json.dumps(filtered_data, ensure_ascii=False, indent=2)
        
        elif isinstance(data, list):
            return json.dumps([self._filter_sensitive_data(item) for item in data], ensure_ascii=False, indent=2)
        
        else:
            return json.dumps(data, ensure_ascii=False, indent=2)
