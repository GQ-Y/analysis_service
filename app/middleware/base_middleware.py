#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: base_middleware.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 基础中间件类

定义中间件的基础接口和通用功能。
所有中间件都应该继承此基类。

本文件是分析服务项目的一部分。
"""

from abc import ABC, abstractmethod
from typing import Callable, Any, Optional
from fastapi import Request, Response
import time
import uuid


class BaseMiddleware(ABC):
    """中间件基类"""
    
    def __init__(self, **kwargs):
        """初始化中间件
        
        Args:
            **kwargs: 中间件参数
        """
        self.params = kwargs
        self.enabled = kwargs.get('enabled', True)
    
    @abstractmethod
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """中间件处理方法
        
        Args:
            request: HTTP请求对象
            call_next: 下一个中间件或处理器
            
        Returns:
            Response: HTTP响应对象
        """
        pass
    
    def is_enabled(self) -> bool:
        """检查中间件是否启用
        
        Returns:
            bool: 是否启用
        """
        return self.enabled
    
    def should_process(self, request: Request) -> bool:
        """判断是否应该处理此请求
        
        Args:
            request: HTTP请求对象
            
        Returns:
            bool: 是否应该处理
        """
        # 检查排除路径
        exclude_paths = self.params.get('exclude_paths', [])
        if request.url.path in exclude_paths:
            return False
            
        # 检查包含路径
        include_paths = self.params.get('include_paths', [])
        if include_paths and request.url.path not in include_paths:
            return False
            
        return True
    
    def get_request_id(self, request: Request) -> str:
        """获取或生成请求ID
        
        Args:
            request: HTTP请求对象
            
        Returns:
            str: 请求ID
        """
        request_id = request.headers.get('X-Request-ID')
        if not request_id:
            request_id = str(uuid.uuid4())
            # 将请求ID存储到请求状态中
            if hasattr(request, 'state'):
                request.state.request_id = request_id
        return request_id
    
    def get_client_ip(self, request: Request) -> str:
        """获取客户端IP地址
        
        Args:
            request: HTTP请求对象
            
        Returns:
            str: 客户端IP地址
        """
        # 检查代理头
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
            
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
            
        # 使用客户端IP
        if request.client:
            return request.client.host
            
        return 'unknown'
    
    async def before_request(self, request: Request) -> Optional[Response]:
        """请求前处理
        
        Args:
            request: HTTP请求对象
            
        Returns:
            Optional[Response]: 如果返回Response则中断处理链
        """
        return None
    
    async def after_request(self, request: Request, response: Response) -> Response:
        """请求后处理
        
        Args:
            request: HTTP请求对象
            response: HTTP响应对象
            
        Returns:
            Response: 处理后的响应对象
        """
        return response
    
    async def on_exception(self, request: Request, exception: Exception) -> Optional[Response]:
        """异常处理
        
        Args:
            request: HTTP请求对象
            exception: 异常对象
            
        Returns:
            Optional[Response]: 如果返回Response则处理异常
        """
        return None


class MiddlewarePipeline:
    """中间件管道"""
    
    def __init__(self):
        self.middlewares = []
    
    def add_middleware(self, middleware: BaseMiddleware):
        """添加中间件
        
        Args:
            middleware: 中间件实例
        """
        if middleware.is_enabled():
            self.middlewares.append(middleware)
    
    def add_middlewares(self, middlewares: list):
        """批量添加中间件
        
        Args:
            middlewares: 中间件列表
        """
        for middleware in middlewares:
            self.add_middleware(middleware)
    
    async def process(self, request: Request, handler: Callable) -> Response:
        """处理请求
        
        Args:
            request: HTTP请求对象
            handler: 最终处理器
            
        Returns:
            Response: HTTP响应对象
        """
        async def call_next(index: int = 0):
            if index >= len(self.middlewares):
                # 到达管道末端，调用最终处理器
                return await handler(request)
            
            middleware = self.middlewares[index]
            
            # 检查是否应该处理此请求
            if not middleware.should_process(request):
                return await call_next(index + 1)
            
            try:
                # 请求前处理
                early_response = await middleware.before_request(request)
                if early_response:
                    return early_response
                
                # 调用下一个中间件
                response = await middleware(request, lambda req: call_next(index + 1))
                
                # 请求后处理
                return await middleware.after_request(request, response)
                
            except Exception as e:
                # 异常处理
                exception_response = await middleware.on_exception(request, e)
                if exception_response:
                    return exception_response
                raise
        
        return await call_next()
