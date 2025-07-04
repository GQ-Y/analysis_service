#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: cors_middleware.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: CORS中间件

处理跨域资源共享(CORS)请求，设置相应的响应头。

本文件是分析服务项目的一部分。
"""

from typing import Callable, List
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from .base_middleware import BaseMiddleware


class CorsMiddleware(BaseMiddleware):
    """CORS中间件"""
    
    def __init__(self, **kwargs):
        """初始化CORS中间件
        
        Args:
            allow_origins: 允许的源列表
            allow_methods: 允许的HTTP方法列表
            allow_headers: 允许的请求头列表
            allow_credentials: 是否允许携带凭证
            max_age: 预检请求缓存时间
        """
        super().__init__(**kwargs)
        
        self.allow_origins = kwargs.get('allow_origins', ['*'])
        self.allow_methods = kwargs.get('allow_methods', ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
        self.allow_headers = kwargs.get('allow_headers', ['*'])
        self.allow_credentials = kwargs.get('allow_credentials', True)
        self.max_age = kwargs.get('max_age', 86400)  # 24小时
        
        # 处理通配符
        self.allow_all_origins = '*' in self.allow_origins
        self.allow_all_headers = '*' in self.allow_headers
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """处理CORS请求
        
        Args:
            request: HTTP请求对象
            call_next: 下一个中间件或处理器
            
        Returns:
            Response: HTTP响应对象
        """
        if not self.should_process(request):
            return await call_next(request)
        
        origin = request.headers.get('origin')
        
        # 处理预检请求
        if request.method == 'OPTIONS':
            return self._handle_preflight_request(request, origin)
        
        # 处理实际请求
        response = await call_next(request)
        return self._add_cors_headers(response, origin)
    
    def _handle_preflight_request(self, request: Request, origin: str) -> Response:
        """处理预检请求
        
        Args:
            request: HTTP请求对象
            origin: 请求源
            
        Returns:
            Response: 预检响应
        """
        # 检查源是否被允许
        if not self._is_origin_allowed(origin):
            return JSONResponse(
                status_code=403,
                content={"error": "CORS policy violation: Origin not allowed"}
            )
        
        # 检查请求方法是否被允许
        requested_method = request.headers.get('access-control-request-method')
        if requested_method and requested_method not in self.allow_methods:
            return JSONResponse(
                status_code=403,
                content={"error": "CORS policy violation: Method not allowed"}
            )
        
        # 检查请求头是否被允许
        requested_headers = request.headers.get('access-control-request-headers')
        if requested_headers and not self.allow_all_headers:
            requested_header_list = [h.strip() for h in requested_headers.split(',')]
            for header in requested_header_list:
                if header.lower() not in [h.lower() for h in self.allow_headers]:
                    return JSONResponse(
                        status_code=403,
                        content={"error": f"CORS policy violation: Header '{header}' not allowed"}
                    )
        
        # 创建预检响应
        response = Response(status_code=200)
        return self._add_cors_headers(response, origin, is_preflight=True)
    
    def _add_cors_headers(self, response: Response, origin: str, is_preflight: bool = False) -> Response:
        """添加CORS响应头
        
        Args:
            response: HTTP响应对象
            origin: 请求源
            is_preflight: 是否为预检请求
            
        Returns:
            Response: 添加了CORS头的响应
        """
        # 设置允许的源
        if self.allow_all_origins:
            response.headers['Access-Control-Allow-Origin'] = '*'
        elif self._is_origin_allowed(origin):
            response.headers['Access-Control-Allow-Origin'] = origin
        
        # 设置是否允许凭证
        if self.allow_credentials and not self.allow_all_origins:
            response.headers['Access-Control-Allow-Credentials'] = 'true'
        
        # 预检请求的额外头
        if is_preflight:
            # 设置允许的方法
            response.headers['Access-Control-Allow-Methods'] = ', '.join(self.allow_methods)
            
            # 设置允许的头
            if self.allow_all_headers:
                response.headers['Access-Control-Allow-Headers'] = '*'
            else:
                response.headers['Access-Control-Allow-Headers'] = ', '.join(self.allow_headers)
            
            # 设置缓存时间
            response.headers['Access-Control-Max-Age'] = str(self.max_age)
        
        # 设置暴露的头
        response.headers['Access-Control-Expose-Headers'] = 'X-Request-ID, X-Response-Time'
        
        return response
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """检查源是否被允许
        
        Args:
            origin: 请求源
            
        Returns:
            bool: 是否被允许
        """
        if not origin:
            return True  # 同源请求
        
        if self.allow_all_origins:
            return True
        
        return origin in self.allow_origins
    
    async def before_request(self, request: Request) -> None:
        """请求前处理
        
        Args:
            request: HTTP请求对象
        """
        # 记录CORS请求信息
        origin = request.headers.get('origin')
        if origin:
            request.state.cors_origin = origin
            request.state.is_cors_request = True
        else:
            request.state.is_cors_request = False
