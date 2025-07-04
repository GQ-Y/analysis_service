#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: route.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 路由装饰器

提供路由定义的装饰器，简化路由配置。
支持HTTP方法、路径参数、查询参数、中间件等配置。

本文件是分析服务项目的一部分。
"""

from typing import List, Dict, Any, Optional, Callable, Union
from functools import wraps


def route(
    path: str,
    methods: List[str] = None,
    middleware: List[str] = None,
    tags: List[str] = None,
    summary: str = None,
    description: str = None,
    response_model: Any = None,
    status_code: int = 200,
    **kwargs
):
    """路由装饰器
    
    Args:
        path: 路由路径
        methods: HTTP方法列表
        middleware: 中间件列表
        tags: API标签
        summary: API摘要
        description: API描述
        response_model: 响应模型
        status_code: 状态码
        **kwargs: 其他参数
    """
    def decorator(func: Callable):
        # 存储路由元数据
        func._route_meta = {
            'path': path,
            'methods': methods or ['GET'],
            'middleware': middleware or [],
            'tags': tags or [],
            'summary': summary or func.__name__,
            'description': description or func.__doc__,
            'response_model': response_model,
            'status_code': status_code,
            **kwargs
        }
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def get(
    path: str,
    middleware: List[str] = None,
    tags: List[str] = None,
    summary: str = None,
    description: str = None,
    response_model: Any = None,
    **kwargs
):
    """GET路由装饰器
    
    Args:
        path: 路由路径
        middleware: 中间件列表
        tags: API标签
        summary: API摘要
        description: API描述
        response_model: 响应模型
        **kwargs: 其他参数
    """
    return route(
        path=path,
        methods=['GET'],
        middleware=middleware,
        tags=tags,
        summary=summary,
        description=description,
        response_model=response_model,
        **kwargs
    )


def post(
    path: str,
    middleware: List[str] = None,
    tags: List[str] = None,
    summary: str = None,
    description: str = None,
    response_model: Any = None,
    status_code: int = 201,
    **kwargs
):
    """POST路由装饰器
    
    Args:
        path: 路由路径
        middleware: 中间件列表
        tags: API标签
        summary: API摘要
        description: API描述
        response_model: 响应模型
        status_code: 状态码
        **kwargs: 其他参数
    """
    return route(
        path=path,
        methods=['POST'],
        middleware=middleware,
        tags=tags,
        summary=summary,
        description=description,
        response_model=response_model,
        status_code=status_code,
        **kwargs
    )


def put(
    path: str,
    middleware: List[str] = None,
    tags: List[str] = None,
    summary: str = None,
    description: str = None,
    response_model: Any = None,
    **kwargs
):
    """PUT路由装饰器
    
    Args:
        path: 路由路径
        middleware: 中间件列表
        tags: API标签
        summary: API摘要
        description: API描述
        response_model: 响应模型
        **kwargs: 其他参数
    """
    return route(
        path=path,
        methods=['PUT'],
        middleware=middleware,
        tags=tags,
        summary=summary,
        description=description,
        response_model=response_model,
        **kwargs
    )


def delete(
    path: str,
    middleware: List[str] = None,
    tags: List[str] = None,
    summary: str = None,
    description: str = None,
    response_model: Any = None,
    status_code: int = 204,
    **kwargs
):
    """DELETE路由装饰器
    
    Args:
        path: 路由路径
        middleware: 中间件列表
        tags: API标签
        summary: API摘要
        description: API描述
        response_model: 响应模型
        status_code: 状态码
        **kwargs: 其他参数
    """
    return route(
        path=path,
        methods=['DELETE'],
        middleware=middleware,
        tags=tags,
        summary=summary,
        description=description,
        response_model=response_model,
        status_code=status_code,
        **kwargs
    )


def patch(
    path: str,
    middleware: List[str] = None,
    tags: List[str] = None,
    summary: str = None,
    description: str = None,
    response_model: Any = None,
    **kwargs
):
    """PATCH路由装饰器
    
    Args:
        path: 路由路径
        middleware: 中间件列表
        tags: API标签
        summary: API摘要
        description: API描述
        response_model: 响应模型
        **kwargs: 其他参数
    """
    return route(
        path=path,
        methods=['PATCH'],
        middleware=middleware,
        tags=tags,
        summary=summary,
        description=description,
        response_model=response_model,
        **kwargs
    )


def websocket(
    path: str,
    middleware: List[str] = None,
    **kwargs
):
    """WebSocket路由装饰器
    
    Args:
        path: 路由路径
        middleware: 中间件列表
        **kwargs: 其他参数
    """
    def decorator(func: Callable):
        # 存储WebSocket路由元数据
        func._websocket_meta = {
            'path': path,
            'middleware': middleware or [],
            **kwargs
        }
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class RouteRegistry:
    """路由注册表"""
    
    def __init__(self):
        self.routes = []
        self.websocket_routes = []
    
    def register_route(self, handler: Callable, meta: Dict[str, Any]):
        """注册路由
        
        Args:
            handler: 处理函数
            meta: 路由元数据
        """
        self.routes.append({
            'handler': handler,
            'meta': meta
        })
    
    def register_websocket(self, handler: Callable, meta: Dict[str, Any]):
        """注册WebSocket路由
        
        Args:
            handler: 处理函数
            meta: 路由元数据
        """
        self.websocket_routes.append({
            'handler': handler,
            'meta': meta
        })
    
    def get_routes(self) -> List[Dict[str, Any]]:
        """获取所有路由
        
        Returns:
            List[Dict[str, Any]]: 路由列表
        """
        return self.routes.copy()
    
    def get_websocket_routes(self) -> List[Dict[str, Any]]:
        """获取所有WebSocket路由
        
        Returns:
            List[Dict[str, Any]]: WebSocket路由列表
        """
        return self.websocket_routes.copy()
    
    def clear(self):
        """清空注册表"""
        self.routes.clear()
        self.websocket_routes.clear()


# 全局路由注册表
route_registry = RouteRegistry()


def collect_routes_from_class(cls):
    """从类中收集路由
    
    Args:
        cls: 控制器类
    """
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        
        # 检查是否有路由元数据
        if hasattr(attr, '_route_meta'):
            route_registry.register_route(attr, attr._route_meta)
        
        # 检查是否有WebSocket元数据
        if hasattr(attr, '_websocket_meta'):
            route_registry.register_websocket(attr, attr._websocket_meta)
