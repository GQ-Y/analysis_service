#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: __init__.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 装饰器包

包含各种装饰器，用于路由定义、中间件绑定、权限控制等。
提供Python特色的简化配置方式。

本文件是分析服务项目的一部分。
"""

from .route import route, get, post, put, delete, patch, websocket
from .middleware import middleware, use_middleware
from .auth import require_auth, require_role, require_permission
from .validation import validate_request, validate_response
from .cache import cache, cache_key
from .rate_limit import rate_limit
from .log import log_execution, log_performance

__all__ = [
    # 路由装饰器
    'route', 'get', 'post', 'put', 'delete', 'patch', 'websocket',
    
    # 中间件装饰器
    'middleware', 'use_middleware',
    
    # 认证装饰器
    'require_auth', 'require_role', 'require_permission',
    
    # 验证装饰器
    'validate_request', 'validate_response',
    
    # 缓存装饰器
    'cache', 'cache_key',
    
    # 限流装饰器
    'rate_limit',
    
    # 日志装饰器
    'log_execution', 'log_performance',
]
