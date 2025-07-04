#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: middleware.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 中间件配置文件

定义中间件的配置，包括全局中间件、路由中间件、控制器中间件等。
参考ThinkPHP的中间件配置方式，适配Python环境。

本文件是分析服务项目的一部分。
"""

from typing import List, Dict, Any

# 全局中间件 (按顺序执行)
GLOBAL_MIDDLEWARE = [
    "app.middleware.cors_middleware.CorsMiddleware",
    "app.middleware.logging_middleware.LoggingMiddleware", 
    "app.middleware.exception_middleware.ExceptionMiddleware",
]

# 路由中间件别名
MIDDLEWARE_ALIAS = {
    "auth": "app.middleware.auth_middleware.AuthMiddleware",
    "rate_limit": "app.middleware.rate_limit_middleware.RateLimitMiddleware",
    "response": "app.middleware.response_middleware.ResponseMiddleware",
    "cors": "app.middleware.cors_middleware.CorsMiddleware",
    "logging": "app.middleware.logging_middleware.LoggingMiddleware",
}

# 中间件组
MIDDLEWARE_GROUPS = {
    "api": ["auth", "rate_limit", "response"],
    "admin": ["auth:admin", "response"],
    "public": ["cors", "logging", "response"],
    "analysis": ["auth", "rate_limit", "logging", "response"],
}

# 控制器中间件配置
CONTROLLER_MIDDLEWARE = {
    "app.controllers.task_controller.TaskController": {
        "middleware": ["analysis"],
        "except": ["health"],  # 排除的方法
        "only": ["start", "stop", "show", "index"],  # 仅包含的方法
    },
    "app.controllers.stream_controller.StreamController": {
        "middleware": ["analysis"],
        "except": [],
        "only": ["index", "create", "show", "delete"],
    },
    "app.controllers.health_controller.HealthController": {
        "middleware": ["public"],
        "except": [],
        "only": ["check", "status"],
    },
    "app.controllers.discovery_controller.DiscoveryController": {
        "middleware": ["public"],
        "except": [],
        "only": ["services", "info"],
    },
}

# 中间件参数配置
MIDDLEWARE_PARAMS = {
    "auth": {
        "default_role": "user",
        "token_header": "Authorization",
        "token_prefix": "Bearer ",
    },
    "rate_limit": {
        "requests_per_minute": 60,
        "burst_size": 10,
        "key_func": "ip",  # ip, user, endpoint
    },
    "cors": {
        "allow_origins": ["*"],
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["*"],
        "allow_credentials": True,
        "max_age": 86400,  # 24小时
    },
    "logging": {
        "log_requests": True,
        "log_responses": True,
        "log_body": False,  # 是否记录请求体
        "exclude_paths": ["/health", "/docs", "/redoc"],
    },
}

# 中间件优先级 (数字越小优先级越高)
MIDDLEWARE_PRIORITY = {
    "cors": 1,
    "logging": 2,
    "exception": 3,
    "auth": 4,
    "rate_limit": 5,
    "response": 6,
}

# 中间件开关配置
MIDDLEWARE_ENABLED = {
    "cors": True,
    "logging": True,
    "exception": True,
    "auth": True,
    "rate_limit": True,
    "response": True,
}


def setup_middleware(app):
    """设置中间件

    Args:
        app: FastAPI应用实例
    """
    # 这里可以添加中间件设置逻辑
    # 由于我们还没有实现具体的中间件类，暂时跳过
    pass
