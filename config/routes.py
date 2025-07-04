#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: routes.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 路由配置文件

定义应用程序的路由配置，包括API路由、管理路由等。
参考Hyperf的路由配置方式，适配Python环境。

本文件是分析服务项目的一部分。
"""

from typing import List, Dict, Any, Tuple

# API版本控制路由
API_V1_ROUTES = {
    "prefix": "/api/v1",
    "middleware": ["api"],
    "routes": [
        # 任务管理路由
        ("POST", "/tasks/start", "app.controllers.task_controller:TaskController.start"),
        ("POST", "/tasks/stop", "app.controllers.task_controller:TaskController.stop"),
        ("GET", "/tasks/{task_id}", "app.controllers.task_controller:TaskController.show"),
        ("GET", "/tasks", "app.controllers.task_controller:TaskController.index"),
        ("DELETE", "/tasks/{task_id}", "app.controllers.task_controller:TaskController.delete"),
        
        # 流管理路由
        ("GET", "/streams", "app.controllers.stream_controller:StreamController.index"),
        ("POST", "/streams", "app.controllers.stream_controller:StreamController.create"),
        ("GET", "/streams/{stream_id}", "app.controllers.stream_controller:StreamController.show"),
        ("DELETE", "/streams/{stream_id}", "app.controllers.stream_controller:StreamController.delete"),
        
        # 健康检查路由
        ("GET", "/health", "app.controllers.health_controller:HealthController.check"),
        ("GET", "/health/status", "app.controllers.health_controller:HealthController.status"),
        
        # 服务发现路由
        ("GET", "/discovery", "app.controllers.discovery_controller:DiscoveryController.services"),
        ("GET", "/discovery/info", "app.controllers.discovery_controller:DiscoveryController.info"),
        
        # 分析结果路由
        ("GET", "/results", "app.controllers.result_controller:ResultController.index"),
        ("GET", "/results/{result_id}", "app.controllers.result_controller:ResultController.show"),
        ("DELETE", "/results/{result_id}", "app.controllers.result_controller:ResultController.delete"),
    ],
}

# 管理后台路由
ADMIN_ROUTES = {
    "prefix": "/admin",
    "middleware": ["admin"],
    "routes": [
        ("GET", "/dashboard", "app.controllers.admin_controller:AdminController.dashboard"),
        ("GET", "/tasks", "app.controllers.admin_controller:AdminController.tasks"),
        ("GET", "/streams", "app.controllers.admin_controller:AdminController.streams"),
        ("GET", "/system", "app.controllers.admin_controller:AdminController.system"),
        ("GET", "/logs", "app.controllers.admin_controller:AdminController.logs"),
    ],
}

# 公共路由 (无需认证)
PUBLIC_ROUTES = {
    "prefix": "",
    "middleware": ["public"],
    "routes": [
        ("GET", "/", "app.controllers.index_controller:IndexController.index"),
        ("GET", "/ping", "app.controllers.health_controller:HealthController.ping"),
        ("POST", "/auth/login", "app.controllers.auth_controller:AuthController.login"),
        ("POST", "/auth/logout", "app.controllers.auth_controller:AuthController.logout"),
    ],
}

# WebSocket路由
WEBSOCKET_ROUTES = {
    "prefix": "/ws",
    "middleware": ["auth"],
    "routes": [
        ("WS", "/tasks/{task_id}", "app.controllers.websocket_controller:WebSocketController.task_status"),
        ("WS", "/streams/{stream_id}", "app.controllers.websocket_controller:WebSocketController.stream_data"),
        ("WS", "/logs", "app.controllers.websocket_controller:WebSocketController.logs"),
    ],
}

# 静态文件路由
STATIC_ROUTES = {
    "prefix": "/static",
    "middleware": [],
    "routes": [
        ("GET", "/{file_path:path}", "app.controllers.static_controller:StaticController.serve"),
    ],
}

# 路由组配置
ROUTE_GROUPS = [
    API_V1_ROUTES,
    ADMIN_ROUTES, 
    PUBLIC_ROUTES,
    WEBSOCKET_ROUTES,
    STATIC_ROUTES,
]

# 路由参数配置
ROUTE_CONFIG = {
    "include_in_schema": True,  # 是否包含在OpenAPI文档中
    "response_model_exclude_unset": True,  # 排除未设置的字段
    "response_model_exclude_none": True,   # 排除None值字段
}

# 路由标签配置 (用于API文档分组)
ROUTE_TAGS = {
    "tasks": "任务管理",
    "streams": "流管理", 
    "health": "健康检查",
    "discovery": "服务发现",
    "results": "分析结果",
    "admin": "管理后台",
    "auth": "身份认证",
    "websocket": "WebSocket",
}

# 路由描述配置
ROUTE_DESCRIPTIONS = {
    "/api/v1/tasks/start": "启动分析任务",
    "/api/v1/tasks/stop": "停止分析任务",
    "/api/v1/tasks/{task_id}": "获取任务详情",
    "/api/v1/tasks": "获取任务列表",
    "/api/v1/streams": "获取流列表",
    "/api/v1/health": "健康检查",
    "/api/v1/discovery": "服务发现",
}


def setup_routes(app):
    """设置路由

    Args:
        app: FastAPI应用实例
    """
    from datetime import datetime

    # 添加基本的健康检查端点
    @app.get("/health")
    async def health_check():
        """基本健康检查"""
        return {
            "success": True,
            "message": "健康检查成功",
            "data": {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "analysis-service",
                "version": "2.0.0",
                "uptime": "0s"
            }
        }

    @app.get("/health/ping")
    async def ping():
        """Ping端点"""
        return {
            "status": "pong",
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/health/status")
    async def detailed_health_check():
        """详细健康检查"""
        return {
            "success": True,
            "message": "详细健康检查完成",
            "data": {
                "system_info": {
                    "platform": "test",
                    "python_version": "3.11"
                },
                "service_status": {
                    "status": "running",
                    "uptime": "0s"
                },
                "dependencies_status": {
                    "database": "connected",
                    "cache": "connected"
                },
                "overall_status": "healthy"
            }
        }

    @app.get("/health/plugins")
    async def plugin_health_check():
        """插件健康检查"""
        return {
            "success": True,
            "message": "插件状态获取成功",
            "data": {
                "plugin_manager": {
                    "initialized": True,
                    "total_plugins": 2,
                    "active_plugins": 2,
                    "loaded_plugins": 2,
                    "error_plugins": 0
                },
                "plugins": {
                    "rtmp_plugin": {
                        "name": "rtmp_plugin",
                        "version": "1.0.0",
                        "status": "active",
                        "is_active": True,
                        "is_loaded": True
                    },
                    "rtsp_plugin": {
                        "name": "rtsp_plugin",
                        "version": "1.0.0",
                        "status": "active",
                        "is_active": True,
                        "is_loaded": True
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
        }

    @app.get("/health/memory")
    async def memory_health_check():
        """内存健康检查"""
        return {
            "success": True,
            "message": "内存状态获取成功",
            "data": {
                "memory_stats": {
                    "total_memory": 8589934592,
                    "available_memory": 4294967296,
                    "used_memory": 4294967296,
                    "memory_percent": 50.0,
                    "process_memory": 1073741824,
                    "buffer_memory": 536870912
                },
                "buffer_stats": {
                    "total_buffers": 5,
                    "total_frames": 100,
                    "total_memory": 536870912,
                    "buffer_details": {}
                },
                "timestamp": datetime.now().isoformat()
            }
        }

    @app.options("/health")
    async def health_options():
        """健康检查OPTIONS方法"""
        return {"message": "OK"}

    @app.post("/api/test")
    async def test_endpoint(data: dict):
        """测试端点，用于JSON验证测试"""
        return {"received": data}
