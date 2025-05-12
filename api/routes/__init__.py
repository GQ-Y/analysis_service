"""
路由模块
包含所有API路由定义
"""

from .task import router as task_router
from .health import router as health_router

__all__ = [
    "task_router",
    "health_router"
]
