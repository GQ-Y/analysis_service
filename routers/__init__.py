"""
分析服务路由包
"""
from .task import router as task_router
from .health import router as health_router
from .stream import router as stream_router

__all__ = [
    "task_router",
    "health_router",
    "stream_router"
]