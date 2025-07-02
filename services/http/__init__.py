"""
HTTP服务模块
提供基于HTTP的API服务功能
"""

from .zero_copy_task_service import ZeroCopyTaskService

__all__ = [
    "ZeroCopyTaskService"
]
