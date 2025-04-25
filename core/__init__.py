"""
Core 模块
包含核心功能实现
"""

from .exceptions import (
    AnalysisException,
    InvalidInputException,
    ModelLoadException,
    ProcessingException,
    DatabaseException,
    ResourceNotFoundException,
    ValidationException,
    StorageException
)

from .models import (
    StandardResponse,
    AnalysisType,
    RoiType,
    AnalysisStatus,
    BoundingBox,
    DetectionResult,
    SegmentationResult,
    TrackingResult,
    CrossCameraResult
)

from .redis_manager import RedisManager
from .resource import ResourceMonitor
from .config import settings

# 导出任务管理相关组件
from .task_management.utils.status import TaskStatus

__all__ = [
    # 异常类
    "AnalysisException",
    "InvalidInputException", 
    "ModelLoadException",
    "ProcessingException",
    "DatabaseException",
    "ResourceNotFoundException",
    "ValidationException",
    "StorageException",
    
    # 数据模型
    "StandardResponse",
    "AnalysisType",
    "RoiType",
    "AnalysisStatus",
    "BoundingBox",
    "DetectionResult",
    "SegmentationResult",
    "TrackingResult",
    "CrossCameraResult",
    
    # 任务管理
    "TaskStatus",
    
    # 其他核心组件
    "RedisManager",
    "ResourceMonitor",
    "settings"
] 