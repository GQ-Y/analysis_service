#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: base_model.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 基础模型类

定义所有数据模型的基础类，提供通用的字段和方法。
使用Pydantic进行数据验证和序列化。

本文件是分析服务项目的一部分。
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field, ConfigDict, validator
from enum import Enum


class BaseEntity(BaseModel):
    """基础实体模型"""
    
    model_config = ConfigDict(
        # 允许使用任意类型
        arbitrary_types_allowed=True,
        # 验证赋值
        validate_assignment=True,
        # 使用枚举值
        use_enum_values=True,
        # 保护命名空间
        protected_namespaces=(),
        # 序列化时排除未设置的字段
        exclude_unset=True,
        # 序列化时排除None值
        exclude_none=True
    )
    
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="唯一标识符",
        example="123e4567-e89b-12d3-a456-426614174000"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间"
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="更新时间"
    )
    
    def update_timestamp(self):
        """更新时间戳"""
        self.updated_at = datetime.now()
    
    def to_dict(self, exclude_unset: bool = True, exclude_none: bool = True) -> Dict[str, Any]:
        """转换为字典
        
        Args:
            exclude_unset: 是否排除未设置的字段
            exclude_none: 是否排除None值
            
        Returns:
            Dict[str, Any]: 字典表示
        """
        return self.model_dump(
            exclude_unset=exclude_unset,
            exclude_none=exclude_none,
            mode='json'
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """从字典创建实例
        
        Args:
            data: 字典数据
            
        Returns:
            BaseEntity: 模型实例
        """
        return cls(**data)


class StatusEnum(str, Enum):
    """状态枚举基类"""
    pass


class TaskStatusEnum(StatusEnum):
    """任务状态枚举"""
    PENDING = "pending"          # 等待中
    STARTING = "starting"        # 启动中
    RUNNING = "running"          # 运行中
    STOPPING = "stopping"       # 停止中
    STOPPED = "stopped"          # 已停止
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"           # 失败
    CANCELLED = "cancelled"      # 已取消


class StreamStatusEnum(StatusEnum):
    """流状态枚举"""
    CREATED = "created"          # 已创建
    CONNECTING = "connecting"    # 连接中
    ONLINE = "online"           # 在线
    OFFLINE = "offline"         # 离线
    ERROR = "error"             # 错误


class AnalysisTypeEnum(str, Enum):
    """分析类型枚举"""
    DETECTION = "detection"              # 目标检测
    CLASSIFICATION = "classification"    # 图像分类
    SEGMENTATION = "segmentation"        # 图像分割
    TRACKING = "tracking"               # 目标跟踪
    POSE_ESTIMATION = "pose_estimation" # 姿态估计
    FACE_RECOGNITION = "face_recognition" # 人脸识别


class StreamTypeEnum(str, Enum):
    """流类型枚举"""
    RTSP = "rtsp"               # RTSP流
    HTTP = "http"               # HTTP流
    HTTPS = "https"             # HTTPS流
    FILE = "file"               # 文件
    CAMERA = "camera"           # 摄像头
    RTMP = "rtmp"               # RTMP流


class DeviceTypeEnum(str, Enum):
    """设备类型枚举"""
    CPU = "cpu"                 # CPU
    GPU = "gpu"                 # GPU
    AUTO = "auto"               # 自动选择


class PriorityEnum(str, Enum):
    """优先级枚举"""
    LOW = "low"                 # 低优先级
    NORMAL = "normal"           # 普通优先级
    HIGH = "high"               # 高优先级
    URGENT = "urgent"           # 紧急优先级


class BaseConfig(BaseModel):
    """基础配置模型"""
    
    model_config = ConfigDict(
        extra='allow',  # 允许额外字段
        validate_assignment=True,
        use_enum_values=True
    )
    
    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        """合并配置
        
        Args:
            other: 其他配置
            
        Returns:
            BaseConfig: 合并后的配置
        """
        merged_data = self.model_dump()
        other_data = other.model_dump()
        merged_data.update(other_data)
        return self.__class__(**merged_data)


class ROIConfig(BaseConfig):
    """ROI配置模型"""
    
    roi_type: int = Field(
        default=0,
        description="ROI类型: 0-无ROI, 1-矩形, 2-多边形, 3-线段",
        ge=0,
        le=3
    )
    
    roi_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="ROI数据，根据roi_type不同而不同"
    )
    
    @validator('roi_data')
    def validate_roi_data(cls, v, values):
        """验证ROI数据"""
        roi_type = values.get('roi_type', 0)
        
        if roi_type == 0:  # 无ROI
            return None
        elif roi_type == 1:  # 矩形
            if v and all(key in v for key in ['x1', 'y1', 'x2', 'y2']):
                return v
        elif roi_type == 2:  # 多边形
            if v and 'points' in v and isinstance(v['points'], list):
                return v
        elif roi_type == 3:  # 线段
            if v and 'points' in v and isinstance(v['points'], list) and len(v['points']) == 2:
                return v
        
        if roi_type > 0 and not v:
            raise ValueError(f"ROI类型为{roi_type}时，roi_data不能为空")
        
        return v


class AnalysisConfig(BaseConfig):
    """分析配置模型"""
    
    confidence: float = Field(
        default=0.5,
        description="置信度阈值",
        ge=0.0,
        le=1.0
    )
    
    iou_threshold: float = Field(
        default=0.45,
        description="IoU阈值",
        ge=0.0,
        le=1.0
    )
    
    classes: Optional[List[str]] = Field(
        default=None,
        description="需要检测的类别列表"
    )
    
    image_size: Dict[str, int] = Field(
        default={"width": 640, "height": 640},
        description="输入图像尺寸"
    )
    
    device: DeviceTypeEnum = Field(
        default=DeviceTypeEnum.AUTO,
        description="推理设备类型"
    )
    
    batch_size: int = Field(
        default=1,
        description="批处理大小",
        ge=1,
        le=32
    )
    
    max_detections: int = Field(
        default=100,
        description="最大检测数量",
        ge=1,
        le=1000
    )
    
    roi_config: Optional[ROIConfig] = Field(
        default=None,
        description="ROI配置"
    )


class PaginationMeta(BaseModel):
    """分页元数据模型"""
    
    total: int = Field(description="总数量", ge=0)
    page: int = Field(description="当前页码", ge=1)
    page_size: int = Field(description="每页大小", ge=1, le=100)
    total_pages: int = Field(description="总页数", ge=0)
    has_next: bool = Field(description="是否有下一页")
    has_prev: bool = Field(description="是否有上一页")
    
    @classmethod
    def create(cls, total: int, page: int, page_size: int) -> 'PaginationMeta':
        """创建分页元数据
        
        Args:
            total: 总数量
            page: 当前页码
            page_size: 每页大小
            
        Returns:
            PaginationMeta: 分页元数据
        """
        total_pages = (total + page_size - 1) // page_size
        
        return cls(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )


class BaseResponse(BaseModel):
    """基础响应模型"""
    
    success: bool = Field(description="是否成功")
    code: int = Field(description="响应码")
    message: str = Field(description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")
    request_id: Optional[str] = Field(default=None, description="请求ID")


class SuccessResponse(BaseResponse):
    """成功响应模型"""
    
    success: bool = Field(default=True)
    code: int = Field(default=200)
    message: str = Field(default="操作成功")
    data: Optional[Any] = Field(default=None, description="响应数据")


class ErrorResponse(BaseResponse):
    """错误响应模型"""
    
    success: bool = Field(default=False)
    code: int = Field(default=400)
    message: str = Field(default="操作失败")
    error_type: Optional[str] = Field(default=None, description="错误类型")
    details: Optional[Any] = Field(default=None, description="错误详情")


class PaginatedResponse(SuccessResponse):
    """分页响应模型"""
    
    data: Dict[str, Any] = Field(description="分页数据")
    
    @classmethod
    def create(
        cls,
        items: List[Any],
        pagination: PaginationMeta,
        message: str = "获取成功",
        request_id: str = None
    ) -> 'PaginatedResponse':
        """创建分页响应
        
        Args:
            items: 数据列表
            pagination: 分页元数据
            message: 响应消息
            request_id: 请求ID
            
        Returns:
            PaginatedResponse: 分页响应
        """
        return cls(
            data={
                "items": items,
                "pagination": pagination.model_dump()
            },
            message=message,
            request_id=request_id
        )
