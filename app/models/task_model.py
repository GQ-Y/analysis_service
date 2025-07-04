#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: task_model.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 任务模型

定义视频分析任务相关的数据模型，包括任务实体、配置、状态等。

本文件是分析服务项目的一部分。
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import Field, validator

from .base_model import (
    BaseEntity, 
    TaskStatusEnum, 
    AnalysisTypeEnum, 
    DeviceTypeEnum,
    PriorityEnum,
    AnalysisConfig
)


class TaskModel(BaseEntity):
    """任务模型"""
    
    task_name: str = Field(
        description="任务名称",
        min_length=1,
        max_length=100,
        example="视频分析任务001"
    )
    
    stream_url: str = Field(
        description="视频流URL",
        example="rtsp://192.168.1.100:554/stream"
    )
    
    analysis_type: AnalysisTypeEnum = Field(
        description="分析类型",
        example=AnalysisTypeEnum.DETECTION
    )
    
    model_code: str = Field(
        description="模型代码",
        example="yolo_v8_detection"
    )
    
    status: TaskStatusEnum = Field(
        default=TaskStatusEnum.PENDING,
        description="任务状态"
    )
    
    priority: PriorityEnum = Field(
        default=PriorityEnum.NORMAL,
        description="任务优先级"
    )
    
    user_id: Optional[str] = Field(
        default=None,
        description="用户ID"
    )
    
    created_by: Optional[str] = Field(
        default=None,
        description="创建者"
    )
    
    description: Optional[str] = Field(
        default=None,
        description="任务描述",
        max_length=500
    )
    
    # 配置相关
    analysis_config: Optional[AnalysisConfig] = Field(
        default=None,
        description="分析配置"
    )
    
    callback_url: Optional[str] = Field(
        default=None,
        description="回调URL"
    )
    
    enable_callback: bool = Field(
        default=False,
        description="是否启用回调"
    )
    
    save_result: bool = Field(
        default=True,
        description="是否保存结果"
    )
    
    save_images: bool = Field(
        default=False,
        description="是否保存图像"
    )
    
    # 时间相关
    start_time: Optional[datetime] = Field(
        default=None,
        description="开始时间"
    )
    
    stop_time: Optional[datetime] = Field(
        default=None,
        description="停止时间"
    )
    
    duration: Optional[float] = Field(
        default=None,
        description="运行时长（秒）",
        ge=0
    )
    
    # 错误信息
    error_message: Optional[str] = Field(
        default=None,
        description="错误信息"
    )
    
    error_code: Optional[str] = Field(
        default=None,
        description="错误代码"
    )
    
    # 统计信息
    processed_frames: int = Field(
        default=0,
        description="已处理帧数",
        ge=0
    )
    
    detected_objects: int = Field(
        default=0,
        description="检测到的对象数量",
        ge=0
    )
    
    # 批次信息
    batch_id: Optional[str] = Field(
        default=None,
        description="批次ID"
    )
    
    batch_index: Optional[int] = Field(
        default=None,
        description="批次索引"
    )
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="元数据"
    )
    
    @validator('stream_url')
    def validate_stream_url(cls, v):
        """验证流URL"""
        if not v:
            raise ValueError("流URL不能为空")
        
        valid_schemes = ['rtsp://', 'http://', 'https://', 'rtmp://', 'file://']
        if not any(v.startswith(scheme) for scheme in valid_schemes):
            raise ValueError("无效的流URL格式")
        
        return v
    
    @validator('callback_url')
    def validate_callback_url(cls, v, values):
        """验证回调URL"""
        enable_callback = values.get('enable_callback', False)
        
        if enable_callback and not v:
            raise ValueError("启用回调时，回调URL不能为空")
        
        if v and not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError("回调URL必须是HTTP或HTTPS协议")
        
        return v
    
    def start_task(self):
        """启动任务"""
        self.status = TaskStatusEnum.RUNNING
        self.start_time = datetime.now()
        self.update_timestamp()
    
    def stop_task(self):
        """停止任务"""
        self.status = TaskStatusEnum.STOPPED
        self.stop_time = datetime.now()
        if self.start_time:
            self.duration = (self.stop_time - self.start_time).total_seconds()
        self.update_timestamp()
    
    def complete_task(self):
        """完成任务"""
        self.status = TaskStatusEnum.COMPLETED
        self.stop_time = datetime.now()
        if self.start_time:
            self.duration = (self.stop_time - self.start_time).total_seconds()
        self.update_timestamp()
    
    def fail_task(self, error_message: str, error_code: str = None):
        """任务失败"""
        self.status = TaskStatusEnum.FAILED
        self.error_message = error_message
        self.error_code = error_code
        self.stop_time = datetime.now()
        if self.start_time:
            self.duration = (self.stop_time - self.start_time).total_seconds()
        self.update_timestamp()
    
    def is_running(self) -> bool:
        """检查任务是否正在运行"""
        return self.status == TaskStatusEnum.RUNNING
    
    def is_finished(self) -> bool:
        """检查任务是否已结束"""
        return self.status in [
            TaskStatusEnum.COMPLETED,
            TaskStatusEnum.STOPPED,
            TaskStatusEnum.FAILED,
            TaskStatusEnum.CANCELLED
        ]
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """获取运行时信息"""
        runtime_seconds = 0
        if self.start_time:
            end_time = self.stop_time or datetime.now()
            runtime_seconds = (end_time - self.start_time).total_seconds()
        
        return {
            'task_id': self.id,
            'status': self.status,
            'runtime_seconds': runtime_seconds,
            'processed_frames': self.processed_frames,
            'detected_objects': self.detected_objects,
            'fps': self.processed_frames / runtime_seconds if runtime_seconds > 0 else 0
        }


class TaskQueue(BaseEntity):
    """任务队列模型"""
    
    task_id: str = Field(
        description="关联的任务ID"
    )
    
    priority: PriorityEnum = Field(
        default=PriorityEnum.NORMAL,
        description="队列优先级"
    )
    
    status: TaskStatusEnum = Field(
        default=TaskStatusEnum.PENDING,
        description="队列状态"
    )
    
    retry_count: int = Field(
        default=0,
        description="重试次数",
        ge=0
    )
    
    max_retries: int = Field(
        default=3,
        description="最大重试次数",
        ge=0
    )
    
    scheduled_at: Optional[datetime] = Field(
        default=None,
        description="计划执行时间"
    )
    
    started_at: Optional[datetime] = Field(
        default=None,
        description="开始执行时间"
    )
    
    completed_at: Optional[datetime] = Field(
        default=None,
        description="完成时间"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="错误信息"
    )
    
    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """增加重试次数"""
        self.retry_count += 1
        self.update_timestamp()


class TaskResult(BaseEntity):
    """任务结果模型"""
    
    task_id: str = Field(
        description="关联的任务ID"
    )
    
    frame_number: int = Field(
        description="帧号",
        ge=0
    )
    
    timestamp: datetime = Field(
        description="结果时间戳"
    )
    
    detections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="检测结果列表"
    )
    
    image_path: Optional[str] = Field(
        default=None,
        description="图像保存路径"
    )
    
    confidence_avg: Optional[float] = Field(
        default=None,
        description="平均置信度",
        ge=0.0,
        le=1.0
    )
    
    processing_time: Optional[float] = Field(
        default=None,
        description="处理时间（毫秒）",
        ge=0
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="结果元数据"
    )


class TaskStatistics(BaseEntity):
    """任务统计模型"""
    
    task_id: str = Field(
        description="关联的任务ID"
    )
    
    total_frames: int = Field(
        default=0,
        description="总帧数",
        ge=0
    )
    
    processed_frames: int = Field(
        default=0,
        description="已处理帧数",
        ge=0
    )
    
    total_detections: int = Field(
        default=0,
        description="总检测数量",
        ge=0
    )
    
    avg_processing_time: Optional[float] = Field(
        default=None,
        description="平均处理时间（毫秒）",
        ge=0
    )
    
    avg_confidence: Optional[float] = Field(
        default=None,
        description="平均置信度",
        ge=0.0,
        le=1.0
    )
    
    fps: Optional[float] = Field(
        default=None,
        description="处理帧率",
        ge=0
    )
    
    start_time: Optional[datetime] = Field(
        default=None,
        description="统计开始时间"
    )
    
    end_time: Optional[datetime] = Field(
        default=None,
        description="统计结束时间"
    )
    
    def calculate_fps(self) -> float:
        """计算FPS"""
        if self.start_time and self.end_time and self.processed_frames > 0:
            duration = (self.end_time - self.start_time).total_seconds()
            if duration > 0:
                return self.processed_frames / duration
        return 0.0
    
    def update_statistics(self, processing_time: float, confidence: float):
        """更新统计信息"""
        self.processed_frames += 1
        
        # 更新平均处理时间
        if self.avg_processing_time is None:
            self.avg_processing_time = processing_time
        else:
            self.avg_processing_time = (
                (self.avg_processing_time * (self.processed_frames - 1) + processing_time) 
                / self.processed_frames
            )
        
        # 更新平均置信度
        if self.avg_confidence is None:
            self.avg_confidence = confidence
        else:
            self.avg_confidence = (
                (self.avg_confidence * (self.processed_frames - 1) + confidence) 
                / self.processed_frames
            )
        
        # 更新FPS
        self.fps = self.calculate_fps()
        
        self.update_timestamp()
