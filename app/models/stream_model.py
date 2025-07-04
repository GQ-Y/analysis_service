#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: stream_model.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 流模型

定义视频流相关的数据模型，包括流实体、配置、状态等。

本文件是分析服务项目的一部分。
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import Field, field_validator

from .base_model import (
    BaseEntity, 
    StreamStatusEnum, 
    StreamTypeEnum,
    BaseConfig
)


class StreamConfig(BaseConfig):
    """流配置模型"""
    
    # 连接配置
    timeout: int = Field(
        default=30,
        description="连接超时时间（秒）",
        ge=1,
        le=300
    )
    
    retry_count: int = Field(
        default=3,
        description="重试次数",
        ge=0,
        le=10
    )
    
    retry_interval: int = Field(
        default=5,
        description="重试间隔（秒）",
        ge=1,
        le=60
    )
    
    # 视频参数
    frame_rate: Optional[int] = Field(
        default=None,
        description="帧率限制",
        ge=1,
        le=60
    )
    
    resolution: Optional[Dict[str, int]] = Field(
        default=None,
        description="分辨率设置",
        example={"width": 1920, "height": 1080}
    )
    
    # 缓冲配置
    buffer_size: int = Field(
        default=10,
        description="缓冲区大小（帧数）",
        ge=1,
        le=100
    )
    
    # 认证配置
    username: Optional[str] = Field(
        default=None,
        description="用户名"
    )
    
    password: Optional[str] = Field(
        default=None,
        description="密码"
    )
    
    # 其他配置
    enable_audio: bool = Field(
        default=False,
        description="是否启用音频"
    )
    
    transport_protocol: Optional[str] = Field(
        default=None,
        description="传输协议（TCP/UDP）",
        pattern="^(tcp|udp|TCP|UDP)$"
    )


class StreamModel(BaseEntity):
    """流模型"""
    
    stream_name: str = Field(
        description="流名称",
        min_length=1,
        max_length=100,
        example="摄像头001"
    )
    
    stream_url: str = Field(
        description="流URL",
        example="rtsp://192.168.1.100:554/stream"
    )
    
    stream_type: StreamTypeEnum = Field(
        description="流类型",
        example=StreamTypeEnum.RTSP
    )
    
    status: StreamStatusEnum = Field(
        default=StreamStatusEnum.CREATED,
        description="流状态"
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
        description="流描述",
        max_length=500
    )
    
    # 配置
    config: Optional[StreamConfig] = Field(
        default=None,
        description="流配置"
    )
    
    # 位置信息
    location: Optional[str] = Field(
        default=None,
        description="位置信息",
        max_length=200
    )
    
    # 标签
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description="标签列表"
    )
    
    # 连接信息
    last_connected_at: Optional[datetime] = Field(
        default=None,
        description="最后连接时间"
    )
    
    last_disconnected_at: Optional[datetime] = Field(
        default=None,
        description="最后断开时间"
    )
    
    connection_count: int = Field(
        default=0,
        description="连接次数",
        ge=0
    )
    
    # 错误信息
    last_error: Optional[str] = Field(
        default=None,
        description="最后错误信息"
    )
    
    error_count: int = Field(
        default=0,
        description="错误次数",
        ge=0
    )
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="元数据"
    )
    
    @field_validator('stream_url')
    @classmethod
    def validate_stream_url(cls, v):
        """验证流URL"""
        if not v:
            raise ValueError("流URL不能为空")
        
        # 基本URL格式验证
        if not any(v.startswith(prefix) for prefix in ['rtsp://', 'rtmp://', 'http://', 'https://', '/']):
            raise ValueError("不支持的流URL格式")
        
        return v
    
    def connect(self):
        """连接流"""
        self.status = StreamStatusEnum.CONNECTING
        self.connection_count += 1
        self.update_timestamp()
    
    def connected(self):
        """连接成功"""
        self.status = StreamStatusEnum.ONLINE
        self.last_connected_at = datetime.now()
        self.update_timestamp()
    
    def disconnect(self, error_message: str = None):
        """断开连接"""
        self.status = StreamStatusEnum.OFFLINE
        self.last_disconnected_at = datetime.now()
        
        if error_message:
            self.last_error = error_message
            self.error_count += 1
            self.status = StreamStatusEnum.ERROR
        
        self.update_timestamp()
    
    def is_online(self) -> bool:
        """检查是否在线"""
        return self.status == StreamStatusEnum.ONLINE
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return self.status in [StreamStatusEnum.CREATED, StreamStatusEnum.ONLINE]
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        uptime = None
        if self.last_connected_at and self.status == StreamStatusEnum.ONLINE:
            uptime = (datetime.now() - self.last_connected_at).total_seconds()
        
        return {
            'stream_id': self.id,
            'status': self.status,
            'connection_count': self.connection_count,
            'error_count': self.error_count,
            'last_connected_at': self.last_connected_at,
            'last_disconnected_at': self.last_disconnected_at,
            'uptime_seconds': uptime,
            'last_error': self.last_error
        }


class StreamStatistics(BaseEntity):
    """流统计模型"""
    
    stream_id: str = Field(
        description="关联的流ID"
    )
    
    # 连接统计
    total_connections: int = Field(
        default=0,
        description="总连接次数",
        ge=0
    )
    
    successful_connections: int = Field(
        default=0,
        description="成功连接次数",
        ge=0
    )
    
    failed_connections: int = Field(
        default=0,
        description="失败连接次数",
        ge=0
    )
    
    # 时间统计
    total_uptime: float = Field(
        default=0.0,
        description="总在线时间（秒）",
        ge=0
    )
    
    avg_connection_duration: Optional[float] = Field(
        default=None,
        description="平均连接时长（秒）",
        ge=0
    )
    
    # 数据统计
    total_frames: int = Field(
        default=0,
        description="总帧数",
        ge=0
    )
    
    total_bytes: int = Field(
        default=0,
        description="总字节数",
        ge=0
    )
    
    avg_frame_rate: Optional[float] = Field(
        default=None,
        description="平均帧率",
        ge=0
    )
    
    avg_bitrate: Optional[float] = Field(
        default=None,
        description="平均比特率（bps）",
        ge=0
    )
    
    # 质量统计
    frame_drops: int = Field(
        default=0,
        description="丢帧数",
        ge=0
    )
    
    frame_drop_rate: Optional[float] = Field(
        default=None,
        description="丢帧率",
        ge=0.0,
        le=1.0
    )
    
    # 时间范围
    start_time: Optional[datetime] = Field(
        default=None,
        description="统计开始时间"
    )
    
    end_time: Optional[datetime] = Field(
        default=None,
        description="统计结束时间"
    )
    
    def calculate_connection_success_rate(self) -> float:
        """计算连接成功率"""
        if self.total_connections == 0:
            return 0.0
        return self.successful_connections / self.total_connections
    
    def calculate_frame_drop_rate(self) -> float:
        """计算丢帧率"""
        if self.total_frames == 0:
            return 0.0
        return self.frame_drops / self.total_frames
    
    def update_frame_stats(self, frame_count: int, bytes_count: int, duration: float):
        """更新帧统计"""
        self.total_frames += frame_count
        self.total_bytes += bytes_count
        
        if duration > 0:
            current_fps = frame_count / duration
            if self.avg_frame_rate is None:
                self.avg_frame_rate = current_fps
            else:
                # 使用指数移动平均
                self.avg_frame_rate = 0.9 * self.avg_frame_rate + 0.1 * current_fps
            
            current_bitrate = (bytes_count * 8) / duration  # bps
            if self.avg_bitrate is None:
                self.avg_bitrate = current_bitrate
            else:
                self.avg_bitrate = 0.9 * self.avg_bitrate + 0.1 * current_bitrate
        
        self.frame_drop_rate = self.calculate_frame_drop_rate()
        self.update_timestamp()
    
    def update_connection_stats(self, success: bool, duration: float = None):
        """更新连接统计"""
        self.total_connections += 1
        
        if success:
            self.successful_connections += 1
            if duration is not None:
                self.total_uptime += duration
                if self.avg_connection_duration is None:
                    self.avg_connection_duration = duration
                else:
                    # 计算平均连接时长
                    total_duration = self.avg_connection_duration * (self.successful_connections - 1) + duration
                    self.avg_connection_duration = total_duration / self.successful_connections
        else:
            self.failed_connections += 1
        
        self.update_timestamp()


class StreamHealth(BaseEntity):
    """流健康状态模型"""
    
    stream_id: str = Field(
        description="关联的流ID"
    )
    
    is_healthy: bool = Field(
        description="是否健康"
    )
    
    health_score: float = Field(
        description="健康分数",
        ge=0.0,
        le=100.0
    )
    
    last_check_time: datetime = Field(
        description="最后检查时间"
    )
    
    response_time: Optional[float] = Field(
        default=None,
        description="响应时间（毫秒）",
        ge=0
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="错误信息"
    )
    
    check_details: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="检查详情"
    )
    
    def update_health(self, is_healthy: bool, response_time: float = None, error_message: str = None):
        """更新健康状态"""
        self.is_healthy = is_healthy
        self.response_time = response_time
        self.error_message = error_message
        self.last_check_time = datetime.now()
        
        # 计算健康分数
        if is_healthy:
            base_score = 100.0
            if response_time is not None:
                # 响应时间越长，分数越低
                if response_time > 5000:  # 5秒
                    base_score = 50.0
                elif response_time > 2000:  # 2秒
                    base_score = 75.0
                elif response_time > 1000:  # 1秒
                    base_score = 90.0
            self.health_score = base_score
        else:
            self.health_score = 0.0
        
        self.update_timestamp()
