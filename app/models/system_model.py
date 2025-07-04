#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: system_model.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 系统模型

定义系统相关的数据模型，包括系统配置、监控、日志等。

本文件是分析服务项目的一部分。
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import Field, validator
from enum import Enum

from .base_model import BaseEntity, BaseConfig


class LogLevelEnum(str, Enum):
    """日志级别枚举"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ServiceStatusEnum(str, Enum):
    """服务状态枚举"""
    HEALTHY = "healthy"         # 健康
    DEGRADED = "degraded"       # 降级
    UNHEALTHY = "unhealthy"     # 不健康
    UNKNOWN = "unknown"         # 未知


class SystemConfig(BaseConfig):
    """系统配置模型"""
    
    # 服务配置
    service_name: str = Field(
        default="analysis-service",
        description="服务名称"
    )
    
    service_version: str = Field(
        default="2.0.0",
        description="服务版本"
    )
    
    debug_mode: bool = Field(
        default=False,
        description="调试模式"
    )
    
    # 性能配置
    max_concurrent_tasks: int = Field(
        default=10,
        description="最大并发任务数",
        ge=1,
        le=100
    )
    
    max_memory_usage: float = Field(
        default=0.8,
        description="最大内存使用率",
        ge=0.1,
        le=0.95
    )
    
    max_cpu_usage: float = Field(
        default=0.8,
        description="最大CPU使用率",
        ge=0.1,
        le=0.95
    )
    
    # 日志配置
    log_level: LogLevelEnum = Field(
        default=LogLevelEnum.INFO,
        description="日志级别"
    )
    
    log_retention_days: int = Field(
        default=30,
        description="日志保留天数",
        ge=1,
        le=365
    )
    
    # 监控配置
    health_check_interval: int = Field(
        default=30,
        description="健康检查间隔（秒）",
        ge=10,
        le=300
    )
    
    metrics_collection_interval: int = Field(
        default=60,
        description="指标收集间隔（秒）",
        ge=30,
        le=600
    )
    
    # 安全配置
    enable_authentication: bool = Field(
        default=True,
        description="启用身份验证"
    )
    
    session_timeout: int = Field(
        default=3600,
        description="会话超时时间（秒）",
        ge=300,
        le=86400
    )
    
    # 其他配置
    custom_config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="自定义配置"
    )


class SystemInfo(BaseEntity):
    """系统信息模型"""
    
    hostname: str = Field(
        description="主机名"
    )
    
    platform: str = Field(
        description="平台信息"
    )
    
    python_version: str = Field(
        description="Python版本"
    )
    
    service_version: str = Field(
        description="服务版本"
    )
    
    start_time: datetime = Field(
        description="启动时间"
    )
    
    uptime_seconds: float = Field(
        description="运行时间（秒）",
        ge=0
    )
    
    # 硬件信息
    cpu_count: int = Field(
        description="CPU核心数",
        ge=1
    )
    
    memory_total: int = Field(
        description="总内存（字节）",
        ge=0
    )
    
    disk_total: int = Field(
        description="总磁盘空间（字节）",
        ge=0
    )
    
    # 网络信息
    ip_address: str = Field(
        description="IP地址"
    )
    
    port: int = Field(
        description="服务端口",
        ge=1,
        le=65535
    )
    
    # 环境信息
    environment: str = Field(
        description="运行环境",
        example="production"
    )
    
    timezone: str = Field(
        description="时区"
    )
    
    def calculate_uptime(self) -> float:
        """计算运行时间"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def update_uptime(self):
        """更新运行时间"""
        self.uptime_seconds = self.calculate_uptime()
        self.update_timestamp()


class SystemMetrics(BaseEntity):
    """系统指标模型"""
    
    # CPU指标
    cpu_usage_percent: float = Field(
        description="CPU使用率（%）",
        ge=0.0,
        le=100.0
    )
    
    cpu_load_average: Optional[List[float]] = Field(
        default=None,
        description="CPU负载平均值（1分钟、5分钟、15分钟）"
    )
    
    # 内存指标
    memory_usage_percent: float = Field(
        description="内存使用率（%）",
        ge=0.0,
        le=100.0
    )
    
    memory_used: int = Field(
        description="已使用内存（字节）",
        ge=0
    )
    
    memory_available: int = Field(
        description="可用内存（字节）",
        ge=0
    )
    
    # 磁盘指标
    disk_usage_percent: float = Field(
        description="磁盘使用率（%）",
        ge=0.0,
        le=100.0
    )
    
    disk_used: int = Field(
        description="已使用磁盘空间（字节）",
        ge=0
    )
    
    disk_available: int = Field(
        description="可用磁盘空间（字节）",
        ge=0
    )
    
    # 网络指标
    network_bytes_sent: int = Field(
        default=0,
        description="网络发送字节数",
        ge=0
    )
    
    network_bytes_received: int = Field(
        default=0,
        description="网络接收字节数",
        ge=0
    )
    
    # 应用指标
    active_tasks: int = Field(
        default=0,
        description="活跃任务数",
        ge=0
    )
    
    active_streams: int = Field(
        default=0,
        description="活跃流数",
        ge=0
    )
    
    active_connections: int = Field(
        default=0,
        description="活跃连接数",
        ge=0
    )
    
    # 性能指标
    requests_per_second: float = Field(
        default=0.0,
        description="每秒请求数",
        ge=0.0
    )
    
    avg_response_time: float = Field(
        default=0.0,
        description="平均响应时间（毫秒）",
        ge=0.0
    )
    
    error_rate: float = Field(
        default=0.0,
        description="错误率",
        ge=0.0,
        le=1.0
    )
    
    def is_healthy(self) -> bool:
        """检查系统是否健康"""
        return (
            self.cpu_usage_percent < 80.0 and
            self.memory_usage_percent < 80.0 and
            self.disk_usage_percent < 90.0 and
            self.error_rate < 0.05
        )
    
    def get_health_score(self) -> float:
        """计算健康分数"""
        cpu_score = max(0, 100 - self.cpu_usage_percent)
        memory_score = max(0, 100 - self.memory_usage_percent)
        disk_score = max(0, 100 - self.disk_usage_percent)
        error_score = max(0, 100 - self.error_rate * 100)
        
        return (cpu_score + memory_score + disk_score + error_score) / 4


class SystemHealth(BaseEntity):
    """系统健康状态模型"""
    
    overall_status: ServiceStatusEnum = Field(
        description="整体状态"
    )
    
    health_score: float = Field(
        description="健康分数",
        ge=0.0,
        le=100.0
    )
    
    last_check_time: datetime = Field(
        description="最后检查时间"
    )
    
    # 组件状态
    components: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="组件状态"
    )
    
    # 检查详情
    checks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="检查详情"
    )
    
    # 警告和错误
    warnings: List[str] = Field(
        default_factory=list,
        description="警告列表"
    )
    
    errors: List[str] = Field(
        default_factory=list,
        description="错误列表"
    )
    
    def add_component_status(self, component: str, status: str, details: Dict[str, Any] = None):
        """添加组件状态"""
        self.components[component] = {
            'status': status,
            'details': details or {},
            'checked_at': datetime.now().isoformat()
        }
    
    def add_check_result(self, check_name: str, success: bool, message: str, details: Dict[str, Any] = None):
        """添加检查结果"""
        self.checks.append({
            'name': check_name,
            'success': success,
            'message': message,
            'details': details or {},
            'checked_at': datetime.now().isoformat()
        })
    
    def add_warning(self, message: str):
        """添加警告"""
        self.warnings.append(message)
    
    def add_error(self, message: str):
        """添加错误"""
        self.errors.append(message)
    
    def calculate_overall_status(self):
        """计算整体状态"""
        if self.errors:
            self.overall_status = ServiceStatusEnum.UNHEALTHY
        elif self.warnings:
            self.overall_status = ServiceStatusEnum.DEGRADED
        elif self.health_score >= 80:
            self.overall_status = ServiceStatusEnum.HEALTHY
        elif self.health_score >= 60:
            self.overall_status = ServiceStatusEnum.DEGRADED
        else:
            self.overall_status = ServiceStatusEnum.UNHEALTHY


class SystemLog(BaseEntity):
    """系统日志模型"""
    
    level: LogLevelEnum = Field(
        description="日志级别"
    )
    
    logger_name: str = Field(
        description="日志器名称"
    )
    
    message: str = Field(
        description="日志消息"
    )
    
    module: Optional[str] = Field(
        default=None,
        description="模块名称"
    )
    
    function: Optional[str] = Field(
        default=None,
        description="函数名称"
    )
    
    line_number: Optional[int] = Field(
        default=None,
        description="行号"
    )
    
    # 请求信息
    request_id: Optional[str] = Field(
        default=None,
        description="请求ID"
    )
    
    user_id: Optional[str] = Field(
        default=None,
        description="用户ID"
    )
    
    ip_address: Optional[str] = Field(
        default=None,
        description="IP地址"
    )
    
    # 异常信息
    exception_type: Optional[str] = Field(
        default=None,
        description="异常类型"
    )
    
    exception_message: Optional[str] = Field(
        default=None,
        description="异常消息"
    )
    
    stack_trace: Optional[str] = Field(
        default=None,
        description="堆栈跟踪"
    )
    
    # 额外数据
    extra_data: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="额外数据"
    )
    
    @classmethod
    def create_log(
        cls,
        level: LogLevelEnum,
        logger_name: str,
        message: str,
        module: str = None,
        function: str = None,
        line_number: int = None,
        request_id: str = None,
        user_id: str = None,
        ip_address: str = None,
        exception_type: str = None,
        exception_message: str = None,
        stack_trace: str = None,
        **extra_data
    ) -> 'SystemLog':
        """创建日志记录"""
        return cls(
            level=level,
            logger_name=logger_name,
            message=message,
            module=module,
            function=function,
            line_number=line_number,
            request_id=request_id,
            user_id=user_id,
            ip_address=ip_address,
            exception_type=exception_type,
            exception_message=exception_message,
            stack_trace=stack_trace,
            extra_data=extra_data
        )
