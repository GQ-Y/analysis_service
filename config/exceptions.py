#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: exceptions.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 异常处理配置文件

定义异常处理器的配置，包括不同类型异常的处理器。
参考Hyperf的异常处理器配置方式，适配Python环境。

本文件是分析服务项目的一部分。
"""

from typing import List, Dict, Any

# 异常处理器配置 (按优先级顺序)
EXCEPTION_HANDLERS = {
    "http": [
        "app.exceptions.handlers.validation_exception_handler.ValidationExceptionHandler",
        "app.exceptions.handlers.business_exception_handler.BusinessExceptionHandler", 
        "app.exceptions.handlers.http_exception_handler.HttpExceptionHandler",
        "app.exceptions.handlers.app_exception_handler.AppExceptionHandler",
    ],
    "websocket": [
        "app.exceptions.handlers.websocket_exception_handler.WebSocketExceptionHandler",
    ],
}

# 异常类型映射
EXCEPTION_TYPE_MAPPING = {
    # 验证异常
    "ValidationError": "app.exceptions.handlers.validation_exception_handler.ValidationExceptionHandler",
    "RequestValidationError": "app.exceptions.handlers.validation_exception_handler.ValidationExceptionHandler",
    
    # 业务异常
    "BusinessException": "app.exceptions.handlers.business_exception_handler.BusinessExceptionHandler",
    "TaskException": "app.exceptions.handlers.business_exception_handler.BusinessExceptionHandler",
    "StreamException": "app.exceptions.handlers.business_exception_handler.BusinessExceptionHandler",
    
    # HTTP异常
    "HTTPException": "app.exceptions.handlers.http_exception_handler.HttpExceptionHandler",
    "StarletteHTTPException": "app.exceptions.handlers.http_exception_handler.HttpExceptionHandler",
    
    # 系统异常
    "SystemException": "app.exceptions.handlers.app_exception_handler.AppExceptionHandler",
    "Exception": "app.exceptions.handlers.app_exception_handler.AppExceptionHandler",
}

# 异常响应格式配置
EXCEPTION_RESPONSE_FORMAT = {
    "include_traceback": False,  # 是否包含堆栈跟踪
    "include_request_id": True,  # 是否包含请求ID
    "include_timestamp": True,   # 是否包含时间戳
    "default_message": "服务器内部错误",
    "default_code": 500,
}

# 异常日志配置
EXCEPTION_LOGGING = {
    "log_level": "ERROR",
    "log_format": "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    "include_traceback": True,
    "include_request_info": True,
    "exclude_exceptions": [
        "HTTPException",  # HTTP异常不记录到错误日志
        "ValidationError",  # 验证异常不记录到错误日志
    ],
}

# 错误码定义
ERROR_CODES = {
    # 通用错误码 (1000-1999)
    "UNKNOWN_ERROR": {"code": 1000, "message": "未知错误"},
    "INVALID_PARAMETER": {"code": 1001, "message": "参数无效"},
    "MISSING_PARAMETER": {"code": 1002, "message": "缺少必需参数"},
    "UNAUTHORIZED": {"code": 1003, "message": "未授权访问"},
    "FORBIDDEN": {"code": 1004, "message": "禁止访问"},
    "NOT_FOUND": {"code": 1005, "message": "资源不存在"},
    "METHOD_NOT_ALLOWED": {"code": 1006, "message": "方法不允许"},
    "RATE_LIMIT_EXCEEDED": {"code": 1007, "message": "请求频率超限"},
    
    # 任务相关错误码 (2000-2999)
    "TASK_NOT_FOUND": {"code": 2001, "message": "任务不存在"},
    "TASK_ALREADY_RUNNING": {"code": 2002, "message": "任务已在运行"},
    "TASK_NOT_RUNNING": {"code": 2003, "message": "任务未运行"},
    "TASK_START_FAILED": {"code": 2004, "message": "任务启动失败"},
    "TASK_STOP_FAILED": {"code": 2005, "message": "任务停止失败"},
    "TASK_CONFIG_INVALID": {"code": 2006, "message": "任务配置无效"},
    
    # 流相关错误码 (3000-3999)
    "STREAM_NOT_FOUND": {"code": 3001, "message": "流不存在"},
    "STREAM_CONNECTION_FAILED": {"code": 3002, "message": "流连接失败"},
    "STREAM_FORMAT_UNSUPPORTED": {"code": 3003, "message": "流格式不支持"},
    "STREAM_TIMEOUT": {"code": 3004, "message": "流连接超时"},
    
    # 分析相关错误码 (4000-4999)
    "ANALYZER_NOT_FOUND": {"code": 4001, "message": "分析器不存在"},
    "ANALYZER_INIT_FAILED": {"code": 4002, "message": "分析器初始化失败"},
    "ANALYSIS_FAILED": {"code": 4003, "message": "分析失败"},
    "MODEL_LOAD_FAILED": {"code": 4004, "message": "模型加载失败"},
    
    # 系统相关错误码 (5000-5999)
    "MEMORY_INSUFFICIENT": {"code": 5001, "message": "内存不足"},
    "DISK_SPACE_INSUFFICIENT": {"code": 5002, "message": "磁盘空间不足"},
    "SERVICE_UNAVAILABLE": {"code": 5003, "message": "服务不可用"},
    "DATABASE_CONNECTION_FAILED": {"code": 5004, "message": "数据库连接失败"},
}

# 异常处理器参数配置
EXCEPTION_HANDLER_PARAMS = {
    "validation": {
        "return_details": True,  # 是否返回详细的验证错误信息
        "max_errors": 10,        # 最大错误数量
    },
    "business": {
        "log_level": "WARNING",  # 业务异常日志级别
        "include_context": True, # 是否包含上下文信息
    },
    "http": {
        "custom_messages": {     # 自定义HTTP状态码消息
            404: "请求的资源不存在",
            405: "请求方法不被允许",
            422: "请求参数验证失败",
            429: "请求过于频繁，请稍后再试",
            500: "服务器内部错误",
            502: "网关错误",
            503: "服务暂时不可用",
        },
    },
    "app": {
        "send_notification": False,  # 是否发送通知
        "notification_threshold": "ERROR",  # 通知阈值
    },
}


def setup_exception_handlers(app):
    """设置异常处理器

    Args:
        app: FastAPI应用实例
    """
    # 这里可以添加异常处理器设置逻辑
    # 由于我们还没有实现具体的异常处理器，暂时跳过
    pass
