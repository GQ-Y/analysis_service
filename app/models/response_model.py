#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: response_model.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 响应模型

定义API响应的标准格式。

本文件是分析服务项目的一部分。
"""

from typing import Any, Optional, Generic, TypeVar
from datetime import datetime
from pydantic import BaseModel, Field

T = TypeVar('T')


class ResponseModel(BaseModel, Generic[T]):
    """标准API响应模型"""
    
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[T] = Field(None, description="响应数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponseModel(BaseModel):
    """错误响应模型"""
    
    success: bool = Field(False, description="是否成功")
    message: str = Field(..., description="错误消息")
    error_code: Optional[str] = Field(None, description="错误代码")
    details: Optional[dict] = Field(None, description="错误详情")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PaginationResponseModel(BaseModel, Generic[T]):
    """分页响应模型"""
    
    success: bool = Field(True, description="是否成功")
    message: str = Field("获取成功", description="响应消息")
    data: dict = Field(..., description="分页数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")
    
    def __init__(self, items: list[T], total: int, page: int = 1, page_size: int = 10, **kwargs):
        data = {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }
        super().__init__(data=data, **kwargs)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# 常用的响应创建函数
def create_success_response(data: Any = None, message: str = "操作成功") -> ResponseModel:
    """创建成功响应"""
    return ResponseModel(success=True, message=message, data=data)


def create_error_response(message: str, error_code: str = None, details: dict = None) -> ErrorResponseModel:
    """创建错误响应"""
    return ErrorResponseModel(
        message=message,
        error_code=error_code,
        details=details
    )


def create_pagination_response(
    items: list,
    total: int,
    page: int = 1,
    page_size: int = 10,
    message: str = "获取成功"
) -> PaginationResponseModel:
    """创建分页响应"""
    return PaginationResponseModel(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        message=message
    )
