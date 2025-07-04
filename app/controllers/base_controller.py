#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: base_controller.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 基础控制器类

定义控制器的基础功能，包括依赖注入、响应格式化、异常处理等。
所有控制器都应该继承此基类。

本文件是分析服务项目的一部分。
"""

import time
import uuid
from typing import Any, Dict, Optional, Union
from fastapi import Request, HTTPException
from pydantic import BaseModel
import logging

from config.dependencies import get_container
from app.exceptions.business_exception import BusinessException


class BaseController:
    """基础控制器类"""
    
    def __init__(self):
        """初始化控制器"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._container = None

    @property
    def container(self):
        """获取依赖容器

        Returns:
            依赖容器实例
        """
        if self._container is None:
            self._container = get_container()
        return self._container
    
    def success(
        self,
        data: Any = None,
        message: str = "操作成功",
        code: int = 200,
        request_id: str = None
    ) -> Dict[str, Any]:
        """成功响应
        
        Args:
            data: 响应数据
            message: 响应消息
            code: 响应码
            request_id: 请求ID
            
        Returns:
            Dict[str, Any]: 标准响应格式
        """
        return {
            "success": True,
            "code": code,
            "message": message,
            "data": data,
            "timestamp": int(time.time()),
            "request_id": request_id or str(uuid.uuid4())
        }
    
    def error(
        self,
        message: str = "操作失败",
        code: int = 400,
        data: Any = None,
        request_id: str = None
    ) -> Dict[str, Any]:
        """错误响应
        
        Args:
            message: 错误消息
            code: 错误码
            data: 错误数据
            request_id: 请求ID
            
        Returns:
            Dict[str, Any]: 标准错误响应格式
        """
        return {
            "success": False,
            "code": code,
            "message": message,
            "data": data,
            "timestamp": int(time.time()),
            "request_id": request_id or str(uuid.uuid4())
        }
    
    def paginated_response(
        self,
        items: list,
        total: int,
        page: int = 1,
        page_size: int = 20,
        message: str = "获取成功",
        request_id: str = None
    ) -> Dict[str, Any]:
        """分页响应
        
        Args:
            items: 数据列表
            total: 总数量
            page: 当前页码
            page_size: 每页大小
            message: 响应消息
            request_id: 请求ID
            
        Returns:
            Dict[str, Any]: 分页响应格式
        """
        total_pages = (total + page_size - 1) // page_size
        
        pagination_data = {
            "items": items,
            "pagination": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }
        
        return self.success(
            data=pagination_data,
            message=message,
            request_id=request_id
        )
    
    def get_request_id(self, request: Request) -> str:
        """获取请求ID
        
        Args:
            request: 请求对象
            
        Returns:
            str: 请求ID
        """
        # 从请求头获取
        request_id = request.headers.get('X-Request-ID')
        if request_id:
            return request_id
        
        # 从请求状态获取
        if hasattr(request.state, 'request_id'):
            return request.state.request_id
        
        # 生成新的请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        return request_id
    
    def get_user(self, request: Request) -> Optional[Dict[str, Any]]:
        """获取当前用户信息
        
        Args:
            request: 请求对象
            
        Returns:
            Optional[Dict[str, Any]]: 用户信息
        """
        if hasattr(request.state, 'user'):
            return request.state.user
        return None
    
    def require_user(self, request: Request) -> Dict[str, Any]:
        """要求用户已认证
        
        Args:
            request: 请求对象
            
        Returns:
            Dict[str, Any]: 用户信息
            
        Raises:
            HTTPException: 用户未认证
        """
        user = self.get_user(request)
        if not user:
            raise HTTPException(status_code=401, detail="用户未认证")
        return user
    
    def validate_pagination(
        self,
        page: int = 1,
        page_size: int = 20,
        max_page_size: int = 100
    ) -> tuple[int, int]:
        """验证分页参数
        
        Args:
            page: 页码
            page_size: 每页大小
            max_page_size: 最大每页大小
            
        Returns:
            tuple[int, int]: 验证后的页码和每页大小
            
        Raises:
            BusinessException: 参数无效
        """
        if page < 1:
            raise BusinessException("页码必须大于0")
        
        if page_size < 1:
            raise BusinessException("每页大小必须大于0")
        
        if page_size > max_page_size:
            raise BusinessException(f"每页大小不能超过{max_page_size}")
        
        return page, page_size
    
    def log_request(self, request: Request, action: str, **kwargs):
        """记录请求日志
        
        Args:
            request: 请求对象
            action: 操作名称
            **kwargs: 额外参数
        """
        request_id = self.get_request_id(request)
        user = self.get_user(request)
        user_id = user.get('user_id') if user else None
        
        self.logger.info(
            f"请求操作: {action}",
            extra={
                'request_id': request_id,
                'user_id': user_id,
                'method': request.method,
                'path': request.url.path,
                'action': action,
                **kwargs
            }
        )
    
    def log_error(self, request: Request, error: Exception, action: str = None):
        """记录错误日志
        
        Args:
            request: 请求对象
            error: 异常对象
            action: 操作名称
        """
        request_id = self.get_request_id(request)
        user = self.get_user(request)
        user_id = user.get('user_id') if user else None
        
        self.logger.error(
            f"请求错误: {action or 'unknown'} - {str(error)}",
            extra={
                'request_id': request_id,
                'user_id': user_id,
                'method': request.method,
                'path': request.url.path,
                'action': action,
                'error_type': type(error).__name__,
                'error_message': str(error)
            },
            exc_info=True
        )
    
    async def handle_service_call(
        self,
        request: Request,
        service_method,
        *args,
        action: str = None,
        **kwargs
    ) -> Any:
        """处理服务调用
        
        Args:
            request: 请求对象
            service_method: 服务方法
            *args: 位置参数
            action: 操作名称
            **kwargs: 关键字参数
            
        Returns:
            Any: 服务方法返回值
        """
        try:
            # 记录请求日志
            if action:
                self.log_request(request, action, args=args, kwargs=kwargs)
            
            # 调用服务方法
            if hasattr(service_method, '__call__'):
                if hasattr(service_method, '__code__') and service_method.__code__.co_flags & 0x80:
                    # 异步方法
                    result = await service_method(*args, **kwargs)
                else:
                    # 同步方法
                    result = service_method(*args, **kwargs)
            else:
                raise ValueError("service_method必须是可调用对象")
            
            return result
            
        except Exception as e:
            # 记录错误日志
            self.log_error(request, e, action)
            raise
    
    def serialize_model(self, model: BaseModel) -> Dict[str, Any]:
        """序列化Pydantic模型
        
        Args:
            model: Pydantic模型实例
            
        Returns:
            Dict[str, Any]: 序列化后的字典
        """
        return model.dict(exclude_unset=True, exclude_none=True)
    
    def serialize_models(self, models: list[BaseModel]) -> list[Dict[str, Any]]:
        """序列化Pydantic模型列表
        
        Args:
            models: Pydantic模型实例列表
            
        Returns:
            list[Dict[str, Any]]: 序列化后的字典列表
        """
        return [self.serialize_model(model) for model in models]
