#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: task_controller.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 任务控制器

处理任务相关的HTTP请求，包括任务创建、查询、启动、停止等操作。
使用装饰器驱动的路由定义和依赖注入。

本文件是分析服务项目的一部分。
"""

from typing import Dict, Any, Optional, List
from fastapi import Request, HTTPException, Query, Path
from pydantic import BaseModel

from .base_controller import BaseController
from app.decorators.route import get, post, delete
from app.decorators.auth import require_auth, Permissions
from app.decorators.validation import validate_request
from app.schemas.task_schemas import (
    TaskCreateSchema,
    TaskUpdateSchema,
    TaskResponseSchema,
    BatchTaskCreateSchema
)
from app.exceptions.business_exception import BusinessException


class TaskController(BaseController):
    """任务控制器"""
    
    @get(
        path="/api/v1/tasks",
        tags=["任务管理"],
        summary="获取任务列表",
        description="获取当前用户的任务列表，支持分页和筛选"
    )
    async def index(
        self,
        request: Request,
        page: int = Query(1, ge=1, description="页码"),
        page_size: int = Query(20, ge=1, le=100, description="每页大小"),
        status: Optional[str] = Query(None, description="任务状态筛选"),
        model_code: Optional[str] = Query(None, description="模型代码筛选"),
        task_name: Optional[str] = Query(None, description="任务名称筛选")
    ) -> Dict[str, Any]:
        """获取任务列表
        
        Args:
            request: 请求对象
            page: 页码
            page_size: 每页大小
            status: 状态筛选
            model_code: 模型代码筛选
            task_name: 任务名称筛选
            
        Returns:
            Dict[str, Any]: 任务列表响应
        """
        try:
            # 验证分页参数
            page, page_size = self.validate_pagination(page, page_size)
            
            # 获取用户信息
            user = self.get_user(request)
            user_id = user.get('user_id') if user else None
            
            # 构建筛选条件
            filters = {
                'user_id': user_id,
                'status': status,
                'model_code': model_code,
                'task_name': task_name
            }
            # 移除None值
            filters = {k: v for k, v in filters.items() if v is not None}
            
            # 调用任务服务
            result = await self.handle_service_call(
                request,
                self.dependencies.task_service.get_tasks,
                page=page,
                page_size=page_size,
                filters=filters,
                action="获取任务列表"
            )
            
            # 返回分页响应
            return self.paginated_response(
                items=result['items'],
                total=result['total'],
                page=page,
                page_size=page_size,
                message="获取任务列表成功",
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "获取任务列表")
            raise HTTPException(status_code=500, detail="获取任务列表失败")
    
    @get(
        path="/api/v1/tasks/{task_id}",
        tags=["任务管理"],
        summary="获取任务详情",
        description="根据任务ID获取任务的详细信息"
    )
    async def show(
        self,
        request: Request,
        task_id: str = Path(..., description="任务ID")
    ) -> Dict[str, Any]:
        """获取任务详情
        
        Args:
            request: 请求对象
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 任务详情响应
        """
        try:
            # 调用任务服务
            task = await self.handle_service_call(
                request,
                self.dependencies.task_service.get_task,
                task_id,
                action="获取任务详情"
            )
            
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")
            
            return self.success(
                data=task,
                message="获取任务详情成功",
                request_id=self.get_request_id(request)
            )
            
        except HTTPException:
            raise
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "获取任务详情")
            raise HTTPException(status_code=500, detail="获取任务详情失败")
    
    @post(
        path="/api/v1/tasks/start",
        tags=["任务管理"],
        summary="启动分析任务",
        description="创建并启动一个新的视频分析任务",
        status_code=201
    )
    @require_auth(permissions=[Permissions.TASK_CREATE, Permissions.TASK_START])
    @validate_request(TaskCreateSchema)
    async def start(
        self,
        request: Request,
        validated_data: TaskCreateSchema
    ) -> Dict[str, Any]:
        """启动分析任务
        
        Args:
            request: 请求对象
            validated_data: 验证后的任务数据
            
        Returns:
            Dict[str, Any]: 任务创建响应
        """
        try:
            # 获取用户信息
            user = self.require_user(request)
            
            # 准备任务数据
            task_data = validated_data.dict()
            task_data['user_id'] = user.get('user_id')
            task_data['created_by'] = user.get('username')
            
            # 调用任务服务启动任务
            task = await self.handle_service_call(
                request,
                self.dependencies.task_service.start_task,
                task_data,
                action="启动任务"
            )
            
            return self.success(
                data=task,
                message="任务启动成功",
                code=201,
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "启动任务")
            raise HTTPException(status_code=500, detail="任务启动失败")
    
    @post(
        path="/api/v1/tasks/{task_id}/stop",
        tags=["任务管理"],
        summary="停止分析任务",
        description="停止指定的视频分析任务"
    )
    @require_auth(permissions=[Permissions.TASK_STOP])
    async def stop(
        self,
        request: Request,
        task_id: str = Path(..., description="任务ID")
    ) -> Dict[str, Any]:
        """停止分析任务
        
        Args:
            request: 请求对象
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 任务停止响应
        """
        try:
            # 获取用户信息（用于权限验证）
            user = self.require_user(request)
            
            # 转换task_id为整数
            task_id_int = int(task_id)
            
            # 调用任务服务停止任务（只传递task_id）
            result = await self.handle_service_call(
                request,
                self.dependencies.task_service.stop_task,
                task_id_int,
                action="停止任务"
            )
            
            return self.success(
                data=result,
                message="任务停止成功",
                request_id=self.get_request_id(request)
            )
            
        except ValueError as e:
            # 处理task_id转换错误或业务逻辑错误
            raise HTTPException(status_code=400, detail=str(e))
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "停止任务")
            raise HTTPException(status_code=500, detail="任务停止失败")
    
    @delete(
        path="/api/v1/tasks/{task_id}",
        tags=["任务管理"],
        summary="删除任务",
        description="删除指定的任务（仅限已停止的任务）",
        status_code=204
    )
    @require_auth(permissions=[Permissions.TASK_DELETE])
    async def delete(
        self,
        request: Request,
        task_id: str = Path(..., description="任务ID")
    ) -> Dict[str, Any]:
        """删除任务
        
        Args:
            request: 请求对象
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 删除响应
        """
        try:
            # 获取用户信息
            user = self.require_user(request)
            
            # 调用任务服务删除任务
            await self.handle_service_call(
                request,
                self.dependencies.task_service.delete_task,
                task_id,
                user_id=user.get('user_id'),
                action="删除任务"
            )
            
            return self.success(
                message="任务删除成功",
                code=204,
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "删除任务")
            raise HTTPException(status_code=500, detail="任务删除失败")
    
    @post(
        path="/api/v1/tasks/batch/start",
        tags=["任务管理"],
        summary="批量启动任务",
        description="批量创建并启动多个视频分析任务",
        status_code=201
    )
    @require_auth(permissions=[Permissions.TASK_CREATE, Permissions.TASK_START])
    @validate_request(BatchTaskCreateSchema)
    async def batch_start(
        self,
        request: Request,
        validated_data: BatchTaskCreateSchema
    ) -> Dict[str, Any]:
        """批量启动任务
        
        Args:
            request: 请求对象
            validated_data: 验证后的批量任务数据
            
        Returns:
            Dict[str, Any]: 批量任务创建响应
        """
        try:
            # 获取用户信息
            user = self.require_user(request)
            
            # 准备批量任务数据
            batch_data = validated_data.dict()
            batch_data['user_id'] = user.get('user_id')
            batch_data['created_by'] = user.get('username')
            
            # 调用任务服务批量启动任务
            results = await self.handle_service_call(
                request,
                self.dependencies.task_service.batch_start_tasks,
                batch_data,
                action="批量启动任务"
            )
            
            return self.success(
                data=results,
                message="批量任务启动成功",
                code=201,
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "批量启动任务")
            raise HTTPException(status_code=500, detail="批量任务启动失败")
