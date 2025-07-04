#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: stream_controller.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 流控制器

处理视频流相关的HTTP请求，包括流的创建、查询、管理等操作。

本文件是分析服务项目的一部分。
"""

from typing import Dict, Any, Optional
from fastapi import Request, HTTPException, Query, Path

from .base_controller import BaseController
from app.decorators.route import get, post, delete
from app.decorators.auth import require_auth, Permissions
from app.decorators.validation import validate_request
from app.schemas.stream_schemas import StreamCreateSchema, StreamUpdateSchema
from app.exceptions.business_exception import BusinessException


class StreamController(BaseController):
    """流控制器"""
    
    @get(
        path="/api/v1/streams",
        tags=["流管理"],
        summary="获取流列表",
        description="获取当前用户的视频流列表"
    )
    async def index(
        self,
        request: Request,
        page: int = Query(1, ge=1, description="页码"),
        page_size: int = Query(20, ge=1, le=100, description="每页大小"),
        status: Optional[str] = Query(None, description="流状态筛选"),
        stream_type: Optional[str] = Query(None, description="流类型筛选")
    ) -> Dict[str, Any]:
        """获取流列表
        
        Args:
            request: 请求对象
            page: 页码
            page_size: 每页大小
            status: 状态筛选
            stream_type: 流类型筛选
            
        Returns:
            Dict[str, Any]: 流列表响应
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
                'stream_type': stream_type
            }
            # 移除None值
            filters = {k: v for k, v in filters.items() if v is not None}
            
            # 调用流服务
            result = await self.handle_service_call(
                request,
                self.dependencies.stream_service.get_streams,
                page=page,
                page_size=page_size,
                filters=filters,
                action="获取流列表"
            )
            
            # 返回分页响应
            return self.paginated_response(
                items=result['items'],
                total=result['total'],
                page=page,
                page_size=page_size,
                message="获取流列表成功",
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "获取流列表")
            raise HTTPException(status_code=500, detail="获取流列表失败")
    
    @get(
        path="/api/v1/streams/{stream_id}",
        tags=["流管理"],
        summary="获取流详情",
        description="根据流ID获取流的详细信息"
    )
    async def show(
        self,
        request: Request,
        stream_id: str = Path(..., description="流ID")
    ) -> Dict[str, Any]:
        """获取流详情
        
        Args:
            request: 请求对象
            stream_id: 流ID
            
        Returns:
            Dict[str, Any]: 流详情响应
        """
        try:
            # 调用流服务
            stream = await self.handle_service_call(
                request,
                self.dependencies.stream_service.get_stream,
                stream_id,
                action="获取流详情"
            )
            
            if not stream:
                raise HTTPException(status_code=404, detail="流不存在")
            
            return self.success(
                data=stream,
                message="获取流详情成功",
                request_id=self.get_request_id(request)
            )
            
        except HTTPException:
            raise
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "获取流详情")
            raise HTTPException(status_code=500, detail="获取流详情失败")
    
    @post(
        path="/api/v1/streams",
        tags=["流管理"],
        summary="创建流",
        description="创建一个新的视频流配置",
        status_code=201
    )
    @require_auth(permissions=[Permissions.STREAM_CREATE])
    @validate_request(StreamCreateSchema)
    async def create(
        self,
        request: Request,
        validated_data: StreamCreateSchema
    ) -> Dict[str, Any]:
        """创建流
        
        Args:
            request: 请求对象
            validated_data: 验证后的流数据
            
        Returns:
            Dict[str, Any]: 流创建响应
        """
        try:
            # 获取用户信息
            user = self.require_user(request)
            
            # 准备流数据
            stream_data = validated_data.dict()
            stream_data['user_id'] = user.get('user_id')
            stream_data['created_by'] = user.get('username')
            
            # 调用流服务创建流
            stream = await self.handle_service_call(
                request,
                self.dependencies.stream_service.create_stream,
                stream_data,
                action="创建流"
            )
            
            return self.success(
                data=stream,
                message="流创建成功",
                code=201,
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "创建流")
            raise HTTPException(status_code=500, detail="流创建失败")
    
    @delete(
        path="/api/v1/streams/{stream_id}",
        tags=["流管理"],
        summary="删除流",
        description="删除指定的视频流配置",
        status_code=204
    )
    @require_auth(permissions=[Permissions.STREAM_DELETE])
    async def delete(
        self,
        request: Request,
        stream_id: str = Path(..., description="流ID")
    ) -> Dict[str, Any]:
        """删除流
        
        Args:
            request: 请求对象
            stream_id: 流ID
            
        Returns:
            Dict[str, Any]: 删除响应
        """
        try:
            # 获取用户信息
            user = self.require_user(request)
            
            # 调用流服务删除流
            await self.handle_service_call(
                request,
                self.dependencies.stream_service.delete_stream,
                stream_id,
                user_id=user.get('user_id'),
                action="删除流"
            )
            
            return self.success(
                message="流删除成功",
                code=204,
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "删除流")
            raise HTTPException(status_code=500, detail="流删除失败")
    
    @post(
        path="/api/v1/streams/{stream_id}/test",
        tags=["流管理"],
        summary="测试流连接",
        description="测试指定流的连接状态"
    )
    @require_auth(permissions=[Permissions.STREAM_READ])
    async def test_connection(
        self,
        request: Request,
        stream_id: str = Path(..., description="流ID")
    ) -> Dict[str, Any]:
        """测试流连接
        
        Args:
            request: 请求对象
            stream_id: 流ID
            
        Returns:
            Dict[str, Any]: 测试结果响应
        """
        try:
            # 调用流服务测试连接
            result = await self.handle_service_call(
                request,
                self.dependencies.stream_service.test_stream_connection,
                stream_id,
                action="测试流连接"
            )
            
            return self.success(
                data=result,
                message="流连接测试完成",
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "测试流连接")
            raise HTTPException(status_code=500, detail="流连接测试失败")
    
    @get(
        path="/api/v1/streams/{stream_id}/stats",
        tags=["流管理"],
        summary="获取流统计信息",
        description="获取指定流的统计信息"
    )
    @require_auth(permissions=[Permissions.STREAM_READ])
    async def get_stats(
        self,
        request: Request,
        stream_id: str = Path(..., description="流ID")
    ) -> Dict[str, Any]:
        """获取流统计信息
        
        Args:
            request: 请求对象
            stream_id: 流ID
            
        Returns:
            Dict[str, Any]: 统计信息响应
        """
        try:
            # 调用流服务获取统计信息
            stats = await self.handle_service_call(
                request,
                self.dependencies.stream_service.get_stream_stats,
                stream_id,
                action="获取流统计信息"
            )
            
            return self.success(
                data=stats,
                message="获取流统计信息成功",
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "获取流统计信息")
            raise HTTPException(status_code=500, detail="获取流统计信息失败")
