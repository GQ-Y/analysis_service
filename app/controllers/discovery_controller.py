#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: discovery_controller.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 服务发现控制器

提供服务发现、服务注册、服务状态查询等功能。

本文件是分析服务项目的一部分。
"""

from typing import Dict, Any, Optional
from fastapi import Request, HTTPException, Query

from .base_controller import BaseController
from app.decorators.route import get, post
from app.decorators.auth import require_auth, optional_auth
from app.exceptions.business_exception import BusinessException


class DiscoveryController(BaseController):
    """服务发现控制器"""
    
    @get(
        path="/api/v1/discovery/services",
        tags=["服务发现"],
        summary="获取服务列表",
        description="获取所有已注册的服务列表"
    )
    @optional_auth()
    async def services(
        self,
        request: Request,
        service_type: Optional[str] = Query(None, description="服务类型筛选"),
        status: Optional[str] = Query(None, description="服务状态筛选")
    ) -> Dict[str, Any]:
        """获取服务列表
        
        Args:
            request: 请求对象
            service_type: 服务类型筛选
            status: 服务状态筛选
            
        Returns:
            Dict[str, Any]: 服务列表响应
        """
        try:
            # 构建筛选条件
            filters = {
                'service_type': service_type,
                'status': status
            }
            # 移除None值
            filters = {k: v for k, v in filters.items() if v is not None}
            
            # 调用发现服务
            services = await self.handle_service_call(
                request,
                self.dependencies.discovery_service.get_services,
                filters=filters,
                action="获取服务列表"
            )
            
            return self.success(
                data=services,
                message="获取服务列表成功",
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "获取服务列表")
            raise HTTPException(status_code=500, detail="获取服务列表失败")
    
    @get(
        path="/api/v1/discovery/info",
        tags=["服务发现"],
        summary="获取当前服务信息",
        description="获取当前分析服务的详细信息"
    )
    async def info(self, request: Request) -> Dict[str, Any]:
        """获取当前服务信息
        
        Args:
            request: 请求对象
            
        Returns:
            Dict[str, Any]: 服务信息响应
        """
        try:
            # 调用发现服务获取当前服务信息
            service_info = await self.handle_service_call(
                request,
                self.dependencies.discovery_service.get_current_service_info,
                action="获取当前服务信息"
            )
            
            return self.success(
                data=service_info,
                message="获取服务信息成功",
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "获取服务信息")
            raise HTTPException(status_code=500, detail="获取服务信息失败")
    
    @get(
        path="/api/v1/discovery/nodes",
        tags=["服务发现"],
        summary="获取服务节点",
        description="获取指定服务的所有节点信息"
    )
    @optional_auth()
    async def nodes(
        self,
        request: Request,
        service_name: str = Query(..., description="服务名称")
    ) -> Dict[str, Any]:
        """获取服务节点
        
        Args:
            request: 请求对象
            service_name: 服务名称
            
        Returns:
            Dict[str, Any]: 服务节点响应
        """
        try:
            # 调用发现服务获取节点信息
            nodes = await self.handle_service_call(
                request,
                self.dependencies.discovery_service.get_service_nodes,
                service_name,
                action="获取服务节点"
            )
            
            return self.success(
                data=nodes,
                message="获取服务节点成功",
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "获取服务节点")
            raise HTTPException(status_code=500, detail="获取服务节点失败")
    
    @post(
        path="/api/v1/discovery/register",
        tags=["服务发现"],
        summary="注册服务",
        description="注册当前服务到服务发现中心"
    )
    @require_auth(roles=['admin'])
    async def register(self, request: Request) -> Dict[str, Any]:
        """注册服务
        
        Args:
            request: 请求对象
            
        Returns:
            Dict[str, Any]: 注册响应
        """
        try:
            # 调用发现服务注册当前服务
            result = await self.handle_service_call(
                request,
                self.dependencies.discovery_service.register_service,
                action="注册服务"
            )
            
            return self.success(
                data=result,
                message="服务注册成功",
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "注册服务")
            raise HTTPException(status_code=500, detail="服务注册失败")
    
    @post(
        path="/api/v1/discovery/deregister",
        tags=["服务发现"],
        summary="注销服务",
        description="从服务发现中心注销当前服务"
    )
    @require_auth(roles=['admin'])
    async def deregister(self, request: Request) -> Dict[str, Any]:
        """注销服务
        
        Args:
            request: 请求对象
            
        Returns:
            Dict[str, Any]: 注销响应
        """
        try:
            # 调用发现服务注销当前服务
            result = await self.handle_service_call(
                request,
                self.dependencies.discovery_service.deregister_service,
                action="注销服务"
            )
            
            return self.success(
                data=result,
                message="服务注销成功",
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "注销服务")
            raise HTTPException(status_code=500, detail="服务注销失败")
    
    @get(
        path="/api/v1/discovery/health/{service_name}",
        tags=["服务发现"],
        summary="检查服务健康状态",
        description="检查指定服务的健康状态"
    )
    @optional_auth()
    async def check_health(
        self,
        request: Request,
        service_name: str
    ) -> Dict[str, Any]:
        """检查服务健康状态
        
        Args:
            request: 请求对象
            service_name: 服务名称
            
        Returns:
            Dict[str, Any]: 健康状态响应
        """
        try:
            # 调用发现服务检查健康状态
            health_status = await self.handle_service_call(
                request,
                self.dependencies.discovery_service.check_service_health,
                service_name,
                action="检查服务健康状态"
            )
            
            return self.success(
                data=health_status,
                message="健康状态检查完成",
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "检查服务健康状态")
            raise HTTPException(status_code=500, detail="健康状态检查失败")
    
    @get(
        path="/api/v1/discovery/metrics",
        tags=["服务发现"],
        summary="获取服务指标",
        description="获取服务发现相关的指标信息"
    )
    @require_auth(roles=['admin'])
    async def metrics(self, request: Request) -> Dict[str, Any]:
        """获取服务指标
        
        Args:
            request: 请求对象
            
        Returns:
            Dict[str, Any]: 指标信息响应
        """
        try:
            # 调用发现服务获取指标
            metrics = await self.handle_service_call(
                request,
                self.dependencies.discovery_service.get_metrics,
                action="获取服务指标"
            )
            
            return self.success(
                data=metrics,
                message="获取服务指标成功",
                request_id=self.get_request_id(request)
            )
            
        except BusinessException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.log_error(request, e, "获取服务指标")
            raise HTTPException(status_code=500, detail="获取服务指标失败")
