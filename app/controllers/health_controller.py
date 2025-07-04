#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: health_controller.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 健康检查控制器

提供服务健康状态检查、系统监控等功能。
不需要认证，用于监控系统和负载均衡器检查。

本文件是分析服务项目的一部分。
"""

import time
import psutil
from typing import Dict, Any
from fastapi import Request
from datetime import datetime

from .base_controller import BaseController
from app.decorators.route import get


class HealthController(BaseController):
    """健康检查控制器"""
    
    @get(
        path="/health",
        tags=["健康检查"],
        summary="基础健康检查",
        description="检查服务是否正常运行"
    )
    async def check(self, request: Request) -> Dict[str, Any]:
        """基础健康检查
        
        Args:
            request: 请求对象
            
        Returns:
            Dict[str, Any]: 健康状态响应
        """
        try:
            # 基础健康检查
            health_data = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime": self._get_uptime(),
                "version": "2.0.0",
                "service": "analysis-service"
            }
            
            return self.success(
                data=health_data,
                message="服务运行正常",
                request_id=self.get_request_id(request)
            )
            
        except Exception as e:
            self.log_error(request, e, "健康检查")
            return self.error(
                message="服务异常",
                code=503,
                request_id=self.get_request_id(request)
            )
    
    @get(
        path="/health/ping",
        tags=["健康检查"],
        summary="简单ping检查",
        description="最简单的存活检查"
    )
    async def ping(self, request: Request) -> Dict[str, str]:
        """简单ping检查
        
        Args:
            request: 请求对象
            
        Returns:
            Dict[str, str]: ping响应
        """
        return {"status": "pong", "timestamp": datetime.now().isoformat()}
    
    @get(
        path="/health/plugins",
        tags=["健康检查"],
        summary="插件状态检查",
        description="获取插件系统状态信息"
    )
    async def plugins(self, request: Request) -> Dict[str, Any]:
        """插件状态检查

        Args:
            request: 请求对象

        Returns:
            Dict[str, Any]: 插件状态响应
        """
        try:
            from app.plugins import get_plugin_manager

            plugin_manager = get_plugin_manager()
            plugin_status = plugin_manager.get_all_plugin_status()
            manager_stats = plugin_manager.get_manager_stats()

            plugin_data = {
                "plugin_manager": {
                    "initialized": manager_stats["initialized"],
                    "total_plugins": manager_stats["total_plugins"],
                    "active_plugins": manager_stats["active_plugins"],
                    "loaded_plugins": manager_stats["loaded_plugins"],
                    "error_plugins": manager_stats["error_plugins"]
                },
                "plugins": plugin_status,
                "timestamp": datetime.now().isoformat()
            }

            return self.success(
                data=plugin_data,
                message="插件状态获取成功",
                request_id=self.get_request_id(request)
            )

        except Exception as e:
            self.log_error(request, e, "插件状态检查")
            return self.error(
                message=f"获取插件状态失败: {str(e)}",
                request_id=self.get_request_id(request)
            )

    @get(
        path="/health/memory",
        tags=["健康检查"],
        summary="内存状态检查",
        description="获取内存管理状态信息"
    )
    async def memory(self, request: Request) -> Dict[str, Any]:
        """内存状态检查

        Args:
            request: 请求对象

        Returns:
            Dict[str, Any]: 内存状态响应
        """
        try:
            from app.core.memory import get_memory_manager

            memory_manager = get_memory_manager()
            memory_stats = memory_manager.get_memory_stats()
            buffer_stats = memory_manager.get_buffer_stats()

            memory_data = {
                "memory_stats": {
                    "total_memory": memory_stats.total_memory,
                    "available_memory": memory_stats.available_memory,
                    "used_memory": memory_stats.used_memory,
                    "memory_percent": memory_stats.memory_percent,
                    "process_memory": memory_stats.process_memory,
                    "buffer_memory": memory_stats.buffer_memory
                },
                "buffer_stats": buffer_stats,
                "timestamp": datetime.now().isoformat()
            }

            return self.success(
                data=memory_data,
                message="内存状态获取成功",
                request_id=self.get_request_id(request)
            )

        except Exception as e:
            self.log_error(request, e, "内存状态检查")
            return self.error(
                message=f"获取内存状态失败: {str(e)}",
                request_id=self.get_request_id(request)
            )

    @get(
        path="/health/status",
        tags=["健康检查"],
        summary="详细状态检查",
        description="获取详细的系统状态信息"
    )
    async def status(self, request: Request) -> Dict[str, Any]:
        """详细状态检查
        
        Args:
            request: 请求对象
            
        Returns:
            Dict[str, Any]: 详细状态响应
        """
        try:
            # 获取系统信息
            system_info = self._get_system_info()
            
            # 获取服务状态
            service_status = await self._get_service_status()
            
            # 获取依赖状态
            dependencies_status = await self._get_dependencies_status()
            
            status_data = {
                "overall_status": self._calculate_overall_status(service_status, dependencies_status),
                "timestamp": datetime.now().isoformat(),
                "system": system_info,
                "service": service_status,
                "dependencies": dependencies_status
            }
            
            return self.success(
                data=status_data,
                message="状态检查完成",
                request_id=self.get_request_id(request)
            )
            
        except Exception as e:
            self.log_error(request, e, "详细状态检查")
            return self.error(
                message="状态检查失败",
                code=503,
                request_id=self.get_request_id(request)
            )
    
    @get(
        path="/health/ready",
        tags=["健康检查"],
        summary="就绪检查",
        description="检查服务是否已准备好接收请求"
    )
    async def ready(self, request: Request) -> Dict[str, Any]:
        """就绪检查
        
        Args:
            request: 请求对象
            
        Returns:
            Dict[str, Any]: 就绪状态响应
        """
        try:
            # 检查关键依赖
            dependencies_ready = await self._check_dependencies_ready()
            
            # 检查服务组件
            services_ready = await self._check_services_ready()
            
            is_ready = dependencies_ready and services_ready
            
            ready_data = {
                "ready": is_ready,
                "timestamp": datetime.now().isoformat(),
                "checks": {
                    "dependencies": dependencies_ready,
                    "services": services_ready
                }
            }
            
            if is_ready:
                return self.success(
                    data=ready_data,
                    message="服务已就绪",
                    request_id=self.get_request_id(request)
                )
            else:
                return self.error(
                    message="服务未就绪",
                    code=503,
                    data=ready_data,
                    request_id=self.get_request_id(request)
                )
                
        except Exception as e:
            self.log_error(request, e, "就绪检查")
            return self.error(
                message="就绪检查失败",
                code=503,
                request_id=self.get_request_id(request)
            )
    
    def _get_uptime(self) -> float:
        """获取系统运行时间
        
        Returns:
            float: 运行时间（秒）
        """
        try:
            return time.time() - psutil.boot_time()
        except Exception:
            return 0.0
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息
        
        Returns:
            Dict[str, Any]: 系统信息
        """
        try:
            # CPU信息
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # 内存信息
            memory = psutil.virtual_memory()
            
            # 磁盘信息
            disk = psutil.disk_usage('/')
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "usage_percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "usage_percent": (disk.used / disk.total) * 100
                },
                "uptime": self._get_uptime()
            }
        except Exception as e:
            self.logger.error(f"获取系统信息失败: {e}")
            return {"error": "无法获取系统信息"}
    
    async def _get_service_status(self) -> Dict[str, Any]:
        """获取服务状态
        
        Returns:
            Dict[str, Any]: 服务状态
        """
        try:
            # 检查任务服务
            task_service_status = await self._check_task_service()
            
            # 检查分析器状态
            analyzer_status = await self._check_analyzer_status()
            
            # 检查内存管理器状态
            memory_status = await self._check_memory_status()
            
            return {
                "task_service": task_service_status,
                "analyzer": analyzer_status,
                "memory_manager": memory_status
            }
        except Exception as e:
            self.logger.error(f"获取服务状态失败: {e}")
            return {"error": "无法获取服务状态"}
    
    async def _get_dependencies_status(self) -> Dict[str, Any]:
        """获取依赖状态
        
        Returns:
            Dict[str, Any]: 依赖状态
        """
        try:
            # 检查Redis连接
            redis_status = await self._check_redis_status()
            
            return {
                "redis": redis_status
            }
        except Exception as e:
            self.logger.error(f"获取依赖状态失败: {e}")
            return {"error": "无法获取依赖状态"}
    
    async def _check_redis_status(self) -> Dict[str, Any]:
        """检查Redis状态
        
        Returns:
            Dict[str, Any]: Redis状态
        """
        try:
            redis_client = self.dependencies.redis
            await redis_client.ping()
            return {"status": "healthy", "message": "Redis连接正常"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Redis连接失败: {e}"}
    
    async def _check_task_service(self) -> Dict[str, Any]:
        """检查任务服务状态
        
        Returns:
            Dict[str, Any]: 任务服务状态
        """
        try:
            task_service = self.dependencies.task_service
            # 这里可以添加具体的服务健康检查逻辑
            return {"status": "healthy", "message": "任务服务正常"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"任务服务异常: {e}"}
    
    async def _check_analyzer_status(self) -> Dict[str, Any]:
        """检查分析器状态
        
        Returns:
            Dict[str, Any]: 分析器状态
        """
        try:
            analyzer_factory = self.dependencies.analyzer_factory
            # 这里可以添加具体的分析器健康检查逻辑
            return {"status": "healthy", "message": "分析器正常"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"分析器异常: {e}"}
    
    async def _check_memory_status(self) -> Dict[str, Any]:
        """检查内存管理器状态
        
        Returns:
            Dict[str, Any]: 内存管理器状态
        """
        try:
            memory_manager = self.dependencies.memory_manager
            # 这里可以添加具体的内存管理器健康检查逻辑
            return {"status": "healthy", "message": "内存管理器正常"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"内存管理器异常: {e}"}
    
    async def _check_dependencies_ready(self) -> bool:
        """检查依赖是否就绪
        
        Returns:
            bool: 依赖是否就绪
        """
        try:
            # 检查Redis
            redis_status = await self._check_redis_status()
            return redis_status["status"] == "healthy"
        except Exception:
            return False
    
    async def _check_services_ready(self) -> bool:
        """检查服务是否就绪
        
        Returns:
            bool: 服务是否就绪
        """
        try:
            # 检查关键服务
            task_status = await self._check_task_service()
            analyzer_status = await self._check_analyzer_status()
            memory_status = await self._check_memory_status()
            
            return all([
                task_status["status"] == "healthy",
                analyzer_status["status"] == "healthy",
                memory_status["status"] == "healthy"
            ])
        except Exception:
            return False
    
    def _calculate_overall_status(self, service_status: Dict, dependencies_status: Dict) -> str:
        """计算整体状态
        
        Args:
            service_status: 服务状态
            dependencies_status: 依赖状态
            
        Returns:
            str: 整体状态
        """
        try:
            # 检查所有服务状态
            all_healthy = True
            
            for status_dict in [service_status, dependencies_status]:
                for component, status in status_dict.items():
                    if isinstance(status, dict) and status.get("status") != "healthy":
                        all_healthy = False
                        break
                if not all_healthy:
                    break
            
            return "healthy" if all_healthy else "unhealthy"
        except Exception:
            return "unknown"
