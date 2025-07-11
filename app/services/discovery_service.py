#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: discovery_service.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 服务发现服务

处理服务发现相关的业务逻辑，包括服务注册、发现、健康检查等功能。

本文件是分析服务项目的一部分。
"""

import socket
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .base_service import CacheableService
from app.exceptions.business_exception import BusinessException


class DiscoveryService(CacheableService):
    """服务发现服务"""
    
    def __init__(self):
        """初始化服务发现服务"""
        super().__init__(cache_ttl=60)  # 1分钟缓存
        self.service_id = str(uuid.uuid4())
        self.service_info = None
        self._initialize_service_info()
    
    def _initialize_service_info(self):
        """初始化当前服务信息"""
        self.service_info = {
            'service_id': self.service_id,
            'service_name': 'analysis-service',
            'service_type': 'video-analysis',
            'version': '2.0.0',
            'host': self._get_local_ip(),
            'port': 8002,
            'status': 'healthy',
            'registered_at': datetime.now().isoformat(),
            'last_heartbeat': datetime.now().isoformat(),
            'metadata': {
                'description': '智能视频分析服务',
                'capabilities': ['object_detection', 'video_analysis', 'stream_processing'],
                'protocols': ['http', 'websocket'],
                'endpoints': {
                    'health': '/health',
                    'api': '/api/v1',
                    'docs': '/docs'
                }
            }
        }
    
    async def get_services(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """获取服务列表
        
        Args:
            filters: 筛选条件
            
        Returns:
            List[Dict[str, Any]]: 服务列表
        """
        try:
            # 构建缓存键
            cache_key = self.generate_cache_key('services_list', **filters or {})
            
            # 尝试从缓存获取
            cached_services = await self.get_cached_data(cache_key)
            if cached_services:
                self.log_info("从缓存获取服务列表")
                return cached_services
            
            # 从注册中心获取服务列表
            services = await self._get_registered_services(filters)
            
            # 缓存结果
            await self.set_cached_data(cache_key, services, ttl=30)  # 30秒缓存
            
            self.log_info(f"获取服务列表成功，共{len(services)}个服务")
            return services
            
        except Exception as e:
            self.log_error("获取服务列表失败", e)
            raise BusinessException(f"获取服务列表失败: {str(e)}")
    
    async def get_current_service_info(self) -> Dict[str, Any]:
        """获取当前服务信息
        
        Returns:
            Dict[str, Any]: 当前服务信息
        """
        try:
            # 更新心跳时间
            self.service_info['last_heartbeat'] = datetime.now().isoformat()
            
            # 获取运行时信息
            runtime_info = await self._get_runtime_info()
            self.service_info['runtime'] = runtime_info
            
            self.log_info("获取当前服务信息成功")
            return self.service_info.copy()
            
        except Exception as e:
            self.log_error("获取当前服务信息失败", e)
            raise BusinessException(f"获取当前服务信息失败: {str(e)}")
    
    async def get_service_nodes(self, service_name: str) -> List[Dict[str, Any]]:
        """获取指定服务的节点列表
        
        Args:
            service_name: 服务名称
            
        Returns:
            List[Dict[str, Any]]: 节点列表
        """
        try:
            # 构建缓存键
            cache_key = self.generate_cache_key('service_nodes', service_name)
            
            # 尝试从缓存获取
            cached_nodes = await self.get_cached_data(cache_key)
            if cached_nodes:
                self.log_info("从缓存获取服务节点", service_name=service_name)
                return cached_nodes
            
            # 从注册中心获取节点
            nodes = await self._get_service_nodes_from_registry(service_name)
            
            # 缓存结果
            await self.set_cached_data(cache_key, nodes, ttl=30)
            
            self.log_info(f"获取服务节点成功，服务{service_name}共{len(nodes)}个节点")
            return nodes
            
        except Exception as e:
            self.log_error("获取服务节点失败", e, service_name=service_name)
            raise BusinessException(f"获取服务节点失败: {str(e)}")
    
    async def register_service(self) -> Dict[str, Any]:
        """注册当前服务
        
        Returns:
            Dict[str, Any]: 注册结果
        """
        try:
            # 更新服务状态
            self.service_info['status'] = 'registering'
            self.service_info['registered_at'] = datetime.now().isoformat()
            
            # 执行服务注册
            registration_result = await self._register_to_registry()
            
            # 更新状态
            self.service_info['status'] = 'registered'
            
            result = {
                'service_id': self.service_id,
                'service_name': self.service_info['service_name'],
                'registered_at': self.service_info['registered_at'],
                'registration_result': registration_result
            }
            
            self.log_info("服务注册成功", service_id=self.service_id)
            return result
            
        except Exception as e:
            self.service_info['status'] = 'registration_failed'
            self.log_error("服务注册失败", e)
            raise BusinessException(f"服务注册失败: {str(e)}")
    
    async def deregister_service(self) -> Dict[str, Any]:
        """注销当前服务
        
        Returns:
            Dict[str, Any]: 注销结果
        """
        try:
            # 更新服务状态
            self.service_info['status'] = 'deregistering'
            
            # 执行服务注销
            deregistration_result = await self._deregister_from_registry()
            
            # 更新状态
            self.service_info['status'] = 'deregistered'
            
            result = {
                'service_id': self.service_id,
                'service_name': self.service_info['service_name'],
                'deregistered_at': datetime.now().isoformat(),
                'deregistration_result': deregistration_result
            }
            
            self.log_info("服务注销成功", service_id=self.service_id)
            return result
            
        except Exception as e:
            self.service_info['status'] = 'deregistration_failed'
            self.log_error("服务注销失败", e)
            raise BusinessException(f"服务注销失败: {str(e)}")
    
    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """检查指定服务的健康状态
        
        Args:
            service_name: 服务名称
            
        Returns:
            Dict[str, Any]: 健康状态
        """
        try:
            # 获取服务节点
            nodes = await self.get_service_nodes(service_name)
            
            if not nodes:
                return {
                    'service_name': service_name,
                    'status': 'not_found',
                    'message': '服务未找到',
                    'checked_at': datetime.now().isoformat()
                }
            
            # 检查每个节点的健康状态
            health_results = []
            healthy_count = 0
            
            for node in nodes:
                node_health = await self._check_node_health(node)
                health_results.append(node_health)
                if node_health['healthy']:
                    healthy_count += 1
            
            # 计算整体健康状态
            total_nodes = len(nodes)
            health_ratio = healthy_count / total_nodes
            
            if health_ratio >= 0.8:
                overall_status = 'healthy'
            elif health_ratio >= 0.5:
                overall_status = 'degraded'
            else:
                overall_status = 'unhealthy'
            
            result = {
                'service_name': service_name,
                'status': overall_status,
                'total_nodes': total_nodes,
                'healthy_nodes': healthy_count,
                'health_ratio': health_ratio,
                'nodes': health_results,
                'checked_at': datetime.now().isoformat()
            }
            
            self.log_info(f"服务健康检查完成，{service_name}状态: {overall_status}")
            return result
            
        except Exception as e:
            self.log_error("服务健康检查失败", e, service_name=service_name)
            raise BusinessException(f"服务健康检查失败: {str(e)}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取服务发现相关指标
        
        Returns:
            Dict[str, Any]: 指标信息
        """
        try:
            # 获取所有服务
            all_services = await self.get_services()
            
            # 统计指标
            total_services = len(all_services)
            service_types = {}
            healthy_services = 0
            
            for service in all_services:
                service_type = service.get('service_type', 'unknown')
                service_types[service_type] = service_types.get(service_type, 0) + 1
                
                if service.get('status') == 'healthy':
                    healthy_services += 1
            
            metrics = {
                'total_services': total_services,
                'healthy_services': healthy_services,
                'unhealthy_services': total_services - healthy_services,
                'health_ratio': healthy_services / total_services if total_services > 0 else 0,
                'service_types': service_types,
                'current_service': {
                    'service_id': self.service_id,
                    'status': self.service_info['status'],
                    'uptime': self._calculate_uptime()
                },
                'collected_at': datetime.now().isoformat()
            }
            
            self.log_info("获取服务发现指标成功")
            return metrics
            
        except Exception as e:
            self.log_error("获取服务发现指标失败", e)
            raise BusinessException(f"获取服务发现指标失败: {str(e)}")
    
    def _get_local_ip(self) -> str:
        """获取本地IP地址
        
        Returns:
            str: 本地IP地址
        """
        try:
            # 创建一个UDP socket连接到外部地址来获取本地IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    async def _get_runtime_info(self) -> Dict[str, Any]:
        """获取运行时信息
        
        Returns:
            Dict[str, Any]: 运行时信息
        """
        import psutil
        
        try:
            # 获取系统信息
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_total': memory.total,
                'memory_available': memory.available,
                'uptime': self._calculate_uptime(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.log_warning(f"获取运行时信息失败: {e}")
            return {'error': '无法获取运行时信息'}
    
    def _calculate_uptime(self) -> str:
        """计算服务运行时间
        
        Returns:
            str: 运行时间
        """
        try:
            registered_time = datetime.fromisoformat(self.service_info['registered_at'])
            uptime = datetime.now() - registered_time
            
            days = uptime.days
            hours, remainder = divmod(uptime.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            return f"{days}d {hours}h {minutes}m {seconds}s"
        except Exception:
            return "unknown"
    
    async def _get_registered_services(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """从注册中心获取服务列表
        
        Args:
            filters: 筛选条件
            
        Returns:
            List[Dict[str, Any]]: 服务列表
        """
        # 实际环境中应该从服务注册中心获取服务列表
        services = [self.service_info.copy()]
        
        # 应用筛选条件
        if filters:
            filtered_services = []
            for service in services:
                match = True
                for key, value in filters.items():
                    if key in service and service[key] != value:
                        match = False
                        break
                if match:
                    filtered_services.append(service)
            return filtered_services
        
        return services
    
    async def _get_service_nodes_from_registry(self, service_name: str) -> List[Dict[str, Any]]:
        """从注册中心获取服务节点
        
        Args:
            service_name: 服务名称
            
        Returns:
            List[Dict[str, Any]]: 节点列表
        """
        # 实际环境中应该从服务注册中心获取节点列表
        if service_name == self.service_info['service_name']:
            return [self.service_info.copy()]
        return []
    
    async def _register_to_registry(self) -> Dict[str, Any]:
        """向注册中心注册服务
        
        Returns:
            Dict[str, Any]: 注册结果
        """
        # 实际环境中应该实现服务注册逻辑
        return {
            'success': True,
            'message': '服务注册成功',
            'registry_endpoint': 'mock://registry'
        }
    
    async def _deregister_from_registry(self) -> Dict[str, Any]:
        """从注册中心注销服务
        
        Returns:
            Dict[str, Any]: 注销结果
        """
        # 实际环境中应该实现服务注销逻辑
        return {
            'success': True,
            'message': '服务注销成功',
            'registry_endpoint': 'mock://registry'
        }
    
    async def _check_node_health(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """检查节点健康状态
        
        Args:
            node: 节点信息
            
        Returns:
            Dict[str, Any]: 健康状态
        """
        # 实际环境中应该实现节点健康检查逻辑
        return {
            'node_id': node.get('service_id'),
            'host': node.get('host'),
            'port': node.get('port'),
            'healthy': True,
            'response_time': 0.05,
            'checked_at': datetime.now().isoformat()
        }
