#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: stream_service.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 流服务

处理视频流相关的业务逻辑，包括流的创建、管理、连接测试等功能。

本文件是分析服务项目的一部分。
"""

import uuid
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp

from .base_service import CacheableService
from app.exceptions.business_exception import BusinessException


class StreamService(CacheableService):
    """流服务"""
    
    def __init__(self):
        """初始化流服务"""
        super().__init__(cache_ttl=300)  # 5分钟缓存
        self.stream_repository = None
    
    async def _initialize_service(self):
        """初始化服务依赖"""
        self.stream_repository = self.get_dependency('stream_repository')
    
    async def get_streams(
        self,
        page: int = 1,
        page_size: int = 20,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """获取流列表
        
        Args:
            page: 页码
            page_size: 每页大小
            filters: 筛选条件
            
        Returns:
            Dict[str, Any]: 流列表和分页信息
        """
        try:
            # 验证分页参数
            page, page_size = self.validate_pagination(page, page_size)
            
            # 构建缓存键
            cache_key = self.generate_cache_key(
                'streams_list',
                page=page,
                page_size=page_size,
                **filters or {}
            )
            
            # 尝试从缓存获取
            cached_result = await self.get_cached_data(cache_key)
            if cached_result:
                self.log_info("从缓存获取流列表", cache_key=cache_key)
                return cached_result
            
            # 从数据库获取
            total = await self.stream_repository.count_streams(filters)
            streams = await self.stream_repository.get_streams(page, page_size, filters)
            
            # 构建结果
            result = {
                'items': streams,
                'pagination': self.calculate_pagination(total, page, page_size)
            }
            
            # 缓存结果
            await self.set_cached_data(cache_key, result, ttl=300)
            
            self.log_info(f"获取流列表成功，共{total}条记录")
            return result
            
        except Exception as e:
            self.log_error("获取流列表失败", e)
            raise BusinessException(f"获取流列表失败: {str(e)}")
    
    async def get_stream(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取流详情
        
        Args:
            stream_id: 流ID
            
        Returns:
            Optional[Dict[str, Any]]: 流详情
        """
        try:
            if not stream_id:
                raise BusinessException("流ID不能为空")
            
            # 构建缓存键
            cache_key = self.generate_cache_key('stream_detail', stream_id)
            
            # 尝试从缓存获取
            cached_stream = await self.get_cached_data(cache_key)
            if cached_stream:
                self.log_info("从缓存获取流详情", stream_id=stream_id)
                return cached_stream
            
            # 从数据库获取
            stream = await self.stream_repository.get_stream(stream_id)
            if not stream:
                return None
            
            # 缓存结果
            await self.set_cached_data(cache_key, stream, ttl=600)
            
            self.log_info("获取流详情成功", stream_id=stream_id)
            return stream
            
        except Exception as e:
            self.log_error("获取流详情失败", e, stream_id=stream_id)
            raise BusinessException(f"获取流详情失败: {str(e)}")
    
    async def create_stream(self, stream_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建流
        
        Args:
            stream_data: 流数据
            
        Returns:
            Dict[str, Any]: 创建的流信息
        """
        try:
            # 验证必需参数
            required_fields = ['stream_name', 'stream_url', 'stream_type']
            self.validate_required_params(stream_data, required_fields)
            
            # 生成流ID
            stream_id = str(uuid.uuid4())
            
            # 准备流数据
            stream_info = {
                'stream_id': stream_id,
                'stream_name': stream_data['stream_name'],
                'stream_url': stream_data['stream_url'],
                'stream_type': stream_data['stream_type'],
                'user_id': stream_data.get('user_id'),
                'created_by': stream_data.get('created_by'),
                'status': 'created',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'description': stream_data.get('description', ''),
                'config': stream_data.get('config', {}),
                'metadata': stream_data.get('metadata', {}),
            }
            
            # 验证流URL
            await self._validate_stream_url(stream_info['stream_url'], stream_info['stream_type'])
            
            # 保存流到数据库
            await self.stream_repository.create_stream(stream_info)
            
            # 清除相关缓存
            await self._clear_stream_caches(stream_id)
            
            self.log_info("流创建成功", stream_id=stream_id, stream_name=stream_info['stream_name'])
            return stream_info
            
        except Exception as e:
            self.log_error("流创建失败", e, stream_data=stream_data)
            raise BusinessException(f"流创建失败: {str(e)}")
    
    async def delete_stream(self, stream_id: str, user_id: str = None) -> bool:
        """删除流
        
        Args:
            stream_id: 流ID
            user_id: 用户ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            if not stream_id:
                raise BusinessException("流ID不能为空")
            
            # 获取流信息
            stream = await self.get_stream(stream_id)
            if not stream:
                raise BusinessException("流不存在")
            
            # 检查权限
            if user_id and stream.get('user_id') != user_id:
                raise BusinessException("无权限操作此流")
            
            # 检查是否有关联的任务
            if await self._has_active_tasks(stream_id):
                raise BusinessException("流正在被任务使用，无法删除")
            
            # 删除流
            await self.stream_repository.delete_stream(stream_id)
            
            # 清除相关缓存
            await self._clear_stream_caches(stream_id)
            
            self.log_info("流删除成功", stream_id=stream_id)
            return True
            
        except Exception as e:
            self.log_error("流删除失败", e, stream_id=stream_id)
            raise BusinessException(f"流删除失败: {str(e)}")
    
    async def test_stream_connection(self, stream_id: str) -> Dict[str, Any]:
        """测试流连接
        
        Args:
            stream_id: 流ID
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        try:
            if not stream_id:
                raise BusinessException("流ID不能为空")
            
            # 获取流信息
            stream = await self.get_stream(stream_id)
            if not stream:
                raise BusinessException("流不存在")
            
            stream_url = stream['stream_url']
            stream_type = stream['stream_type']
            
            # 执行连接测试
            test_result = await self._test_connection(stream_url, stream_type)
            
            # 更新流状态
            new_status = 'online' if test_result['success'] else 'offline'
            await self.stream_repository.update_stream_status(stream_id, new_status)
            
            # 清除缓存
            await self._clear_stream_caches(stream_id)
            
            result = {
                'stream_id': stream_id,
                'test_time': datetime.now().isoformat(),
                'success': test_result['success'],
                'message': test_result['message'],
                'details': test_result.get('details', {}),
                'status': new_status
            }
            
            self.log_info("流连接测试完成", stream_id=stream_id, success=test_result['success'])
            return result
            
        except Exception as e:
            self.log_error("流连接测试失败", e, stream_id=stream_id)
            raise BusinessException(f"流连接测试失败: {str(e)}")
    
    async def get_stream_stats(self, stream_id: str) -> Dict[str, Any]:
        """获取流统计信息
        
        Args:
            stream_id: 流ID
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            if not stream_id:
                raise BusinessException("流ID不能为空")
            
            # 构建缓存键
            cache_key = self.generate_cache_key('stream_stats', stream_id)
            
            # 尝试从缓存获取
            cached_stats = await self.get_cached_data(cache_key)
            if cached_stats:
                self.log_info("从缓存获取流统计信息", stream_id=stream_id)
                return cached_stats
            
            # 获取流信息
            stream = await self.get_stream(stream_id)
            if not stream:
                raise BusinessException("流不存在")
            
            # 获取统计信息
            stats = await self._collect_stream_stats(stream_id, stream)
            
            # 缓存结果
            await self.set_cached_data(cache_key, stats, ttl=60)  # 1分钟缓存
            
            self.log_info("获取流统计信息成功", stream_id=stream_id)
            return stats
            
        except Exception as e:
            self.log_error("获取流统计信息失败", e, stream_id=stream_id)
            raise BusinessException(f"获取流统计信息失败: {str(e)}")
    
    async def _validate_stream_url(self, stream_url: str, stream_type: str):
        """验证流URL
        
        Args:
            stream_url: 流URL
            stream_type: 流类型
            
        Raises:
            BusinessException: URL无效
        """
        if not stream_url:
            raise BusinessException("流URL不能为空")
        
        # 根据流类型验证URL格式
        if stream_type == 'rtsp':
            if not stream_url.startswith('rtsp://'):
                raise BusinessException("RTSP流URL必须以rtsp://开头")
        elif stream_type == 'http':
            if not stream_url.startswith(('http://', 'https://')):
                raise BusinessException("HTTP流URL必须以http://或https://开头")
        elif stream_type == 'file':
            # 文件路径验证
            pass
        else:
            raise BusinessException(f"不支持的流类型: {stream_type}")
    
    async def _test_connection(self, stream_url: str, stream_type: str) -> Dict[str, Any]:
        """测试连接
        
        Args:
            stream_url: 流URL
            stream_type: 流类型
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        try:
            if stream_type in ['http', 'https']:
                return await self._test_http_stream(stream_url)
            elif stream_type == 'rtsp':
                return await self._test_rtsp_stream(stream_url)
            else:
                return {
                    'success': False,
                    'message': f"不支持的流类型测试: {stream_type}"
                }
        except Exception as e:
            return {
                'success': False,
                'message': f"连接测试异常: {str(e)}"
            }
    
    async def _test_http_stream(self, stream_url: str) -> Dict[str, Any]:
        """测试HTTP流
        
        Args:
            stream_url: 流URL
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.head(stream_url) as response:
                    if response.status == 200:
                        return {
                            'success': True,
                            'message': '连接成功',
                            'details': {
                                'status_code': response.status,
                                'content_type': response.headers.get('content-type', ''),
                                'content_length': response.headers.get('content-length', '')
                            }
                        }
                    else:
                        return {
                            'success': False,
                            'message': f'HTTP错误: {response.status}',
                            'details': {'status_code': response.status}
                        }
        except asyncio.TimeoutError:
            return {
                'success': False,
                'message': '连接超时'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'连接失败: {str(e)}'
            }
    
    async def _test_rtsp_stream(self, stream_url: str) -> Dict[str, Any]:
        """测试RTSP流
        
        Args:
            stream_url: 流URL
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        # 这里应该实现RTSP连接测试
        # 暂时返回模拟结果
        return {
            'success': True,
            'message': 'RTSP连接测试（模拟）',
            'details': {'protocol': 'rtsp'}
        }
    
    async def _has_active_tasks(self, stream_id: str) -> bool:
        """检查是否有活跃的任务使用此流
        
        Args:
            stream_id: 流ID
            
        Returns:
            bool: 是否有活跃任务
        """
        # 这里应该检查任务表
        # 暂时返回False
        return False
    
    async def _collect_stream_stats(self, stream_id: str, stream: Dict[str, Any]) -> Dict[str, Any]:
        """收集流统计信息
        
        Args:
            stream_id: 流ID
            stream: 流信息
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        # 这里应该收集实际的统计信息
        # 暂时返回模拟数据
        return {
            'stream_id': stream_id,
            'stream_name': stream['stream_name'],
            'status': stream['status'],
            'uptime': '00:00:00',
            'frame_rate': 0,
            'bitrate': 0,
            'resolution': 'unknown',
            'codec': 'unknown',
            'last_update': datetime.now().isoformat()
        }
    
    async def _clear_stream_caches(self, stream_id: str):
        """清除流相关缓存
        
        Args:
            stream_id: 流ID
        """
        cache_keys = [
            self.generate_cache_key('stream_detail', stream_id),
            self.generate_cache_key('stream_stats', stream_id),
            'streams_list:*'  # 清除所有流列表缓存
        ]
        
        for cache_key in cache_keys:
            await self.delete_cached_data(cache_key)
