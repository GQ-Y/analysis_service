#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: task_service.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 任务服务

处理视频分析任务的业务逻辑，包括任务创建、启动、停止、查询等功能。

本文件是分析服务项目的一部分。
"""

import uuid
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_service import CacheableService
from app.exceptions.business_exception import BusinessException


class TaskService(CacheableService):
    """任务服务"""
    
    def __init__(self):
        """初始化任务服务"""
        super().__init__(cache_ttl=600)  # 10分钟缓存
        self.task_repository = None
        self.analyzer_factory = None
        self.memory_manager = None
    
    async def _initialize_service(self):
        """初始化服务依赖"""
        self.task_repository = self.get_dependency('task_repository')
        self.analyzer_factory = self.get_dependency('analyzer_factory')
        self.memory_manager = self.get_dependency('memory_manager')
    
    async def get_tasks(
        self,
        page: int = 1,
        page_size: int = 20,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """获取任务列表
        
        Args:
            page: 页码
            page_size: 每页大小
            filters: 筛选条件
            
        Returns:
            Dict[str, Any]: 任务列表和分页信息
        """
        try:
            # 验证分页参数
            page, page_size = self.validate_pagination(page, page_size)
            
            # 构建缓存键
            cache_key = self.generate_cache_key(
                'tasks_list',
                page=page,
                page_size=page_size,
                **filters or {}
            )
            
            # 尝试从缓存获取
            cached_result = await self.get_cached_data(cache_key)
            if cached_result:
                self.log_info("从缓存获取任务列表", cache_key=cache_key)
                return cached_result
            
            # 从数据库获取
            total = await self.task_repository.count_tasks(filters)
            tasks = await self.task_repository.get_tasks(page, page_size, filters)
            
            # 构建结果
            result = {
                'items': tasks,
                'pagination': self.calculate_pagination(total, page, page_size)
            }
            
            # 缓存结果
            await self.set_cached_data(cache_key, result, ttl=300)  # 5分钟缓存
            
            self.log_info(f"获取任务列表成功，共{total}条记录")
            return result
            
        except Exception as e:
            self.log_error("获取任务列表失败", e)
            raise BusinessException(f"获取任务列表失败: {str(e)}")
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务详情
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[Dict[str, Any]]: 任务详情
        """
        try:
            if not task_id:
                raise BusinessException("任务ID不能为空")
            
            # 构建缓存键
            cache_key = self.generate_cache_key('task_detail', task_id)
            
            # 尝试从缓存获取
            cached_task = await self.get_cached_data(cache_key)
            if cached_task:
                self.log_info("从缓存获取任务详情", task_id=task_id)
                return cached_task
            
            # 从数据库获取
            task = await self.task_repository.get_task(task_id)
            if not task:
                return None
            
            # 缓存结果
            await self.set_cached_data(cache_key, task, ttl=600)  # 10分钟缓存
            
            self.log_info("获取任务详情成功", task_id=task_id)
            return task
            
        except Exception as e:
            self.log_error("获取任务详情失败", e, task_id=task_id)
            raise BusinessException(f"获取任务详情失败: {str(e)}")
    
    async def start_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """启动分析任务
        
        Args:
            task_data: 任务数据
            
        Returns:
            Dict[str, Any]: 创建的任务信息
        """
        try:
            # 验证必需参数
            required_fields = ['task_name', 'stream_url', 'analysis_type', 'model_code']
            self.validate_required_params(task_data, required_fields)
            
            # 生成任务ID
            task_id = str(uuid.uuid4())
            
            # 准备任务数据
            task_info = {
                'task_id': task_id,
                'task_name': task_data['task_name'],
                'stream_url': task_data['stream_url'],
                'analysis_type': task_data['analysis_type'],
                'model_code': task_data['model_code'],
                'user_id': task_data.get('user_id'),
                'created_by': task_data.get('created_by'),
                'status': 'starting',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'config': task_data.get('config', {}),
                'roi_config': task_data.get('roi_config', {}),
                'callback_url': task_data.get('callback_url'),
            }
            
            # 验证流URL
            await self._validate_stream_url(task_info['stream_url'])
            
            # 创建分析器
            analyzer = await self._create_analyzer(
                task_info['analysis_type'],
                task_info['model_code']
            )
            
            # 保存任务到数据库
            await self.task_repository.create_task(task_info)
            
            # 启动任务处理
            await self._start_task_processing(task_id, analyzer, task_info)
            
            # 更新任务状态
            await self.task_repository.update_task_status(task_id, 'running')
            task_info['status'] = 'running'
            
            # 清除相关缓存
            await self._clear_task_caches(task_id)
            
            self.log_info("任务启动成功", task_id=task_id, task_name=task_info['task_name'])
            return task_info
            
        except Exception as e:
            self.log_error("任务启动失败", e, task_data=task_data)
            raise BusinessException(f"任务启动失败: {str(e)}")
    
    async def stop_task(self, task_id: str, user_id: str = None) -> Dict[str, Any]:
        """停止分析任务
        
        Args:
            task_id: 任务ID
            user_id: 用户ID
            
        Returns:
            Dict[str, Any]: 停止结果
        """
        try:
            if not task_id:
                raise BusinessException("任务ID不能为空")
            
            # 获取任务信息
            task = await self.get_task(task_id)
            if not task:
                raise BusinessException("任务不存在")
            
            # 检查权限
            if user_id and task.get('user_id') != user_id:
                raise BusinessException("无权限操作此任务")
            
            # 检查任务状态
            if task['status'] not in ['running', 'starting']:
                raise BusinessException(f"任务状态为{task['status']}，无法停止")
            
            # 停止任务处理
            await self._stop_task_processing(task_id)
            
            # 更新任务状态
            await self.task_repository.update_task_status(task_id, 'stopped')
            
            # 清除相关缓存
            await self._clear_task_caches(task_id)
            
            result = {
                'task_id': task_id,
                'status': 'stopped',
                'stopped_at': datetime.now().isoformat()
            }
            
            self.log_info("任务停止成功", task_id=task_id)
            return result
            
        except Exception as e:
            self.log_error("任务停止失败", e, task_id=task_id)
            raise BusinessException(f"任务停止失败: {str(e)}")
    
    async def delete_task(self, task_id: str, user_id: str = None) -> bool:
        """删除任务
        
        Args:
            task_id: 任务ID
            user_id: 用户ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            if not task_id:
                raise BusinessException("任务ID不能为空")
            
            # 获取任务信息
            task = await self.get_task(task_id)
            if not task:
                raise BusinessException("任务不存在")
            
            # 检查权限
            if user_id and task.get('user_id') != user_id:
                raise BusinessException("无权限操作此任务")
            
            # 检查任务状态
            if task['status'] == 'running':
                raise BusinessException("运行中的任务无法删除，请先停止任务")
            
            # 删除任务
            await self.task_repository.delete_task(task_id)
            
            # 清除相关缓存
            await self._clear_task_caches(task_id)
            
            self.log_info("任务删除成功", task_id=task_id)
            return True
            
        except Exception as e:
            self.log_error("任务删除失败", e, task_id=task_id)
            raise BusinessException(f"任务删除失败: {str(e)}")
    
    async def batch_start_tasks(self, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """批量启动任务
        
        Args:
            batch_data: 批量任务数据
            
        Returns:
            List[Dict[str, Any]]: 批量任务结果
        """
        try:
            tasks_data = batch_data.get('tasks', [])
            if not tasks_data:
                raise BusinessException("批量任务数据不能为空")
            
            if len(tasks_data) > 10:  # 限制批量数量
                raise BusinessException("批量任务数量不能超过10个")
            
            results = []
            success_count = 0
            
            for i, task_data in enumerate(tasks_data):
                try:
                    # 添加批量任务的公共信息
                    task_data.update({
                        'user_id': batch_data.get('user_id'),
                        'created_by': batch_data.get('created_by'),
                        'batch_id': batch_data.get('batch_id', str(uuid.uuid4())),
                        'batch_index': i
                    })
                    
                    # 启动单个任务
                    task_result = await self.start_task(task_data)
                    results.append({
                        'index': i,
                        'success': True,
                        'task': task_result
                    })
                    success_count += 1
                    
                except Exception as e:
                    results.append({
                        'index': i,
                        'success': False,
                        'error': str(e),
                        'task_data': task_data
                    })
            
            self.log_info(f"批量任务启动完成，成功{success_count}个，失败{len(tasks_data) - success_count}个")
            return results
            
        except Exception as e:
            self.log_error("批量任务启动失败", e)
            raise BusinessException(f"批量任务启动失败: {str(e)}")
    
    async def _validate_stream_url(self, stream_url: str):
        """验证流URL
        
        Args:
            stream_url: 流URL
            
        Raises:
            BusinessException: URL无效
        """
        # 这里可以添加流URL验证逻辑
        if not stream_url or not stream_url.startswith(('rtsp://', 'http://', 'https://')):
            raise BusinessException("无效的流URL格式")
    
    async def _create_analyzer(self, analysis_type: str, model_code: str):
        """创建分析器
        
        Args:
            analysis_type: 分析类型
            model_code: 模型代码
            
        Returns:
            分析器实例
        """
        try:
            analyzer = await self.analyzer_factory.create_analyzer(analysis_type, model_code)
            return analyzer
        except Exception as e:
            raise BusinessException(f"创建分析器失败: {str(e)}")
    
    async def _start_task_processing(self, task_id: str, analyzer, task_info: Dict[str, Any]):
        """启动任务处理
        
        Args:
            task_id: 任务ID
            analyzer: 分析器实例
            task_info: 任务信息
        """
        # 这里应该启动实际的任务处理逻辑
        # 暂时只是模拟
        self.log_info("启动任务处理", task_id=task_id)
    
    async def _stop_task_processing(self, task_id: str):
        """停止任务处理
        
        Args:
            task_id: 任务ID
        """
        # 这里应该停止实际的任务处理逻辑
        # 暂时只是模拟
        self.log_info("停止任务处理", task_id=task_id)
    
    async def _clear_task_caches(self, task_id: str):
        """清除任务相关缓存
        
        Args:
            task_id: 任务ID
        """
        cache_keys = [
            self.generate_cache_key('task_detail', task_id),
            'tasks_list:*'  # 清除所有任务列表缓存
        ]
        
        for cache_key in cache_keys:
            await self.delete_cached_data(cache_key)
