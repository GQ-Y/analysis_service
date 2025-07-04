#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: task_repository.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 任务仓储

处理任务相关的数据访问操作，包括任务的增删改查、状态管理等。

本文件是分析服务项目的一部分。
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .base_repository import InMemoryRepository
from app.models.task_model import TaskModel, TaskQueue, TaskResult, TaskStatistics
from app.models.base_model import TaskStatusEnum, PriorityEnum
from app.exceptions.business_exception import BusinessException


class TaskRepository(InMemoryRepository[TaskModel]):
    """任务仓储"""
    
    def __init__(self):
        """初始化任务仓储"""
        super().__init__(TaskModel)
    
    async def find_by_status(self, status: TaskStatusEnum) -> List[TaskModel]:
        """根据状态查找任务
        
        Args:
            status: 任务状态
            
        Returns:
            List[TaskModel]: 任务列表
        """
        return await self.find_by_field('status', status)
    
    async def find_by_user(self, user_id: str) -> List[TaskModel]:
        """根据用户ID查找任务
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[TaskModel]: 任务列表
        """
        return await self.find_by_field('user_id', user_id)
    
    async def find_running_tasks(self) -> List[TaskModel]:
        """查找运行中的任务
        
        Returns:
            List[TaskModel]: 运行中的任务列表
        """
        return await self.find_by_status(TaskStatusEnum.RUNNING)
    
    async def find_pending_tasks(self) -> List[TaskModel]:
        """查找等待中的任务
        
        Returns:
            List[TaskModel]: 等待中的任务列表
        """
        return await self.find_by_status(TaskStatusEnum.PENDING)
    
    async def find_by_priority(self, priority: PriorityEnum) -> List[TaskModel]:
        """根据优先级查找任务
        
        Args:
            priority: 优先级
            
        Returns:
            List[TaskModel]: 任务列表
        """
        return await self.find_by_field('priority', priority)
    
    async def find_by_batch(self, batch_id: str) -> List[TaskModel]:
        """根据批次ID查找任务
        
        Args:
            batch_id: 批次ID
            
        Returns:
            List[TaskModel]: 任务列表
        """
        return await self.find_by_field('batch_id', batch_id)
    
    async def find_by_date_range(self, start_date: datetime, end_date: datetime) -> List[TaskModel]:
        """根据日期范围查找任务
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[TaskModel]: 任务列表
        """
        tasks = []
        for task in self._data.values():
            if start_date <= task.created_at <= end_date:
                tasks.append(task)
        return tasks
    
    async def update_status(self, task_id: str, status: TaskStatusEnum) -> bool:
        """更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            
        Returns:
            bool: 是否更新成功
        """
        task = await self.get_by_id(task_id)
        if not task:
            return False
        
        task.status = status
        task.updated_at = datetime.now()
        
        # 根据状态更新时间字段
        if status == TaskStatusEnum.RUNNING:
            task.start_time = datetime.now()
        elif status in [TaskStatusEnum.COMPLETED, TaskStatusEnum.STOPPED, TaskStatusEnum.FAILED]:
            task.stop_time = datetime.now()
            if task.start_time:
                task.duration = (task.stop_time - task.start_time).total_seconds()
        
        await self.update(task)
        return True
    
    async def update_progress(self, task_id: str, processed_frames: int, detected_objects: int) -> bool:
        """更新任务进度
        
        Args:
            task_id: 任务ID
            processed_frames: 已处理帧数
            detected_objects: 检测到的对象数
            
        Returns:
            bool: 是否更新成功
        """
        task = await self.get_by_id(task_id)
        if not task:
            return False
        
        task.processed_frames = processed_frames
        task.detected_objects = detected_objects
        task.updated_at = datetime.now()
        
        await self.update(task)
        return True
    
    async def set_error(self, task_id: str, error_message: str, error_code: str = None) -> bool:
        """设置任务错误
        
        Args:
            task_id: 任务ID
            error_message: 错误消息
            error_code: 错误代码
            
        Returns:
            bool: 是否设置成功
        """
        task = await self.get_by_id(task_id)
        if not task:
            return False
        
        task.status = TaskStatusEnum.FAILED
        task.error_message = error_message
        task.error_code = error_code
        task.stop_time = datetime.now()
        task.updated_at = datetime.now()
        
        if task.start_time:
            task.duration = (task.stop_time - task.start_time).total_seconds()
        
        await self.update(task)
        return True
    
    async def get_user_task_count(self, user_id: str, status: TaskStatusEnum = None) -> int:
        """获取用户任务数量
        
        Args:
            user_id: 用户ID
            status: 任务状态（可选）
            
        Returns:
            int: 任务数量
        """
        filters = {'user_id': user_id}
        if status:
            filters['status'] = status
        
        return await self.count(filters)
    
    async def get_statistics(self, user_id: str = None, days: int = 30) -> Dict[str, Any]:
        """获取任务统计信息
        
        Args:
            user_id: 用户ID（可选）
            days: 统计天数
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        start_date = datetime.now() - timedelta(days=days)
        
        # 构建过滤条件
        filters = {}
        if user_id:
            filters['user_id'] = user_id
        
        # 获取所有任务
        all_tasks = await self.find_all(filters)
        
        # 过滤日期范围内的任务
        recent_tasks = [task for task in all_tasks if task.created_at >= start_date]
        
        # 统计各状态任务数量
        status_counts = {}
        for status in TaskStatusEnum:
            status_counts[status.value] = len([task for task in recent_tasks if task.status == status])
        
        # 统计优先级分布
        priority_counts = {}
        for priority in PriorityEnum:
            priority_counts[priority.value] = len([task for task in recent_tasks if task.priority == priority])
        
        # 计算成功率
        completed_tasks = status_counts.get(TaskStatusEnum.COMPLETED.value, 0)
        failed_tasks = status_counts.get(TaskStatusEnum.FAILED.value, 0)
        total_finished = completed_tasks + failed_tasks
        success_rate = (completed_tasks / total_finished) if total_finished > 0 else 0
        
        # 计算平均运行时间
        finished_tasks = [task for task in recent_tasks 
                         if task.status in [TaskStatusEnum.COMPLETED, TaskStatusEnum.FAILED] 
                         and task.duration is not None]
        avg_duration = sum(task.duration for task in finished_tasks) / len(finished_tasks) if finished_tasks else 0
        
        return {
            'total_tasks': len(recent_tasks),
            'status_distribution': status_counts,
            'priority_distribution': priority_counts,
            'success_rate': success_rate,
            'average_duration_seconds': avg_duration,
            'period_days': days,
            'user_id': user_id
        }
    
    async def cleanup_old_tasks(self, days: int = 90) -> int:
        """清理旧任务
        
        Args:
            days: 保留天数
            
        Returns:
            int: 清理的任务数量
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        old_tasks = []
        for task in self._data.values():
            if (task.created_at < cutoff_date and 
                task.status in [TaskStatusEnum.COMPLETED, TaskStatusEnum.FAILED, TaskStatusEnum.CANCELLED]):
                old_tasks.append(task.id)
        
        # 删除旧任务
        for task_id in old_tasks:
            await self.delete(task_id)
        
        self.logger.info(f"清理了 {len(old_tasks)} 个旧任务")
        return len(old_tasks)


class TaskQueueRepository(InMemoryRepository[TaskQueue]):
    """任务队列仓储"""
    
    def __init__(self):
        """初始化任务队列仓储"""
        super().__init__(TaskQueue)
    
    async def find_by_task_id(self, task_id: str) -> Optional[TaskQueue]:
        """根据任务ID查找队列项
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[TaskQueue]: 队列项
        """
        return await self.find_one_by_field('task_id', task_id)
    
    async def find_pending_queue_items(self) -> List[TaskQueue]:
        """查找等待中的队列项
        
        Returns:
            List[TaskQueue]: 队列项列表
        """
        return await self.find_by_field('status', TaskStatusEnum.PENDING)
    
    async def find_by_priority(self, priority: PriorityEnum) -> List[TaskQueue]:
        """根据优先级查找队列项
        
        Args:
            priority: 优先级
            
        Returns:
            List[TaskQueue]: 队列项列表
        """
        return await self.find_by_field('priority', priority)
    
    async def get_next_task(self) -> Optional[TaskQueue]:
        """获取下一个待执行的任务
        
        Returns:
            Optional[TaskQueue]: 队列项
        """
        # 获取所有等待中的任务
        pending_tasks = await self.find_pending_queue_items()
        
        if not pending_tasks:
            return None
        
        # 按优先级和创建时间排序
        priority_order = {
            PriorityEnum.URGENT: 0,
            PriorityEnum.HIGH: 1,
            PriorityEnum.NORMAL: 2,
            PriorityEnum.LOW: 3
        }
        
        pending_tasks.sort(key=lambda x: (
            priority_order.get(x.priority, 999),
            x.created_at
        ))
        
        return pending_tasks[0]


class TaskResultRepository(InMemoryRepository[TaskResult]):
    """任务结果仓储"""
    
    def __init__(self):
        """初始化任务结果仓储"""
        super().__init__(TaskResult)
    
    async def find_by_task_id(self, task_id: str) -> List[TaskResult]:
        """根据任务ID查找结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            List[TaskResult]: 结果列表
        """
        return await self.find_by_field('task_id', task_id)
    
    async def find_by_frame_range(self, task_id: str, start_frame: int, end_frame: int) -> List[TaskResult]:
        """根据帧范围查找结果
        
        Args:
            task_id: 任务ID
            start_frame: 开始帧号
            end_frame: 结束帧号
            
        Returns:
            List[TaskResult]: 结果列表
        """
        results = []
        for result in self._data.values():
            if (result.task_id == task_id and 
                start_frame <= result.frame_number <= end_frame):
                results.append(result)
        
        # 按帧号排序
        results.sort(key=lambda x: x.frame_number)
        return results
    
    async def get_latest_result(self, task_id: str) -> Optional[TaskResult]:
        """获取最新结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[TaskResult]: 最新结果
        """
        results = await self.find_by_task_id(task_id)
        if not results:
            return None
        
        return max(results, key=lambda x: x.frame_number)
    
    async def cleanup_old_results(self, task_id: str, keep_count: int = 1000) -> int:
        """清理旧结果
        
        Args:
            task_id: 任务ID
            keep_count: 保留数量
            
        Returns:
            int: 清理的结果数量
        """
        results = await self.find_by_task_id(task_id)
        
        if len(results) <= keep_count:
            return 0
        
        # 按帧号排序，保留最新的
        results.sort(key=lambda x: x.frame_number, reverse=True)
        old_results = results[keep_count:]
        
        # 删除旧结果
        for result in old_results:
            await self.delete(result.id)
        
        return len(old_results)


class TaskStatisticsRepository(InMemoryRepository[TaskStatistics]):
    """任务统计仓储"""
    
    def __init__(self):
        """初始化任务统计仓储"""
        super().__init__(TaskStatistics)
    
    async def find_by_task_id(self, task_id: str) -> Optional[TaskStatistics]:
        """根据任务ID查找统计
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[TaskStatistics]: 统计信息
        """
        return await self.find_one_by_field('task_id', task_id)
    
    async def update_statistics(self, task_id: str, processing_time: float, confidence: float) -> bool:
        """更新统计信息
        
        Args:
            task_id: 任务ID
            processing_time: 处理时间
            confidence: 置信度
            
        Returns:
            bool: 是否更新成功
        """
        stats = await self.find_by_task_id(task_id)
        
        if not stats:
            # 创建新的统计记录
            stats = TaskStatistics(
                task_id=task_id,
                start_time=datetime.now()
            )
            await self.create(stats)
        
        # 更新统计信息
        stats.update_statistics(processing_time, confidence)
        await self.update(stats)
        
        return True
