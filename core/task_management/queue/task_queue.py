"""
任务队列管理器
负责任务的添加、获取、状态更新等操作
"""
import json
import time
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
from core.redis_manager import RedisManager
from core.config import settings
from core.task_management.utils.status import TaskStatus

logger = logging.getLogger(__name__)

class TaskQueue:
    """任务队列管理器"""
    def __init__(self):
        self.redis = RedisManager()
        self.max_concurrent = settings.TASK_QUEUE_MAX_CONCURRENT
        self.max_retries = settings.TASK_QUEUE_MAX_RETRIES
        self.retry_delay = settings.TASK_QUEUE_RETRY_DELAY
        self.result_ttl = settings.TASK_QUEUE_RESULT_TTL
        
    async def add_task(self, task_data: Dict[str, Any], priority: float = 0, task_id: Optional[str] = None) -> str:
        """添加任务到队列"""
        try:
            if task_id is not None:
                task_data['id'] = task_id
            else:
                task_id = task_data.get('id')
                if not task_id:
                    task_id = task_data.get('task_id')
                if not task_id:
                    raise ValueError("任务ID不能为空")
            
            logger.info(f"TaskQueue.add_task - 添加任务: {task_id}")
            
            if 'status' not in task_data:
                task_data['status'] = TaskStatus.WAITING
            if 'created_at' not in task_data:
                task_data['created_at'] = datetime.now().isoformat()
            
            task_key = f"task:{task_id}"
            await self.redis.set_value(task_key, task_data)
            await self.redis.zadd_task("task_queue:waiting", task_id, priority)
            
            return task_id
            
        except Exception as e:
            logger.error(f"TaskQueue.add_task - 添加任务失败: {str(e)}")
            raise
            
    async def get_task(self, task_id: str) -> Optional[Dict]:
        """获取任务信息"""
        try:
            task_key = f"task:{task_id}"
            return await self.redis.get_value(task_key, as_json=True)
        except Exception as e:
            logger.error(f"TaskQueue.get_task - 获取任务失败: {str(e)}")
            return None
            
    async def update_task_status(
        self, 
        task_id: str, 
        status: int,
        result: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        """更新任务状态"""
        try:
            task_key = f"task:{task_id}"
            task_data = await self.redis.get_value(task_key, as_json=True)
            if not task_data:
                raise ValueError(f"任务不存在: {task_id}")
                
            task_data['status'] = status
            task_data['updated_at'] = datetime.now().isoformat()
            
            if status == TaskStatus.PROCESSING:
                task_data['started_at'] = datetime.now().isoformat()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT]:
                task_data['finished_at'] = datetime.now().isoformat()
                
            if result:
                result_key = f"result:{task_id}"
                await self.redis.set_value(result_key, result, ex=self.result_ttl)
                task_data['has_result'] = True
                
            if error:
                task_data['error'] = error
                
            await self.redis.set_value(task_key, task_data)
            
            if status != TaskStatus.WAITING:
                await self.redis.delete_key(f"task_queue:waiting:{task_id}")
                
            logger.info(f"任务状态更新成功: {task_id} -> {status}")
            
        except Exception as e:
            logger.error(f"更新任务状态失败: {str(e)}")
            raise
            
    async def get_next_task(self) -> Optional[Dict]:
        """获取下一个待处理任务"""
        try:
            tasks = await self.redis.zget_tasks("task_queue:waiting", 0, 0)
            if not tasks:
                return None
                
            task_id = tasks[0]
            return await self.get_task(task_id)
            
        except Exception as e:
            logger.error(f"获取下一个任务失败: {str(e)}")
            return None
            
    async def cleanup_expired_results(self):
        """清理过期的结果"""
        try:
            pattern = "result:*"
            await self.redis.delete_pattern(pattern)
        except Exception as e:
            logger.error(f"清理过期结果失败: {str(e)}")
            raise 