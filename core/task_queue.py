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

logger = logging.getLogger(__name__)

class TaskStatus:
    """任务状态定义"""
    WAITING = 0      # 等待中
    PROCESSING = 1   # 处理中
    COMPLETED = 2    # 已完成
    FAILED = -1      # 失败
    TIMEOUT = -2     # 超时
    CANCELLED = -3   # 已取消
    STOPPING = -4    # 停止中
    STOPPED = -5     # 已停止

class TaskQueue:
    """任务队列管理器"""
    def __init__(self):
        self.redis = RedisManager()
        self.running_tasks: Dict[str, Dict] = {}
        self.max_concurrent = settings.TASK_QUEUE_MAX_CONCURRENT
        self.max_retries = settings.TASK_QUEUE_MAX_RETRIES
        self.retry_delay = settings.TASK_QUEUE_RETRY_DELAY
        self.result_ttl = settings.TASK_QUEUE_RESULT_TTL
        
    async def add_task(self, task_data: Dict[str, Any], priority: float = 0, task_id: Optional[str] = None) -> str:
        """添加任务到队列
        
        Args:
            task_data: 任务数据
            priority: 任务优先级，默认为0
            task_id: 可选的任务ID，如果不提供则从task_data中获取
            
        Returns:
            str: 任务ID
        """
        try:
            # 获取或使用提供的任务ID
            if task_id is not None:
                task_data['id'] = task_id
            else:
                task_id = task_data.get('id')
                if not task_id:
                    task_id = task_data.get('task_id')  # 兼容不同字段名
                if not task_id:
                    raise ValueError("任务ID不能为空")
            
            logger.info(f"TaskQueue.add_task - 添加任务: {task_id}")
            
            # 设置任务状态和创建时间
            if 'status' not in task_data:
                task_data['status'] = TaskStatus.WAITING
            if 'created_at' not in task_data:
                task_data['created_at'] = datetime.now().isoformat()
            
            # 保存任务数据
            task_key = f"task:{task_id}"
            logger.info(f"TaskQueue.add_task - 保存任务到Redis: key={task_key}")
            
            try:
                await self.redis.set_value(task_key, task_data)
                logger.info(f"TaskQueue.add_task - 保存任务成功: {task_id}")
            except Exception as e:
                logger.error(f"TaskQueue.add_task - 保存任务数据失败: {str(e)}")
                raise
            
            # 添加到等待队列
            try:
                await self.redis.zadd_task("task_queue:waiting", task_id, priority)
                logger.info(f"TaskQueue.add_task - 添加到等待队列成功: {task_id}")
            except Exception as e:
                logger.error(f"TaskQueue.add_task - 添加到等待队列失败: {str(e)}")
                raise
            
            # 验证任务是否正确保存
            try:
                saved_task = await self.get_task(task_id)
                if saved_task:
                    logger.info(f"TaskQueue.add_task - 验证任务已保存: {task_id}")
                else:
                    logger.warning(f"TaskQueue.add_task - 警告: 验证失败，未找到已保存的任务: {task_id}")
            except Exception as e:
                logger.error(f"TaskQueue.add_task - 验证任务保存时发生错误: {str(e)}")
            
            logger.info(f"TaskQueue.add_task - 任务添加成功: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"TaskQueue.add_task - 添加任务失败: {str(e)}")
            raise
            
    async def get_task(self, task_id: str) -> Optional[Dict]:
        """获取任务信息"""
        try:
            logger.info(f"TaskQueue.get_task - 获取任务: {task_id}")
            task_key = f"task:{task_id}"
            
            try:
                task_data = await self.redis.get_value(task_key, as_json=True)
                if task_data:
                    logger.info(f"TaskQueue.get_task - 找到任务: {task_id}")
                    return task_data
                else:
                    logger.warning(f"TaskQueue.get_task - 任务不存在: {task_id}")
                    return None
            except Exception as e:
                logger.error(f"TaskQueue.get_task - 从Redis获取任务数据时出错: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"TaskQueue.get_task - 获取任务信息失败: {str(e)}")
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
                
            # 更新状态
            task_data['status'] = status
            task_data['updated_at'] = datetime.now().isoformat()
            
            if status == TaskStatus.PROCESSING:
                task_data['started_at'] = datetime.now().isoformat()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT]:
                task_data['finished_at'] = datetime.now().isoformat()
                
            if result:
                # 保存结果，设置过期时间
                result_key = f"result:{task_id}"
                await self.redis.set_value(result_key, result, ex=self.result_ttl)
                task_data['has_result'] = True
                
            if error:
                task_data['error'] = error
                
            # 更新任务数据
            await self.redis.set_value(task_key, task_data)
            
            # 从等待队列移除
            if status != TaskStatus.WAITING:
                await self.redis.delete_key(f"task_queue:waiting:{task_id}")
                
            logger.info(f"任务状态更新成功: {task_id} -> {status}")
            
        except Exception as e:
            logger.error(f"更新任务状态失败: {str(e)}")
            raise
            
    async def get_next_task(self) -> Optional[Dict]:
        """获取下一个待处理任务"""
        try:
            # 检查是否达到最大并发
            if len(self.running_tasks) >= self.max_concurrent:
                return None
                
            # 获取优先级最高的任务
            tasks = await self.redis.zget_tasks("task_queue:waiting", 0, 0)
            if not tasks:
                return None
                
            task_id = tasks[0]
            task_data = await self.get_task(task_id)
            if not task_data:
                return None
                
            # 更新状态为处理中
            await self.update_task_status(task_id, TaskStatus.PROCESSING)
            
            # 添加到运行中任务
            self.running_tasks[task_id] = task_data
            
            return task_data
            
        except Exception as e:
            logger.error(f"获取下一个任务失败: {str(e)}")
            return None
            
    async def complete_task(self, task_id: str, result: Dict):
        """完成任务"""
        try:
            await self.update_task_status(task_id, TaskStatus.COMPLETED, result=result)
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
        except Exception as e:
            logger.error(f"完成任务失败: {str(e)}")
            raise
            
    async def fail_task(self, task_id: str, error: str):
        """标记任务失败"""
        try:
            task_data = await self.get_task(task_id)
            if not task_data:
                raise ValueError(f"任务不存在: {task_id}")
                
            retries = task_data.get('retries', 0)
            if retries < self.max_retries:
                # 重试任务
                task_data['retries'] = retries + 1
                task_data['last_error'] = error
                task_data['retry_at'] = (datetime.now().timestamp() + self.retry_delay)
                
                await self.update_task_status(task_id, TaskStatus.WAITING)
                await self.redis.zadd_task(
                    "task_queue:waiting",
                    task_id,
                    task_data['retry_at']
                )
            else:
                # 达到最大重试次数，标记为失败
                await self.update_task_status(task_id, TaskStatus.FAILED, error=error)
                
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
                
        except Exception as e:
            logger.error(f"标记任务失败时出错: {str(e)}")
            raise
            
    async def cancel_task(self, task_id: str):
        """取消任务"""
        try:
            await self.update_task_status(task_id, TaskStatus.CANCELLED)
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
        except Exception as e:
            logger.error(f"取消任务失败: {str(e)}")
            raise
            
    async def cleanup_expired_results(self):
        """清理过期的结果"""
        try:
            pattern = "result:*"
            await self.redis.delete_pattern(pattern)
        except Exception as e:
            logger.error(f"清理过期结果失败: {str(e)}")
            raise
            
    async def start_cleanup_task(self):
        """启动清理任务"""
        while True:
            try:
                await self.cleanup_expired_results()
            except Exception as e:
                logger.error(f"执行清理任务失败: {str(e)}")
            await asyncio.sleep(settings.TASK_QUEUE_CLEANUP_INTERVAL)