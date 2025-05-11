"""
基于Redis的任务存储服务
"""
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
import aioredis
from models.task import TaskBase, QueueTask
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class TaskStore:
    """任务存储服务"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """初始化存储服务
        
        Args:
            redis_url: Redis连接URL
        """
        self.redis_url = redis_url
        self._redis: Optional[aioredis.Redis] = None
        
    async def connect(self):
        """连接Redis"""
        if not self._redis:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
    async def disconnect(self):
        """断开Redis连接"""
        if self._redis:
            await self._redis.close()
            self._redis = None
            
    def _task_key(self, task_id: str) -> str:
        """生成任务键名"""
        return f"task:{task_id}"
        
    def _queue_key(self, task_id: str) -> str:
        """生成队列任务键名"""
        return f"queue:{task_id}"
        
    async def save_task(self, task: TaskBase) -> bool:
        """保存任务
        
        Args:
            task: 任务对象
            
        Returns:
            bool: 是否保存成功
        """
        try:
            # 确保已连接
            await self.connect()
            
            # 转换为JSON并保存
            task_data = task.model_dump()
            # 转换datetime为ISO格式字符串
            for field in ["start_time", "stop_time", "created_at", "updated_at"]:
                if task_data.get(field):
                    task_data[field] = task_data[field].isoformat()
                    
            await self._redis.set(
                self._task_key(task.id),
                json.dumps(task_data)
            )
            return True
            
        except Exception as e:
            logger.error(f"保存任务失败: {str(e)}")
            return False
            
    async def get_task(self, task_id: str) -> Optional[TaskBase]:
        """获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[TaskBase]: 任务对象，不存在返回None
        """
        try:
            # 确保已连接
            await self.connect()
            
            # 获取任务数据
            task_data = await self._redis.get(self._task_key(task_id))
            if not task_data:
                return None
                
            # 转换回Python对象
            task_dict = json.loads(task_data)
            # 转换ISO格式字符串为datetime
            for field in ["start_time", "stop_time", "created_at", "updated_at"]:
                if task_dict.get(field):
                    task_dict[field] = datetime.fromisoformat(task_dict[field])
                    
            return TaskBase(**task_dict)
            
        except Exception as e:
            logger.error(f"获取任务失败: {str(e)}")
            return None
            
    async def save_queue_task(self, task: QueueTask) -> bool:
        """保存队列任务
        
        Args:
            task: 队列任务对象
            
        Returns:
            bool: 是否保存成功
        """
        try:
            # 确保已连接
            await self.connect()
            
            # 保存任务数据
            await self._redis.set(
                self._queue_key(task.id),
                json.dumps(task.to_dict())
            )
            
            # 同时将任务ID加入优先级队列
            await self._redis.zadd(
                "task_queue",
                {task.id: task.priority}
            )
            return True
            
        except Exception as e:
            logger.error(f"保存队列任务失败: {str(e)}")
            return False
            
    async def get_queue_task(self, task_id: str) -> Optional[QueueTask]:
        """获取队列任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[QueueTask]: 队列任务对象，不存在返回None
        """
        try:
            # 确保已连接
            await self.connect()
            
            # 获取任务数据
            task_data = await self._redis.get(self._queue_key(task_id))
            if not task_data:
                return None
                
            # 转换为对象
            return QueueTask.from_dict(json.loads(task_data))
            
        except Exception as e:
            logger.error(f"获取队列任务失败: {str(e)}")
            return None
            
    async def get_pending_tasks(self, count: int = 10) -> List[QueueTask]:
        """获取待处理的任务
        
        Args:
            count: 获取数量
            
        Returns:
            List[QueueTask]: 队列任务列表
        """
        try:
            # 确保已连接
            await self.connect()
            
            # 从优先级队列中获取任务ID
            task_ids = await self._redis.zrange(
                "task_queue",
                0,
                count - 1,
                desc=True  # 按优先级降序
            )
            
            # 获取任务详情
            tasks = []
            for task_id in task_ids:
                task = await self.get_queue_task(task_id)
                if task:
                    tasks.append(task)
                    
            return tasks
            
        except Exception as e:
            logger.error(f"获取待处理任务失败: {str(e)}")
            return []
            
    async def remove_from_queue(self, task_id: str) -> bool:
        """从队列中移除任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否移除成功
        """
        try:
            # 确保已连接
            await self.connect()
            
            # 从优先级队列中移除
            await self._redis.zrem("task_queue", task_id)
            return True
            
        except Exception as e:
            logger.error(f"从队列移除任务失败: {str(e)}")
            return False
            
    async def update_task_status(
        self,
        task_id: str,
        status: int,
        error_message: Optional[str] = None
    ) -> bool:
        """更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            error_message: 错误信息
            
        Returns:
            bool: 是否更新成功
        """
        try:
            # 确保已连接
            await self.connect()
            
            # 获取当前任务
            task = await self.get_task(task_id)
            if not task:
                return False
                
            # 更新状态
            task.status = status
            if error_message:
                task.error_message = error_message
                
            # 更新时间戳
            if status == 1:  # 开始运行
                task.start_time = datetime.now()
            elif status in [2, -1]:  # 完成或失败
                task.stop_time = datetime.now()
                if task.start_time:
                    task.duration = (task.stop_time - task.start_time).total_seconds() / 60
                    
            task.updated_at = datetime.now()
            
            # 保存更新后的任务
            return await self.save_task(task)
            
        except Exception as e:
            logger.error(f"更新任务状态失败: {str(e)}")
            return False 