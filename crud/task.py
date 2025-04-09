"""
任务CRUD操作
"""
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from analysis_service.models.task import TaskBase, QueueTask
from analysis_service.services.task_store import TaskStore
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class TaskCRUD:
    """任务CRUD操作类"""
    
    def __init__(self, task_store: TaskStore):
        """初始化
        
        Args:
            task_store: 任务存储服务
        """
        self.store = task_store
        
    async def create_task(
        self,
        task_id: Optional[str] = None,
        model_code: str = None,
        stream_url: str = None,
        callback_urls: Optional[str] = None,
        output_url: Optional[str] = None,
        task_name: Optional[str] = None,
        analysis_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enable_callback: bool = False,
        save_result: bool = False
    ) -> Optional[TaskBase]:
        """创建任务
        
        Args:
            task_id: 任务ID，不指定则自动生成
            model_code: 模型代码
            stream_url: 流URL
            callback_urls: 回调地址
            output_url: 输出URL
            task_name: 任务名称
            analysis_type: 分析类型
            config: 分析配置
            enable_callback: 是否启用回调
            save_result: 是否保存结果
            
        Returns:
            Optional[TaskBase]: 创建的任务对象
        """
        try:
            # 生成任务ID
            if not task_id:
                task_id = str(uuid.uuid4())
                
            # 创建任务对象
            task = TaskBase(
                id=task_id,
                task_name=task_name,
                model_code=model_code,
                stream_url=stream_url,
                callback_urls=callback_urls,
                output_url=output_url,
                analysis_type=analysis_type,
                config=config,
                enable_callback=enable_callback,
                save_result=save_result
            )
            
            # 保存任务
            if await self.store.save_task(task):
                return task
            return None
            
        except Exception as e:
            logger.error(f"创建任务失败: {str(e)}")
            return None
            
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
        return await self.store.update_task_status(task_id, status, error_message)
        
    async def get_task(self, task_id: str) -> Optional[TaskBase]:
        """获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[TaskBase]: 任务对象
        """
        return await self.store.get_task(task_id)
        
    async def create_queue_task(
        self,
        task: TaskBase,
        parent_task_id: Optional[str] = None,
        priority: int = 0
    ) -> Optional[QueueTask]:
        """创建队列任务
        
        Args:
            task: 关联的任务对象
            parent_task_id: 父任务ID
            priority: 优先级
            
        Returns:
            Optional[QueueTask]: 创建的队列任务对象
        """
        try:
            # 创建队列任务对象
            queue_task = QueueTask(
                id=str(uuid.uuid4()),
                task_id=task.id,
                parent_task_id=parent_task_id,
                priority=priority
            )
            
            # 保存队列任务
            if await self.store.save_queue_task(queue_task):
                return queue_task
            return None
            
        except Exception as e:
            logger.error(f"创建队列任务失败: {str(e)}")
            return None 