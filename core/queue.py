"""
任务队列管理
"""
import asyncio
from typing import Optional, Tuple, Dict
from datetime import datetime, timedelta
import uuid
from analysis_service.models.task import TaskBase, QueueTask
from analysis_service.core.resource import ResourceMonitor
from analysis_service.core.detector import YOLODetector
from shared.utils.logger import setup_logger
from analysis_service.crud.task import TaskCRUD
from analysis_service.models.analysis_type import AnalysisType
from analysis_service.services.task_store import TaskStore
from analysis_service.models.task_status import TaskStatus

logger = setup_logger(__name__)

class RetryPolicy:
    """重试策略"""
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # 秒
    
    @classmethod
    def should_retry(cls, task: QueueTask) -> bool:
        """判断是否应该重试"""
        return (
            task.status == -1 and  # 失败状态
            (task.retry_count or 0) < cls.MAX_RETRIES
        )
        
    @classmethod
    async def wait_before_retry(cls, retry_count: int):
        """重试前等待"""
        delay = cls.RETRY_DELAY * (2 ** retry_count)  # 指数退避
        await asyncio.sleep(delay)

class TaskQueueManager:
    """任务队列管理器"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """初始化队列管理器
        
        Args:
            redis_url: Redis连接URL
        """
        self.task_store = TaskStore(redis_url)
        self.task_crud = TaskCRUD(self.task_store)
        self.detector = YOLODetector()
        self.resource_monitor = ResourceMonitor()
        self.running_tasks = {}  # 存储正在运行的任务
        self.max_concurrent = 3  # 最大并发数
        self.is_running = False  # 运行状态标志
        
    async def start(self):
        """启动队列管理器"""
        if not self.is_running:
            self.is_running = True
            # 连接Redis
            await self.task_store.connect()
            # 检查并启动之前未完成的任务
            await self._recover_tasks()
            logger.info("任务队列管理器已启动")
            
    async def stop(self):
        """停止队列管理器"""
        self.is_running = False
        # 停止所有运行中的任务
        for task_id in list(self.running_tasks.keys()):
            await self.cancel_task(task_id)
        # 断开Redis连接
        await self.task_store.disconnect()
        logger.info("任务队列管理器已停止")
        
    async def _recover_tasks(self):
        """恢复之前未完成的任务"""
        try:
            # 获取所有待处理的任务
            pending_tasks = await self.task_store.get_pending_tasks()
            
            # 启动等待中的任务(考虑并发限制)
            for task in pending_tasks:
                if len(self.running_tasks) < self.max_concurrent:
                    asyncio.create_task(self._process_task(task.id))
                    
        except Exception as e:
            logger.error(f"任务恢复失败: {str(e)}")
            
    async def add_task(
        self, 
        task: TaskBase,
        parent_task_id: str = None,
        analyze_interval: int = None,
        alarm_interval: int = None,
        random_interval: Tuple[int, int] = None,
        confidence_threshold: float = None,
        push_interval: int = None
    ) -> Optional[QueueTask]:
        """添加并立即执行任务"""
        try:
            # 检查资源
            if not self.resource_monitor.has_available_resource():
                raise Exception("没有可用资源")
                
            # 创建队列任务
            queue_task = await self.task_crud.create_queue_task(
                task=task,
                parent_task_id=parent_task_id
            )
            if not queue_task:
                raise Exception("创建队列任务失败")
            
            # 检查当前运行任务数
            if len(self.running_tasks) >= self.max_concurrent:
                logger.warning("已达到最大并发数，等待中...")
                return queue_task
                
            # 立即启动任务
            if self.is_running:
                asyncio.create_task(self._process_task(
                    queue_task.id,
                    analyze_interval=analyze_interval,
                    alarm_interval=alarm_interval,
                    random_interval=random_interval,
                    confidence_threshold=confidence_threshold,
                    push_interval=push_interval
                ))
            
            return queue_task
            
        except Exception as e:
            logger.error(f"添加任务失败: {str(e)}")
            return None
        
    async def _process_task(
        self,
        queue_task_id: str,
        analyze_interval: int = None,
        alarm_interval: int = None,
        random_interval: Tuple[int, int] = None,
        confidence_threshold: float = None,
        push_interval: int = None
    ):
        """处理单个任务"""
        try:
            # 获取队列任务
            queue_task = await self.task_store.get_queue_task(queue_task_id)
            if not queue_task:
                logger.error(f"找不到队列任务: {queue_task_id}")
                return
                
            # 获取关联的主任务
            task = await self.task_store.get_task(queue_task.task_id)
            if not task:
                logger.error(f"找不到关联的任务记录: {queue_task.task_id}")
                return
            
            # 更新状态
            queue_task.status = 1
            queue_task.started_at = datetime.now()
            await self.task_store.save_queue_task(queue_task)
            await self.task_crud.update_task_status(task.id, 1)
            
            # 记录到运行中的任务
            self.running_tasks[queue_task_id] = task.id
            
            # 构建配置字典
            config = task.config or {}
            if confidence_threshold is not None:
                config['confidence_threshold'] = confidence_threshold
            
            # 执行任务
            await self.detector.start_stream_analysis(
                task_id=queue_task_id,
                stream_url=task.stream_url,
                model_code=task.model_code,
                callback_urls=task.callback_urls,
                analyze_interval=analyze_interval,
                alarm_interval=alarm_interval,
                random_interval=random_interval,
                config=config,
                push_interval=push_interval
            )
            
            # 更新状态为完成
            queue_task.status = 2
            queue_task.completed_at = datetime.now()
            await self.task_store.save_queue_task(queue_task)
            await self.task_crud.update_task_status(task.id, 2)
            
            # 从运行中任务移除
            self.running_tasks.pop(queue_task_id, None)
            
            # 从队列中移除
            await self.task_store.remove_from_queue(queue_task_id)
            
        except Exception as e:
            logger.error(f"处理任务失败: {str(e)}")
            # 更新状态为失败
            if queue_task:
                queue_task.status = -1
                queue_task.error_message = str(e)
                await self.task_store.save_queue_task(queue_task)
            if task:
                await self.task_crud.update_task_status(task.id, -1, str(e))
            # 从运行中任务移除
            self.running_tasks.pop(queue_task_id, None)
            
    async def update_priority(self, queue_task_id: str, priority: int) -> bool:
        """更新任务优先级"""
        try:
            # 获取任务
            task = await self.task_store.get_queue_task(queue_task_id)
            if not task:
                return False
                
            # 更新优先级
            task.priority = priority
            return await self.task_store.save_queue_task(task)
            
        except Exception as e:
            logger.error(f"更新任务优先级失败: {str(e)}")
            return False
            
    async def get_task_status(self, queue_task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        try:
            # 获取队列任务
            queue_task = await self.task_store.get_queue_task(queue_task_id)
            if not queue_task:
                return None
                
            # 获取关联的主任务
            task = await self.task_store.get_task(queue_task.task_id)
            if not task:
                return None
                
            return {
                "queue_task": queue_task.to_dict(),
                "task": task.model_dump()
            }
            
        except Exception as e:
            logger.error(f"获取任务状态失败: {str(e)}")
            return None
            
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        try:
            # 停止检测器
            await self.detector.stop_stream_analysis(task_id)
            
            # 更新状态
            await self.task_crud.update_task_status(task_id, -1, "任务已取消")
            
            # 从运行中任务移除
            self.running_tasks.pop(task_id, None)
            
            return True
            
        except Exception as e:
            logger.error(f"取消任务失败: {str(e)}")
            return False
            
    async def is_task_running(self, task_id: str) -> bool:
        """检查任务是否正在运行"""
        return task_id in self.running_tasks
        
    async def stop_task(self, task_id: str) -> bool:
        """停止任务"""
        try:
            # 更新状态为停止中
            await self.task_crud.update_task_status(task_id, TaskStatus.STOPPING, "任务正在停止")
            
            # 停止检测器
            await self.detector.stop_stream_analysis(task_id)
            
            # 从运行中任务移除
            self.running_tasks.pop(task_id, None)
            
            return True
            
        except Exception as e:
            logger.error(f"停止任务失败: {str(e)}")
            return False
        
    async def create_stream_task(
        self,
        task_id: str,
        model_code: str,
        stream_url: str,
        analysis_type: AnalysisType,
        callback_urls: Optional[str] = None,
        config: Optional[Dict] = None,
        task_name: Optional[str] = None,
        enable_callback: bool = True,
        save_result: bool = False
    ) -> Optional[TaskBase]:
        """创建流分析任务"""
        try:
            # 创建主任务
            task = await self.task_crud.create_task(
                task_id=task_id,
                model_code=model_code,
                stream_url=stream_url,
                callback_urls=callback_urls,
                task_name=task_name,
                analysis_type=analysis_type.value,
                config=config,
                enable_callback=enable_callback,
                save_result=save_result
            )
            
            if not task:
                raise Exception("创建任务失败")
                
            # 添加到队列
            queue_task = await self.add_task(task)
            if not queue_task:
                raise Exception("添加到队列失败")
                
            return task
            
        except Exception as e:
            logger.error(f"创建流分析任务失败: {str(e)}")
            return None
