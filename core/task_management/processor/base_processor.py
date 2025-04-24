"""
任务处理器基类
定义任务处理的基本接口和通用方法
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from core.task_management.utils.status import TaskStatus
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseTaskProcessor(ABC):
    """任务处理器基类"""
    
    def __init__(self):
        """初始化基类"""
        self.active_tasks = {}
        
    @abstractmethod
    async def start_task(self, task_id: str, task_config: Dict[str, Any]) -> bool:
        """启动任务"""
        pass
        
    @abstractmethod
    async def stop_task(self, task_id: str) -> Dict[str, Any]:
        """停止任务"""
        pass
        
    @abstractmethod
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        pass
        
    def is_task_active(self, task_id: str) -> bool:
        """检查任务是否处于活动状态"""
        return task_id in self.active_tasks
        
    def remove_active_task(self, task_id: str):
        """从活动任务列表中移除任务"""
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
            logger.info(f"任务已从活动列表中移除: {task_id}")
            
    def add_active_task(self, task_id: str, task_info: Dict[str, Any]):
        """添加任务到活动列表"""
        self.active_tasks[task_id] = task_info
        logger.info(f"任务已添加到活动列表: {task_id}") 