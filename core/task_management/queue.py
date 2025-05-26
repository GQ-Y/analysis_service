"""
任务队列
负责任务的排队和调度
"""
from typing import Dict, Any, List, Optional
import asyncio
import queue
import threading
import time
from datetime import datetime

from core.config import settings
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class TaskQueue:
    """任务队列"""

    def __init__(self, max_size: int = None):
        """
        初始化任务队列

        Args:
            max_size: 队列最大容量，如果为None则使用配置文件中的值
        """
        if max_size is None:
            max_size = settings.TASK_QUEUE_MAX_SIZE

        self.queue = queue.PriorityQueue(maxsize=max_size)
        self.processing = {}
        self.lock = threading.Lock()
        self.max_concurrent = settings.TASK_QUEUE_MAX_CONCURRENT
        self.max_retries = settings.TASK_QUEUE_MAX_RETRIES
        self.retry_delay = settings.TASK_QUEUE_RETRY_DELAY

    async def initialize(self):
        """初始化任务队列"""
        normal_logger.info("任务队列初始化完成")
        return True

    async def shutdown(self):
        """关闭任务队列"""
        normal_logger.info("任务队列已关闭")
        return True

    def put(self, task_id: str, task_data: Dict[str, Any], priority: int = 0) -> bool:
        """
        添加任务到队列

        Args:
            task_id: 任务ID
            task_data: 任务数据
            priority: 优先级，数字越小优先级越高

        Returns:
            bool: 是否添加成功
        """
        try:
            # 检查队列是否已满
            if self.queue.full():
                normal_logger.warning("任务队列已满")
                return False

            # 添加任务到队列
            self.queue.put((priority, task_id, task_data))
            normal_logger.info(f"任务已添加到队列: {task_id}")
            return True

        except Exception as e:
            exception_logger.exception(f"添加任务到队列失败: {str(e)}")
            return False

    def get(self) -> Optional[Dict[str, Any]]:
        """
        从队列获取任务

        Returns:
            Optional[Dict[str, Any]]: 任务数据，如果队列为空则返回None
        """
        try:
            # 检查队列是否为空
            if self.queue.empty():
                return None

            # 获取任务
            priority, task_id, task_data = self.queue.get(block=False)

            # 标记为处理中
            with self.lock:
                self.processing[task_id] = {
                    "data": task_data,
                    "start_time": time.time()
                }

            normal_logger.info(f"从队列获取任务: {task_id}")
            return {
                "task_id": task_id,
                "data": task_data
            }

        except queue.Empty:
            return None
        except Exception as e:
            exception_logger.exception(f"从队列获取任务失败: {str(e)}")
            return None

    def complete(self, task_id: str) -> bool:
        """
        标记任务为已完成

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否标记成功
        """
        try:
            # 检查任务是否在处理中
            with self.lock:
                if task_id not in self.processing:
                    normal_logger.warning(f"任务不在处理中: {task_id}")
                    return False

                # 移除任务
                del self.processing[task_id]

            # 标记队列项为已处理
            self.queue.task_done()

            normal_logger.info(f"任务已完成: {task_id}")
            return True

        except Exception as e:
            exception_logger.exception(f"标记任务为已完成失败: {str(e)}")
            return False

    def fail(self, task_id: str, error: str) -> bool:
        """
        标记任务为失败

        Args:
            task_id: 任务ID
            error: 错误信息

        Returns:
            bool: 是否标记成功
        """
        try:
            # 检查任务是否在处理中
            with self.lock:
                if task_id not in self.processing:
                    normal_logger.warning(f"任务不在处理中: {task_id}")
                    return False

                # 移除任务
                del self.processing[task_id]

            # 标记队列项为已处理
            self.queue.task_done()

            normal_logger.info(f"任务已失败: {task_id}, 错误: {error}")
            return True

        except Exception as e:
            exception_logger.exception(f"标记任务为失败失败: {str(e)}")
            return False

    def size(self) -> int:
        """
        获取队列大小

        Returns:
            int: 队列大小
        """
        return self.queue.qsize()

    def processing_count(self) -> int:
        """
        获取处理中的任务数量

        Returns:
            int: 处理中的任务数量
        """
        with self.lock:
            return len(self.processing)

    def is_empty(self) -> bool:
        """
        检查队列是否为空

        Returns:
            bool: 队列是否为空
        """
        return self.queue.empty()

    def is_full(self) -> bool:
        """
        检查队列是否已满

        Returns:
            bool: 队列是否已满
        """
        return self.queue.full()

    def get_processing_tasks(self) -> List[Dict[str, Any]]:
        """
        获取处理中的任务列表

        Returns:
            List[Dict[str, Any]]: 处理中的任务列表
        """
        with self.lock:
            return [
                {
                    "task_id": task_id,
                    "data": task_data["data"],
                    "start_time": task_data["start_time"],
                    "duration": time.time() - task_data["start_time"]
                }
                for task_id, task_data in self.processing.items()
            ]
