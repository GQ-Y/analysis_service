"""
任务管理器
管理分析任务的生命周期和状态
"""
import os
import time
import uuid
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
import threading
from enum import Enum
from core.config import settings
from shared.utils.logger import setup_logger
from .processor import TaskProcessor

from .utils.status import TaskStatus

logger = setup_logger(__name__)

class TaskManager:
    """任务管理器"""

    def __init__(self, max_tasks: int = None):
        """初始化任务管理器

        Args:
            max_tasks: 最大任务数，默认从配置中读取
        """
        # 设置最大任务数
        self.max_tasks = max_tasks or settings.TASK_QUEUE_MAX_CONCURRENT

        # 初始化任务字典
        self.tasks: Dict[str, Dict[str, Any]] = {}

        # 设置输出目录
        self.output_dir = settings.OUTPUT.save_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.task_timeout = settings.TASK_QUEUE_RESULT_TTL
        self.cleanup_interval = settings.TASK_QUEUE_CLEANUP_INTERVAL
        self.last_cleanup = time.time()

        # 初始化任务锁
        self.task_lock = threading.Lock()

        # 创建并持有 TaskProcessor 实例
        self.processor = TaskProcessor(task_manager=self)

        # 清理线程
        self.cleanup_thread = None
        self.running = False

        logger.info(f"任务管理器初始化完成，最大任务数: {self.max_tasks}")

    async def initialize(self):
        """初始化任务管理器"""
        try:
            # 初始化处理器
            await self.processor.initialize()

            # 启动清理线程
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()

            logger.info("任务管理器初始化成功")
            return True
        except Exception as e:
            logger.error(f"任务管理器初始化失败: {str(e)}")
            return False

    async def shutdown(self):
        """关闭任务管理器"""
        try:
            # 停止清理线程
            self.running = False
            if self.cleanup_thread and self.cleanup_thread.is_alive():
                self.cleanup_thread.join(timeout=2.0)

            # 停止所有任务
            for task_id in list(self.tasks.keys()):
                await self.stop_task(task_id)

            # 关闭处理器
            await self.processor.shutdown()

            logger.info("任务管理器已关闭")
            return True
        except Exception as e:
            logger.error(f"任务管理器关闭失败: {str(e)}")
            return False

    def _cleanup_loop(self):
        """清理过期任务的循环"""
        while self.running:
            try:
                self.cleanup_tasks()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"清理任务失败: {str(e)}")
                time.sleep(60)  # 出错时等待1分钟再重试

    def cleanup_tasks(self):
        """清理过期任务"""
        try:
            current_time = time.time()

            # 检查是否需要清理
            if current_time - self.last_cleanup < self.cleanup_interval:
                return

            expired_tasks = []
            for task_id, task in self.tasks.items():
                # 检查任务是否过期
                if current_time - task["created_at"] > self.task_timeout:
                    expired_tasks.append(task_id)

            # 删除过期任务
            for task_id in expired_tasks:
                del self.tasks[task_id]
                logger.info(f"清理过期任务: {task_id}")

            self.last_cleanup = current_time

        except Exception as e:
            logger.error(f"清理任务失败: {str(e)}")

    def add_task(self, task_id: str, task_data: Dict[str, Any]) -> bool:
        """
        添加新任务

        Args:
            task_id: 任务ID
            task_data: 任务数据

        Returns:
            bool: 是否添加成功
        """
        try:
            if len(self.tasks) >= self.max_tasks:
                logger.warning(f"达到最大任务数限制 ({self.max_tasks})")
                return False

            if task_id in self.tasks:
                logger.warning(f"任务ID已存在: {task_id}")
                return False

            # 添加任务
            self.tasks[task_id] = {
                "id": task_id,
                "data": task_data,
                "status": TaskStatus.WAITING,
                "created_at": time.time(),
                "updated_at": time.time(),
                "result": None,
                "error": None
            }

            logger.info(f"添加新任务: {task_id}")
            return True

        except Exception as e:
            logger.error(f"添加任务失败: {str(e)}")
            return False

    def update_task_status(self, task_id: str, status: TaskStatus, result: Dict = None, error: str = None) -> bool:
        """
        更新任务状态

        Args:
            task_id: 任务ID
            status: 新状态
            result: 任务结果
            error: 错误信息

        Returns:
            bool: 是否更新成功
        """
        try:
            if task_id not in self.tasks:
                logger.warning(f"任务不存在: {task_id}")
                return False

            self.tasks[task_id].update({
                "status": status,
                "updated_at": time.time(),
                "result": result,
                "error": error
            })

            # logger.info(f"更新任务状态: {task_id} -> {status}")
            return True

        except Exception as e:
            logger.error(f"更新任务状态失败: {str(e)}")
            return False

    def get_task_status(self, task_id: str) -> Dict:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            Dict: 任务状态信息
        """
        try:
            if task_id not in self.tasks:
                logger.warning(f"任务不存在: {task_id}")
                return None

            return self.tasks[task_id]

        except Exception as e:
            logger.error(f"获取任务状态失败: {str(e)}")
            return None

    def get_all_tasks(self) -> List[Dict]:
        """
        获取所有任务

        Returns:
            List[Dict]: 任务列表
        """
        return list(self.tasks.values())

    def remove_task(self, task_id: str) -> bool:
        """
        移除任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否移除成功
        """
        try:
            if task_id not in self.tasks:
                logger.warning(f"任务不存在: {task_id}")
                return False

            del self.tasks[task_id]
            logger.info(f"移除任务: {task_id}")
            return True

        except Exception as e:
            logger.error(f"移除任务失败: {str(e)}")
            return False

    def clear_tasks(self):
        """清空所有任务"""
        try:
            self.tasks.clear()
            logger.info("清空所有任务")
        except Exception as e:
            logger.error(f"清空任务失败: {str(e)}")

    def get_task_count(self) -> int:
        """
        获取当前任务数量

        Returns:
            int: 任务数量
        """
        return len(self.tasks)

    def get_active_tasks(self) -> List[Dict]:
        """
        获取活跃任务列表

        Returns:
            List[Dict]: 活跃任务列表
        """
        return [task for task in self.tasks.values() if task["status"] == TaskStatus.PROCESSING]

    def get_completed_tasks(self) -> List[Dict]:
        """
        获取已完成任务列表

        Returns:
            List[Dict]: 已完成任务列表
        """
        return [task for task in self.tasks.values() if task["status"] == TaskStatus.COMPLETED]

    def get_failed_tasks(self) -> List[Dict]:
        """
        获取失败任务列表

        Returns:
            List[Dict]: 失败任务列表
        """
        return [task for task in self.tasks.values() if task["status"] == TaskStatus.FAILED]

    def create_task(self, task_type: str, params: Dict[str, Any]) -> str:
        """
        创建新任务

        Args:
            task_type: 任务类型
            params: 任务参数

        Returns:
            str: 任务ID
        """
        task_id = str(uuid.uuid4())
        task_data = {
            "id": task_id,
            "type": task_type,
            "params": params,
            "created_at": time.time()
        }

        if self.add_task(task_id, task_data):
            logger.info(f"创建任务成功: {task_id}, 类型: {task_type}")
            return task_id
        else:
            logger.error(f"创建任务失败: {task_id}, 类型: {task_type}")
            return None

    def has_task(self, task_id: str) -> bool:
        """
        检查任务是否存在

        Args:
            task_id: 任务ID

        Returns:
            bool: 任务是否存在
        """
        return task_id in self.tasks

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务信息

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 任务信息
        """
        try:
            if task_id not in self.tasks:
                logger.warning(f"任务不存在: {task_id}")
                return None

            return self.tasks[task_id]

        except Exception as e:
            logger.error(f"获取任务信息失败: {str(e)}")
            return None

    def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新任务信息

        Args:
            task_id: 任务ID
            updates: 更新内容

        Returns:
            bool: 是否更新成功
        """
        try:
            if task_id not in self.tasks:
                logger.warning(f"任务不存在: {task_id}")
                return False

            # 更新任务信息
            self.tasks[task_id]["data"].update(updates)
            self.tasks[task_id]["updated_at"] = time.time()

            logger.info(f"更新任务信息成功: {task_id}")
            return True

        except Exception as e:
            logger.error(f"更新任务信息失败: {str(e)}")
            return False

    def delete_task(self, task_id: str) -> bool:
        """
        删除任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否删除成功
        """
        try:
            if task_id not in self.tasks:
                logger.warning(f"任务不存在: {task_id}")
                return False

            # 删除任务
            del self.tasks[task_id]

            logger.info(f"删除任务成功: {task_id}")
            return True

        except Exception as e:
            logger.error(f"删除任务失败: {str(e)}")
            return False

    def get_all_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取所有任务

        Args:
            status: 任务状态过滤

        Returns:
            List[Dict[str, Any]]: 任务列表
        """
        try:
            tasks = list(self.tasks.values())

            # 按状态过滤
            if status:
                tasks = [task for task in tasks if task["status"] == status]

            return tasks

        except Exception as e:
            logger.error(f"获取任务列表失败: {str(e)}")
            return []

    def get_task_count(self) -> int:
        """
        获取任务总数

        Returns:
            int: 任务总数
        """
        with self.task_lock:
            return len(self.tasks)

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        获取活动任务

        Returns:
            List[Dict[str, Any]]: 活动任务列表
        """
        try:
            with self.task_lock:
                return [
                    task for task in self.tasks.values()
                    if task.get("status") in [TaskStatus.WAITING, TaskStatus.PROCESSING]
                ]
        except Exception as e:
            logger.error(f"获取活动任务失败: {str(e)}")
            return []

    def get_completed_tasks(self) -> List[Dict[str, Any]]:
        """
        获取已完成任务

        Returns:
            List[Dict[str, Any]]: 已完成任务列表
        """
        try:
            with self.task_lock:
                return [
                    task for task in self.tasks.values()
                    if task.get("status") == TaskStatus.COMPLETED
                ]
        except Exception as e:
            logger.error(f"获取已完成任务失败: {str(e)}")
            return []

    def get_failed_tasks(self) -> List[Dict[str, Any]]:
        """
        获取失败任务

        Returns:
            List[Dict[str, Any]]: 失败任务列表
        """
        try:
            with self.task_lock:
                return [
                    task for task in self.tasks.values()
                    if task.get("status") in [TaskStatus.FAILED, TaskStatus.STOPPED]
                ]
        except Exception as e:
            logger.error(f"获取失败任务失败: {str(e)}")
            return []

    def get_task_output_path(self, task_id: str, filename: str) -> str:
        """
        获取任务输出文件路径

        Args:
            task_id: 任务ID
            filename: 文件名

        Returns:
            str: 文件路径
        """
        try:
            if task_id not in self.tasks:
                logger.warning(f"任务不存在: {task_id}")
                return None

            # 创建任务输出目录
            task_output_dir = os.path.join(self.output_dir, task_id)
            os.makedirs(task_output_dir, exist_ok=True)

            # 返回文件路径
            return os.path.join(task_output_dir, filename)

        except Exception as e:
            logger.error(f"获取任务输出路径失败: {str(e)}")
            return None

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """
        清理旧任务

        Args:
            max_age_hours: 最大保留时间(小时)
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            expired_tasks = []
            for task_id, task in self.tasks.items():
                # 检查任务是否过期
                if current_time - task["created_at"] > max_age_seconds:
                    expired_tasks.append(task_id)

            # 删除过期任务
            for task_id in expired_tasks:
                del self.tasks[task_id]
                logger.info(f"清理过期任务: {task_id}")

            logger.info(f"清理完成，已删除 {len(expired_tasks)} 个过期任务")

        except Exception as e:
            logger.error(f"清理旧任务失败: {str(e)}")

    async def start_task(self, task_id: str) -> bool:
        """
        启动任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否启动成功
        """
        try:
            # 检查任务是否存在
            if task_id not in self.tasks:
                logger.warning(f"任务不存在: {task_id}")
                return False

            # 获取任务配置
            task_config = self.tasks[task_id]["data"]["params"]

            # 尝试启动流任务
            result = await self.processor.start_stream_analysis(task_id, task_config)

            if result:
                logger.info(f"启动任务成功: {task_id}")
                return True
            else:
                logger.error(f"启动任务失败: {task_id}")
                return False

        except Exception as e:
            logger.error(f"启动任务异常: {str(e)}")
            return False

    async def start_stream_task(self, task_id: str, task_config: Dict[str, Any]) -> bool:
        """
        启动流任务

        Args:
            task_id: 任务ID
            task_config: 任务配置

        Returns:
            bool: 是否启动成功
        """
        try:
            # 检查任务是否存在
            if task_id not in self.tasks:
                logger.warning(f"任务不存在: {task_id}")
                return False

            # 尝试启动流任务
            result = await self.processor.start_stream_analysis(task_id, task_config)

            if result:
                logger.info(f"启动流任务成功: {task_id}")
                return True
            else:
                logger.error(f"启动流任务失败: {task_id}")
                return False

        except Exception as e:
            logger.error(f"启动流任务异常: {str(e)}")
            return False

    async def stop_task(self, task_id: str) -> bool:
        """
        停止任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否停止成功
        """
        try:
            # 检查任务是否存在
            if task_id not in self.tasks:
                logger.warning(f"任务不存在: {task_id}")
                return False

            # 尝试停止任务
            result = await self.processor.stop_task(task_id)

            if result:
                logger.info(f"停止任务成功: {task_id}")
                return True
            else:
                logger.error(f"停止任务失败: {task_id}")
                return False

        except Exception as e:
            logger.error(f"停止任务异常: {str(e)}")
            return False

    async def pause_task(self, task_id: str, reason: str = None) -> bool:
        """暂停任务
        
        Args:
            task_id: 任务ID
            reason: 暂停原因
            
        Returns:
            bool: 是否成功暂停
        """
        try:
            if task_id not in self.tasks:
                logger.warning(f"任务不存在: {task_id}")
                return False
                
            if self.tasks[task_id]["status"] != TaskStatus.PROCESSING:
                logger.warning(f"只能暂停处理中的任务: {task_id}, 当前状态: {self.tasks[task_id]['status']}")
                return False
                
            with self.task_lock:
                self.tasks[task_id].update({
                    "status": TaskStatus.PAUSED,
                    "updated_at": time.time(),
                    "pause_reason": reason
                })
                
            # 通知任务处理器
            await self.processor.pause_task(task_id)
            
            logger.info(f"任务已暂停: {task_id}, 原因: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"暂停任务失败: {str(e)}")
            return False
            
    async def resume_task(self, task_id: str) -> bool:
        """恢复暂停的任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否成功恢复
        """
        try:
            if task_id not in self.tasks:
                logger.warning(f"任务不存在: {task_id}")
                return False
                
            if self.tasks[task_id]["status"] != TaskStatus.PAUSED:
                logger.warning(f"只能恢复暂停的任务: {task_id}, 当前状态: {self.tasks[task_id]['status']}")
                return False
                
            with self.task_lock:
                self.tasks[task_id].update({
                    "status": TaskStatus.PROCESSING,
                    "updated_at": time.time(),
                    "pause_reason": None
                })
                
            # 通知任务处理器
            await self.processor.resume_task(task_id)
            
            logger.info(f"任务已恢复: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"恢复任务失败: {str(e)}")
            return False