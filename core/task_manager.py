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
from enum import Enum
import asyncio
import threading
from core.config import settings
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"       # 等待处理
    PROCESSING = "processing" # 处理中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"        # 失败
    CANCELLED = "cancelled"  # 已取消
    STOPPING = "stopping"    # 停止中
    STOPPED = "stopped"      # 已停止

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
        
        # 启动清理线程
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"任务管理器初始化完成，最大任务数: {self.max_tasks}")
        
    def _cleanup_loop(self):
        """清理过期任务的循环"""
        while True:
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
                "status": TaskStatus.PENDING,
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
            
            logger.info(f"更新任务状态: {task_id} -> {status}")
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

    def create_task(self, task_type: str, params: Dict[str, Any], protocol: str = "mqtt") -> str:
        """
        创建新任务
        
        参数:
            task_type: 任务类型 (image, video, stream)
            params: 任务参数
            protocol: 使用的协议 (http, mqtt)
            
        返回:
            str: 任务ID
        """
        task_id = str(uuid.uuid4())
        
        with self.task_lock:
            self.tasks[task_id] = {
                "id": task_id,
                "type": task_type,
                "params": params,
                "status": "pending",
                "protocol": protocol,
                "create_time": int(time.time()),
                "progress": 0
            }
            
        logger.info(f"已创建任务: {task_id}, 类型: {task_type}, 协议: {protocol}")
        return task_id
        
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务信息
        
        参数:
            task_id: 任务ID
            
        返回:
            Dict | None: 任务信息字典，如果任务不存在则返回None
        """
        with self.task_lock:
            task = self.tasks.get(task_id)
            
        if task:
            # 返回任务的副本，避免直接修改
            return dict(task)
        
        logger.warning(f"未找到任务: {task_id}")
        return None
        
    def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新任务信息
        
        参数:
            task_id: 任务ID
            updates: 要更新的字段和值
            
        返回:
            bool: 更新是否成功
        """
        with self.task_lock:
            if task_id not in self.tasks:
                logger.warning(f"尝试更新不存在的任务: {task_id}")
                return False
                
            # 更新任务信息
            for key, value in updates.items():
                self.tasks[task_id][key] = value
                
            # 如果状态有变化，添加时间戳
            if "status" in updates:
                status = updates["status"]
                self.tasks[task_id][f"{status}_time"] = int(time.time())
                
        logger.debug(f"已更新任务: {task_id}, 更新: {updates}")
        return True
        
    def delete_task(self, task_id: str) -> bool:
        """
        删除任务
        
        参数:
            task_id: 任务ID
            
        返回:
            bool: 删除是否成功
        """
        with self.task_lock:
            if task_id not in self.tasks:
                logger.warning(f"尝试删除不存在的任务: {task_id}")
                return False
                
            del self.tasks[task_id]
            
        logger.info(f"已删除任务: {task_id}")
        return True
        
    def get_all_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取所有任务列表
        
        参数:
            status: 可选的状态过滤器
            
        返回:
            List[Dict]: 任务信息列表
        """
        with self.task_lock:
            if status:
                tasks = [dict(task) for task in self.tasks.values() if task.get("status") == status]
            else:
                tasks = [dict(task) for task in self.tasks.values()]
                
        return tasks
        
    def get_tasks_by_protocol(self, protocol: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        按协议获取任务
        
        参数:
            protocol: 协议 (http, mqtt)
            status: 可选的状态过滤器
            
        返回:
            List[Dict]: 任务信息列表
        """
        with self.task_lock:
            if status:
                tasks = [
                    dict(task) for task in self.tasks.values() 
                    if task.get("protocol") == protocol and task.get("status") == status
                ]
            else:
                tasks = [
                    dict(task) for task in self.tasks.values() 
                    if task.get("protocol") == protocol
                ]
                
        return tasks
        
    def get_task_output_path(self, task_id: str, filename: str) -> str:
        """
        获取任务输出文件路径
        
        参数:
            task_id: 任务ID
            filename: 文件名
            
        返回:
            str: 输出文件的完整路径
        """
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"任务不存在: {task_id}")
            
        task_type = task.get("type")
        
        if task_type == "image":
            return os.path.join(self.output_dir, "images", filename)
        elif task_type == "video" or task_type == "stream":
            return os.path.join(self.output_dir, "videos", filename)
        else:
            return os.path.join(self.output_dir, filename)
            
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """
        清理旧任务
        
        参数:
            max_age_hours: 任务的最大保留时间（小时）
        """
        current_time = int(time.time())
        max_age_seconds = max_age_hours * 3600
        
        with self.task_lock:
            tasks_to_delete = []
            
            for task_id, task in self.tasks.items():
                create_time = task.get("create_time", 0)
                
                # 如果任务已经完成/失败且时间超过最大保留时间
                if (task.get("status") in ["completed", "failed", "stopped"] and
                    current_time - create_time > max_age_seconds):
                    tasks_to_delete.append(task_id)
                    
            # 删除旧任务
            for task_id in tasks_to_delete:
                del self.tasks[task_id]
                
        if tasks_to_delete:
            logger.info(f"已清理 {len(tasks_to_delete)} 个旧任务") 