"""
任务管理器
负责创建、获取和管理分析任务
"""
import os
import uuid
import time
from typing import Dict, List, Any, Optional
from threading import Lock

from shared.utils.logger import setup_logger
from core.config import settings

logger = setup_logger(__name__)

class TaskManager:
    """任务管理器单例类"""
    
    _instance = None
    _lock = Lock()
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = TaskManager()
        return cls._instance
    
    def __init__(self):
        """初始化任务管理器"""
        # 确保不直接实例化
        if TaskManager._instance is not None:
            raise RuntimeError("请使用 get_instance() 方法获取 TaskManager 实例")
            
        self.tasks = {}  # 任务字典 {task_id: task_info}
        self.task_lock = Lock()  # 用于保护任务字典的锁
        
        # 确保输出目录存在
        self.output_dir = settings.OUTPUT.save_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)
        os.makedirs(f"{self.output_dir}/videos", exist_ok=True)
        
        logger.info("任务管理器已初始化")
        
    def create_task(self, task_type: str, params: Dict[str, Any], protocol: str = "http") -> str:
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