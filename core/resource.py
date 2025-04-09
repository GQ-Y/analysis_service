"""
资源监控模块
"""
import psutil
import torch
from typing import Dict
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class ResourceMonitor:
    """资源监控"""
    
    def __init__(self):
        self.cpu_threshold = 0.95
        self.memory_threshold = 0.95
        self.gpu_memory_threshold = 0.95
        self.disk_threshold = 0.95
        self.max_tasks = 100  # 最大任务数
        self._running_tasks = 0
        self._waiting_tasks = 0
        
    def get_resource_usage(self) -> Dict:
        """获取资源使用情况"""
        try:
            # CPU使用率
            cpu_percent = sum(psutil.cpu_percent(interval=0.1, percpu=True)) / psutil.cpu_count() / 100
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100
            
            # GPU使用率
            gpu_percent = 0
            gpu_memory_percent = 0
            if torch.cuda.is_available():
                # GPU内存使用率
                gpu_memory = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_percent = gpu_memory / gpu_memory_total
                
                # GPU使用率（如果可用）
                try:
                    gpu_percent = torch.cuda.utilization() / 100
                except:
                    gpu_percent = 0
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent / 100
                
            logger.debug(f"资源使用情况:")
            logger.debug(f"  - CPU: {cpu_percent*100:.1f}%")
            logger.debug(f"  - 内存: {memory_percent*100:.1f}%")
            logger.debug(f"  - GPU: {gpu_percent*100:.1f}%")
            logger.debug(f"  - GPU内存: {gpu_memory_percent*100:.1f}%")
            logger.debug(f"  - 磁盘: {disk_percent*100:.1f}%")
            logger.debug(f"  - 运行中任务: {self._running_tasks}")
            logger.debug(f"  - 等待中任务: {self._waiting_tasks}")
                
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "gpu_percent": gpu_percent,
                "gpu_memory_percent": gpu_memory_percent,
                "disk_percent": disk_percent,
                "running_tasks": self._running_tasks,
                "waiting_tasks": self._waiting_tasks
            }
            
        except Exception as e:
            logger.error(f"获取资源使用情况失败: {str(e)}", exc_info=True)
            return {
                "cpu_percent": 1,
                "memory_percent": 1,
                "gpu_percent": 1,
                "gpu_memory_percent": 1,
                "disk_percent": 1,
                "running_tasks": 0,
                "waiting_tasks": 0
            }
            
    def has_available_resource(self) -> bool:
        """检查是否有可用资源"""
        usage = self.get_resource_usage()
        
        if usage["cpu_percent"] > self.cpu_threshold:
            logger.warning(f"CPU使用率超过阈值: {usage['cpu_percent']*100:.1f}% > {self.cpu_threshold*100}%")
            return False
            
        if usage["memory_percent"] > self.memory_threshold:
            logger.warning(f"内存使用率超过阈值: {usage['memory_percent']*100:.1f}% > {self.memory_threshold*100}%")
            return False
            
        if torch.cuda.is_available():
            if usage["gpu_memory_percent"] > self.gpu_memory_threshold:
                logger.warning(f"GPU内存使用率超过阈值: {usage['gpu_memory_percent']*100:.1f}% > {self.gpu_memory_threshold*100}%")
                return False
                
        if usage["disk_percent"] > self.disk_threshold:
            logger.warning(f"磁盘使用率超过阈值: {usage['disk_percent']*100:.1f}% > {self.disk_threshold*100}%")
            return False
            
        if self._running_tasks + self._waiting_tasks >= self.max_tasks:
            logger.warning(f"任务数超过阈值: {self._running_tasks + self._waiting_tasks} >= {self.max_tasks}")
            return False
            
        logger.info("资源检查通过")
        return True
        
    def increment_running_tasks(self):
        """增加运行中任务数"""
        self._running_tasks += 1
        
    def decrement_running_tasks(self):
        """减少运行中任务数"""
        self._running_tasks = max(0, self._running_tasks - 1)
        
    def increment_waiting_tasks(self):
        """增加等待中任务数"""
        self._waiting_tasks += 1
        
    def decrement_waiting_tasks(self):
        """减少等待中任务数"""
        self._waiting_tasks = max(0, self._waiting_tasks - 1)
        
    def move_task_to_running(self):
        """将一个等待中的任务移动到运行中"""
        if self._waiting_tasks > 0:
            self._waiting_tasks -= 1
            self._running_tasks += 1
