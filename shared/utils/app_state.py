"""
全局应用状态管理器
提供全局服务实例的统一管理和访问
"""
from typing import Optional, Any, Dict
import threading
from shared.utils.logger import get_normal_logger

normal_logger = get_normal_logger(__name__)


class AppStateManager:
    """全局应用状态管理器"""
    
    def __init__(self):
        """初始化应用状态管理器"""
        self._services: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._initialized = False
        
    def initialize(self):
        """初始化应用状态管理器"""
        with self._lock:
            if not self._initialized:
                self._services.clear()
                self._initialized = True
                normal_logger.info("全局应用状态管理器已初始化")
    
    def register_service(self, name: str, service: Any):
        """
        注册服务实例
        
        Args:
            name: 服务名称
            service: 服务实例
        """
        with self._lock:
            if not self._initialized:
                self.initialize()
            self._services[name] = service
            normal_logger.debug(f"注册服务: {name} -> {type(service).__name__}")
    
    def get_service(self, name: str) -> Optional[Any]:
        """
        获取服务实例
        
        Args:
            name: 服务名称
            
        Returns:
            Optional[Any]: 服务实例，如果不存在则返回None
        """
        with self._lock:
            return self._services.get(name)
    
    def unregister_service(self, name: str):
        """
        注销服务实例
        
        Args:
            name: 服务名称
        """
        with self._lock:
            if name in self._services:
                del self._services[name]
                normal_logger.debug(f"注销服务: {name}")
    
    def get_video_service(self):
        """获取视频服务实例"""
        return self.get_service('video_service')
    
    def register_video_service(self, video_service):
        """注册视频服务实例"""
        self.register_service('video_service', video_service)
    
    def get_task_manager(self):
        """获取任务管理器实例"""
        return self.get_service('task_manager')
    
    def register_task_manager(self, task_manager):
        """注册任务管理器实例"""
        self.register_service('task_manager', task_manager)
    
    def get_stream_manager(self):
        """获取流管理器实例"""
        return self.get_service('stream_manager')
    
    def register_stream_manager(self, stream_manager):
        """注册流管理器实例"""
        self.register_service('stream_manager', stream_manager)
    
    def get_stream_task_bridge(self):
        """获取流任务桥接器实例"""
        return self.get_service('stream_task_bridge')
    
    def register_stream_task_bridge(self, stream_task_bridge):
        """注册流任务桥接器实例"""
        self.register_service('stream_task_bridge', stream_task_bridge)
    
    def shutdown(self):
        """关闭应用状态管理器"""
        with self._lock:
            self._services.clear()
            self._initialized = False
            normal_logger.info("全局应用状态管理器已关闭")
    
    def list_services(self) -> Dict[str, str]:
        """
        列出所有已注册的服务
        
        Returns:
            Dict[str, str]: 服务名称到类型名称的映射
        """
        with self._lock:
            return {name: type(service).__name__ for name, service in self._services.items()}


# 全局应用状态管理器实例
app_state_manager = AppStateManager() 