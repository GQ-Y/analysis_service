"""
事件系统模块
负责管理事件订阅和触发
"""

import asyncio
import threading
from typing import Dict, Any, Callable, Set, List, Optional, Union
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class EventSystem:
    """事件系统，负责处理事件的订阅和发布"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EventSystem, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化事件系统"""
        if self._initialized:
            return
            
        # 同步事件处理器映射，格式: 事件名 -> {处理器1, 处理器2, ...}
        self._sync_handlers: Dict[str, Set[Callable]] = {}
        
        # 异步事件处理器映射，格式: 事件名 -> {处理器1, 处理器2, ...}
        self._async_handlers: Dict[str, Set[Callable]] = {}
        
        # 锁，用于保护处理器映射
        self._handlers_lock = threading.Lock()
        
        self._initialized = True
        
        normal_logger.debug("事件系统初始化完成")
    
    def register_sync_handler(self, event_name: str, handler: Callable) -> None:
        """注册同步事件处理器
        
        Args:
            event_name: 事件名
            handler: 处理函数，格式: handler(event_data)
        """
        with self._handlers_lock:
            if event_name not in self._sync_handlers:
                self._sync_handlers[event_name] = set()
            
            self._sync_handlers[event_name].add(handler)
            normal_logger.debug(f"注册同步事件处理器: {event_name}")
    
    def register_async_handler(self, event_name: str, handler: Callable) -> None:
        """注册异步事件处理器
        
        Args:
            event_name: 事件名
            handler: 异步处理函数，格式: async handler(event_data)
        """
        with self._handlers_lock:
            if event_name not in self._async_handlers:
                self._async_handlers[event_name] = set()
            
            self._async_handlers[event_name].add(handler)
            normal_logger.debug(f"注册异步事件处理器: {event_name}")
    
    def unregister_sync_handler(self, event_name: str, handler: Callable) -> bool:
        """取消注册同步事件处理器
        
        Args:
            event_name: 事件名
            handler: 处理函数
            
        Returns:
            bool: 是否成功取消注册
        """
        with self._handlers_lock:
            if event_name in self._sync_handlers and handler in self._sync_handlers[event_name]:
                self._sync_handlers[event_name].remove(handler)
                normal_logger.debug(f"取消注册同步事件处理器: {event_name}")
                return True
            
            return False
    
    def unregister_async_handler(self, event_name: str, handler: Callable) -> bool:
        """取消注册异步事件处理器
        
        Args:
            event_name: 事件名
            handler: 处理函数
            
        Returns:
            bool: 是否成功取消注册
        """
        with self._handlers_lock:
            if event_name in self._async_handlers and handler in self._async_handlers[event_name]:
                self._async_handlers[event_name].remove(handler)
                normal_logger.debug(f"取消注册异步事件处理器: {event_name}")
                return True
            
            return False
    
    def trigger_event(self, event_name: str, event_data: Any = None) -> None:
        """触发同步事件
        
        Args:
            event_name: 事件名
            event_data: 事件数据
        """
        # 获取同步处理器
        sync_handlers = set()
        with self._handlers_lock:
            if event_name in self._sync_handlers:
                sync_handlers = self._sync_handlers[event_name].copy()
        
        # 调用同步处理器
        for handler in sync_handlers:
            try:
                handler(event_data)
            except Exception as e:
                exception_logger.exception(f"同步事件处理器异常: {event_name}, {str(e)}")
    
    async def trigger_async_event(self, event_name: str, event_data: Any = None) -> None:
        """触发异步事件
        
        Args:
            event_name: 事件名
            event_data: 事件数据
        """
        # 获取同步和异步处理器
        sync_handlers = set()
        async_handlers = set()
        
        with self._handlers_lock:
            if event_name in self._sync_handlers:
                sync_handlers = self._sync_handlers[event_name].copy()
            
            if event_name in self._async_handlers:
                async_handlers = self._async_handlers[event_name].copy()
        
        # 调用同步处理器
        for handler in sync_handlers:
            try:
                handler(event_data)
            except Exception as e:
                exception_logger.exception(f"同步事件处理器异常: {event_name}, {str(e)}")
        
        # 调用异步处理器
        if async_handlers:
            # 创建任务列表
            tasks = []
            for handler in async_handlers:
                try:
                    tasks.append(asyncio.create_task(handler(event_data)))
                except Exception as e:
                    exception_logger.exception(f"创建异步事件处理任务异常: {event_name}, {str(e)}")
            
            # 等待所有任务完成
            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    exception_logger.exception(f"等待异步事件处理任务异常: {event_name}, {str(e)}")

# 单例实例
event_system = EventSystem()
