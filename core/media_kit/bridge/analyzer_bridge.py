"""
分析器桥接模块
负责协调流状态和分析任务，当流状态变化时通知相关分析任务
"""

import asyncio
import threading
from typing import Dict, Set, Any, Optional, Callable, List
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

from ..base.stream_interface import StreamStatus, StreamHealthStatus

class AnalyzerBridge:
    """分析器桥接类，协调流状态和分析任务"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AnalyzerBridge, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化桥接器"""
        if self._initialized:
            return
            
        # 流与分析任务的映射关系
        self._stream_tasks: Dict[str, Set[str]] = {}  # stream_id -> {task_id1, task_id2, ...}
        self._task_streams: Dict[str, str] = {}  # task_id -> stream_id
        
        # 任务状态回调
        self._task_callbacks: Dict[str, Dict[str, Callable]] = {}  # task_id -> {event_type: callback}
        
        # 状态锁
        self._lock = asyncio.Lock()
        
        # 流管理器引用
        self._stream_manager = None
        
        # 事件系统引用
        self._event_system = None
        
        self._initialized = True
        
        normal_logger.info("分析器桥接器初始化完成")
    
    async def initialize(self) -> None:
        """初始化桥接器"""
        # 导入流管理器
        from ..base.stream_manager import stream_manager
        self._stream_manager = stream_manager
        
        # 导入事件系统
        from ..base.event_system import event_system
        self._event_system = event_system
        
        # 注册流状态变化事件
        self._event_system.register_async_handler(
            "stream_status_changed", 
            self._on_stream_status_changed
        )
        
        normal_logger.info("分析器桥接器事件注册完成")
    
    async def register_analysis_task(self, task_id: str, stream_id: str, 
                                   ready_callback: Optional[Callable] = None,
                                   error_callback: Optional[Callable] = None) -> bool:
        """注册分析任务
        
        Args:
            task_id: 分析任务ID
            stream_id: 流ID
            ready_callback: 流就绪回调函数，格式: callback(task_id, stream_id)
            error_callback: 流错误回调函数，格式: callback(task_id, stream_id, error_message)
            
        Returns:
            bool: 是否成功注册
        """
        async with self._lock:
            # 记录任务与流的关系
            if stream_id not in self._stream_tasks:
                self._stream_tasks[stream_id] = set()
            
            self._stream_tasks[stream_id].add(task_id)
            self._task_streams[task_id] = stream_id
            
            # 注册回调
            if ready_callback or error_callback:
                self._task_callbacks[task_id] = {}
                
                if ready_callback:
                    self._task_callbacks[task_id]["ready"] = ready_callback
                
                if error_callback:
                    self._task_callbacks[task_id]["error"] = error_callback
            
            # 获取当前流状态
            stream_status = await self._stream_manager.get_stream_status(stream_id)
            
            # 如果流已经就绪，立即通知分析任务
            if stream_status == StreamStatus.RUNNING:
                await self._notify_task_stream_ready(task_id, stream_id)
                return True
            elif stream_status == StreamStatus.ERROR:
                # 流状态异常，通知任务
                await self._notify_task_stream_error(task_id, stream_id, "流状态异常")
                return False
            else:
                # 流状态为其他（如CONNECTING），等待状态变化
                return True
    
    async def unregister_analysis_task(self, task_id: str) -> bool:
        """取消注册分析任务
        
        Args:
            task_id: 分析任务ID
            
        Returns:
            bool: 是否成功取消注册
        """
        async with self._lock:
            # 检查任务是否存在
            if task_id not in self._task_streams:
                return False
            
            # 获取流ID
            stream_id = self._task_streams[task_id]
            
            # 移除任务与流的关系
            if stream_id in self._stream_tasks:
                self._stream_tasks[stream_id].discard(task_id)
                
                # 如果流没有关联任务了，可以考虑停止流
                if not self._stream_tasks[stream_id]:
                    del self._stream_tasks[stream_id]
            
            del self._task_streams[task_id]
            
            # 移除回调
            if task_id in self._task_callbacks:
                del self._task_callbacks[task_id]
            
            return True
    
    async def get_task_stream_status(self, task_id: str) -> Optional[StreamStatus]:
        """获取任务关联的流状态
        
        Args:
            task_id: 分析任务ID
            
        Returns:
            Optional[StreamStatus]: 流状态，如果任务不存在则返回None
        """
        async with self._lock:
            # 检查任务是否存在
            if task_id not in self._task_streams:
                return None
            
            # 获取流ID
            stream_id = self._task_streams[task_id]
            
            # 获取流状态
            return await self._stream_manager.get_stream_status(stream_id)
    
    async def get_task_stream_health(self, task_id: str) -> Optional[StreamHealthStatus]:
        """获取任务关联的流健康状态
        
        Args:
            task_id: 分析任务ID
            
        Returns:
            Optional[StreamHealthStatus]: 流健康状态，如果任务不存在则返回None
        """
        async with self._lock:
            # 检查任务是否存在
            if task_id not in self._task_streams:
                return None
            
            # 获取流ID
            stream_id = self._task_streams[task_id]
            
            # 获取流状态
            return await self._stream_manager.get_stream_health(stream_id)
    
    async def _on_stream_status_changed(self, event_data: Dict[str, Any]) -> None:
        """流状态变化事件处理
        
        Args:
            event_data: 事件数据，格式: {"stream_id": "xxx", "status": StreamStatus.xxx}
        """
        # 获取流ID和状态
        stream_id = event_data.get("stream_id")
        status = event_data.get("status")
        
        if not stream_id or not status:
            return
        
        # 获取依赖该流的所有任务
        async with self._lock:
            tasks = list(self._stream_tasks.get(stream_id, set()))
        
        if not tasks:
            return
        
        if status == StreamStatus.RUNNING:
            # 流就绪，通知所有相关任务
            for task_id in tasks:
                await self._notify_task_stream_ready(task_id, stream_id)
        elif status in [StreamStatus.ERROR, StreamStatus.STOPPED]:
            # 流异常或停止，通知所有相关任务
            error_message = event_data.get("error_message", "流状态异常")
            for task_id in tasks:
                await self._notify_task_stream_error(task_id, stream_id, error_message)
    
    async def _notify_task_stream_ready(self, task_id: str, stream_id: str) -> None:
        """通知任务流已就绪
        
        Args:
            task_id: 分析任务ID
            stream_id: 流ID
        """
        # 获取任务的就绪回调
        callback = None
        if task_id in self._task_callbacks and "ready" in self._task_callbacks[task_id]:
            callback = self._task_callbacks[task_id]["ready"]
        
        # 调用回调
        if callback:
            try:
                callback(task_id, stream_id)
            except Exception as e:
                exception_logger.exception(f"任务 {task_id} 流就绪回调异常: {str(e)}")
    
    async def _notify_task_stream_error(self, task_id: str, stream_id: str, error_message: str) -> None:
        """通知任务流出错
        
        Args:
            task_id: 分析任务ID
            stream_id: 流ID
            error_message: 错误信息
        """
        # 获取任务的错误回调
        callback = None
        if task_id in self._task_callbacks and "error" in self._task_callbacks[task_id]:
            callback = self._task_callbacks[task_id]["error"]
        
        # 调用回调
        if callback:
            try:
                callback(task_id, stream_id, error_message)
            except Exception as e:
                exception_logger.exception(f"任务 {task_id} 流错误回调异常: {str(e)}")

# 单例实例
analyzer_bridge = AnalyzerBridge()
