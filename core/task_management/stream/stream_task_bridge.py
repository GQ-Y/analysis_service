"""
视频流与任务桥接模块
负责处理视频流状态变化对任务的影响，实现视频流与分析任务的解耦
"""
import asyncio
import json
import threading
from typing import Dict, Any, Set, Optional
import pickle
from concurrent.futures import ThreadPoolExecutor
import time

from core.redis_manager import RedisManager
from shared.utils.logger import get_normal_logger, get_exception_logger
from core.task_management.utils.status import TaskStatus
from .status import StreamStatus

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class StreamTaskBridge:
    """视频流与任务桥接器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StreamTaskBridge, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化桥接器"""
        if self._initialized:
            return
            
        self._initialized = True
        self._stream_tasks = {}  # stream_id -> {task_id1, task_id2, ...}
        self._task_streams = {}  # task_id -> stream_id
        self._redis = None
        self._subscriptions = {}  # channel -> asyncio.Task
        self._task_manager = None  # 延迟初始化
        
        # 添加线程池执行器，用于处理耗时操作
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        normal_logger.info("视频流与任务桥接器初始化完成")
        
    async def initialize(self, task_manager=None):
        """初始化桥接器
        
        Args:
            task_manager: 任务管理器实例
        """
        # 保存任务管理器引用
        self._task_manager = task_manager
        
        # 初始化Redis
        self._redis = RedisManager()
        
        # 序列化初始状态
        self._serialize_state()
        
        # 启动监听
        await self._start_stream_status_listener()
        
        normal_logger.info("视频流与任务桥接器初始化完成")
        
    def _serialize_state(self):
        """序列化状态以确保多进程安全"""
        try:
            # 如果Redis不可用，跳过序列化
            if not self._redis or not self._redis.redis_client:
                return
                
            # 序列化状态
            serialized_stream_tasks = pickle.dumps(self._stream_tasks)
            serialized_task_streams = pickle.dumps(self._task_streams)
            
            # 使用异步任务进行序列化，避免在同步方法中创建新事件循环
            asyncio.create_task(self._async_serialize_state(serialized_stream_tasks, serialized_task_streams))
            
            normal_logger.debug("桥接器状态已序列化")
        except Exception as e:
            exception_logger.exception(f"序列化状态失败: {str(e)}")
    
    async def _async_serialize_state(self, serialized_stream_tasks, serialized_task_streams):
        """异步执行序列化状态操作"""
        try:
            await self._redis.redis_client.set('stream_task_bridge:stream_tasks', serialized_stream_tasks)
            await self._redis.redis_client.set('stream_task_bridge:task_streams', serialized_task_streams)
        except Exception as e:
            exception_logger.exception(f"异步序列化状态失败: {str(e)}")
            
    def _deserialize_state(self):
        """反序列化状态以确保多进程安全"""
        try:
            # 如果Redis不可用，跳过反序列化
            if not self._redis or not self._redis.redis_client:
                return
                
            # 使用异步任务进行反序列化，避免在同步方法中创建新事件循环
            # 创建一个异步任务并立即执行，但后续不等待它的结果
            # 注意这种方式可能导致反序列化结果不立即可用
            asyncio.create_task(self._async_deserialize_state())
                
            normal_logger.debug("桥接器状态反序列化已启动")
        except Exception as e:
            exception_logger.exception(f"反序列化状态失败: {str(e)}")
            
    async def _async_deserialize_state(self):
        """异步执行反序列化状态操作"""
        try:
            # 从Redis获取状态
            serialized_stream_tasks = await self._redis.redis_client.get('stream_task_bridge:stream_tasks')
            serialized_task_streams = await self._redis.redis_client.get('stream_task_bridge:task_streams')
            
            # 反序列化状态
            if serialized_stream_tasks:
                self._stream_tasks = pickle.loads(serialized_stream_tasks)
            if serialized_task_streams:
                self._task_streams = pickle.loads(serialized_task_streams)
                
            normal_logger.debug("桥接器状态已完成反序列化")
        except Exception as e:
            exception_logger.exception(f"异步反序列化状态失败: {str(e)}")
            
    async def _start_stream_status_listener(self):
        """启动流状态监听"""
        try:
            # 创建监听任务
            listen_task = asyncio.create_task(self._listen_stream_status())
            self._subscriptions["stream_status"] = listen_task
            
            normal_logger.info("流状态监听已启动")
            
        except Exception as e:
            exception_logger.exception(f"启动流状态监听失败: {str(e)}")
            
    async def _listen_stream_status(self):
        """监听流状态变化"""
        pubsub = None
        try:
            # 确保Redis实例已创建
            if self._redis is None:
                self._redis = RedisManager()
                
            # 创建订阅
            pubsub = self._redis.redis_client.pubsub()
            await pubsub.psubscribe("stream_status:*")
            
            normal_logger.info("开始监听流状态变化")
            
            # 监听消息
            while True:
                try:
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=0.5  # 添加超时，使循环更容易被中断
                    )
                    
                    if message is None:
                        # 检查是否有取消请求
                        await asyncio.sleep(0.1)
                        continue
                        
                    try:
                        # 解析消息
                        channel = message["channel"]
                        if isinstance(channel, bytes):
                            channel = channel.decode("utf-8")
                            
                        # 获取任务ID
                        task_id = channel.split(":")[-1]
                        
                        # 解析消息内容
                        data = json.loads(message["data"])
                        stream_id = data.get("stream_id")
                        status = data.get("status")
                        
                        # 处理流状态变化
                        await self._handle_stream_status_change(stream_id, task_id, status)
                        
                    except json.JSONDecodeError:
                        normal_logger.warning(f"无效的流状态消息: {message}")
                    except Exception as e:
                        exception_logger.exception(f"处理流状态消息异常: {str(e)}")
                        
                except asyncio.TimeoutError:
                    # 超时是正常的，继续循环
                    continue
                except asyncio.CancelledError:
                    # 任务被取消，退出循环
                    normal_logger.info("流状态监听任务被取消")
                    break
                        
        except asyncio.CancelledError:
            normal_logger.info("流状态监听任务被取消")
            
        except Exception as e:
            exception_logger.exception(f"流状态监听异常: {str(e)}")
            
        finally:
            # 确保取消订阅
            if pubsub is not None:
                try:
                    await pubsub.punsubscribe("stream_status:*")
                    await pubsub.close()
                    normal_logger.info("Redis订阅已关闭")
                except Exception as e:
                    exception_logger.exception(f"关闭Redis订阅异常: {str(e)}")
            
    async def _handle_stream_status_change(self, stream_id: str, task_id: str, status: int):
        """处理流状态变化
        
        Args:
            stream_id: 流ID
            task_id: 任务ID
            status: 流状态
        """
        try:
            # 检查任务管理器是否已初始化
            if self._task_manager is None:
                normal_logger.warning("任务管理器未初始化，无法处理流状态变化")
                return
                
            # 处理不同状态
            if status == int(StreamStatus.OFFLINE):
                # 流离线，暂停任务
                normal_logger.warning(f"流 {stream_id} 离线，暂停任务: {task_id}")
                await self._task_manager.pause_task(task_id, reason="视频流离线")
                
            elif status == int(StreamStatus.ERROR):
                # 流错误，暂停任务
                normal_logger.error(f"流 {stream_id} 错误，暂停任务: {task_id}")
                await self._task_manager.pause_task(task_id, reason="视频流错误")
                
            elif status == int(StreamStatus.ONLINE):
                # 流恢复在线，检查任务是否因流离线而暂停
                task_status = self._task_manager.get_task_status(task_id)
                if task_status and task_status.get("status") == TaskStatus.PAUSED:
                    pause_reason = task_status.get("pause_reason", "")
                    if "视频流" in pause_reason:
                        normal_logger.info(f"流 {stream_id} 恢复在线，恢复任务: {task_id}")
                        await self._task_manager.resume_task(task_id)
                        
        except Exception as e:
            exception_logger.exception(f"处理流状态变化异常: {str(e)}")
            
    def register_task_stream(self, task_id: str, stream_id: str):
        """注册任务与流的关系
        
        Args:
            task_id: 任务ID
            stream_id: 流ID
        """
        # 反序列化当前状态
        self._deserialize_state()
        
        # 添加到映射
        if stream_id not in self._stream_tasks:
            self._stream_tasks[stream_id] = set()
        self._stream_tasks[stream_id].add(task_id)
        self._task_streams[task_id] = stream_id
        
        # 序列化新状态
        self._serialize_state()
        
        normal_logger.info(f"任务 {task_id} 已关联到流 {stream_id}")
        
    def unregister_task_stream(self, task_id: str):
        """取消注册任务与流的关系
        
        Args:
            task_id: 任务ID
        """
        # 反序列化当前状态
        self._deserialize_state()
        
        # 从映射中移除
        if task_id in self._task_streams:
            stream_id = self._task_streams[task_id]
            if stream_id in self._stream_tasks:
                self._stream_tasks[stream_id].discard(task_id)
                if not self._stream_tasks[stream_id]:
                    del self._stream_tasks[stream_id]
            del self._task_streams[task_id]
            
            # 序列化新状态
            self._serialize_state()
            
            normal_logger.info(f"任务 {task_id} 已取消关联流 {stream_id}")
            
    def get_stream_tasks(self, stream_id: str) -> Set[str]:
        """获取流关联的任务
        
        Args:
            stream_id: 流ID
            
        Returns:
            Set[str]: 任务ID集合
        """
        # 反序列化当前状态
        self._deserialize_state()
        return self._stream_tasks.get(stream_id, set())
        
    def get_task_stream(self, task_id: str) -> Optional[str]:
        """获取任务关联的流
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[str]: 流ID
        """
        # 反序列化当前状态
        self._deserialize_state()
        return self._task_streams.get(task_id)
        
    async def shutdown(self):
        """关闭桥接器"""
        try:
            normal_logger.info("开始关闭视频流与任务桥接器...")
            
            # 取消所有订阅任务
            for channel, task in list(self._subscriptions.items()):
                if not task.done():
                    normal_logger.info(f"正在取消订阅任务: {channel}")
                    task.cancel()
                    try:
                        # 设置超时，避免无限等待
                        await asyncio.wait_for(task, timeout=5.0)
                    except asyncio.TimeoutError:
                        normal_logger.warning(f"取消订阅任务超时: {channel}")
                    except asyncio.CancelledError:
                        normal_logger.info(f"订阅任务已成功取消: {channel}")
                    except Exception as e:
                        exception_logger.exception(f"取消订阅任务异常: {str(e)}")
            
            # 重置订阅
            self._subscriptions.clear()
                    
            normal_logger.info("视频流与任务桥接器已关闭")
            return True
        except Exception as e:
            exception_logger.exception(f"关闭视频流与任务桥接器异常: {str(e)}")
            return False

# 单例实例将在需要时创建 