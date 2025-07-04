"""
结果推送模块 - 流水线架构组件
负责收集分析结果，按时序排列并推送到Redis队列
"""
import asyncio
import time
import threading
import json
import redis.asyncio as redis
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import uuid

@dataclass
class ResultEntry:
    """结果条目"""
    frame_id: str
    stream_id: str
    timestamp: float
    frame_index: int
    processing_time: float
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None
    received_time: float = 0.0
    
    def __post_init__(self):
        if self.received_time == 0.0:
            self.received_time = time.time()

@dataclass
class StreamResultBuffer:
    """流结果缓冲区"""
    stream_id: str
    buffer: deque  # 存储结果条目
    next_expected_index: int = 0
    max_buffer_size: int = 100
    timeout_seconds: float = 5.0
    last_push_time: float = 0.0
    
    def __post_init__(self):
        if not hasattr(self, 'buffer') or self.buffer is None:
            self.buffer = deque(maxlen=self.max_buffer_size)
        self.last_push_time = time.time()

class ResultModule:
    """
    结果推送模块
    负责收集分析结果，按时序排列并推送到Redis队列
    """
    
    def __init__(self, 
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 result_queue_prefix: str = "analysis_results",
                 max_buffer_time: float = 5.0,
                 order_timeout: float = 10.0):
        
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.result_queue_prefix = result_queue_prefix
        self.max_buffer_time = max_buffer_time
        self.order_timeout = order_timeout
        
        # Redis连接
        self.redis_client: Optional[redis.Redis] = None
        
        # 流结果缓冲区
        self.stream_buffers: Dict[str, StreamResultBuffer] = {}
        self.buffers_lock = asyncio.Lock()
        
        # 后台任务
        self.running = False
        self.push_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # 统计信息
        self.stats = {
            "total_results_received": 0,
            "total_results_pushed": 0,
            "in_order_results": 0,
            "out_of_order_results": 0,
            "timeout_results": 0,
            "active_streams": 0,
            "average_push_delay": 0.0
        }
        
        # 外部回调
        self.result_callbacks: List[Callable] = []
        
        print(f"[结果模块] 初始化完成 - Redis: {redis_host}:{redis_port}/{redis_db}")
    
    async def start(self):
        """启动结果模块"""
        if self.running:
            return
        
        self.running = True
        
        # 连接Redis
        await self._connect_redis()
        
        # 启动后台任务
        self.push_task = asyncio.create_task(self._push_results_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_results())
        
        print("[结果模块] 启动完成")
    
    async def stop(self):
        """停止结果模块"""
        if not self.running:
            return
        
        self.running = False
        
        # 停止后台任务
        for task in [self.push_task, self.cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # 推送剩余结果
        await self._flush_all_buffers()
        
        # 关闭Redis连接
        if self.redis_client:
            await self.redis_client.close()
        
        print("[结果模块] 停止完成")
    
    async def _connect_redis(self):
        """连接Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True
            )
            
            # 测试连接
            await self.redis_client.ping()
            print(f"[结果模块] Redis连接成功: {self.redis_host}:{self.redis_port}")
            
        except Exception as e:
            print(f"[结果模块] Redis连接失败: {e}")
            # 创建模拟Redis客户端
            self.redis_client = self._create_mock_redis()
    
    def _create_mock_redis(self):
        """创建模拟Redis客户端"""
        class MockRedis:
            def __init__(self):
                self.data = {}
            
            async def ping(self):
                return True
            
            async def lpush(self, key: str, *values):
                if key not in self.data:
                    self.data[key] = []
                self.data[key].extend(values)
                return len(self.data[key])
            
            async def close(self):
                pass
        
        print("[结果模块] 使用模拟Redis客户端")
        return MockRedis()
    
    async def add_result(self, result_data: Dict[str, Any]):
        """
        添加分析结果
        
        Args:
            result_data: 包含分析结果的字典
        """
        try:
            # 创建结果条目
            result_entry = ResultEntry(
                frame_id=result_data.get("frame_id", ""),
                stream_id=result_data.get("stream_id", ""),
                timestamp=result_data.get("timestamp", time.time()),
                frame_index=result_data.get("frame_index", 0),
                processing_time=result_data.get("processing_time", 0.0),
                results=result_data.get("results", {}),
                metadata=result_data.get("metadata", {}),
                success=result_data.get("success", True),
                error_message=result_data.get("error_message")
            )
            
            # 添加到流缓冲区
            await self._add_to_stream_buffer(result_entry)
            
            # 更新统计
            self.stats["total_results_received"] += 1
            
            # 调用外部回调
            for callback in self.result_callbacks:
                try:
                    callback(result_entry)
                except Exception as e:
                    print(f"[结果模块] 回调异常: {e}")
            
        except Exception as e:
            print(f"[结果模块] 添加结果异常: {e}")
    
    async def _add_to_stream_buffer(self, result_entry: ResultEntry):
        """添加结果到流缓冲区"""
        try:
            async with self.buffers_lock:
                stream_id = result_entry.stream_id
                
                # 创建或获取流缓冲区
                if stream_id not in self.stream_buffers:
                    self.stream_buffers[stream_id] = StreamResultBuffer(
                        stream_id=stream_id,
                        buffer=deque(maxlen=100),
                        timeout_seconds=self.order_timeout
                    )
                
                stream_buffer = self.stream_buffers[stream_id]
                
                # 检查是否是按序到达
                if result_entry.frame_index == stream_buffer.next_expected_index:
                    # 按序到达，立即标记为可推送
                    result_entry.metadata["in_order"] = True
                    stream_buffer.next_expected_index += 1
                    self.stats["in_order_results"] += 1
                else:
                    # 乱序到达，缓存等待
                    result_entry.metadata["in_order"] = False
                    self.stats["out_of_order_results"] += 1
                
                # 添加到缓冲区
                stream_buffer.buffer.append(result_entry)
                
                # 更新活跃流数量
                self.stats["active_streams"] = len(self.stream_buffers)
                
        except Exception as e:
            print(f"[结果模块] 添加到流缓冲区异常: {e}")
    
    async def _push_results_loop(self):
        """推送结果循环"""
        while self.running:
            try:
                await asyncio.sleep(0.5)  # 每500ms检查一次
                
                # 处理所有流的缓冲区
                async with self.buffers_lock:
                    for stream_id, stream_buffer in list(self.stream_buffers.items()):
                        await self._process_stream_buffer(stream_buffer)
                
            except Exception as e:
                print(f"[结果模块] 推送循环异常: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_stream_buffer(self, stream_buffer: StreamResultBuffer):
        """处理流缓冲区"""
        try:
            current_time = time.time()
            to_push = []
            remaining = deque()
            
            # 按时间戳排序缓冲区
            sorted_results = sorted(stream_buffer.buffer, key=lambda x: x.timestamp)
            
            for result_entry in sorted_results:
                # 检查是否超时
                age = current_time - result_entry.received_time
                
                if (result_entry.metadata.get("in_order", False) or 
                    age > stream_buffer.timeout_seconds):
                    
                    # 标记为推送
                    to_push.append(result_entry)
                    
                    if age > stream_buffer.timeout_seconds:
                        self.stats["timeout_results"] += 1
                        result_entry.metadata["timeout"] = True
                
                else:
                    # 继续等待
                    remaining.append(result_entry)
            
            # 更新缓冲区
            stream_buffer.buffer = remaining
            
            # 推送结果
            if to_push:
                await self._push_results_to_redis(stream_buffer.stream_id, to_push)
                
                # 更新推送时间
                stream_buffer.last_push_time = current_time
                
                # 更新统计
                self.stats["total_results_pushed"] += len(to_push)
                
                # 计算平均推送延迟
                total_delay = sum(
                    current_time - result.received_time 
                    for result in to_push
                )
                avg_delay = total_delay / len(to_push)
                
                # 更新平均推送延迟（使用指数移动平均）
                if self.stats["average_push_delay"] == 0:
                    self.stats["average_push_delay"] = avg_delay
                else:
                    self.stats["average_push_delay"] = (
                        self.stats["average_push_delay"] * 0.9 + avg_delay * 0.1
                    )
                
                print(f"[结果模块] 推送结果: {stream_buffer.stream_id}, "
                      f"数量: {len(to_push)}, 平均延迟: {avg_delay:.3f}s")
            
        except Exception as e:
            print(f"[结果模块] 处理流缓冲区异常: {stream_buffer.stream_id}, {e}")
    
    async def _push_results_to_redis(self, stream_id: str, results: List[ResultEntry]):
        """推送结果到Redis"""
        try:
            if not self.redis_client:
                return
            
            # 生成Redis队列键
            queue_key = f"{self.result_queue_prefix}:{stream_id}"
            
            # 准备推送数据
            push_data = []
            for result in results:
                result_dict = asdict(result)
                result_dict["push_time"] = time.time()
                push_data.append(json.dumps(result_dict, ensure_ascii=False))
            
            # 批量推送到Redis
            if push_data:
                await self.redis_client.lpush(queue_key, *push_data)
                
        except Exception as e:
            print(f"[结果模块] 推送到Redis异常: {stream_id}, {e}")
    
    async def _cleanup_expired_results(self):
        """清理过期结果的后台任务"""
        while self.running:
            try:
                await asyncio.sleep(30.0)  # 每30秒清理一次
                
                current_time = time.time()
                expired_streams = []
                
                async with self.buffers_lock:
                    for stream_id, stream_buffer in self.stream_buffers.items():
                        # 清理过期结果
                        remaining = deque()
                        expired_count = 0
                        
                        for result_entry in stream_buffer.buffer:
                            age = current_time - result_entry.received_time
                            if age > self.order_timeout * 2:  # 超过2倍超时时间
                                expired_count += 1
                            else:
                                remaining.append(result_entry)
                        
                        stream_buffer.buffer = remaining
                        
                        if expired_count > 0:
                            print(f"[结果模块] 清理过期结果: {stream_id}, {expired_count}个")
                        
                        # 检查是否为空闲流
                        idle_time = current_time - stream_buffer.last_push_time
                        if idle_time > 300.0 and not stream_buffer.buffer:  # 5分钟无活动
                            expired_streams.append(stream_id)
                    
                    # 移除空闲流
                    for stream_id in expired_streams:
                        del self.stream_buffers[stream_id]
                        print(f"[结果模块] 移除空闲流: {stream_id}")
                
                # 更新统计
                self.stats["active_streams"] = len(self.stream_buffers)
                
            except Exception as e:
                print(f"[结果模块] 清理异常: {e}")
                await asyncio.sleep(30.0)
    
    async def _flush_all_buffers(self):
        """刷新所有缓冲区"""
        try:
            async with self.buffers_lock:
                for stream_buffer in self.stream_buffers.values():
                    if stream_buffer.buffer:
                        # 强制推送所有剩余结果
                        all_results = list(stream_buffer.buffer)
                        await self._push_results_to_redis(stream_buffer.stream_id, all_results)
                        
                        print(f"[结果模块] 刷新缓冲区: {stream_buffer.stream_id}, "
                              f"剩余: {len(all_results)}个")
                        
                        stream_buffer.buffer.clear()
            
        except Exception as e:
            print(f"[结果模块] 刷新缓冲区异常: {e}")
    
    def add_result_callback(self, callback: Callable[[ResultEntry], None]):
        """添加结果回调"""
        self.result_callbacks.append(callback)
    
    def get_result_statistics(self) -> Dict[str, Any]:
        """获取结果统计信息"""
        stream_info = {}
        for stream_id, stream_buffer in self.stream_buffers.items():
            stream_info[stream_id] = {
                "buffer_size": len(stream_buffer.buffer),
                "next_expected_index": stream_buffer.next_expected_index,
                "last_push_time": stream_buffer.last_push_time,
                "idle_time": time.time() - stream_buffer.last_push_time
            }
        
        return {
            "global_stats": self.stats,
            "stream_info": stream_info,
            "redis_connected": self.redis_client is not None
        }
    
    async def get_queue_status(self, stream_id: str) -> Dict[str, Any]:
        """获取队列状态"""
        try:
            if not self.redis_client:
                return {"error": "Redis未连接"}
            
            queue_key = f"{self.result_queue_prefix}:{stream_id}"
            queue_length = await self.redis_client.llen(queue_key)
            
            return {
                "stream_id": stream_id,
                "queue_key": queue_key,
                "queue_length": queue_length,
                "buffer_size": len(self.stream_buffers.get(stream_id, StreamResultBuffer("")).buffer)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def force_push_stream_results(self, stream_id: str) -> int:
        """强制推送指定流的所有结果"""
        try:
            async with self.buffers_lock:
                if stream_id not in self.stream_buffers:
                    return 0
                
                stream_buffer = self.stream_buffers[stream_id]
                result_count = len(stream_buffer.buffer)
                
                if result_count > 0:
                    all_results = list(stream_buffer.buffer)
                    await self._push_results_to_redis(stream_id, all_results)
                    stream_buffer.buffer.clear()
                    
                    print(f"[结果模块] 强制推送: {stream_id}, {result_count}个结果")
                
                return result_count
                
        except Exception as e:
            print(f"[结果模块] 强制推送异常: {stream_id}, {e}")
            return 0 