"""
零拷贝任务处理器
重构TaskProcessor支持帧引用传递和批量处理优化
"""
import asyncio
import time
import threading
import queue
from typing import Dict, Optional, Any, List
import traceback

from .processor import TaskProcessor
from ..interfaces.zero_copy_stream_interface import AsyncFrameReferenceQueue
from ..frame.frame_reference import FrameReference
from ..memory.memory_pool import MemoryPool
from .stream.zero_copy_manager import ZeroCopyStreamManager

# 使用项目现有的日志系统
try:
    from shared.utils.logger import get_normal_logger, get_exception_logger
    normal_logger = get_normal_logger(__name__)
    exception_logger = get_exception_logger(__name__)
except ImportError:
    import logging
    normal_logger = logging.getLogger(__name__)
    exception_logger = logging.getLogger(__name__)


class ZeroCopyTaskProcessor(TaskProcessor):
    """
    零拷贝任务处理器
    扩展基础任务处理器，支持帧引用传递和批量处理
    """
    
    def __init__(self, task_manager, memory_pool: MemoryPool):
        """
        初始化零拷贝任务处理器
        
        Args:
            task_manager: 任务管理器
            memory_pool: 内存池实例
        """
        super().__init__(task_manager)
        
        # 零拷贝相关组件
        self.memory_pool = memory_pool
        self.zero_copy_stream_manager = None
        
        # 零拷贝任务跟踪
        self._zero_copy_tasks: Dict[str, Dict[str, Any]] = {}
        
        # 批处理配置
        self._batch_configs: Dict[str, Dict[str, Any]] = {}
        
        # 性能统计
        self._zero_copy_stats = {
            "total_frames_processed": 0,
            "batch_operations": 0,
            "zero_copy_operations": 0,
            "memory_allocation_time": 0.0,
            "frame_processing_time": 0.0,
        }
        
        normal_logger.info("零拷贝任务处理器初始化完成")
    
    async def initialize(self):
        """初始化零拷贝任务处理器"""
        # 调用父类初始化
        await super().initialize()
        
        # 初始化零拷贝流管理器
        self.zero_copy_stream_manager = ZeroCopyStreamManager(self.memory_pool)
        
        normal_logger.info("零拷贝任务处理器初始化完成")
        return True
    
    async def start_task_zero_copy(self, task_id: str, task_config: Dict[str, Any]) -> bool:
        """
        启动零拷贝任务
        
        Args:
            task_id: 任务ID
            task_config: 任务配置
            
        Returns:
            bool: 是否成功启动
        """
        try:
            # 检查是否启用零拷贝模式
            enable_zero_copy = task_config.get("enable_zero_copy", True)
            if not enable_zero_copy:
                normal_logger.info(f"任务 {task_id} 未启用零拷贝，使用传统模式")
                return await self.start_task(task_id, task_config)
            
            # 检查内存池状态
            if not self.memory_pool or not self.memory_pool.initialized:
                normal_logger.error(f"内存池未初始化，任务 {task_id} 无法使用零拷贝模式")
                return False
            
            # 配置批处理参数
            batch_config = {
                "enable_batch": task_config.get("enable_batch_processing", True),
                "batch_size": task_config.get("batch_size", 4),
                "batch_timeout": task_config.get("batch_timeout", 0.1),
            }
            self._batch_configs[task_id] = batch_config
            
            # 创建停止和暂停事件
            stop_event = threading.Event()
            pause_event = threading.Event()
            self.stop_events[task_id] = stop_event
            self.pause_events[task_id] = pause_event
            
            # 创建结果队列
            result_queue = queue.Queue()
            
            # 获取流ID并订阅零拷贝流
            stream_id = task_config.get("stream_id", "")
            if not stream_id:
                normal_logger.error(f"任务 {task_id} 缺少stream_id配置")
                return False
            
            # 订阅零拷贝流
            success, frame_ref_queue = await self.zero_copy_stream_manager.subscribe_stream_zero_copy(
                stream_id, task_id, task_config
            )
            
            if not success or frame_ref_queue is None:
                normal_logger.error(f"订阅零拷贝流失败: {stream_id}")
                return False
            
            # 保存零拷贝任务信息
            self._zero_copy_tasks[task_id] = {
                "frame_ref_queue": frame_ref_queue,
                "batch_config": batch_config,
                "stream_id": stream_id,
                "start_time": time.time(),
            }
            
            # 创建并启动零拷贝任务线程
            thread = threading.Thread(
                target=asyncio.run,
                args=(self.process_stream_worker_zero_copy(
                    task_id, task_config, result_queue, stop_event, pause_event, frame_ref_queue
                ),),
                daemon=True
            )
            thread.start()
            
            # 保存任务信息
            self.running_tasks[task_id] = {
                "thread": thread,
                "config": task_config,
                "start_time": time.time(),
                "status": "PROCESSING",
                "stream_id": stream_id,
                "zero_copy_enabled": True,
            }
            self.task_threads[task_id] = thread
            
            # 创建并启动结果处理器
            result_handler = asyncio.create_task(self._handle_results(task_id, result_queue))
            self.result_handlers[task_id] = result_handler
            
            normal_logger.info(f"零拷贝任务启动成功: {task_id}")
            return True
            
        except Exception as e:
            exception_logger.exception(f"启动零拷贝任务失败: {task_id}, {str(e)}")
            return False
    
    async def process_stream_worker_zero_copy(self, 
                                            task_id: str, 
                                            task_config: Dict[str, Any],
                                            result_queue: queue.Queue,
                                            stop_event: threading.Event,
                                            pause_event: threading.Event,
                                            frame_ref_queue: AsyncFrameReferenceQueue) -> None:
        """
        零拷贝流处理工作器
        
        Args:
            task_id: 任务ID
            task_config: 任务配置
            result_queue: 结果队列
            stop_event: 停止事件
            pause_event: 暂停事件
            frame_ref_queue: 帧引用队列
        """
        normal_logger.info(f"启动零拷贝流处理工作器: {task_id}")
        
        try:
            # 获取分析器
            analyzer = await self._get_analyzer(task_config)
            if not analyzer:
                normal_logger.error(f"无法获取分析器: {task_id}")
                return
            
            # 获取批处理配置
            batch_config = self._batch_configs.get(task_id, {})
            enable_batch = batch_config.get("enable_batch", True)
            batch_size = batch_config.get("batch_size", 4)
            batch_timeout = batch_config.get("batch_timeout", 0.1)
            
            # 分析间隔配置
            analysis_interval = task_config.get("analysis_interval", 1)
            
            frame_counter = 0
            batch_buffer = []
            last_batch_time = time.time()
            
            while not stop_event.is_set():
                try:
                    # 检查暂停状态
                    if pause_event.is_set():
                        await asyncio.sleep(0.1)
                        continue
                    
                    # 获取帧引用
                    frame_ref = await frame_ref_queue.get()
                    if frame_ref is None:
                        continue
                    
                    frame_counter += 1
                    
                    # 应用分析间隔逻辑
                    if analysis_interval > 1 and (frame_counter - 1) % analysis_interval != 0:
                        frame_ref.release()  # 释放跳过的帧引用
                        continue
                    
                    if enable_batch:
                        # 批处理模式
                        batch_buffer.append(frame_ref)
                        
                        # 检查是否需要处理批次
                        current_time = time.time()
                        should_process_batch = (
                            len(batch_buffer) >= batch_size or
                            (batch_buffer and current_time - last_batch_time >= batch_timeout)
                        )
                        
                        if should_process_batch:
                            await self._process_frame_batch(
                                task_id, batch_buffer, analyzer, result_queue
                            )
                            batch_buffer.clear()
                            last_batch_time = current_time
                            self._zero_copy_stats["batch_operations"] += 1
                    else:
                        # 单帧处理模式
                        await self._process_single_frame_reference(
                            task_id, frame_ref, analyzer, result_queue, frame_counter
                        )
                        self._zero_copy_stats["zero_copy_operations"] += 1
                    
                    self._zero_copy_stats["total_frames_processed"] += 1
                    
                except Exception as e:
                    exception_logger.exception(f"零拷贝帧处理异常: {task_id}, {str(e)}")
                    continue
            
            # 处理剩余的批次
            if batch_buffer:
                await self._process_frame_batch(
                    task_id, batch_buffer, analyzer, result_queue
                )
            
        except Exception as e:
            exception_logger.exception(f"零拷贝流处理工作器异常: {task_id}, {str(e)}")
        finally:
            normal_logger.info(f"零拷贝流处理工作器结束: {task_id}")
    
    async def _process_single_frame_reference(self,
                                            task_id: str,
                                            frame_ref: FrameReference,
                                            analyzer: Any,
                                            result_queue: queue.Queue,
                                            frame_index: int) -> None:
        """
        处理单个帧引用

        Args:
            task_id: 任务ID
            frame_ref: 帧引用
            analyzer: 分析器
            result_queue: 结果队列
            frame_index: 帧索引
        """
        try:
            start_time = time.time()

            # 获取帧数据（零拷贝）
            frame_data = frame_ref.get_data()
            if frame_data is None:
                normal_logger.warning(f"任务 {task_id}: 无法获取帧数据")
                return

            # 执行分析
            analysis_data = await analyzer.process_video_frame(
                frame_data, frame_index=frame_index
            )

            # 记录处理时间
            processing_time = time.time() - start_time
            self._zero_copy_stats["frame_processing_time"] += processing_time

            # 构建结果
            result = {
                "task_id": task_id,
                "frame_index": frame_index,
                "frame_metadata": frame_ref.get_metadata().to_dict(),
                "analysis_data": analysis_data,
                "processing_time": processing_time,
                "timestamp": time.time(),
            }

            # 放入结果队列
            result_queue.put(result)

        except Exception as e:
            exception_logger.exception(f"处理帧引用异常: {task_id}, {str(e)}")
        finally:
            # 释放帧引用
            frame_ref.release()

    async def _process_frame_batch(self,
                                 task_id: str,
                                 frame_refs: List[FrameReference],
                                 analyzer: Any,
                                 result_queue: queue.Queue) -> None:
        """
        批量处理帧引用

        Args:
            task_id: 任务ID
            frame_refs: 帧引用列表
            analyzer: 分析器
            result_queue: 结果队列
        """
        try:
            start_time = time.time()

            # 提取帧数据（零拷贝）
            frame_data_list = []
            metadata_list = []

            for frame_ref in frame_refs:
                frame_data = frame_ref.get_data()
                if frame_data is not None:
                    frame_data_list.append(frame_data)
                    metadata_list.append(frame_ref.get_metadata())

            if not frame_data_list:
                normal_logger.warning(f"任务 {task_id}: 批次中无有效帧数据")
                return

            # 批量分析（如果分析器支持）
            if hasattr(analyzer, 'process_video_frames_batch'):
                analysis_results = await analyzer.process_video_frames_batch(frame_data_list)
            else:
                # 逐个处理
                analysis_results = []
                for i, frame_data in enumerate(frame_data_list):
                    result = await analyzer.process_video_frame(frame_data, frame_index=i)
                    analysis_results.append(result)

            # 记录处理时间
            processing_time = time.time() - start_time
            self._zero_copy_stats["frame_processing_time"] += processing_time

            # 构建批量结果
            for i, (metadata, analysis_data) in enumerate(zip(metadata_list, analysis_results)):
                result = {
                    "task_id": task_id,
                    "frame_index": i,
                    "frame_metadata": metadata.to_dict(),
                    "analysis_data": analysis_data,
                    "processing_time": processing_time / len(analysis_results),
                    "timestamp": time.time(),
                    "batch_size": len(frame_refs),
                }
                result_queue.put(result)

        except Exception as e:
            exception_logger.exception(f"批量处理帧引用异常: {task_id}, {str(e)}")
        finally:
            # 释放所有帧引用
            for frame_ref in frame_refs:
                frame_ref.release()

    async def _get_analyzer(self, task_config: Dict[str, Any]):
        """获取分析器（从父类复制的辅助方法）"""
        # 这里需要实现获取分析器的逻辑，可以从父类复制相关代码
        # 暂时返回None，需要后续完善
        return None

    def get_zero_copy_stats(self) -> Dict[str, Any]:
        """
        获取零拷贝统计信息
        
        Returns:
            Dict[str, Any]: 零拷贝统计信息
        """
        return {
            "zero_copy_stats": self._zero_copy_stats.copy(),
            "active_zero_copy_tasks": len(self._zero_copy_tasks),
            "memory_pool_stats": self.memory_pool.get_stats() if self.memory_pool else {},
            "stream_manager_stats": (
                self.zero_copy_stream_manager.get_memory_stats() 
                if self.zero_copy_stream_manager else {}
            ),
        }
