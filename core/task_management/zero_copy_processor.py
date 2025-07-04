"""
零拷贝任务处理器
重构TaskProcessor支持帧引用传递和批量处理优化
"""
import asyncio
import time
import threading
import queue
import json
from typing import Dict, Optional, Any, List
import traceback

# 移除普通TaskProcessor的导入，因为已经被完全替换
from ..frame.frame_reference import FrameReference
from ..memory.memory_pool import MemoryPool
from .stream.zero_copy_manager import ZeroCopyStreamManager
from core.timeline_architecture.timeline_manager import TimelineManager
from core.timeline_architecture.processor_module import ProcessorModule
from core.timeline_architecture.result_module import ResultModule # Added import # Added import

# 使用项目现有的日志系统
try:
    from shared.utils.logger import get_normal_logger, get_exception_logger, get_analysis_logger
    normal_logger = get_normal_logger(__name__)
    exception_logger = get_exception_logger(__name__)
    analysis_logger = get_analysis_logger()
except ImportError:
    import logging
    normal_logger = logging.getLogger(__name__)
    exception_logger = logging.getLogger(__name__)
    analysis_logger = logging.getLogger(__name__)


class ZeroCopyTaskProcessor:
    """
    零拷贝任务处理器（唯一的任务处理器实现）
    完全替换普通任务处理器，支持帧引用传递和批量处理
    """

    def __init__(self, task_manager, memory_pool: MemoryPool):
        """
        初始化零拷贝任务处理器

        Args:
            task_manager: 任务管理器
            memory_pool: 内存池实例
        """
        # 保存任务管理器引用
        self.task_manager = task_manager
        # 零拷贝相关组件
        self.memory_pool = memory_pool
        self.zero_copy_stream_manager = None
        self.timeline_manager: Optional[TimelineManager] = None
        self.processor_module: Optional[ProcessorModule] = None
        self.result_module: Optional[ResultModule] = None

        # 移除旧的属性，这些属性现在由TimelineManager, ProcessorModule, ResultModule管理
        # self.running_tasks = {}
        # self.task_threads = {}
        # self.result_handlers = {}
        # self.stop_events = {}
        # self.pause_events = {}
        # self.stream_subscribers = {}
        # self.redis = None # RedisManager现在由ResultModule管理
        # self.preview_frames = {}
        # self._zero_copy_tasks = {}
        # self._batch_configs = {}
        # self._zero_copy_stats = {}
        # self._last_stats_log_time = time.time()
        # self._stats_log_interval = 60 # 默认60秒记录一次
        # self._result_handler_tasks = {}

        normal_logger.info("零拷贝任务处理器初始化完成（唯一任务处理器）")
    
    async def initialize(self):
        """初始化零拷贝任务处理器"""
        # 基础任务处理器初始化（从TaskProcessor移植）
        import os
        from core.redis_manager import RedisManager
        from shared.utils.app_state import app_state_manager

        # 确保日志目录存在 (由 LoggingConfig 处理)
        # os.makedirs("logs", exist_ok=True)

        # 设置 FFmpeg 日志级别环境变量 (由 ZLMediaKit 或 Stream 模块处理)
        # os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"  # 禁用颜色输出
        # os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "error"

        # 初始化 Redis 实例 (现在由 ResultModule 内部处理)

        # 使用全局的零拷贝流管理器实例，而不是创建新的
        self.zero_copy_stream_manager = app_state_manager.get_service("stream_manager")
        if not self.zero_copy_stream_manager:
            normal_logger.error("无法获取全局流管理器实例")
            return False

        self.timeline_manager = self.zero_copy_stream_manager.timeline_manager
        if not self.timeline_manager:
            normal_logger.error("无法获取全局时间轴管理器实例")
            return False
        await self.timeline_manager.start() # Start the TimelineManager

        # 创建内存模块实例
        from core.timeline_architecture.memory_module import MemoryModule
        self.memory_module = MemoryModule(max_memory_mb=1024, cleanup_interval=0.5)
        await self.memory_module.start()

        self.processor_module = ProcessorModule(
            timeline_manager=self.timeline_manager, # Pass timeline_manager
            memory_module=self.memory_module, # Pass memory_module
            target_cpu_utilization=0.8 # Use a default or configurable value
        )
        await self.processor_module.start() # Start the ProcessorModule

        self.result_module = ResultModule(
            # Redis 连接信息现在由 ResultModule 内部管理或从配置中获取
            # redis_host=...,
            # redis_port=...,
            # redis_db=...,
            result_queue_prefix="analysis_results" # Use a default or configurable value
        )
        await self.result_module.start() # Start the ResultModule

        normal_logger.info("零拷贝任务处理器初始化完成（唯一任务处理器）")
        return True

    async def shutdown(self):
        """关闭任务处理器"""
        # 任务停止现在由 ProcessorModule 统一管理，无需在此处迭代 running_tasks
        # for task_id in list(self.running_tasks.keys()):
        #     await self.stop_task_zero_copy(task_id)

        # 按正确顺序停止组件：先停止处理器，再停止时间轴管理器
        if self.processor_module:
            await self.processor_module.stop()

        if self.result_module:
            await self.result_module.stop()

        if hasattr(self, 'memory_module') and self.memory_module:
            await self.memory_module.stop()

        # 最后停止时间轴管理器，确保所有使用锁的任务都已停止
        if hasattr(self, 'timeline_manager') and self.timeline_manager:
            await self.timeline_manager.stop()

        normal_logger.info("零拷贝任务处理器已关闭")
        return True

    async def start_stream_analysis(self, task_id: str, task_config: Dict[str, Any]) -> bool:
        """
        启动流分析任务（统一入口，替换普通任务的start_stream_analysis）

        Args:
            task_id: 任务ID
            task_config: 任务配置

        Returns:
            bool: 是否启动成功
        """
        # 直接使用零拷贝模式启动任务
        return await self.start_task_zero_copy(task_id, task_config)

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
            
            stream_id = self._extract_stream_id(task_config)
            if not stream_id:
                normal_logger.error(f"任务 {task_id} 缺少流ID，无法启动零拷贝模式")
                return False

            # 订阅零拷贝流 (现在帧将通过 TimelineManager 获取)
            success, _ = await self.zero_copy_stream_manager.subscribe_stream_zero_copy(
                stream_id, task_id, task_config
            )
            
            if not success:
                normal_logger.error(f"订阅零拷贝流失败: {stream_id}")
                return False
            
            # ProcessorModule 管理处理器池，不需要注册具体任务
            # 任务处理由 TimelineManager 和流订阅机制自动处理

            # 任务启动和状态管理现在由 ProcessorModule 负责
            # self.running_tasks[task_id] = {
            #     "task_config": task_config,
            #     "status": TaskStatus.RUNNING,
            #     "start_time": time.time(),
            #     "analyzer": analyzer # 存储分析器实例
            # }

            normal_logger.info(f"零拷贝任务启动成功: {task_id}")
            return True

        except Exception as e:
            exception_logger.exception(f"启动零拷贝任务失败: {task_id}, {str(e)}")
            return False
    
    async def process_stream_worker_zero_copy(self, 
                                            task_id: str, 
                                            task_config: Dict[str, Any]) -> None:
        """
        零拷贝流处理工作器 (现在由 ProcessorModule 管理)
        
        Args:
            task_id: 任务ID
            task_config: 任务配置
        """
        normal_logger.info(f"零拷贝流处理工作器 (ProcessorModule) 启动: {task_id}")

        try:
            # This worker is now primarily a placeholder or for future direct task management
            # The actual frame processing loop is managed by ProcessorModule's AnalysisProcessor
            # This method might be removed or refactored depending on how task lifecycle is managed
            # For now, it will just log and exit, as ProcessorModule handles the loop.
            normal_logger.info(f"任务 {task_id} 的帧处理已委托给 ProcessorModule")
            # Keep the thread alive for a short period or until explicitly stopped if needed
            while True:
                await asyncio.sleep(1) # Keep the thread alive

        except asyncio.CancelledError:
            normal_logger.info(f"零拷贝流处理工作器 {task_id} 被取消")
        except Exception as e:
            exception_logger.exception(f"零拷贝流处理工作器异常: {task_id}, {str(e)}")
        finally:
            normal_logger.info(f"零拷贝流处理工作器 {task_id} 结束")
    
    async def _process_single_frame_reference(self,
                                            task_id: str,
                                            frame_ref: FrameReference,
                                            analyzer: Any,
                                            frame_index: int) -> None:
        """
        处理单个帧引用

        Args:
            task_id: 任务ID
            frame_ref: 帧引用
            analyzer: 分析器
            frame_index: 帧索引
        """
        try:
            start_time = time.time()

            # 获取帧数据（零拷贝）
            frame_data = frame_ref.get_data()
            if frame_data is None:
                normal_logger.warning(f"任务 {task_id}: 无法获取帧数据")
                return

            # 添加分析日志：分析器读取帧数据
            metadata = frame_ref.get_metadata()
            analysis_logger.info(f"[帧读取] 任务 {task_id} 分析器成功读取第 {frame_index} 帧数据, "
                               f"帧ID: {metadata.frame_id}, 内存块ID: {metadata.memory_block_ref}, "
                               f"帧大小: {metadata.width}x{metadata.height}, 时间戳: {metadata.timestamp}")

            # 执行分析
            analysis_data = await analyzer.process_video_frame(
                frame_data, frame_index=frame_index
            )

            # 记录处理时间
            processing_time = time.time() - start_time

            # 构建结果
            result = {
                "task_id": task_id,
                "frame_index": frame_index,
                "frame_metadata": metadata.to_dict(),
                "analysis_data": analysis_data,
                "processing_time": processing_time,
                "timestamp": time.time(),
            }

            # 放入结果模块
            if self.result_module:
                await self.result_module.add_result(result)

        except Exception as e:
            exception_logger.exception(f"处理帧引用异常: {task_id}, {str(e)}")
        finally:
            # 释放帧引用
            frame_ref.release()

    async def _process_frame_batch(self,
                                 task_id: str,
                                 frame_refs: List[FrameReference],
                                 analyzer: Any) -> None:
        """
        批量处理帧引用

        Args:
            task_id: 任务ID
            frame_refs: 帧引用列表
            analyzer: 分析器
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
                # 放入结果模块
                if self.result_module:
                    await self.result_module.add_result(result)

        except Exception as e:
            exception_logger.exception(f"批量处理帧引用异常: {task_id}, {str(e)}")
        finally:
            # 释放所有帧引用
            for frame_ref in frame_refs:
                frame_ref.release()

    async def _handle_results(self, task_id: str, result_queue: queue.Queue) -> None:
        """
        处理分析结果队列 (现在由 ResultModule 处理)

        Args:
            task_id: 任务ID
            result_queue: 结果队列
        """
        normal_logger.info(f"结果处理器 (ResultModule) 启动: {task_id}")

        try:
            # This handler is now primarily a placeholder or for future direct result management
            # The actual result processing loop is managed by ResultModule
            # This method might be removed or refactored depending on how result lifecycle is managed.
            normal_logger.info(f"任务 {task_id} 的结果处理已委托给 ResultModule")
            # Keep the thread alive for a short period or until explicitly stopped if needed
            while True:
                await asyncio.sleep(1) # Keep the thread alive

        except asyncio.CancelledError:
            normal_logger.info(f"结果处理器 {task_id} 被取消")
        except Exception as e:
            exception_logger.exception(f"结果处理器异常: {task_id}, {str(e)}")
        finally:
            normal_logger.info(f"结果处理器 {task_id} 结束")

    async def _process_analysis_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """
        处理单个分析结果 (现在委托给 ResultModule)

        Args:
            task_id: 任务ID
            result: 分析结果
        """
        try:
            # 更新预览帧（如果需要）
            if result.get("analysis_data"):
                self.preview_frames[task_id] = result

            # 将结果添加到 ResultModule
            if self.result_module:
                await self.result_module.add_result(result)
            else:
                normal_logger.warning(f"ResultModule 未初始化，无法处理结果: {task_id}")

        except Exception as e:
            exception_logger.exception(f"处理分析结果失败: {task_id}, {str(e)}")

    

    

    

    

    async def _update_video_service(self, task_id: str, result: Dict[str, Any]) -> None:
        """更新视频服务 (现在由 ResultModule 处理)"""
        try:
            # 获取视频服务
            from shared.utils.app_state import app_state_manager
            video_service = app_state_manager.get_service("video_service")

            if video_service and result.get("analysis_data"):
                # 转换结果格式以适配视频服务
                analysis_result = {
                    "detections": result["analysis_data"].get("detections", []),
                    "frame_info": result.get("frame_metadata", {}),
                    "timestamp": result.get("timestamp"),
                    "processing_time": result.get("processing_time"),
                }

                # 更新视频服务
                video_service.update_analysis_result(task_id, analysis_result)

        except Exception as e:
            exception_logger.exception(f"更新视频服务失败: {task_id}, {str(e)}")

    async def _get_analyzer(self, task_config: Dict[str, Any]):
        """获取分析器"""
        try:
            # 优先从任务配置中直接获取分析器
            analyzer = task_config.get("analyzer")
            if not analyzer:
                normal_logger.error("任务配置中缺少analyzer")
                return None

            # 确保分析器已加载模型
            if hasattr(analyzer, 'is_model_loaded') and not analyzer.is_model_loaded():
                normal_logger.info(f"等待分析器模型加载: {task_config.get('task_id')}")
                # 等待模型加载完成
                max_wait = 30  # 最多等待30秒
                wait_time = 0
                while not analyzer.is_model_loaded() and wait_time < max_wait:
                    await asyncio.sleep(0.1)
                    wait_time += 0.1

                if not analyzer.is_model_loaded():
                    normal_logger.error(f"分析器模型加载超时: {task_config.get('task_id')}")
                    return None

                normal_logger.info(f"分析器模型加载完成: {task_config.get('task_id')}")

            return analyzer

        except Exception as e:
            exception_logger.exception(f"获取分析器失败: {str(e)}")
            return None

    def get_zero_copy_stats(self) -> Dict[str, Any]:
        """
        获取零拷贝统计信息 (现在从各个模块获取)
        
        Returns:
            Dict[str, Any]: 零拷贝统计信息
        """
        return {
            "timeline_stats": self.timeline_manager.get_statistics() if self.timeline_manager else {},
            "processor_stats": self.processor_module.get_processor_statistics() if self.processor_module else {},
            "result_stats": self.result_module.get_result_statistics() if self.result_module else {},
            "stream_manager_stats": (
                self.zero_copy_stream_manager.get_memory_stats()
                if self.zero_copy_stream_manager else {}
            ),
        }

    def _extract_stream_id(self, task_config: Dict[str, Any]) -> Optional[str]:
        """
        从任务配置中提取流ID（兼容多种参数格式）

        Args:
            task_config: 任务配置

        Returns:
            Optional[str]: 流ID
        """
        # 优先级顺序：stream_id > video_id > camera_id
        return (task_config.get("stream_id") or
                task_config.get("video_id") or
                task_config.get("camera_id"))

    def _extract_stream_url(self, task_config: Dict[str, Any]) -> Optional[str]:
        """
        从任务配置中提取流URL（兼容多种参数格式）

        Args:
            task_config: 任务配置

        Returns:
            Optional[str]: 流URL
        """
        # 优先级顺序：stream_url > url > rtsp_url
        return (task_config.get("stream_url") or
                task_config.get("url") or
                task_config.get("rtsp_url"))

    async def stop_task(self, task_id: str) -> bool:
        """
        停止任务（统一入口，替换普通任务的stop_task）

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否停止成功
        """
        # 直接使用零拷贝模式停止任务
        return await self.stop_task_zero_copy(task_id)

    async def stop_task_zero_copy(self, task_id: str) -> bool:
        """
        停止零拷贝任务 (现在委托给 ProcessorModule)

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否停止成功
        """
        try:
            normal_logger.info(f"[停止任务] 接收到停止零拷贝任务请求: {task_id}")

            # 由于任务管理现在由其他模块负责，我们直接尝试停止流订阅
            # 使用task_id作为stream_id（根据代码逻辑，它们通常是相同的）
            stream_id = task_id

            # 停止流订阅
            if self.zero_copy_stream_manager:
                try:
                    # 取消订阅流
                    await self.zero_copy_stream_manager.unsubscribe_stream_zero_copy(stream_id, task_id)
                    normal_logger.info(f"已取消流订阅: stream_id={stream_id}, task_id={task_id}")
                except Exception as e:
                    normal_logger.error(f"取消流订阅失败: {e}")

            # **重要：更新TaskManager中的任务状态为STOPPED**
            if self.task_manager:
                from core.task_management.utils.status import TaskStatus
                self.task_manager.update_task_status(task_id, TaskStatus.STOPPED)
                normal_logger.info(f"TaskManager中任务状态已更新为STOPPED: {task_id}")

            normal_logger.info(f"零拷贝任务已停止: {task_id}")
            return True

        except Exception as e:
            exception_logger.exception(f"停止零拷贝任务失败: {task_id}, {str(e)}")
            return False

    def _on_result_task_done(self, task_id: str, task: asyncio.Task) -> None:
        """
        处理结果任务完成后的清理逻辑 (现在由 ProcessorModule 和 ResultModule 管理)
        此方法现在仅作为占位符，未来可能移除或重构。
        """
        normal_logger.info(f"_on_result_task_done 被调用，但清理逻辑已委托给其他模块: {task_id}")
