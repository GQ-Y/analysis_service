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
        # 基础任务处理器功能（从TaskProcessor移植）
        self.task_manager = task_manager
        self.running_tasks = {}
        self.task_threads = {}
        self.result_handlers = {}
        self.stop_events = {}
        self.pause_events = {}
        self.stream_subscribers = {}
        self.redis = None  # 延迟初始化Redis

        # 预览相关
        self.preview_frames = {}  # 存储最新的分析结果帧，用于预览

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

        normal_logger.info("零拷贝任务处理器初始化完成（唯一任务处理器）")
    
    async def initialize(self):
        """初始化零拷贝任务处理器"""
        # 基础任务处理器初始化（从TaskProcessor移植）
        import os
        from core.redis_manager import RedisManager
        from shared.utils.app_state import app_state_manager

        # 确保日志目录存在
        os.makedirs("logs", exist_ok=True)

        # 设置 FFmpeg 日志级别环境变量，只显示错误信息
        os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"  # 禁用颜色输出
        os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "error"

        # 初始化 Redis 实例
        self.redis = RedisManager()

        # 使用全局的零拷贝流管理器实例，而不是创建新的
        self.zero_copy_stream_manager = app_state_manager.get_service("stream_manager")
        if not self.zero_copy_stream_manager:
            normal_logger.error("无法获取全局流管理器实例")
            return False

        normal_logger.info("零拷贝任务处理器初始化完成（唯一任务处理器）")
        return True

    async def shutdown(self):
        """关闭任务处理器"""
        # 停止所有运行中的任务
        for task_id in list(self.running_tasks.keys()):
            await self.stop_task_zero_copy(task_id)

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
            
            # 从任务配置中提取分析器
            analyzer = task_config.get("analyzer")

            # 保存任务信息
            self.running_tasks[task_id] = {
                "thread": thread,
                "config": task_config,
                "analyzer": analyzer,  # 保存分析器引用
                "start_time": time.time(),
                "status": "PROCESSING",
                "stream_id": stream_id,
                "zero_copy_enabled": True,
            }
            self.task_threads[task_id] = thread
            
            # 异步启动结果处理器（不阻塞返回）
            asyncio.ensure_future(self._handle_results(task_id, result_queue))

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

    async def _handle_results(self, task_id: str, result_queue: queue.Queue) -> None:
        """
        处理分析结果队列

        Args:
            task_id: 任务ID
            result_queue: 结果队列
        """
        normal_logger.info(f"启动结果处理器: {task_id}")

        try:
            while True:
                try:
                    # 检查任务是否应该停止
                    if task_id in self.stop_events and self.stop_events[task_id].is_set():
                        normal_logger.info(f"结果处理器收到停止信号: {task_id}")
                        break

                    # 从队列中获取结果（非阻塞）
                    try:
                        result = result_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue

                    # 处理结果
                    await self._process_analysis_result(task_id, result)

                    # 标记任务完成
                    result_queue.task_done()

                except Exception as e:
                    exception_logger.exception(f"处理结果时出错: {task_id}, {str(e)}")
                    continue

        except Exception as e:
            exception_logger.exception(f"结果处理器异常: {task_id}, {str(e)}")
        finally:
            normal_logger.info(f"结果处理器结束: {task_id}")

    async def _process_analysis_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """
        处理单个分析结果

        Args:
            task_id: 任务ID
            result: 分析结果
        """
        try:
            # 获取任务配置
            task_info = self.running_tasks.get(task_id)
            if not task_info:
                normal_logger.warning(f"任务信息不存在: {task_id}")
                return

            task_config = task_info.get("config", {})

            # 更新预览帧（如果需要）
            if result.get("analysis_data"):
                self.preview_frames[task_id] = result

            # 保存结果到Redis（如果配置了）
            if task_config.get("save_result", False):
                # 异步保存，不阻塞主流程
                asyncio.ensure_future(self._save_result_to_redis(task_id, result))
                # 异步保存到数据库
                asyncio.ensure_future(self._save_result_to_database(task_id, result))

            # 保存图像（如果配置了）
            if task_config.get("save_images", False):
                # 异步保存图像，不阻塞主流程
                asyncio.ensure_future(self._save_analysis_image(task_id, result))

            # 发送回调（如果配置了）
            if task_config.get("enable_callback", False):
                await self._send_result_callback(task_id, result)

            # 更新视频服务（如果存在）
            await self._update_video_service(task_id, result)

        except Exception as e:
            exception_logger.exception(f"处理分析结果失败: {task_id}, {str(e)}")

    async def _save_result_to_redis(self, task_id: str, result: Dict[str, Any]) -> None:
        """保存结果到Redis"""
        try:
            if not self.redis:
                return

            # 构建Redis键
            redis_key = f"analysis_result:{task_id}:{result.get('frame_index', 0)}"

            # 保存结果
            await self.redis.setex(
                redis_key,
                3600,  # 1小时过期
                json.dumps(result, default=str)
            )

        except Exception as e:
            exception_logger.exception(f"保存结果到Redis失败: {task_id}, {str(e)}")

    async def _save_result_to_database(self, task_id: str, result: Dict[str, Any]) -> None:
        """保存结果到数据库（异步，不阻塞主流程）"""
        try:
            # 在线程池中执行数据库操作，避免阻塞主线程
            import asyncio
            import concurrent.futures

            def _sync_save_to_db():
                try:
                    from models.database import AnalysisResult
                    from shared.utils.database import get_db_session
                    from datetime import datetime
                    import json

                    # 获取数据库会话
                    db_session = get_db_session()
                    if not db_session:
                        normal_logger.warning("无法获取数据库会话，跳过数据库保存")
                        return

                    # 提取分析数据
                    analysis_data = result.get("analysis_data", {})
                    frame_metadata = result.get("frame_metadata", {})

                    # 构建分析结果记录
                    analysis_result = AnalysisResult(
                        task_id=int(task_id) if task_id.isdigit() else 0,  # 转换为整数
                        subtask_id=int(task_id) if task_id.isdigit() else 0,  # 暂时使用相同值
                        status=1,  # 已完成
                        progress=100,
                        timestamp=int(result.get("timestamp", datetime.now().timestamp())),
                        frame_id=result.get("frame_index", 0),
                        objects=json.dumps(analysis_data.get("detections", [])),
                        frame_info=json.dumps(frame_metadata),
                        image_results=json.dumps(analysis_data.get("image_results", {})),
                        image_path=result.get("image_path"),
                        analysis_info=json.dumps({
                            "processing_time": result.get("processing_time", 0),
                            "inference_time": analysis_data.get("inference_time", 0),
                            "model_info": analysis_data.get("model_info", {})
                        }),
                        scene_understanding=json.dumps(analysis_data.get("scene_understanding", {}))
                    )

                    # 保存到数据库
                    db_session.add(analysis_result)
                    db_session.commit()

                    normal_logger.debug(f"分析结果已保存到数据库: task_id={task_id}, frame_id={result.get('frame_index', 0)}")

                except Exception as e:
                    exception_logger.exception(f"保存结果到数据库失败: {task_id}, {str(e)}")
                    if 'db_session' in locals():
                        db_session.rollback()

            # 在线程池中异步执行数据库操作
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, _sync_save_to_db)

        except Exception as e:
            exception_logger.exception(f"异步保存数据库操作失败: {task_id}, {str(e)}")

    async def _save_analysis_image(self, task_id: str, result: Dict[str, Any]) -> None:
        """保存分析图像（异步，不阻塞主流程）"""
        try:
            # 在线程池中执行文件操作，避免阻塞主线程
            import asyncio
            import concurrent.futures

            def _sync_save_image():
                try:
                    # 检查是否有图像数据
                    image_results = result.get("image_results")
                    if not image_results or not isinstance(image_results, dict):
                        return

                    annotated = image_results.get("annotated")
                    if not annotated or not isinstance(annotated, dict):
                        return

                    base64_data = annotated.get("base64")
                    if not base64_data:
                        return

                    # 解码Base64图像数据
                    import base64
                    import os
                    from datetime import datetime

                    image_bytes = base64.b64decode(base64_data)

                    # 构建保存路径
                    current_date_str = datetime.now().strftime("%Y%m%d")
                    frame_id = result.get("frame_id", 0)
                    timestamp = result.get("timestamp", int(datetime.now().timestamp()))

                    # 创建保存目录
                    save_dir = os.path.join("temp", "analysis_results", task_id, current_date_str)
                    os.makedirs(save_dir, exist_ok=True)

                    # 生成文件名
                    filename = f"{timestamp}_{frame_id}.jpg"
                    full_path = os.path.join(save_dir, filename)

                    # 保存图像文件
                    with open(full_path, "wb") as f:
                        f.write(image_bytes)

                    # 更新结果中的图像路径
                    result["image_path"] = os.path.join("analysis_results", task_id, current_date_str, filename)

                    normal_logger.debug(f"保存分析图像成功: {full_path}")

                except Exception as e:
                    exception_logger.exception(f"保存分析图像失败: {task_id}, {str(e)}")

            # 在线程池中异步执行文件操作
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, _sync_save_image)

        except Exception as e:
            exception_logger.exception(f"异步保存图像操作失败: {task_id}, {str(e)}")

    async def _send_result_callback(self, task_id: str, result: Dict[str, Any]) -> None:
        """发送结果回调"""
        try:
            # 这里可以实现回调发送逻辑
            # 暂时跳过，因为需要与现有的回调系统集成
            pass

        except Exception as e:
            exception_logger.exception(f"发送结果回调失败: {task_id}, {str(e)}")

    async def _update_video_service(self, task_id: str, result: Dict[str, Any]) -> None:
        """更新视频服务"""
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
            task_id = task_config.get("task_id")
            if not task_id:
                normal_logger.error("任务配置中缺少task_id")
                return None

            # 从运行任务中获取分析器
            task_info = self.running_tasks.get(task_id)
            if not task_info:
                normal_logger.error(f"任务信息不存在: {task_id}")
                return None

            analyzer = task_info.get("analyzer")
            if not analyzer:
                normal_logger.error(f"任务中没有分析器: {task_id}")
                return None

            # 确保分析器已加载模型
            if hasattr(analyzer, 'is_model_loaded') and not analyzer.is_model_loaded():
                normal_logger.info(f"等待分析器模型加载: {task_id}")
                # 等待模型加载完成
                max_wait = 30  # 最多等待30秒
                wait_time = 0
                while not analyzer.is_model_loaded() and wait_time < max_wait:
                    await asyncio.sleep(0.1)
                    wait_time += 0.1

                if not analyzer.is_model_loaded():
                    normal_logger.error(f"分析器模型加载超时: {task_id}")
                    return None

                normal_logger.info(f"分析器模型加载完成: {task_id}")

            return analyzer

        except Exception as e:
            exception_logger.exception(f"获取分析器失败: {str(e)}")
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
        停止零拷贝任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否停止成功
        """
        try:
            # 检查任务是否存在
            if task_id not in self.running_tasks:
                normal_logger.warning(f"零拷贝任务不存在: {task_id}")
                return False

            # 设置停止事件
            if task_id in self.stop_events:
                self.stop_events[task_id].set()

            # 等待线程结束
            if task_id in self.task_threads:
                thread = self.task_threads[task_id]
                if thread.is_alive():
                    thread.join(timeout=5.0)

            # 取消结果处理器
            if task_id in self.result_handlers:
                result_handler = self.result_handlers[task_id]
                if not result_handler.done():
                    result_handler.cancel()

            # 取消零拷贝流订阅
            if task_id in self._zero_copy_tasks:
                zero_copy_task_info = self._zero_copy_tasks[task_id]
                stream_id = zero_copy_task_info.get("stream_id")

                if stream_id and self.zero_copy_stream_manager:
                    # 取消零拷贝流订阅
                    await self.zero_copy_stream_manager.unsubscribe_stream_zero_copy(stream_id, task_id)

                # 清理零拷贝任务信息
                del self._zero_copy_tasks[task_id]

            # 清理批处理配置
            if task_id in self._batch_configs:
                del self._batch_configs[task_id]

            # 更新任务状态
            if task_id in self.running_tasks:
                from core.task_management.utils.status import TaskStatus
                self.running_tasks[task_id]["status"] = TaskStatus.STOPPED
                self.running_tasks[task_id]["end_time"] = time.time()

            # **重要：更新TaskManager中的任务状态为STOPPED**
            if self.task_manager:
                from core.task_management.utils.status import TaskStatus
                self.task_manager.update_task_status(task_id, TaskStatus.STOPPED)
                normal_logger.info(f"TaskManager中任务状态已更新为STOPPED: {task_id}")

            # 清理资源
            if task_id in self.stop_events:
                del self.stop_events[task_id]
            if task_id in self.pause_events:
                del self.pause_events[task_id]
            if task_id in self.task_threads:
                del self.task_threads[task_id]
            if task_id in self.result_handlers:
                del self.result_handlers[task_id]
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

            normal_logger.info(f"零拷贝任务已停止: {task_id}")
            return True

        except Exception as e:
            exception_logger.exception(f"停止零拷贝任务失败: {task_id}, {str(e)}")
            return False
