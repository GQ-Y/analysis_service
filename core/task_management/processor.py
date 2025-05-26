"""
任务处理器
负责执行分析任务
"""
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import threading
import time
import queue
import cv2
import numpy as np
from datetime import datetime
import uuid
import logging
import traceback
import os
import sys
import io
import contextlib
import json
import base64

from core.config import settings
from shared.utils.logger import get_normal_logger, get_exception_logger, get_test_logger
from core.task_management.utils.status import TaskStatus
from core.task_management.stream.status import StreamStatus, StreamHealthStatus

# 延迟导入stream_manager，避免循环导入
from core.redis_manager import RedisManager
from core.analyzer.analyzer_factory import AnalyzerFactory

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()

class TaskProcessor:
    """任务处理器"""

    def __init__(self, task_manager):
        """
        初始化任务处理器

        Args:
            task_manager: 任务管理器
        """
        self.task_manager = task_manager
        self.running_tasks = {}
        self.task_threads = {}
        self.result_handlers = {}
        self.stop_events = {}
        self.pause_events = {}  # 新增：任务暂停事件
        self.stream_subscribers = {}  # 新增：任务ID到流ID的映射
        self.redis = None  # 延迟初始化Redis

        # 预览相关
        self.preview_frames = {}  # 存储最新的分析结果帧，用于预览

    async def initialize(self):
        """初始化任务处理器"""
        # 确保日志目录存在
        os.makedirs("logs", exist_ok=True)

        # 设置 FFmpeg 日志级别环境变量，只显示错误信息
        os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"  # 禁用颜色输出
        # 不设置为 quiet，而是设置为 error，这样可以捕获错误信息
        os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "error"

        # 初始化 Redis 实例
        self.redis = RedisManager()

        normal_logger.info("任务处理器初始化完成")
        return True

    async def shutdown(self):
        """关闭任务处理器"""
        # 停止所有运行中的任务
        for task_id in list(self.running_tasks.keys()):
            await self.stop_task(task_id)

        normal_logger.info("任务处理器已关闭")
        return True

    async def start_stream_analysis(self, task_id: str, task_config: Dict[str, Any]) -> bool:
        """
        启动流分析任务

        Args:
            task_id: 任务ID
            task_config: 任务配置

        Returns:
            bool: 是否启动成功
        """
        method_start_time = time.perf_counter()
        normal_logger.info(f"TaskProcessor.start_stream_analysis CALLED: task_id={task_id}")
        try:
            # 检查任务是否已存在
            if task_id in self.running_tasks:
                normal_logger.warning(f"任务已存在: {task_id}")
                method_end_time = time.perf_counter()
                normal_logger.info(f"TaskProcessor.start_stream_analysis RETURNED (already running): task_id={task_id}, 耗时: {(method_end_time - method_start_time) * 1000:.2f} ms")
                return False

            # 创建停止和暂停事件
            stop_event = threading.Event()
            pause_event = threading.Event()  # 新增：暂停事件
            self.stop_events[task_id] = stop_event
            self.pause_events[task_id] = pause_event  # 新增：保存暂停事件

            # 创建结果队列
            result_queue = queue.Queue()

            # 订阅流状态变化
            # 获取流ID
            stream_id = task_config.get("stream_id", "")
            if stream_id:
                self.stream_subscribers[task_id] = stream_id
                # 使用桥接器注册任务与流的关系
                from core.task_management.stream import stream_task_bridge
                stream_task_bridge.register_task_stream(task_id, stream_id)

            # 创建并启动任务线程 - 使用 asyncio.to_thread 运行异步函数
            thread = threading.Thread(
                target=asyncio.run,
                args=(self.process_stream_worker(task_id, task_config, result_queue, stop_event, pause_event),),
                daemon=True
            )
            thread.start()

            # 保存任务信息
            self.running_tasks[task_id] = {
                "thread": thread,
                "config": task_config,
                "start_time": time.time(),
                "status": TaskStatus.PROCESSING,
                "stream_id": stream_id  # 保存流ID
            }
            self.task_threads[task_id] = thread

            # 创建并启动结果处理器
            result_handler = asyncio.create_task(self._handle_results(task_id, result_queue))
            self.result_handlers[task_id] = result_handler

            normal_logger.info(f"流处理进程已启动: {task_id}")
            method_end_time = time.perf_counter()
            normal_logger.info(f"TaskProcessor.start_stream_analysis RETURNED (success): task_id={task_id}, 耗时: {(method_end_time - method_start_time) * 1000:.2f} ms")
            return True

        except Exception as e:
            exception_logger.exception(f"启动流分析任务失败 (详细错误): {task_id}") # Log with traceback
            method_end_time = time.perf_counter()
            normal_logger.error(f"TaskProcessor.start_stream_analysis RETURNED (exception): task_id={task_id}, 耗时: {(method_end_time - method_start_time) * 1000:.2f} ms. Error: {str(e)}")
            return False

    async def stop_task(self, task_id: str) -> bool:
        """
        停止任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否停止成功
        """
        try:
            # 检查任务是否存在
            if task_id not in self.running_tasks:
                normal_logger.warning(f"任务不存在: {task_id}")
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

            # 取消流订阅
            if task_id in self.stream_subscribers:
                stream_id = self.stream_subscribers[task_id]
                # 使用桥接器取消注册任务与流的关系
                from core.task_management.stream import stream_task_bridge
                stream_task_bridge.unregister_task_stream(task_id)
                # 取消订阅流
                from core.task_management.stream import stream_manager
                await stream_manager.unsubscribe_stream(stream_id, task_id)
                del self.stream_subscribers[task_id]

            # 更新任务状态
            if task_id in self.running_tasks:
                self.running_tasks[task_id]["status"] = TaskStatus.STOPPED
                self.running_tasks[task_id]["end_time"] = time.time()

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

            normal_logger.info(f"任务已停止: {task_id}")
            return True

        except Exception as e:
            normal_logger.error(f"停止任务失败: {str(e)}")
            return False

    async def pause_task(self, task_id: str) -> bool:
        """
        暂停任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否暂停成功
        """
        try:
            # 检查任务是否存在
            if task_id not in self.running_tasks:
                normal_logger.warning(f"任务不存在: {task_id}")
                return False

            # 设置暂停事件
            if task_id in self.pause_events:
                self.pause_events[task_id].set()
                normal_logger.info(f"任务暂停事件已设置: {task_id}")
                return True
            else:
                normal_logger.warning(f"任务暂停事件不存在: {task_id}")
                return False

        except Exception as e:
            normal_logger.error(f"暂停任务失败: {str(e)}")
            return False

    async def resume_task(self, task_id: str) -> bool:
        """
        恢复任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否恢复成功
        """
        try:
            # 检查任务是否存在
            if task_id not in self.running_tasks:
                normal_logger.warning(f"任务不存在: {task_id}")
                return False

            # 清除暂停事件
            if task_id in self.pause_events:
                self.pause_events[task_id].clear()
                normal_logger.info(f"任务恢复事件已设置: {task_id}")
                return True
            else:
                normal_logger.warning(f"任务暂停事件不存在: {task_id}")
                return False

        except Exception as e:
            normal_logger.error(f"恢复任务失败: {str(e)}")
            return False

    async def process_stream_worker(self, task_id: str, task_config: Dict[str, Any], result_queue: queue.Queue,
                                   stop_event: threading.Event, pause_event: threading.Event):
        """
        流处理工作线程

        Args:
            task_id: 任务ID
            task_config: 任务配置
            result_queue: 结果队列
            stop_event: 停止事件
            pause_event: 暂停事件
        """
        try:
            # 此时接收到的 task_config 实际上是 _build_task_config() 中的 params_config
            normal_logger.info(f"工作进程 {task_id}: 开始初始化。接收到的 task_config (即params_config): {json.dumps(task_config, indent=2, ensure_ascii=False)}")

            # stream_id 和 stream_url 应该直接在 task_config (即params_config) 中
            stream_id_from_task_config = task_config.get("video_id") # _build_task_config 中stream_config用了 video_id
            if not stream_id_from_task_config: # 向后兼容或备用方案
                 stream_id_from_task_config = task_config.get("stream_id") 
                 if not stream_id_from_task_config:
                      stream_id_from_task_config = f"stream_{task_id}" # 如果都没有，则生成
            
            stream_url = task_config.get("stream_url", "")
            
            # 构建流订阅所需的 stream_config (这部分代码可以保持不变，因为它已经从 task_config 取值了)
            stream_config_for_subscription = {
                "url": stream_url,
                "rtsp_transport": task_config.get("rtsp_transport", "tcp"),
                "reconnect_attempts": task_config.get("reconnect_attempts", settings.STREAMING.reconnect_attempts),
                "reconnect_delay": task_config.get("reconnect_delay", settings.STREAMING.reconnect_delay),
                "frame_buffer_size": task_config.get("frame_buffer_size", settings.STREAMING.frame_buffer_size),
                "task_id": task_id, 
                "video_id": stream_id_from_task_config 
            }
            
            # 获取流配置 (这部分是旧的，上面的 stream_id_from_task_config 和 stream_url 已经获取了)
            # stream_id = task_config.get("stream_id", "") 
            # if not stream_id:
            # stream_id = f"stream_{task_id}"
            # normal_logger.info(f"流ID为空，使用任务ID生成流ID: {stream_id}")
            stream_id_to_use = stream_id_from_task_config # 统一使用此变量

            normal_logger.info(f"工作进程 {task_id}: 使用流ID '{stream_id_to_use}' 和 URL '{stream_url}'")

            # 订阅视频流 (使用 stream_config_for_subscription)
            from core.task_management.stream import stream_manager
            try:
                normal_logger.info(f"开始订阅流 {stream_id_to_use}，URL: {stream_url}")
                success, frame_queue = await stream_manager.subscribe_stream(stream_id_to_use, task_id, stream_config_for_subscription)
                if not success or frame_queue is None:
                    error_msg = f"订阅视频流失败: {stream_id_to_use}"
                    normal_logger.error(error_msg)
                    if hasattr(self.task_manager, 'update_task_status'):
                         self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                    return
                # normal_logger.info(f"成功订阅流 {stream_id_to_use}，等待帧数据...") # 这条日志在原位置已注释
            except Exception as e:
                error_msg = f"订阅视频流时发生异常: {str(e)}"
                normal_logger.error(error_msg)
                normal_logger.error(traceback.format_exc())
                if hasattr(self.task_manager, 'update_task_status'):
                    self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                return

            # 创建分析器
            analyzer = None
            
            # analyzer_type 来自 task_config (即params_config) 中的 subtask.type
            analyzer_type = task_config.get("subtask", {}).get("type")

            if analyzer_type:
                # 构造分析器初始化和处理所需的参数
                # 1. 从 task_config (即params_config) 中的 "analysis"
                analyzer_kwargs = task_config.get("analysis", {}).copy()
                
                # 2. 从 task_config (即params_config) 中的 "model" 获取 model_code, confidence, iou_threshold
                model_params = task_config.get("model", {})
                if "code" in model_params:
                    analyzer_kwargs["model_code"] = model_params["code"]
                if "confidence" not in analyzer_kwargs and "confidence" in model_params:
                    analyzer_kwargs["confidence"] = model_params["confidence"]
                if "iou_threshold" not in analyzer_kwargs and "iou_threshold" in model_params:
                    analyzer_kwargs["iou_threshold"] = model_params["iou_threshold"]

                # 3. 从 task_config (即params_config) 中的 "device" 获取 device
                if "device" in task_config:
                    analyzer_kwargs["device"] = task_config.get("device")

                # 4. 新增：从 task_config (即params_config) 中的 "result" 获取 return_base64，并以 "return_image" 的键名加入
                result_config = task_config.get("result", {})
                if "return_base64" in result_config:
                    analyzer_kwargs["return_image"] = result_config.get("return_base64", False)
                else:
                    analyzer_kwargs["return_image"] = task_config.get("return_base64", False)

                # DIAGNOSTIC LOG:
                normal_logger.info(f"任务 {task_id} [DIAGNOSTIC]: 准备创建分析器. Type: {analyzer_type}, Args: {json.dumps(analyzer_kwargs, indent=2)}")

                try:
                    normal_logger.info(f"任务 {task_id}: 准备创建分析器, 类型: {analyzer_type}, 收集到的参数: {analyzer_kwargs}")
                    analyzer = AnalyzerFactory.create_analyzer(
                        analyzer_type,
                        **analyzer_kwargs
                    )
                    # DIAGNOSTIC LOG:
                    if analyzer:
                        normal_logger.info(f"任务 {task_id} [DIAGNOSTIC]: 分析器创建成功. Analyzer object: {analyzer}")
                    else:
                        normal_logger.error(f"任务 {task_id} [DIAGNOSTIC]: 分析器创建失败 (AnalyzerFactory.create_analyzer 返回 None).")
    
                    if analyzer:
                        # 等待模型加载完成
                        if hasattr(analyzer, 'load_model') and hasattr(analyzer, 'loaded') and not analyzer.loaded: # 检查是否定义了load_model, loaded，且尚未加载
                            model_to_load = analyzer_kwargs.get('model_code') or getattr(analyzer, 'model_code', '未知模型')
                            normal_logger.info(f"任务 {task_id}: 分析器 {analyzer.__class__.__name__} 正在等待模型 '{model_to_load}' 加载...")
                            try:
                                # 调用分析器自身的 load_model 方法，它应该负责设置 self.loaded
                                # YOLODetectionAnalyzer.load_model 确实会调用 self.detector.load_model 并设置 self.loaded
                                loaded_successfully = await analyzer.load_model(model_to_load)
                                if loaded_successfully: # analyzer.loaded 应该已被更新
                                    normal_logger.info(f"任务 {task_id}: 模型 '{model_to_load}' 为分析器 {analyzer.__class__.__name__} 加载成功。")
                                else:
                                    # 如果 analyzer.load_model 返回 False，说明加载失败
                                    error_msg = f"任务 {task_id}: 分析器 {analyzer.__class__.__name__} 的模型 '{model_to_load}' 显式加载失败 (load_model返回False)。"
                                    normal_logger.error(error_msg)
                                    if hasattr(self.task_manager, 'update_task_status'):
                                        self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                                    return # 加载失败，则退出
                            except Exception as load_exc:
                                error_msg = f"任务 {task_id}: 分析器 {analyzer.__class__.__name__} 等待模型 '{model_to_load}' 加载时发生异常: {load_exc}"
                                normal_logger.error(error_msg)
                                normal_logger.error(traceback.format_exc())
                                if hasattr(self.task_manager, 'update_task_status'):
                                    self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                                return # 异常，则退出
                        
                        # 再次检查 loaded 状态，确保模型已成功加载
                        if hasattr(analyzer, 'loaded') and analyzer.loaded:
                            normal_logger.info(f"任务 {task_id}: 分析器 {analyzer.__class__.__name__} 创建并模型加载成功。")
                        elif hasattr(analyzer, 'loaded') and not analyzer.loaded: # 如果显式调用 load_model 后 loaded 仍为 False
                            model_code_check = analyzer_kwargs.get("model_code") or getattr(analyzer, 'model_code', '未知模型')
                            error_msg = f"任务 {task_id}: 分析器 {analyzer.__class__.__name__} 已创建，但模型 '{model_code_check}' 在尝试显式加载后仍未能加载。"
                            normal_logger.error(error_msg)
                            if hasattr(self.task_manager, 'update_task_status'):
                                self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                            return
                        elif not hasattr(analyzer, 'loaded'): # 如果分析器没有 'loaded' 属性 (理论上 BaseAnalyzer 有)
                             normal_logger.warning(f"任务 {task_id}: 分析器 {analyzer.__class__.__name__} 没有 'loaded' 属性，无法确认模型加载状态。")
                        # 之前这里有一个 elif hasattr(analyzer, 'loaded') and not analyzer.loaded: 的分支，现在由上面的逻辑覆盖
                        # else: # 这个else对应 if analyzer: 的情况，即 analyzer 为 None
                        #     normal_logger.info(f"任务 {task_id}: 分析器 {analyzer.__class__.__name__} 创建成功 (无法从属性确认模型加载状态，依赖初始化过程)。")
                    else: # analyzer is None
                        error_msg = f"任务 {task_id}: 创建分析器 {analyzer_type} 失败 (AnalyzerFactory 返回 None)."
                        normal_logger.error(error_msg)
                        if hasattr(self.task_manager, 'update_task_status'):
                            self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                        return
                except Exception as e:
                    error_msg = f"任务 {task_id}: 创建或加载分析器时发生异常: {str(e)}"
                    normal_logger.error(error_msg)
                    normal_logger.error(traceback.format_exc())
                    if hasattr(self.task_manager, 'update_task_status'):
                        self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                    return
            elif not analyzer_type:
                 normal_logger.warning(f"任务 {task_id}: 'analyzer_type' (从params_config.subtask.type获取) 未找到。任务将仅拉取帧数据，不进行分析。")
            # elif not params_from_task_config: # 这个检查现在不需要了，因为 task_config 就是 params_from_task_config
            #     normal_logger.warning(f"任务 {task_id}: 'params' 未在任务配置中找到。任务将仅拉取帧数据，不进行分析。")
    
    
            # 分析循环
            normal_logger.info(f"开始任务 {task_id} 的分析循环")
            frame_index = 0 # 初始化帧索引
            while not stop_event.is_set():
                # 检查暂停事件
                if pause_event.is_set():
                    # 任务已暂停，等待恢复
                    time.sleep(1)
                    continue

                try:
                    # 从帧队列获取帧，增加超时时间
                    frame_data = await asyncio.wait_for(frame_queue.get(), timeout=15.0)  # 将超时时间从10秒增加到15秒
                    
                    # 解包帧数据，通常是(frame, timestamp)的元组
                    if frame_data and isinstance(frame_data, tuple) and len(frame_data) >= 2:
                        frame, timestamp = frame_data[:2]
                    else:
                        normal_logger.warning(f"任务 {task_id}: 无效的帧数据格式")
                        frame, timestamp = None, time.time()

                    # 更新帧计数和状态
                    if hasattr(self, '_frame_counts'):
                        self._frame_counts[task_id] = self._frame_counts.get(task_id, 0) + 1
                        
                        # 每100帧记录一次日志
                        if self._frame_counts[task_id] % 100 == 0:
                            normal_logger.info(f"任务 {task_id}: 已处理 {self._frame_counts[task_id]} 帧")
                        
                        # 重置连续错误计数
                        if hasattr(self, '_frame_error_counts') and task_id in self._frame_error_counts:
                            self._frame_error_counts[task_id] = 0
                        
                        # 重置超时计数
                        if hasattr(self, '_frame_timeout_counts') and task_id in self._frame_timeout_counts:
                            self._frame_timeout_counts[task_id] = 0

                    if frame is None:
                        # 没有收到帧，可能是流离线
                        normal_logger.warning(f"任务 {task_id}: 未接收到视频帧")

                        # 增加错误计数
                        if not hasattr(self, '_frame_error_counts'):
                            self._frame_error_counts = {}

                        self._frame_error_counts[task_id] = self._frame_error_counts.get(task_id, 0) + 1
                        error_count = self._frame_error_counts[task_id]

                        # 如果连续错误次数过多，尝试重新订阅
                        if error_count >= 3:
                            normal_logger.warning(f"任务 {task_id}: 连续 {error_count} 次未接收到视频帧，尝试重新订阅")

                            # 检查流状态
                            from core.task_management.stream import stream_manager
                            stream_info = await stream_manager.get_stream_info(stream_id_to_use)

                            if stream_info:
                                # 尝试重新订阅
                                await stream_manager.unsubscribe_stream(stream_id_to_use, task_id)
                                success, new_frame_queue = await stream_manager.subscribe_stream(stream_id_to_use, task_id, stream_config_for_subscription)

                                if success and new_frame_queue is not None:
                                    normal_logger.info(f"成功重新订阅流 {stream_id_to_use}")
                                    frame_queue = new_frame_queue
                                    self._frame_error_counts[task_id] = 0  # 重置错误计数
                                else:
                                    normal_logger.error(f"重新订阅流 {stream_id_to_use} 失败")

                        time.sleep(1)
                        continue

                    frame_index += 1 # 增加帧索引

                    # 执行分析
                    analysis_data = None
                    if analyzer:
                        try:
                            # process_video_frame 也使用收集到的 analyzer_kwargs
                            # 确保 return_image 标志在这里被传递
                            analysis_data = await analyzer.process_video_frame(
                                frame, 
                                frame_index=frame_index, 
                                **analyzer_kwargs 
                            )
                        except Exception as analysis_exc:
                            normal_logger.error(f"任务 {task_id}: 帧 {frame_index} 分析时发生错误: {analysis_exc}")
                            normal_logger.error(traceback.format_exc())
                            analysis_data = {
                                "success": False,
                                "error": f"帧分析错误: {str(analysis_exc)}",
                                "detections": [], # 确保有空列表以防后续代码出错
                                "frame_index": frame_index,
                                "task_id": task_id,
                                "timestamp": time.time() # 使用当前时间戳
                            }
                    else:
                        # 如果没有分析器 (因为未配置或创建失败)
                        analysis_data = {
                            "success": True, # 获取帧是成功的
                            "task_id": task_id,
                            "timestamp": timestamp, # 原始帧时间戳
                            "frame_index": frame_index,
                            "frame_shape": frame.shape if frame is not None else None,
                            "message": "成功获取帧 (分析器未配置或初始化失败)"
                        }
                    

                    result_queue.put(analysis_data) # 直接放入分析结果

                except asyncio.TimeoutError:
                    normal_logger.warning(f"任务 {task_id}: 获取帧超时")

                    # 增加错误计数
                    if not hasattr(self, '_frame_timeout_counts'):
                        self._frame_timeout_counts = {}

                    self._frame_timeout_counts[task_id] = self._frame_timeout_counts.get(task_id, 0) + 1
                    timeout_count = self._frame_timeout_counts[task_id]

                    # 检查流状态
                    from core.task_management.stream import stream_manager
                    stream_info = await stream_manager.get_stream_info(stream_id_to_use)

                    if stream_info:
                        status = stream_info.get("status")
                        health = stream_info.get("health_status")
                        normal_logger.info(f"流 {stream_id_to_use} 状态: {status}, 健康状态: {health}")

                        # 根据不同情况采取不同策略
                        if status in [StreamStatus.RUNNING, StreamStatus.ONLINE]:
                            # 流状态正常但获取帧超时
                            if timeout_count >= 8:  # 提高连续超时阈值，从5次改为8次
                                normal_logger.info(f"流 {stream_id_to_use} 状态正常但连续 {timeout_count} 次获取帧超时，尝试重新订阅")
                                await stream_manager.unsubscribe_stream(stream_id_to_use, task_id)
                                
                                # 等待一段时间后再重新订阅，避免立即重新订阅
                                await asyncio.sleep(2.0)
                                
                                success, new_frame_queue = await stream_manager.subscribe_stream(stream_id_to_use, task_id, stream_config_for_subscription)

                                if success and new_frame_queue is not None:
                                    normal_logger.info(f"成功重新订阅流 {stream_id_to_use}")
                                    frame_queue = new_frame_queue
                                    self._frame_timeout_counts[task_id] = 0  # 重置超时计数
                                else:
                                    normal_logger.error(f"重新订阅流 {stream_id_to_use} 失败")
                        elif status in [StreamStatus.CONNECTING, StreamStatus.INITIALIZING]:
                            # 流正在连接中，等待
                            normal_logger.info(f"流 {stream_id_to_use} 正在连接中，等待...")
                        elif status in [StreamStatus.OFFLINE, StreamStatus.ERROR]:
                            # 流离线或错误，尝试重新连接
                            if timeout_count % 15 == 0:  # 每15次超时尝试一次重连，增加间隔
                                normal_logger.info(f"流 {stream_id_to_use} 状态异常 ({status})，尝试重新连接")
                                # 通知流管理器重新连接
                                await stream_manager.reconnect_stream(stream_id_to_use)

                    # 短暂等待后继续
                    await asyncio.sleep(1)
                    continue

                except Exception as e:
                    normal_logger.error(f"任务 {task_id} 分析异常: {str(e)}")
                    normal_logger.error(traceback.format_exc())

                    # 增加错误计数
                    if not hasattr(self, '_frame_exception_counts'):
                        self._frame_exception_counts = {}

                    self._frame_exception_counts[task_id] = self._frame_exception_counts.get(task_id, 0) + 1
                    exception_count = self._frame_exception_counts[task_id]

                    # 如果连续异常次数过多，尝试重新订阅
                    if exception_count >= 5:
                        normal_logger.warning(f"任务 {task_id}: 连续 {exception_count} 次处理异常，尝试重新订阅")

                        # 尝试重新订阅
                        from core.task_management.stream import stream_manager
                        await stream_manager.unsubscribe_stream(stream_id_to_use, task_id)
                        success, new_frame_queue = await stream_manager.subscribe_stream(stream_id_to_use, task_id, stream_config_for_subscription)

                        if success and new_frame_queue is not None:
                            normal_logger.info(f"成功重新订阅流 {stream_id_to_use}")
                            frame_queue = new_frame_queue
                            self._frame_exception_counts[task_id] = 0  # 重置异常计数
                        else:
                            normal_logger.error(f"重新订阅流 {stream_id_to_use} 失败")

                    # 短暂等待后继续
                    await asyncio.sleep(1)
                    continue

            # 任务结束，取消订阅
            if stream_id_to_use:
                from core.task_management.stream import stream_manager
                await stream_manager.unsubscribe_stream(stream_id_to_use, task_id)

            normal_logger.info(f"工作进程 {task_id}: 已结束")

        except Exception as e:
            normal_logger.error(f"工作进程 {task_id} 异常: {str(e)}")

    # 已移除_subscribe_to_stream_status、_handle_stream_status和_unsubscribe_from_stream方法
    # 这些功能已由StreamTaskBridge类实现，实现了视频流与分析任务的解耦

    def _process_detection_results(self, results: Dict[str, Any], frame: np.ndarray, task_id: str, frame_count: int) -> Dict[str, Any]:
        """
        处理分析结果

        Args:
            results: 分析结果
            frame: 视频帧
            task_id: 任务ID
            frame_count: 帧计数

        Returns:
            Dict[str, Any]: 处理后的结果
        """
        # 获取图像尺寸
        height, width = frame.shape[:2]

        # 构建基本结果
        processed_results = {
            "task_id": task_id,
            "frame_id": frame_count,
            "timestamp": datetime.now().isoformat(),
            "frame_info": {
                "width": width,
                "height": height
            },
            "analysis_info": {
                "pre_process_time": results.get("pre_process_time", 0),
                "inference_time": results.get("inference_time", 0),
                "post_process_time": results.get("post_process_time", 0)
            }
        }

        # 添加检测结果（如果有）
        if "detections" in results:
            processed_results["detections"] = results["detections"]

        # 添加跟踪结果（如果有）
        if "tracked_objects" in results:
            processed_results["tracked_objects"] = results["tracked_objects"]

        # 添加分割结果（如果有）
        if "segmentations" in results:
            processed_results["segmentations"] = results["segmentations"]

        # 添加跨摄像头跟踪结果（如果有）
        if "cross_camera_objects" in results:
            processed_results["cross_camera_objects"] = results["cross_camera_objects"]

        # 添加越界检测结果（如果有）
        if "crossing_events" in results:
            processed_results["crossing_events"] = results["crossing_events"]
            processed_results["alarm_triggered"] = results.get("alarm_triggered", False)

        # 添加帧索引（如果有）
        if "frame_index" in results:
            processed_results["frame_index"] = results["frame_index"]

        # 存储带有分析结果的帧，用于预览
        # 如果有可视化结果，使用可视化结果
        if "visualized_frame" in results and results["visualized_frame"] is not None:
            self.preview_frames[task_id] = results["visualized_frame"]
        else:
            # 否则，使用原始帧并绘制检测结果
            preview_frame = frame.copy()

            # 绘制检测结果
            if "detections" in results:
                for det in results["detections"]:
                    try:
                        # 获取边界框
                        bbox = det.get("bbox", [0, 0, 0, 0])

                        # 处理不同格式的边界框
                        try:
                            if isinstance(bbox, dict):
                                # 如果是字典格式，尝试获取x1, y1, x2, y2
                                if all(k in bbox for k in ['x1', 'y1', 'x2', 'y2']):
                                    x1, y1, x2, y2 = float(bbox['x1']), float(bbox['y1']), float(bbox['x2']), float(bbox['y2'])
                                elif all(k in bbox for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                                    x1, y1, x2, y2 = float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']), float(bbox['ymax'])
                                else:
                                    normal_logger.debug(f"未知的边界框字典格式: {bbox}")
                                    continue
                            elif isinstance(bbox, list) and len(bbox) == 4:
                                # 如果是列表格式，直接使用
                                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                            else:
                                normal_logger.debug(f"未知的边界框格式: {type(bbox)}, 值: {bbox}")
                                continue

                            # 检查坐标是否已经是像素坐标
                            if x1 > 1.0 or y1 > 1.0 or x2 > 1.0 or y2 > 1.0:
                                # 已经是像素坐标，直接使用
                                x1, y1 = max(0, int(x1)), max(0, int(y1))
                                x2, y2 = min(width, int(x2)), min(height, int(y2))
                            else:
                                # 是归一化坐标，转换为像素坐标
                                x1, y1 = max(0, int(x1 * width)), max(0, int(y1 * height))
                                x2, y2 = min(width, int(x2 * width)), min(height, int(y2 * height))

                            # 确保坐标有效
                            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                                continue

                            # 获取类别和置信度
                            class_name = det.get("class_name", "未知")
                            confidence = float(det.get("confidence", 0))

                            # 绘制边界框
                            cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # 绘制标签
                            label = f"{class_name}: {confidence:.2f}"
                            cv2.putText(preview_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        except (ValueError, TypeError) as e:
                            # 忽略无效的边界框
                            normal_logger.debug(f"忽略无效的检测边界框: {bbox}, 错误: {str(e)}")
                            continue
                    except Exception as e:
                        # 忽略处理单个检测时的错误
                        normal_logger.debug(f"处理检测结果时发生错误: {str(e)}, 检测结果: {det}")
                        continue

            # 绘制跟踪结果
            if "tracked_objects" in results:
                for track in results["tracked_objects"]:
                    try:
                        # 获取边界框
                        bbox = track.get("bbox", [0, 0, 0, 0])

                        # 处理不同格式的边界框
                        try:
                            if isinstance(bbox, dict):
                                # 如果是字典格式，尝试获取x1, y1, x2, y2
                                if all(k in bbox for k in ['x1', 'y1', 'x2', 'y2']):
                                    x1, y1, x2, y2 = float(bbox['x1']), float(bbox['y1']), float(bbox['x2']), float(bbox['y2'])
                                elif all(k in bbox for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                                    x1, y1, x2, y2 = float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']), float(bbox['ymax'])
                                else:
                                    normal_logger.debug(f"未知的边界框字典格式: {bbox}")
                                    continue
                            elif isinstance(bbox, list) and len(bbox) == 4:
                                # 如果是列表格式，直接使用
                                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                            else:
                                normal_logger.debug(f"未知的边界框格式: {type(bbox)}, 值: {bbox}")
                                continue

                            # 检查坐标是否已经是像素坐标
                            if x1 > 1.0 or y1 > 1.0 or x2 > 1.0 or y2 > 1.0:
                                # 已经是像素坐标，直接使用
                                x1, y1 = max(0, int(x1)), max(0, int(y1))
                                x2, y2 = min(width, int(x2)), min(height, int(y2))
                            else:
                                # 是归一化坐标，转换为像素坐标
                                x1, y1 = max(0, int(x1 * width)), max(0, int(y1 * height))
                                x2, y2 = min(width, int(x2 * width)), min(height, int(y2 * height))

                            # 确保坐标有效
                            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                                continue

                            # 获取跟踪ID
                            track_id = track.get("track_id", "未知")

                            # 绘制边界框
                            cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                            # 绘制跟踪ID
                            cv2.putText(preview_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        except (ValueError, TypeError) as e:
                            # 忽略无效的边界框
                            normal_logger.debug(f"忽略无效的跟踪边界框: {bbox}, 错误: {str(e)}")
                            continue
                    except Exception as e:
                        # 忽略处理单个跟踪对象时的错误
                        normal_logger.debug(f"处理跟踪结果时发生错误: {str(e)}, 跟踪结果: {track}")
                        continue

            # 存储预览帧
            self.preview_frames[task_id] = preview_frame

        return processed_results

    def get_preview_frame(self, task_id: str) -> Optional[np.ndarray]:
        """
        获取任务的预览帧

        Args:
            task_id: 任务ID

        Returns:
            Optional[np.ndarray]: 预览帧，如果不存在则返回None
        """
        return self.preview_frames.get(task_id)

    async def _handle_results(self, task_id: str, result_queue: queue.Queue):
        """
        处理分析结果，包括保存、发送回调等。
        现在构建更详细的JSON输出。
        """
        if task_id not in self.running_tasks:
            normal_logger.error(f"处理结果时任务 {task_id} 已不存在或未运行。")
            return

        task_full_config = self.running_tasks[task_id].get("config", {})
        
        
        result_config = task_full_config.get("result", {}) 
        storage_config = result_config.get("storage", {}) 
        
        # 回调URL
        callback_url = result_config.get("callback_url")
        
        # 保存JSON结果的标志和路径
        save_json_enabled = result_config.get("save_result", False)
        json_base_save_path = storage_config.get("save_path", "results") 
        
        # 保存图像的标志和路径
        save_images_enabled = result_config.get("save_images", False)
        image_format = storage_config.get("image_format", "jpg")

        normal_logger.info(f"任务 {task_id}: _handle_results 启动。save_json_enabled={save_json_enabled}, save_images_enabled={save_images_enabled}, callback_url={callback_url}")
        normal_logger.info(f"任务 {task_id}: JSON保存路径基址: {json_base_save_path}, 图像格式: {image_format}")
        normal_logger.debug(f"任务 {task_id}: 完整 task_full_config['result']: {result_config}")

        loop = asyncio.get_running_loop() # 获取当前事件循环

        while True:
            try:
                # 从队列中异步获取结果，设置超时以允许检查停止事件
                raw_result = await loop.run_in_executor(None, result_queue.get, 1.0) # timeout 作为第三个参数

                if raw_result is None:  # 任务结束信号
                    normal_logger.info(f"任务 {task_id} 结果处理器收到结束信号。")
                    break

              
                event_id = str(uuid.uuid4())
                event_timestamp_utc = datetime.utcnow().isoformat() + "Z"

                # 1. 提取 task_info
                task_info = {
                    "task_id": task_id,
                    "task_name": task_full_config.get("name"), # 来自请求的 task_name
                    "stream_url": task_full_config.get("stream_url"), # 直接从顶层获取
                    "model_code": task_full_config.get("model", {}).get("code"), # 从 model 子字典获取
                    "device": task_full_config.get("device", "auto"), # 直接从顶层获取
                    "frame_rate_setting": { 
                        "fps": task_full_config.get("frame_processing_fps"), # 更新键名
                        "skip_frames": task_full_config.get("frame_skip_interval") # 更新键名
                    },
                    "analysis_interval_seconds": task_full_config.get("analysis_interval"), # 直接从顶层获取
                    "output_destination_base": json_base_save_path 
                }

                # 2. 提取 frame_info (来自分析器结果)
                frame_info = raw_result.get("frame_info", {})

                # 3. 提取 analysis_result
                detections_list = raw_result.get("detections", [])
                analysis_result_payload = {
                    "detections": detections_list,
                    "detection_count": len(detections_list)
                }
                
                # 4. 提取 analysis_config_applied (来自分析器结果)
                analysis_config_applied = raw_result.get("applied_config", {})

                # 5. 提取 image_dimensions (来自分析器结果 image_info)
                image_dimensions = raw_result.get("image_info", {})

                # 6. 提取 performance_stats (来自分析器结果 timing_stats)
                # 可能需要计算一个此处的总处理时间
                handler_received_time = time.perf_counter() # 假设这是结果被此handler拿到的时间点
                # 这个时间点不太精确，因为队列可能有延迟。更准确的端到端时间应从帧进入系统开始算。
                # 此处暂时只用分析器提供的详细计时。
                performance_stats = raw_result.get("timing_stats", {})

                # 7. 提取 annotated_image_base64 (来自分析器结果)
                annotated_image_b64_data = raw_result.get("annotated_image_base64")

                # 8. 准备 storage_info
                storage_info_payload = {
                    "json_result_path": None,
                    "annotated_image_path": None
                }

                # 构建最终的回调JSON对象
                callback_json = {
                    "event_id": event_id,
                    "event_timestamp_utc": event_timestamp_utc,
                    "task_info": task_info,
                    "frame_info": frame_info,
                    "analysis_result": analysis_result_payload,
                    "analysis_config_applied": analysis_config_applied,
                    "image_dimensions": image_dimensions,
                    "performance_stats": performance_stats,
                    "storage_info": storage_info_payload # 先初始化，后续填充
                }

                # --- 保存JSON结果 ---
                if save_json_enabled:
                    try:
                        # 确保保存目录存在 (按日期分子目录，再按 task_id)
                        date_str = datetime.now().strftime("%Y-%m-%d")
                        current_save_dir = os.path.join(json_base_save_path, date_str, task_id)
                        os.makedirs(current_save_dir, exist_ok=True)
                        
                        # 文件名：frame_analyzer_counter_eventid.json
                        frame_counter = frame_info.get("analyzer_frame_counter", "unknown_frame")
                        json_filename = f"frame_{frame_counter}_{event_id}.json"
                        full_json_path = os.path.join(current_save_dir, json_filename)
                        
                        json_to_save = callback_json.copy() 
                        
                        # BUG FIX: Update json_result_path in the copy that will be saved
                        if json_to_save.get("storage_info") is None: json_to_save["storage_info"] = {}
                        json_to_save["storage_info"]["json_result_path"] = full_json_path

                        if annotated_image_b64_data and not save_images_enabled: 
                            json_to_save["annotated_image_base64"] = annotated_image_b64_data
                        elif "annotated_image_base64" in json_to_save: 
                             del json_to_save["annotated_image_base64"]


                        with open(full_json_path, 'w', encoding='utf-8') as f:
                            json.dump(json_to_save, f, ensure_ascii=False, indent=4)
                        
                        # Update the original callback_json as well, if it's used later (e.g. for HTTP callback)
                        callback_json["storage_info"]["json_result_path"] = full_json_path
                        normal_logger.info(f"任务 {task_id}: JSON结果已保存到: {full_json_path}")
                    except Exception as e_json:
                        exception_logger.error(f"任务 {task_id}: 保存JSON结果失败: {e_json}")
                        if callback_json.get("error_log") is None: callback_json["error_log"] = []
                        callback_json["error_log"].append(f"Failed to save JSON: {str(e_json)}")

                # --- 保存标注图像 ---
                # 只有当 save_images_enabled 为 True 并且分析器返回了图像数据时才保存
                if save_images_enabled:
                    if annotated_image_b64_data:
                        try:
                            img_data = base64.b64decode(annotated_image_b64_data)
                            # 确保保存目录存在 (与JSON同级或类似结构)
                            date_str = datetime.now().strftime("%Y-%m-%d") # 与JSON保存路径日期一致
                            # 图像可以与JSON文件在同一 task_id 目录下，或者有自己的子目录如 'images'
                            image_save_dir = os.path.join(json_base_save_path, date_str, task_id, "images")
                            os.makedirs(image_save_dir, exist_ok=True)

                            frame_counter = frame_info.get("analyzer_frame_counter", "unknown_frame")
                            image_filename = f"annotated_frame_{frame_counter}_{event_id}.{image_format}"
                            full_image_path = os.path.join(image_save_dir, image_filename)

                            with open(full_image_path, 'wb') as f_img:
                                f_img.write(img_data)
                            callback_json["storage_info"]["annotated_image_path"] = full_image_path
                            normal_logger.info(f"任务 {task_id}: 标注图像已保存到: {full_image_path}")
                            
                            # 如果图像已保存，通常不需要在回调JSON中再带base64数据，除非特定需求
                            # callback_json["annotated_image_base64"] = None # 或从回调中删除
                            if "annotated_image_base64" in callback_json: # 从主回调中移除，因为它已存盘
                                del callback_json["annotated_image_base64"]

                        except Exception as e_img_save:
                            exception_logger.error(f"任务 {task_id}: 保存标注图像失败: {e_img_save}")
                            if callback_json.get("error_log") is None: callback_json["error_log"] = []
                            callback_json["error_log"].append(f"Failed to save image: {str(e_img_save)}")
                    else:
                        normal_logger.warning(f"任务 {task_id}: 配置了 save_images 但分析结果中未找到 'annotated_image_base64' 数据。帧计数: {frame_info.get('analyzer_frame_counter')}")
                        if callback_json.get("error_log") is None: callback_json["error_log"] = []
                        callback_json["error_log"].append("save_images was true, but no annotated_image_base64 found in result.")

                # --- 发送HTTP回调 ---
                if callback_url:
                    # Decide whether to include base64 image in callback
                    # If image was saved, and json also saved, usually we don't send base64 to callback
                    # But if only callback is enabled, user might want it.
                    final_callback_payload = callback_json.copy()
                    if annotated_image_b64_data and not save_images_enabled and not save_json_enabled:
                        # Only callback, no saving, so include image in callback
                        final_callback_payload["annotated_image_base64"] = annotated_image_b64_data
                    elif "annotated_image_base64" in final_callback_payload and (save_images_enabled or save_json_enabled) :
                        # If saved to disk (either as image or in json), remove from direct callback to reduce size
                        del final_callback_payload["annotated_image_base64"]

                    try:
                        # 使用异步HTTP客户端发送回调
                        # TODO: 实现异步HTTP POST请求 (例如使用 httpx)
                        normal_logger.info(f"任务 {task_id}: 准备发送回调到 {callback_url}。数据量级(不含图像): {len(json.dumps(final_callback_payload).encode('utf-8'))} bytes")
                        # Placeholder for actual async post
                        # async with httpx.AsyncClient() as client:
                        #     response = await client.post(callback_url, json=final_callback_payload, timeout=10.0)
                        #     response.raise_for_status() # Raises an exception for 4XX/5XX responses
                        # normal_logger.warning(f"任务 {task_id}: HTTP回调发送功能暂未实现。数据预览: {json.dumps(final_callback_payload, indent=2)[:1000]}...") # 截断预览

                    except Exception as e_callback:
                        exception_logger.error(f"任务 {task_id}: 发送回调到 {callback_url} 失败: {e_callback}")
                        # 可以考虑重试机制

                # 清理，准备下一次迭代
                del raw_result # 释放原始结果占用的内存

            except queue.Empty:
                # 队列为空，继续等待，检查停止事件
                if task_id in self.stop_events and self.stop_events[task_id].is_set():
                    normal_logger.info(f"任务 {task_id} 结果处理器检测到停止信号，正在退出。")
                    break
                await asyncio.sleep(0.01) # 例如，睡眠10毫秒
                continue # 继续外层while循环

            except Exception as e:
                exception_logger.exception(f"任务 {task_id} 处理结果时发生未捕获的错误: {e}")
                # 发生严重错误，也应该尝试检查停止信号
                if task_id in self.stop_events and self.stop_events[task_id].is_set():
                    normal_logger.info(f"任务 {task_id} 结果处理器因错误并检测到停止信号，正在退出。")
                    break
                time.sleep(0.1) # 避免错误刷屏

        normal_logger.info(f"任务 {task_id} 结果处理器已停止。")

    async def process_frame(self, task_id: str, frame: np.ndarray, frame_index: int) -> Dict[str, Any]:
        """
        处理单帧图像
        
        Args:
            task_id: 任务ID
            frame: 图像帧
            frame_index: 帧索引
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        normal_logger.info(f"处理任务 {task_id} 的第 {frame_index} 帧，帧大小: {frame.shape}")
        
        # 获取任务参数
        task_params = self.get_task_params(task_id)
        if not task_params:
            normal_logger.warning(f"找不到任务参数: {task_id}")
            return {"success": False, "error": "找不到任务参数"}
            
        # 获取分析器
        analyzer = self.get_analyzer(task_id)
        if not analyzer:
            normal_logger.warning(f"找不到分析器: {task_id}")
            return {"success": False, "error": "找不到分析器"}
            
        # 检查模型加载状态
        normal_logger.info(f"分析器状态: {analyzer.model_info}")
        
        # 执行分析
        try:
            start_time = time.time()
            result = await analyzer.process_video_frame(frame, frame_index)
            process_time = time.time() - start_time
            
            # 添加处理时间
            if "stats" not in result:
                result["stats"] = {}
            result["stats"]["process_time"] = process_time
            
            # 输出分析结果摘要
            detections_count = len(result.get("detections", []))
            normal_logger.info(f"任务 {task_id} 第 {frame_index} 帧分析完成，检测到 {detections_count} 个目标，耗时: {process_time:.4f}秒")
            
            if detections_count > 0:
                normal_logger.info(f"检测结果示例: {result.get('detections', [])[:2]}")
            
            return result
            
        except Exception as e:
            exception_logger.exception(f"处理帧时出错: {e}")
            return {"success": False, "error": str(e)}
