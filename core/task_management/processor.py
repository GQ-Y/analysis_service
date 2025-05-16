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

from core.config import settings
from shared.utils.logger import setup_logger, setup_stream_error_logger, setup_analysis_logger
from core.task_management.utils.status import TaskStatus
from core.task_management.stream.status import StreamStatus, StreamHealthStatus

# 延迟导入stream_manager，避免循环导入
from core.redis_manager import RedisManager

logger = setup_logger(__name__)
stream_error_logger = setup_stream_error_logger()
analysis_logger = setup_analysis_logger()

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

        logger.info("任务处理器初始化完成")
        return True

    async def shutdown(self):
        """关闭任务处理器"""
        # 停止所有运行中的任务
        for task_id in list(self.running_tasks.keys()):
            await self.stop_task(task_id)

        logger.info("任务处理器已关闭")
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
        try:
            # 检查任务是否已存在
            if task_id in self.running_tasks:
                logger.warning(f"任务已存在: {task_id}")
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

            logger.info(f"流处理进程已启动: {task_id}")
            return True

        except Exception as e:
            logger.error(f"启动流分析任务失败: {str(e)}")
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
                logger.warning(f"任务不存在: {task_id}")
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

            logger.info(f"任务已停止: {task_id}")
            return True

        except Exception as e:
            logger.error(f"停止任务失败: {str(e)}")
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
                logger.warning(f"任务不存在: {task_id}")
                return False

            # 设置暂停事件
            if task_id in self.pause_events:
                self.pause_events[task_id].set()
                logger.info(f"任务暂停事件已设置: {task_id}")
                return True
            else:
                logger.warning(f"任务暂停事件不存在: {task_id}")
                return False

        except Exception as e:
            logger.error(f"暂停任务失败: {str(e)}")
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
                logger.warning(f"任务不存在: {task_id}")
                return False

            # 清除暂停事件
            if task_id in self.pause_events:
                self.pause_events[task_id].clear()
                logger.info(f"任务恢复事件已设置: {task_id}")
                return True
            else:
                logger.warning(f"任务暂停事件不存在: {task_id}")
                return False

        except Exception as e:
            logger.error(f"恢复任务失败: {str(e)}")
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
            logger.info(f"工作进程 {task_id}: 开始初始化")

            # 获取流配置
            stream_id = task_config.get("stream_id", "")
            # 如果stream_id为空，使用task_id作为替代
            if not stream_id:
                stream_id = f"stream_{task_id}"
                logger.info(f"流ID为空，使用任务ID生成流ID: {stream_id}")

            stream_url = task_config.get("stream_url", "")  # 修正字段名，使用stream_url而不是url
            stream_config = {
                "url": stream_url,
                "rtsp_transport": task_config.get("rtsp_transport", "tcp"),
                "reconnect_attempts": task_config.get("reconnect_attempts", settings.STREAMING.reconnect_attempts),
                "reconnect_delay": task_config.get("reconnect_delay", settings.STREAMING.reconnect_delay),
                "frame_buffer_size": task_config.get("frame_buffer_size", settings.STREAMING.frame_buffer_size),
                "task_id": task_id,  # 添加任务ID
                "video_id": stream_id  # 使用stream_id作为视频ID
            }

            # 订阅视频流
            from core.task_management.stream import stream_manager
            try:
                success, frame_queue = await stream_manager.subscribe_stream(stream_id, task_id, stream_config)
                if not success or frame_queue is None:
                    error_msg = f"订阅视频流失败: {stream_id}"
                    logger.error(error_msg)
                    self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                    return
            except Exception as e:
                error_msg = f"订阅视频流时发生异常: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                return

            # 创建分析器
            # ... 原有的分析器创建代码 ...

            # 分析循环
            while not stop_event.is_set():
                # 检查暂停事件
                if pause_event.is_set():
                    # 任务已暂停，等待恢复
                    time.sleep(1)
                    continue

                try:
                    # 从帧队列获取帧
                    logger.info(f"工作进程 {task_id}: 尝试获取帧")
                    frame, timestamp = await asyncio.wait_for(frame_queue.get(), timeout=5.0)

                    # 添加获取帧成功的日志
                    if frame is not None:
                        logger.info(f"工作进程 {task_id}: 成功获取帧，大小: {frame.shape}")

                        # 重置连续错误计数
                        if hasattr(self, '_frame_error_counts') and task_id in self._frame_error_counts:
                            self._frame_error_counts[task_id] = 0

                    if frame is None:
                        # 没有收到帧，可能是流离线
                        logger.warning(f"任务 {task_id}: 未接收到视频帧")

                        # 增加错误计数
                        if not hasattr(self, '_frame_error_counts'):
                            self._frame_error_counts = {}

                        self._frame_error_counts[task_id] = self._frame_error_counts.get(task_id, 0) + 1
                        error_count = self._frame_error_counts[task_id]

                        # 如果连续错误次数过多，尝试重新订阅
                        if error_count >= 3:
                            logger.warning(f"任务 {task_id}: 连续 {error_count} 次未接收到视频帧，尝试重新订阅")

                            # 检查流状态
                            from core.task_management.stream import stream_manager
                            stream_info = await stream_manager.get_stream_info(stream_id)

                            if stream_info:
                                # 尝试重新订阅
                                await stream_manager.unsubscribe_stream(stream_id, task_id)
                                success, new_frame_queue = await stream_manager.subscribe_stream(stream_id, task_id, stream_config)

                                if success and new_frame_queue is not None:
                                    logger.info(f"成功重新订阅流 {stream_id}")
                                    frame_queue = new_frame_queue
                                    self._frame_error_counts[task_id] = 0  # 重置错误计数
                                else:
                                    logger.error(f"重新订阅流 {stream_id} 失败")

                        time.sleep(1)
                        continue

                    # 执行分析
                    # 这里应该有实际的分析代码，但由于我们只是测试帧获取，
                    # 所以创建一个简单的结果对象
                    result = {
                        "task_id": task_id,
                        "timestamp": timestamp,
                        "frame_shape": frame.shape if frame is not None else None,
                        "message": "成功获取帧"
                    }

                    # 将结果放入结果队列
                    result_queue.put(result)

                except asyncio.TimeoutError:
                    logger.warning(f"任务 {task_id}: 获取帧超时")

                    # 增加错误计数
                    if not hasattr(self, '_frame_timeout_counts'):
                        self._frame_timeout_counts = {}

                    self._frame_timeout_counts[task_id] = self._frame_timeout_counts.get(task_id, 0) + 1
                    timeout_count = self._frame_timeout_counts[task_id]

                    # 检查流状态
                    from core.task_management.stream import stream_manager
                    stream_info = await stream_manager.get_stream_info(stream_id)

                    if stream_info:
                        status = stream_info.get("status")
                        health = stream_info.get("health_status")
                        logger.info(f"流 {stream_id} 状态: {status}, 健康状态: {health}")

                        # 根据不同情况采取不同策略
                        if status in [StreamStatus.RUNNING, StreamStatus.ONLINE]:
                            # 流状态正常但获取帧超时
                            if timeout_count >= 3:  # 连续超时3次以上才重新订阅
                                logger.info(f"流 {stream_id} 状态正常但连续 {timeout_count} 次获取帧超时，尝试重新订阅")
                                await stream_manager.unsubscribe_stream(stream_id, task_id)
                                success, new_frame_queue = await stream_manager.subscribe_stream(stream_id, task_id, stream_config)

                                if success and new_frame_queue is not None:
                                    logger.info(f"成功重新订阅流 {stream_id}")
                                    frame_queue = new_frame_queue
                                    self._frame_timeout_counts[task_id] = 0  # 重置超时计数
                                else:
                                    logger.error(f"重新订阅流 {stream_id} 失败")
                        elif status in [StreamStatus.CONNECTING, StreamStatus.INITIALIZING]:
                            # 流正在连接中，等待
                            logger.info(f"流 {stream_id} 正在连接中，等待...")
                        elif status in [StreamStatus.OFFLINE, StreamStatus.ERROR]:
                            # 流离线或错误，尝试重新连接
                            if timeout_count % 5 == 0:  # 每5次超时尝试一次重连
                                logger.info(f"流 {stream_id} 状态异常 ({status})，尝试重新连接")
                                # 通知流管理器重新连接
                                await stream_manager.reconnect_stream(stream_id)

                    # 短暂等待后继续
                    await asyncio.sleep(1)
                    continue

                except Exception as e:
                    logger.error(f"任务 {task_id} 分析异常: {str(e)}")
                    logger.error(traceback.format_exc())

                    # 增加错误计数
                    if not hasattr(self, '_frame_exception_counts'):
                        self._frame_exception_counts = {}

                    self._frame_exception_counts[task_id] = self._frame_exception_counts.get(task_id, 0) + 1
                    exception_count = self._frame_exception_counts[task_id]

                    # 如果连续异常次数过多，尝试重新订阅
                    if exception_count >= 5:
                        logger.warning(f"任务 {task_id}: 连续 {exception_count} 次处理异常，尝试重新订阅")

                        # 尝试重新订阅
                        from core.task_management.stream import stream_manager
                        await stream_manager.unsubscribe_stream(stream_id, task_id)
                        success, new_frame_queue = await stream_manager.subscribe_stream(stream_id, task_id, stream_config)

                        if success and new_frame_queue is not None:
                            logger.info(f"成功重新订阅流 {stream_id}")
                            frame_queue = new_frame_queue
                            self._frame_exception_counts[task_id] = 0  # 重置异常计数
                        else:
                            logger.error(f"重新订阅流 {stream_id} 失败")

                    # 短暂等待后继续
                    await asyncio.sleep(1)
                    continue

            # 任务结束，取消订阅
            if stream_id:
                from core.task_management.stream import stream_manager
                await stream_manager.unsubscribe_stream(stream_id, task_id)

            logger.info(f"工作进程 {task_id}: 已结束")

        except Exception as e:
            logger.error(f"工作进程 {task_id} 异常: {str(e)}")

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
                                    logger.debug(f"未知的边界框字典格式: {bbox}")
                                    continue
                            elif isinstance(bbox, list) and len(bbox) == 4:
                                # 如果是列表格式，直接使用
                                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                            else:
                                logger.debug(f"未知的边界框格式: {type(bbox)}, 值: {bbox}")
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
                            logger.debug(f"忽略无效的检测边界框: {bbox}, 错误: {str(e)}")
                            continue
                    except Exception as e:
                        # 忽略处理单个检测时的错误
                        logger.debug(f"处理检测结果时发生错误: {str(e)}, 检测结果: {det}")
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
                                    logger.debug(f"未知的边界框字典格式: {bbox}")
                                    continue
                            elif isinstance(bbox, list) and len(bbox) == 4:
                                # 如果是列表格式，直接使用
                                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                            else:
                                logger.debug(f"未知的边界框格式: {type(bbox)}, 值: {bbox}")
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
                            logger.debug(f"忽略无效的跟踪边界框: {bbox}, 错误: {str(e)}")
                            continue
                    except Exception as e:
                        # 忽略处理单个跟踪对象时的错误
                        logger.debug(f"处理跟踪结果时发生错误: {str(e)}, 跟踪结果: {track}")
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
        处理结果队列

        Args:
            task_id: 任务ID
            result_queue: 结果队列
        """
        logger.info(f"启动结果处理器: {task_id}")

        # 获取任务配置
        task_config = self.running_tasks.get(task_id, {}).get("config", {})

        # 获取回调配置
        callback_enabled = task_config.get("subtask", {}).get("callback", {}).get("enabled", False)
        callback_url = task_config.get("subtask", {}).get("callback", {}).get("url")

        # 获取回调间隔
        callback_interval = task_config.get("callback_interval", 0)

        # 获取回调服务
        callback_service = None
        try:
            # 尝试从应用状态获取回调服务
            from fastapi import FastAPI
            import inspect

            # 获取当前应用实例
            app = None
            for frame_info in inspect.stack():
                if 'app' in frame_info.frame.f_locals and isinstance(frame_info.frame.f_locals['app'], FastAPI):
                    app = frame_info.frame.f_locals['app']
                    break

            if app and hasattr(app.state, "callback_service"):
                callback_service = app.state.callback_service

                # 注册任务回调
                if callback_enabled and callback_url:
                    callback_service.register_task(
                        task_id=task_id,
                        callback_url=callback_url,
                        enable_callback=callback_enabled,
                        callback_interval=callback_interval
                    )
                    logger.info(f"任务 {task_id} 已注册回调: URL={callback_url}, 间隔={callback_interval}秒")
            else:
                # 如果无法从应用状态获取，创建新的回调服务实例
                from services.http.callback_service import CallbackService
                callback_service = CallbackService()

                # 注册任务回调
                if callback_enabled and callback_url:
                    callback_service.register_task(
                        task_id=task_id,
                        callback_url=callback_url,
                        enable_callback=callback_enabled,
                        callback_interval=callback_interval
                    )
                    logger.info(f"任务 {task_id} 已注册回调: URL={callback_url}, 间隔={callback_interval}秒")
        except Exception as e:
            logger.warning(f"回调服务初始化失败，任务 {task_id} 将不会发送回调: {str(e)}")
            callback_service = None

        try:
            while True:
                # 检查任务是否已停止
                if task_id not in self.running_tasks:
                    logger.info(f"任务已停止，结束结果处理: {task_id}")
                    break

                # 尝试从队列获取结果
                try:
                    result = result_queue.get(block=False)
                except queue.Empty:
                    # 队列为空，等待一段时间后重试
                    await asyncio.sleep(0.1)
                    continue

                # 处理结果
                result_type = result.get("type")

                if result_type == "error":
                    # 处理错误
                    error_message = result.get("message", "未知错误")
                    logger.error(f"任务 {task_id} 发生错误: {error_message}")

                    # 更新任务状态
                    self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_message)

                    # 发送错误通知
                    logger.info(f"任务 {task_id} 发送错误通知: {error_message}")

                    # 发送错误回调
                    if callback_service and callback_enabled and callback_url:
                        error_data = {
                            "success": False,
                            "error": error_message
                        }
                        await callback_service.send_callback(task_id, error_data)

                elif result_type == "result":
                    # 处理分析结果
                    result_data = result.get("data", {})

                    # 获取当前任务状态
                    current_task = self.task_manager.get_task(task_id)
                    current_status = current_task.get("status") if current_task else None

                    # 只有当任务不是重试状态时才更新为处理中
                    # 这样可以避免在重试过程中收到旧的结果时错误地将状态改回处理中
                    if current_status != TaskStatus.RETRYING:
                        # 更新任务状态
                        self.task_manager.update_task_status(task_id, TaskStatus.PROCESSING, result=result_data)

                    # 发送结果回调
                    if callback_service and callback_enabled and callback_url:
                        # 检查是否有检测结果
                        if "detections" in result_data and result_data["detections"]:
                            # 发送回调
                            await callback_service.send_callback(task_id, result_data)

                        # 检查是否有跟踪结果
                        elif "tracked_objects" in result_data and result_data["tracked_objects"]:
                            # 对每个跟踪对象单独处理回调间隔
                            for tracked_obj in result_data["tracked_objects"]:
                                object_id = tracked_obj.get("track_id")
                                if object_id:
                                    # 发送对象级回调
                                    await callback_service.send_callback(
                                        task_id,
                                        result_data,
                                        object_id=str(object_id)
                                    )

                        # 其他类型的结果直接发送
                        elif any(key in result_data for key in ["segmentations", "cross_camera_objects", "crossing_events"]):
                            await callback_service.send_callback(task_id, result_data)

                # 标记队列项为已处理
                result_queue.task_done()

        except asyncio.CancelledError:
            logger.info(f"结果处理器被取消: {task_id}")
            # 取消注册任务回调
            if callback_service:
                callback_service.unregister_task(task_id)

        except Exception as e:
            logger.error(f"结果处理器发生错误: {str(e)}")
            logger.error(traceback.format_exc())
