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

from core.config import settings
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

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

    async def initialize(self):
        """初始化任务处理器"""
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

            # 创建停止事件
            stop_event = threading.Event()
            self.stop_events[task_id] = stop_event

            # 创建结果队列
            result_queue = queue.Queue()

            # 创建并启动任务线程 - 使用 asyncio.to_thread 运行异步函数
            thread = threading.Thread(
                target=asyncio.run,
                args=(self.process_stream_worker(task_id, task_config, result_queue, stop_event),),
                daemon=True
            )
            thread.start()

            # 保存任务信息
            self.running_tasks[task_id] = {
                "thread": thread,
                "config": task_config,
                "start_time": time.time(),
                "status": "running"
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

            # 更新任务状态
            if task_id in self.running_tasks:
                self.running_tasks[task_id]["status"] = "stopped"
                self.running_tasks[task_id]["end_time"] = time.time()

            # 清理资源
            if task_id in self.stop_events:
                del self.stop_events[task_id]
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

    async def process_stream_worker(self, task_id: str, task_config: Dict[str, Any], result_queue: queue.Queue, stop_event: threading.Event):
        """
        流处理工作线程

        Args:
            task_id: 任务ID
            task_config: 任务配置
            result_queue: 结果队列
            stop_event: 停止事件
        """
        try:
            logger.info(f"工作进程 {task_id}: 开始初始化")

            # 获取分析类型
            analysis_type_str = task_config.get("subtask", {}).get("type", "detection")

            # 获取模型配置
            model_config = task_config.get("model", {})
            model_code = model_config.get("code", "yolov8n.pt")

            # 获取引擎类型
            engine_type = task_config.get("engine", 0)  # 默认使用PyTorch

            # 获取YOLO版本
            yolo_version = task_config.get("yolo_version", 0)  # 默认使用YOLOv8n

            # 获取设备
            device = task_config.get("device", "auto")

            # 记录初始化信息
            logger.info(f"工作进程 {task_id}: 开始初始化 {analysis_type_str} 分析器: "
                       f"模型={model_code}, 引擎={engine_type}, YOLO版本={yolo_version}")

            # 使用分析器工厂创建分析器
            try:
                from core.analyzer.analyzer_factory import AnalyzerFactory

                # 将字符串分析类型转换为整数类型
                analysis_type_map = {
                    "detection": 1,
                    "tracking": 2,
                    "segmentation": 3,
                    "cross_camera_tracking": 4,
                    "line_crossing": 5
                }
                analysis_type = analysis_type_map.get(analysis_type_str.lower(), 1)  # 默认使用检测

                # 创建分析器
                detector = AnalyzerFactory.create_analyzer(
                    analysis_type=analysis_type,
                    model_code=model_code,
                    engine_type=engine_type,
                    yolo_version=yolo_version,
                    device=device,
                    **task_config.get("analysis", {})  # 传递分析配置
                )

                logger.info(f"工作进程 {task_id}: 成功创建 {analysis_type_str} 分析器")

            except Exception as e:
                # 创建分析器失败
                error_message = f"创建分析器失败: {str(e)}"
                logger.error(f"工作进程 {task_id}: {error_message}")
                logger.error(traceback.format_exc())
                result_queue.put({
                    "type": "error",
                    "message": error_message
                })
                return

            # 加载模型
            await detector.load_model(model_code)

            # 获取流配置
            stream_url = task_config.get("stream_url")
            if not stream_url:
                logger.error(f"工作进程 {task_id}: 未提供视频流URL")
                result_queue.put({
                    "type": "error",
                    "message": "未提供视频流URL"
                })
                return

            # 打开视频流
            logger.info(f"工作进程 {task_id}: 尝试打开视频流: {stream_url}")
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                logger.error(f"工作进程 {task_id}: 无法打开视频流: {stream_url}")
                result_queue.put({
                    "type": "error",
                    "message": f"无法打开视频流: {stream_url}"
                })
                return

            # 获取视频信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"工作进程 {task_id}: 视频流信息 - 宽度: {width}, 高度: {height}, FPS: {fps}")

            # 获取分析配置
            analysis_config = task_config.get("analysis", {})

            # 获取分析间隔
            analysis_interval = task_config.get("analysis_interval", 1)
            frame_count = 0

            # 主循环
            while not stop_event.is_set():
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"工作进程 {task_id}: 视频流结束或读取失败")
                    break

                # 增加帧计数
                frame_count += 1

                # 根据分析间隔决定是否处理当前帧
                if frame_count % analysis_interval != 0:
                    continue

                # 处理帧
                try:
                    # 执行检测
                    results = await detector.detect(frame, **analysis_config)

                    # 处理结果
                    processed_results = self._process_detection_results(results, frame, task_id, frame_count)

                    # 放入结果队列
                    result_queue.put({
                        "type": "result",
                        "data": processed_results
                    })

                except Exception as e:
                    logger.error(f"工作进程 {task_id}: 处理帧时发生错误: {str(e)}")

            # 释放资源
            cap.release()
            detector.release()

            logger.info(f"工作进程 {task_id}: 退出")

        except Exception as e:
            logger.error(f"工作进程 {task_id}: 发生未处理的严重错误: {str(e)}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            result_queue.put({
                "type": "error",
                "message": f"工作进程发生严重错误: {str(e)}"
            })

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

        return processed_results

    async def _handle_results(self, task_id: str, result_queue: queue.Queue):
        """
        处理结果队列

        Args:
            task_id: 任务ID
            result_queue: 结果队列
        """
        logger.info(f"启动结果处理器: {task_id}")

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
                    self.task_manager.update_task_status(task_id, "failed", error=error_message)

                    # 发送错误通知
                    logger.info(f"任务 {task_id} 发送错误通知: {error_message}")

                elif result_type == "result":
                    # 处理分析结果
                    result_data = result.get("data", {})

                    # 更新任务状态
                    self.task_manager.update_task_status(task_id, "running", result=result_data)

                # 标记队列项为已处理
                result_queue.task_done()

        except asyncio.CancelledError:
            logger.info(f"结果处理器被取消: {task_id}")

        except Exception as e:
            logger.error(f"结果处理器发生错误: {str(e)}")
            logger.error(traceback.format_exc())
