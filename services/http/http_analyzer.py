"""
HTTP分析服务
提供基于HTTP的分析服务功能
"""
import os
import sys
import asyncio
import uuid
import json
import time
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from core.config import settings
from core.task_manager import TaskManager
from core.task_processor import TaskProcessor
from core.task_queue import TaskQueue
from core.analyzer.detection.yolo_detector import YOLODetector
from core.analyzer.segmentation.yolo_segmentor import YOLOSegmentor
from core.task_management import TaskStatus
from services.base_analyzer import BaseAnalyzerService
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class HTTPAnalyzerService(BaseAnalyzerService):
    """HTTP分析服务"""

    def __init__(self):
        """初始化HTTP分析服务"""
        super().__init__()
        normal_logger.info("初始化HTTP分析服务")

        # 初始化任务相关组件
        self.task_queue = None
        self.task_manager = None
        self.task_processor = None

        # 初始化分析器
        self.detector = None
        self.segmentor = None

        # 服务状态
        self.is_running = False
        self.start_time = None

        normal_logger.info("HTTP分析服务初始化完成")

    async def start(self):
        """启动服务"""
        if self.is_running:
            normal_logger.warning("HTTP分析服务已经在运行中")
            return True

        try:
            normal_logger.info("正在启动HTTP分析服务...")

            # 初始化任务队列
            self.task_queue = TaskQueue()

            # 初始化任务管理器
            self.task_manager = TaskManager()

            # 初始化任务处理器
            self.task_processor = TaskProcessor(self.task_manager)

            # 初始化检测器
            self.detector = YOLODetector()

            # 初始化分割器
            self.segmentor = YOLOSegmentor()

            # 更新服务状态
            self.is_running = True
            self.start_time = datetime.now()

            normal_logger.info("HTTP分析服务启动成功")
            return True

        except Exception as e:
            exception_logger.exception(f"启动HTTP分析服务失败: {str(e)}")
            return False

    async def stop(self):
        """停止服务"""
        if not self.is_running:
            normal_logger.warning("HTTP分析服务已经停止")
            return True

        try:
            normal_logger.info("正在停止HTTP分析服务...")

            # 停止所有任务
            if self.task_manager:
                await self.task_manager.stop_all_tasks()

            # 释放资源
            if self.detector:
                self.detector.release()
                self.detector = None

            if self.segmentor:
                self.segmentor.release()
                self.segmentor = None

            # 更新服务状态
            self.is_running = False

            normal_logger.info("HTTP分析服务停止成功")
            return True

        except Exception as e:
            exception_logger.exception(f"停止HTTP分析服务失败: {str(e)}")
            return False

    async def process_image(self, image_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理图像

        Args:
            image_path: 图像路径
            params: 处理参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 检查服务状态
            if not self.is_running:
                await self.start()

            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")

            # 获取分析类型
            analysis_type = params.get("analysis_type", "detection")

            # 获取模型代码
            model_code = params.get("model_code", "yolov8n")

            # 根据分析类型选择分析器
            if analysis_type == "segmentation":
                analyzer = self.segmentor
            else:  # detection 或 tracking
                analyzer = self.detector

            # 加载模型
            await analyzer.load_model(model_code)

            # 执行分析
            start_time = time.time()
            result = await analyzer.detect(image, **params)
            process_time = time.time() - start_time

            # 处理结果
            processed_result = self._process_result(result, image)

            # 添加处理时间
            processed_result["process_time"] = process_time

            return {
                "success": True,
                "message": "图像处理成功",
                "result": processed_result
            }

        except Exception as e:
            exception_logger.exception(f"处理图像失败: {str(e)}")
            return {
                "success": False,
                "message": f"处理图像失败: {str(e)}",
                "result": None
            }

    async def process_video(self, video_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理视频

        Args:
            video_path: 视频路径
            params: 处理参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 检查服务状态
            if not self.is_running:
                await self.start()

            # 检查视频文件
            if not os.path.exists(video_path):
                raise ValueError(f"视频文件不存在: {video_path}")

            # 生成任务ID
            task_id = str(uuid.uuid4())

            # 构建任务配置
            task_config = {
                "video_path": video_path,
                "model": {
                    "code": params.get("model_code", "yolov8n"),
                    "confidence": params.get("confidence", 0.5),
                    "iou_threshold": params.get("iou_threshold", 0.45)
                },
                "analysis": {
                    "type": params.get("analysis_type", "detection"),
                    "classes": params.get("classes"),
                    "roi": params.get("roi"),
                    "roi_type": params.get("roi_type", 0),
                    "track_config": {
                        "enabled": params.get("enable_tracking", False),
                        "tracker_type": params.get("tracker_type", "sort"),
                        "max_age": params.get("max_age", 30),
                        "min_hits": params.get("min_hits", 3),
                        "iou_threshold": params.get("iou_threshold", 0.3)
                    }
                },
                "result": {
                    "save_images": params.get("save_images", False),
                    "return_base64": params.get("return_base64", False),
                    "storage": {
                        "save_path": params.get("save_path", "results")
                    }
                },
                "device": params.get("device", "auto")
            }

            # 启动视频处理任务
            result = await self.task_processor.start_video_analysis(task_id, task_config)

            if not result:
                return {
                    "success": False,
                    "message": "启动视频处理任务失败",
                    "task_id": task_id
                }

            return {
                "success": True,
                "message": "视频处理任务已启动",
                "task_id": task_id
            }

        except Exception as e:
            exception_logger.exception(f"处理视频失败: {str(e)}")
            return {
                "success": False,
                "message": f"处理视频失败: {str(e)}",
                "task_id": None
            }

    async def process_stream(self, stream_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理视频流

        Args:
            stream_url: 视频流URL
            params: 处理参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 检查服务状态
            if not self.is_running:
                await self.start()

            # 生成任务ID
            task_id = params.get("task_id", str(uuid.uuid4()))

            # 构建任务配置 - 直接使用传入的所有参数
            task_config = {
                "stream_url": stream_url,
                # 保留原始参数
                **params
            }

            # 确保基本配置存在
            if "model" not in task_config:
                task_config["model"] = {
                    "code": params.get("model_code", "yolov8n"),
                    "confidence": params.get("confidence", 0.5),
                    "iou_threshold": params.get("iou_threshold", 0.45)
                }

            if "subtask" not in task_config:
                task_config["subtask"] = {
                    "type": params.get("analysis_type", "detection"),
                    "callback": {
                        "enabled": params.get("enable_callback", False),
                        "url": params.get("callback_url")
                    }
                }

            if "analysis" not in task_config:
                task_config["analysis"] = {}

            # 确保跟踪配置存在（如果启用了跟踪）
            if params.get("enable_tracking", False) and "track_config" not in task_config["analysis"]:
                task_config["analysis"]["track_config"] = {
                    "enabled": True,
                    "tracker_type": params.get("tracker_type", "sort"),
                    "max_age": params.get("max_age", 30),
                    "min_hits": params.get("min_hits", 3),
                    "iou_threshold": params.get("iou_threshold", 0.3)
                }

                # 添加跨摄像头跟踪相关配置
                if "related_cameras" in params:
                    task_config["analysis"]["track_config"]["related_cameras"] = params["related_cameras"]

                if "feature_type" in params:
                    task_config["analysis"]["track_config"]["feature_type"] = params["feature_type"]

            if "result" not in task_config:
                task_config["result"] = {
                    "save_images": params.get("save_images", False),
                    "return_base64": params.get("return_base64", False),
                    "storage": {
                        "save_path": params.get("save_path", "results")
                    }
                }

            # 确保分析间隔存在
            if "analysis_interval" not in task_config:
                task_config["analysis_interval"] = params.get("analysis_interval", 1)

            # 确保设备配置存在
            if "device" not in task_config:
                task_config["device"] = params.get("device", "auto")

            # 启动流处理任务
            result = await self.task_processor.start_stream_analysis(task_id, task_config)

            if not result:
                return {
                    "success": False,
                    "message": "启动流处理任务失败",
                    "task_id": task_id
                }

            return {
                "success": True,
                "message": "流处理任务已启动",
                "task_id": task_id
            }

        except Exception as e:
            exception_logger.exception(f"处理流失败: {str(e)}")
            return {
                "success": False,
                "message": f"处理流失败: {str(e)}",
                "task_id": None
            }

    async def get_status(self) -> Dict[str, Any]:
        """
        获取服务状态

        Returns:
            Dict[str, Any]: 服务状态
        """
        try:
            # 获取运行时间
            uptime = None
            if self.start_time:
                uptime = (datetime.now() - self.start_time).total_seconds()

            # 获取任务统计
            task_stats = {}
            if self.task_manager:
                task_stats = {
                    "total": self.task_manager.get_task_count(),
                    "running": self.task_manager.get_task_count(TaskStatus.RUNNING),
                    "waiting": self.task_manager.get_task_count(TaskStatus.WAITING),
                    "completed": self.task_manager.get_task_count(TaskStatus.COMPLETED),
                    "failed": self.task_manager.get_task_count(TaskStatus.FAILED)
                }

            # 构建状态信息
            status = {
                "service": "http_analyzer",
                "status": "running" if self.is_running else "stopped",
                "uptime": uptime,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "version": settings.VERSION,
                "environment": settings.ENVIRONMENT,
                "task_stats": task_stats
            }

            return status

        except Exception as e:
            exception_logger.exception(f"获取服务状态失败: {str(e)}")
            return {
                "service": "http_analyzer",
                "status": "error",
                "error": str(e)
            }
