"""
任务服务
提供任务管理相关的业务逻辑
"""
from typing import Dict, Any, List, Optional, Union
import uuid
import asyncio
from datetime import datetime
import json
import aiohttp

from core.task_management import TaskStatus
from core.task_management.manager import TaskManager
from models.requests import StreamTask, BatchStreamTask, DetectionConfig
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class TaskService:
    """任务服务"""

    def __init__(self, task_manager: TaskManager):
        """
        初始化

        Args:
            task_manager: 任务管理器
        """
        self.task_manager = task_manager

    async def create_task(self, task: StreamTask) -> Dict[str, Any]:
        """
        创建任务

        Args:
            task: 任务参数

        Returns:
            Dict[str, Any]: 创建结果
        """
        try:
            # 生成任务ID
            task_id = str(uuid.uuid4())

            # 构建任务配置
            task_config = self._build_task_config(task)

            # 创建任务
            task_data = {
                "id": task_id,
                "type": task.analysis_type or "detection",
                "params": task_config,
                "created_at": datetime.now().isoformat()
            }

            # 添加任务
            if not self.task_manager.add_task(task_id, task_data):
                return {
                    "success": False,
                    "message": "添加任务失败",
                    "task_id": None
                }

            # 启动任务
            if not await self.task_manager.start_task(task_id):
                return {
                    "success": False,
                    "message": "启动任务失败",
                    "task_id": task_id
                }

            return {
                "success": True,
                "message": "任务创建并启动成功",
                "task_id": task_id
            }

        except Exception as e:
            logger.error(f"创建任务失败: {str(e)}")
            return {
                "success": False,
                "message": f"创建任务失败: {str(e)}",
                "task_id": None
            }

    def _build_task_config(self, task: StreamTask) -> Dict[str, Any]:
        """
        构建任务配置

        Args:
            task: 任务参数

        Returns:
            Dict[str, Any]: 任务配置
        """
        # 构建模型配置
        model_config = {
            "code": task.model_code,
            "confidence": task.config.confidence if task.config and hasattr(task.config, "confidence") else 0.5,
            "iou_threshold": task.config.iou if task.config and hasattr(task.config, "iou") else 0.45
        }

        # 构建子任务配置
        analysis_type = task.analysis_type or "detection"
        # 如果启用了跟踪，将分析类型设置为 tracking
        if task.config and hasattr(task.config, "tracking_type") and task.config.tracking_type > 0:
            analysis_type = "tracking"

        subtask_config = {
            "type": analysis_type,
            "callback": {
                "enabled": task.enable_callback,
                "url": task.callback_url
            }
        }

        # 构建分析配置
        analysis_config = {}
        if task.config:
            if hasattr(task.config, "detect_classes"):
                analysis_config["classes"] = task.config.detect_classes
            if hasattr(task.config, "roi"):
                analysis_config["roi"] = task.config.roi
            if hasattr(task.config, "roi_type"):
                analysis_config["roi_type"] = task.config.roi_type
            if hasattr(task.config, "nested_detection"):
                analysis_config["nested_detection"] = task.config.nested_detection

            # 跟踪配置
            if hasattr(task.config, "tracking_type") and task.config.tracking_type > 0:
                analysis_config["track_config"] = {
                    "enabled": True,
                    "tracker_type": "sort",
                    "max_age": task.config.max_lost_time if hasattr(task.config, "max_lost_time") else 30,
                    "min_hits": 3,
                    "iou_threshold": task.config.iou if hasattr(task.config, "iou") else 0.45
                }

        # 构建结果配置
        result_config = {
            "save_images": task.save_images,
            "return_base64": task.return_base64 if hasattr(task, "return_base64") else True,
            "storage": {
                "save_path": "results"
            }
        }

        # 构建完整配置
        task_config = {
            "model": model_config,
            "subtask": subtask_config,
            "analysis": analysis_config,
            "stream_url": task.stream_url,
            "result": result_config,
            "analysis_interval": task.analyze_interval if hasattr(task, "analyze_interval") else 1,
            "device": task.device if hasattr(task, "device") else "auto"
        }

        return task_config

    async def stop_task(self, task_id: str) -> Dict[str, Any]:
        """
        停止任务

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 停止结果
        """
        try:
            # 获取任务
            task = self.task_manager.get_task(task_id)
            if not task:
                return {
                    "success": False,
                    "message": f"任务不存在: {task_id}",
                    "task_id": task_id
                }

            # 停止任务
            if not await self.task_manager.stop_task(task_id):
                return {
                    "success": False,
                    "message": f"停止任务失败: {task_id}",
                    "task_id": task_id
                }

            return {
                "success": True,
                "message": "任务停止成功",
                "task_id": task_id
            }

        except Exception as e:
            logger.error(f"停止任务失败: {str(e)}")
            return {
                "success": False,
                "message": f"停止任务失败: {str(e)}",
                "task_id": task_id
            }

    async def start_task(self, model_code: str, stream_url: str, task_name: Optional[str] = None,
                     callback_urls: Optional[str] = None, output_url: Optional[str] = None,
                     analysis_type: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
                     enable_callback: bool = False, save_result: bool = False, save_images: bool = False,
                     frame_rate: Optional[int] = None, device: Optional[int] = None,
                     enable_alarm_recording: bool = False, alarm_recording_before: Optional[int] = None,
                     alarm_recording_after: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        启动单个任务

        Args:
            model_code: 模型代码
            stream_url: 流地址
            task_name: 任务名称
            callback_urls: 回调地址
            output_url: 输出地址
            analysis_type: 分析类型
            config: 配置参数
            enable_callback: 是否启用回调
            save_result: 是否保存结果
            save_images: 是否保存图像
            frame_rate: 帧率设置
            device: 设备类型
            enable_alarm_recording: 是否启用报警录像
            alarm_recording_before: 报警前录像时长
            alarm_recording_after: 报警后录像时长
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 启动结果
        """
        try:
            # 创建StreamTask对象
            task = StreamTask(
                model_code=model_code,
                stream_url=stream_url,
                task_name=task_name,
                output_url=output_url,
                analysis_type=analysis_type or "detection",
                enable_callback=enable_callback,
                callback_url=callback_urls,
                save_result=save_result,
                save_images=save_images,
                frame_rate=frame_rate,
                device=device,
                enable_alarm_recording=enable_alarm_recording,
                alarm_recording_before=alarm_recording_before,
                alarm_recording_after=alarm_recording_after
            )

            # 设置配置
            if config:
                task.config = DetectionConfig(**config)

            # 创建任务
            return await self.create_task(task)

        except Exception as e:
            logger.error(f"启动任务异常: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"启动任务异常: {str(e)}",
                "task_id": None
            }

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 任务状态
        """
        try:
            # 获取任务
            task = self.task_manager.get_task(task_id)
            if not task:
                return {
                    "success": False,
                    "message": f"任务不存在: {task_id}"
                }

            # 构建任务状态信息
            task_info = {
                "id": task_id,
                "status": task.get("status", 0),
                "start_time": task.get("start_time"),
                "stop_time": task.get("stop_time"),
                "duration": task.get("duration"),
                "error_message": task.get("error_message")
            }

            return {
                "success": True,
                "message": "获取任务状态成功",
                "task_info": task_info
            }

        except Exception as e:
            logger.error(f"获取任务状态异常: {str(e)}")
            return {
                "success": False,
                "message": f"获取任务状态异常: {str(e)}"
            }

    async def list_tasks(self, status: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """
        获取任务列表

        Args:
            status: 任务状态过滤
            limit: 返回数量限制

        Returns:
            Dict[str, Any]: 任务列表
        """
        try:
            # 获取任务列表
            tasks = self.task_manager.get_all_tasks(status)

            # 限制返回数量
            if limit > 0 and len(tasks) > limit:
                tasks = tasks[:limit]

            return {
                "total": len(tasks),
                "tasks": tasks
            }

        except Exception as e:
            logger.error(f"获取任务列表失败: {str(e)}")
            return {
                "total": 0,
                "tasks": []
            }
