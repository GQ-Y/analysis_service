"""
HTTP任务服务
提供基于HTTP的任务管理服务功能
"""
from typing import Dict, Any, List, Optional, Union
import uuid
import asyncio
from datetime import datetime
import json
import aiohttp
import logging

from core.task_management.utils.status import TaskStatus
from core.task_management.manager import TaskManager
from models.requests import StreamTask, BatchStreamTask
from shared.utils.logger import get_normal_logger, get_exception_logger, get_test_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()

class TaskService:
    """HTTP任务服务"""

    def __init__(self, task_manager: TaskManager = None):
        """
        初始化

        Args:
            task_manager: 任务管理器，如果为None则创建新实例
        """
        self.task_manager = task_manager or TaskManager()

    async def create_task(self, task: StreamTask, task_id: str = None) -> Dict[str, Any]:
        """
        创建任务

        Args:
            task: 任务参数
            task_id: 任务ID（可选）

        Returns:
            Dict[str, Any]: 创建结果
        """
        try:
            # 如果未提供任务ID，则生成一个
            if not task_id:
                task_id = str(uuid.uuid4())

            # 构建任务配置
            task_config = self._build_task_config(task, task_id)

            # 检查是否有测试标记
            if hasattr(task, "test_markers") and task.test_markers:
                normal_logger.info(f"任务包含测试标记: {task.test_markers}")
                task_config["test_markers"] = task.test_markers
                # 添加测试日志标记
                from shared.utils.logger import TEST_LOG_MARKER
                normal_logger.info(f"{TEST_LOG_MARKER} TASK_CREATE_START")

            # 添加任务到管理器
            if not self.task_manager.add_task(task_id, task_config):
                return {
                    "success": False,
                    "message": "添加任务失败",
                    "task_id": task_id
                }

            # 启动任务
            if not await self.task_manager.start_task(task_id):
                return {
                    "success": False,
                    "message": "启动任务失败",
                    "task_id": task_id
                }

            # 记录任务创建成功日志标记
            from shared.utils.logger import TEST_LOG_MARKER
            normal_logger.info(f"{TEST_LOG_MARKER} TASK_CREATE_SUCCESS")

            return {
                "success": True,
                "message": "创建并启动任务成功",
                "task_id": task_id
            }

        except Exception as e:
            exception_logger.exception(f"创建任务异常: {str(e)}")
            return {
                "success": False,
                "message": f"创建任务异常: {str(e)}",
                "task_id": task_id if 'task_id' in locals() else None
            }

    async def start_task(self, model_code: str, stream_url: str, task_name: Optional[str] = None,
                     callback_urls: Optional[str] = None, output_url: Optional[str] = None,
                     analysis_type: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
                     enable_callback: bool = False, save_result: bool = False, save_images: bool = False,
                     frame_rate: Optional[int] = None, device: Optional[int] = None,
                     enable_alarm_recording: bool = False, alarm_recording_before: Optional[int] = None,
                     alarm_recording_after: Optional[int] = None, analysis_interval: Optional[int] = None,
                     callback_interval: Optional[int] = None, **kwargs) -> Dict[str, Any]:
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
            analysis_interval: 分析间隔(帧)，每隔多少帧分析一次
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 启动结果
        """
        try:
            # 确保analysis_interval是有效的整数
            normal_logger.info(f"接收到的分析间隔值: {analysis_interval}, 类型: {type(analysis_interval)}")
            if analysis_interval is not None:
                try:
                    analysis_interval = int(analysis_interval)
                    if analysis_interval < 1:
                        analysis_interval = 1
                        normal_logger.warning(f"无效的分析间隔值: {analysis_interval}，使用默认值1")
                    else:
                        normal_logger.info(f"使用分析间隔值: {analysis_interval}")
                except (ValueError, TypeError):
                    analysis_interval = 1
                    normal_logger.warning(f"无效的分析间隔值: {analysis_interval}，使用默认值1")

            # 处理回调间隔值
            if callback_interval is not None:
                try:
                    callback_interval = int(callback_interval)
                    if callback_interval < 0:
                        callback_interval = 0
                        normal_logger.warning(f"无效的回调间隔值: {callback_interval}，使用默认值0")
                    else:
                        normal_logger.info(f"使用回调间隔值: {callback_interval}秒")
                except (ValueError, TypeError):
                    callback_interval = 0
                    normal_logger.warning(f"无效的回调间隔值: {callback_interval}，使用默认值0")

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
                alarm_recording_after=alarm_recording_after,
                analysis_interval=analysis_interval,
                callback_interval=callback_interval
            )

            # 设置配置
            if config:
                from models.requests import DetectionConfig
                task.config = DetectionConfig(**config)

            # 创建任务
            result = await self.create_task(task)

            if result["success"]:
                test_logger.info("TEST_LOG_MARKER: TASK_CREATE_SUCCESS")

            return result

        except Exception as e:
            exception_logger.exception(f"启动任务异常: {str(e)}")
            return {
                "success": False,
                "message": f"启动任务异常: {str(e)}",
                "task_id": None
            }

    async def create_batch_tasks(self, batch_task: BatchStreamTask) -> Dict[str, Any]:
        """
        批量创建任务

        Args:
            batch_task: 批量任务参数

        Returns:
            Dict[str, Any]: 创建结果
        """
        try:
            # 生成批次ID
            batch_id = str(uuid.uuid4())

            # 创建任务结果
            results = []

            # 处理每个子任务
            for task in batch_task.tasks:
                # 生成任务ID
                task_id = str(uuid.uuid4())

                # 设置全局回调
                if batch_task.callback_urls and not task.callback_url:
                    task.callback_url = batch_task.callback_urls
                    task.enable_callback = True

                # 设置全局分析间隔
                if hasattr(batch_task, "analyze_interval") and batch_task.analyze_interval > 0:
                    task.analysis_interval = batch_task.analyze_interval

                # 创建任务
                result = await self.create_task(task, task_id)

                # 添加到结果列表
                results.append({
                    "stream_url": task.stream_url,
                    "task_id": result.get("task_id"),
                    "success": result.get("success", False),
                    "message": result.get("message", "")
                })

            # 统计成功和失败数量
            success_count = sum(1 for r in results if r.get("success", False))
            failed_count = len(results) - success_count

            if success_count > 0:
                test_logger.info("TEST_LOG_MARKER: BATCH_TASK_CREATE_SUCCESS")

            return {
                "success": success_count > 0,
                "message": f"批量创建任务完成，成功: {success_count}，失败: {failed_count}",
                "batch_id": batch_id,
                "results": results
            }

        except Exception as e:
            exception_logger.exception(f"批量创建任务异常: {str(e)}")
            return {
                "success": False,
                "message": f"批量创建任务异常: {str(e)}",
                "batch_id": None,
                "results": []
            }

    async def stop_task(self, task_id: str) -> Dict[str, Any]:
        """
        停止任务并删除

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 停止结果
        """
        try:
            # 检查任务是否存在
            if not self.task_manager.has_task(task_id):
                return {
                    "success": False,
                    "message": f"任务不存在: {task_id}"
                }

            # 停止任务
            if not await self.task_manager.stop_task(task_id):
                return {
                    "success": False,
                    "message": f"停止任务失败: {task_id}"
                }

            # 删除任务
            if not self.task_manager.delete_task(task_id):
                normal_logger.warning(f"删除任务失败: {task_id}")
                # 即使删除失败也返回停止成功，因为任务已经停止
            else:
                normal_logger.info(f"任务已停止并删除: {task_id}")

            test_logger.info("TEST_LOG_MARKER: TASK_STOP_SUCCESS")

            return {
                "success": True,
                "message": f"任务已停止并删除: {task_id}"
            }

        except Exception as e:
            exception_logger.exception(f"停止任务异常: {str(e)}")
            return {
                "success": False,
                "message": f"停止任务异常: {str(e)}"
            }

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务详情

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 任务详情
        """
        try:
            # 获取任务
            task = self.task_manager.get_task(task_id)
            if not task:
                return {
                    "success": False,
                    "message": f"任务不存在: {task_id}",
                    "task": None
                }

            return {
                "success": True,
                "message": "获取任务成功",
                "task": task
            }

        except Exception as e:
            exception_logger.exception(f"获取任务异常: {str(e)}")
            return {
                "success": False,
                "message": f"获取任务异常: {str(e)}",
                "task": None
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

            test_logger.info("TEST_LOG_MARKER: TASK_STATUS_CHECK_SUCCESS")

            return {
                "success": True,
                "message": "获取任务状态成功",
                "data": task_info
            }

        except Exception as e:
            exception_logger.exception(f"获取任务状态异常: {str(e)}")
            return {
                "success": False,
                "message": f"获取任务状态异常: {str(e)}"
            }

    async def list_tasks(self, status: Optional[int] = None, limit: int = 100) -> Dict[str, Any]:
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

            # 处理任务数据，确保可序列化
            serializable_tasks = []
            for task in tasks:
                # 创建一个新的可序列化字典
                serializable_task = {
                    "id": task.get("id"),
                    "status": task.get("status"),
                    "created_at": task.get("created_at"),
                    "updated_at": task.get("updated_at"),
                    "error": task.get("error")
                }

                # 添加任务数据
                if "data" in task:
                    task_data = task["data"]
                    # 提取基本信息
                    serializable_task["task_name"] = task_data.get("task_name", "")

                    # 提取参数
                    if "params" in task_data:
                        params = task_data["params"]
                        # 从模型配置中获取正确的模型代码
                        serializable_task["model_code"] = params.get("model", {}).get("code", task_data.get("type", ""))
                        serializable_task["stream_url"] = params.get("stream_url", "")
                        serializable_task["output_url"] = params.get("output_url", "")
                        serializable_task["analysis_type"] = params.get("subtask", {}).get("type", "")

                        # 提取回调信息
                        callback = params.get("subtask", {}).get("callback", {})
                        serializable_task["enable_callback"] = callback.get("enabled", False)
                        serializable_task["callback_urls"] = callback.get("url", "")

                # 添加时间信息
                serializable_task["start_time"] = task.get("start_time", None)
                serializable_task["stop_time"] = task.get("stop_time", None)
                serializable_task["duration"] = task.get("duration", None)

                # 添加错误信息
                serializable_task["error_message"] = task.get("error", None)

                serializable_tasks.append(serializable_task)

            test_logger.info("TEST_LOG_MARKER: TASK_LIST_SUCCESS")

            return {
                "success": True,
                "message": "获取任务列表成功",
                "total": len(serializable_tasks),
                "tasks": serializable_tasks
            }

        except Exception as e:
            exception_logger.exception(f"获取任务列表异常: {str(e)}")
            return {
                "success": False,
                "message": f"获取任务列表异常: {str(e)}",
                "total": 0,
                "tasks": []
            }

    def _build_task_config(self, task: StreamTask, task_id: str = None) -> Dict[str, Any]:
        """
        构建任务配置 - 重构版
        仅支持目标检测分析类型

        Args:
            task: 任务参数
            task_id: 任务ID（可选）

        Returns:
            Dict[str, Any]: 任务配置
        """
        # 构建模型配置
        model_config = {
            "code": task.model_code,
            "confidence": task.config.confidence if task.config and hasattr(task.config, "confidence") else 0.5,
            "iou_threshold": task.config.iou_threshold if task.config and hasattr(task.config, "iou_threshold") else 0.45
        }

        # 构建子任务配置
        # 注意：重构后只支持detection类型
        analysis_type = "detection"
        if task.analysis_type and task.analysis_type != "detection":
            normal_logger.warning(f"不支持的分析类型: {task.analysis_type}，使用默认类型: detection")

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
            # 复制所有配置参数
            # 确保 task.config (Pydantic模型) 被转换为纯字典
            if hasattr(task.config, "model_dump") and callable(task.config.model_dump):
                analysis_config = task.config.model_dump(exclude_unset=True) # Pydantic V2+
            elif hasattr(task.config, "dict") and callable(task.config.dict):
                analysis_config = task.config.dict(exclude_unset=True) # Pydantic V1
            else:
                # Fallback or error if not a Pydantic model or unknown version
                # For now, let's try iterating, but this was the source of the issue
                analysis_config = {}
                for attr_name in dir(task.config):
                    if attr_name.startswith('_') or callable(getattr(task.config, attr_name)):
                        continue
                    # 跳过不再支持的跟踪相关参数 (保留，以防万一)
                    if attr_name in ["tracking_type", "max_lost_time", "min_hits", 
                                    "related_cameras", "feature_type"]:
                        continue
                    attr_value = getattr(task.config, attr_name)
                    if attr_value is None:
                        continue
                    analysis_config[attr_name] = attr_value
                normal_logger.warning("task.config 不是预期的Pydantic模型，或版本未知，尝试了属性迭代。")

        # 构建结果配置
        result_config = {
            "save_images": task.save_images,
            "save_result": task.save_result,
            "return_base64": task.return_base64 if hasattr(task, "return_base64") else True,
            "storage": {
                "save_path": task.output_url if hasattr(task, "output_url") and task.output_url else "results"
            }
        }

        # 如果有报警录像配置，添加到结果配置
        if task.enable_alarm_recording:
            result_config["alarm_recording"] = {
                "enabled": True,
                "before_seconds": task.alarm_recording_before if hasattr(task, "alarm_recording_before") else 5,
                "after_seconds": task.alarm_recording_after if hasattr(task, "alarm_recording_after") else 5
            }

        # 记录分析间隔值
        has_interval = hasattr(task, "analysis_interval")
        interval_value = task.analysis_interval if has_interval else None
        normal_logger.info(f"构建任务配置 - 是否有analysis_interval属性: {has_interval}, 值: {interval_value}, 类型: {type(interval_value) if interval_value is not None else 'None'}")

        # 确定最终使用的分析间隔值
        final_interval = task.analysis_interval if hasattr(task, "analysis_interval") and task.analysis_interval is not None and task.analysis_interval > 0 else 1
        normal_logger.info(f"最终使用的分析间隔值: {final_interval}")

        # 获取回调间隔值
        callback_interval = None
        if hasattr(task, "callback_interval") and task.callback_interval is not None:
            callback_interval = task.callback_interval
            normal_logger.info(f"使用回调间隔值: {callback_interval}秒")

        # 构建参数配置（这是TaskManager期望的结构）
        params_config = {
            "model": model_config,
            "subtask": subtask_config,
            "analysis": analysis_config,
            "stream_url": task.stream_url,
            "result": result_config,
            "analysis_interval": final_interval,
            "callback_interval": callback_interval,
            "device": task.device if hasattr(task, "device") else "auto",
            "save_images": task.save_images
        }

        # 添加任务ID（如果有）
        if task_id:
            params_config["task_id"] = task_id

        # 添加输出URL（如果有）
        if hasattr(task, "output_url") and task.output_url:
            params_config["output_url"] = task.output_url

        # 添加帧率设置（如果有）
        if hasattr(task, "frame_rate") and task.frame_rate:
            params_config["frame_rate"] = task.frame_rate

        # 添加任务名称到params中（如果有）
        if hasattr(task, "task_name") and task.task_name:
            params_config["task_name"] = task.task_name

        # 构建完整的任务配置，包含TaskManager期望的结构
        task_config = {
            "params": params_config,  # TaskManager期望的params字段
            "task_name": task.task_name if hasattr(task, "task_name") and task.task_name else "",
            "type": "detection"  # 重构后只支持detection类型
        }

        return task_config
