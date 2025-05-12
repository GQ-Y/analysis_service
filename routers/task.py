"""
任务管理路由
提供任务的创建、查询、停止等API
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, Request
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

from models.requests import StreamTask, BatchStreamTask
from models.responses import BaseResponse
from services.http.task_service import TaskService
from core.task_management.utils.status import TaskStatus
from core.config import settings
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

# 创建路由
router = APIRouter(
    prefix="/api/v1/tasks",
    tags=["任务管理"],
    responses={404: {"description": "Not found"}},
)

# 依赖注入
async def get_task_service(request: Request) -> TaskService:
    """获取任务服务实例"""
    if not hasattr(request.app.state, "task_service"):
        raise HTTPException(status_code=500, detail="任务服务未初始化")
    return request.app.state.task_service

@router.post("/start", response_model=BaseResponse, summary="启动单个流分析任务")
async def start_task(
    task: StreamTask,
    task_service: TaskService = Depends(get_task_service)
) -> BaseResponse:
    """
    启动单个流分析任务

    支持的配置参数包括：
    - model_code: 模型代码，必填
    - stream_url: 流地址，必填
    - task_name: 任务名称，可选
    - output_url: 输出地址，可选
    - save_result: 是否保存结果，默认为false
    - config: 检测配置参数，可选，包括：
        - confidence: 置信度阈值，范围0-1
        - iou: IoU阈值，范围0-1
        - classes: 需要检测的类别ID列表
        - roi: 感兴趣区域，支持矩形、多边形和线段
        - nested_detection: 是否进行嵌套检测

    Args:
        task: 流分析任务参数

    Returns:
        BaseResponse: 响应结果，包含task_id
    """
    request_id = str(uuid.uuid4())

    try:
        # 准备配置参数
        config = {}
        if task.config:
            config = task.config.model_dump()

            # 处理ROI配置
            if hasattr(task.config, "roi") and task.config.roi:
                config["roi"] = task.config.roi

            # 处理检测类别
            if hasattr(task.config, "classes") and task.config.classes:
                config["detect_classes"] = task.config.classes

            # 处理置信度阈值
            if hasattr(task.config, "confidence") and task.config.confidence is not None:
                config["confidence"] = task.config.confidence

            # 处理IoU阈值
            if hasattr(task.config, "iou_threshold") and task.config.iou_threshold is not None:
                config["iou_threshold"] = task.config.iou_threshold

            # 处理嵌套检测
            if hasattr(task.config, "nested_detection"):
                config["nested_detection"] = task.config.nested_detection

            # 处理图像尺寸
            if hasattr(task.config, "image_size") and task.config.image_size:
                config["image_size"] = task.config.image_size

            # 处理推理引擎
            if hasattr(task.config, "engine") and task.config.engine is not None:
                config["engine"] = task.config.engine

            # 处理YOLO版本
            if hasattr(task.config, "yolo_version") and task.config.yolo_version is not None:
                config["yolo_version"] = task.config.yolo_version

            # 处理自定义权重路径
            if hasattr(task.config, "custom_weights_path") and task.config.custom_weights_path:
                config["custom_weights_path"] = task.config.custom_weights_path

            # 处理提示类型和提示内容
            if hasattr(task.config, "prompt_type") and task.config.prompt_type is not None:
                config["prompt_type"] = task.config.prompt_type

                if task.config.prompt_type == 1 and hasattr(task.config, "text_prompt") and task.config.text_prompt:
                    config["text_prompt"] = task.config.text_prompt
                elif task.config.prompt_type == 2 and hasattr(task.config, "visual_prompt") and task.config.visual_prompt:
                    config["visual_prompt"] = task.config.visual_prompt

            # 处理分割
            if hasattr(task.config, "segmentation"):
                config["segmentation"] = task.config.segmentation

            # 处理NMS类型
            if hasattr(task.config, "nms_type") and task.config.nms_type is not None:
                config["nms_type"] = task.config.nms_type

            # 处理最大检测目标数量
            if hasattr(task.config, "max_detections") and task.config.max_detections is not None:
                config["max_detections"] = task.config.max_detections

            # 处理设备类型
            if hasattr(task.config, "device") and task.config.device is not None:
                config["device"] = task.config.device

            # 处理半精度
            if hasattr(task.config, "half_precision"):
                config["half_precision"] = task.config.half_precision

            # 处理跟踪相关参数
            if hasattr(task.config, "tracking_type") and task.config.tracking_type is not None:
                config["tracking_type"] = task.config.tracking_type

            if hasattr(task.config, "max_tracks") and task.config.max_tracks is not None:
                config["max_tracks"] = task.config.max_tracks

            if hasattr(task.config, "max_lost_time") and task.config.max_lost_time is not None:
                config["max_lost_time"] = task.config.max_lost_time

            if hasattr(task.config, "feature_type") and task.config.feature_type is not None:
                config["feature_type"] = task.config.feature_type

            if hasattr(task.config, "related_cameras") and task.config.related_cameras:
                config["related_cameras"] = task.config.related_cameras

            # 处理计数和速度估计
            if hasattr(task.config, "counting_enabled"):
                config["counting_enabled"] = task.config.counting_enabled

            if hasattr(task.config, "time_threshold") and task.config.time_threshold is not None:
                config["time_threshold"] = task.config.time_threshold

            if hasattr(task.config, "speed_estimation"):
                config["speed_estimation"] = task.config.speed_estimation

            if hasattr(task.config, "object_filter") and task.config.object_filter:
                config["object_filter"] = task.config.object_filter

        # 启动任务
        result = await task_service.start_task(
            model_code=task.model_code,
            stream_url=task.stream_url,
            task_name=task.task_name,
            callback_urls=task.callback_url,  # 使用任务中的回调URL
            output_url=task.output_url,
            analysis_type=task.analysis_type,
            config=config,
            enable_callback=task.enable_callback,
            save_result=task.save_result,
            save_images=task.save_images,
            frame_rate=task.frame_rate,
            device=task.device,
            enable_alarm_recording=task.enable_alarm_recording,
            alarm_recording_before=task.alarm_recording_before,
            alarm_recording_after=task.alarm_recording_after
        )

        if not result["success"]:
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/start",
                success=False,
                message=result["message"],
                code=500,
                data=None
            )

        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/start",
            success=True,
            message="任务启动成功",
            code=200,
            data={"task_id": result["task_id"]}
        )

    except Exception as e:
        logger.error(f"启动任务失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/start",
            success=False,
            message=f"启动任务失败: {str(e)}",
            code=500,
            data=None
        )

@router.post("/batch/start", response_model=BaseResponse, summary="批量启动流分析任务")
async def start_batch_tasks(
    batch_task: BatchStreamTask,
    task_service: TaskService = Depends(get_task_service)
) -> BaseResponse:
    """
    批量启动流分析任务

    支持的参数包括：
    - tasks: 流分析任务列表，每个任务包含：
        - model_code: 模型代码，必填
        - stream_url: 流地址，必填
        - task_name: 任务名称，可选
        - output_url: 输出地址，可选
        - save_result: 是否保存结果，默认为false
        - config: 检测配置参数，可选
    - callback_urls: 回调地址，多个用逗号分隔，可选
    - analyze_interval: 分析间隔(秒)，默认为1
    - alarm_interval: 报警间隔(秒)，默认为60
    - random_interval: 随机延迟区间(秒)，默认为(0,0)
    - push_interval: 推送间隔(秒)，默认为5

    全局配置参数会应用到每个子任务中。

    Args:
        batch_task: 批量流分析任务参数

    Returns:
        BaseResponse: 响应结果，包含成功和失败的任务信息
    """
    request_id = str(uuid.uuid4())

    try:
        # 批量启动任务
        task_ids = []
        failed_tasks = []

        for task in batch_task.tasks:
            # 准备配置参数
            config = {}
            if task.config:
                config = task.config.model_dump()

                # 处理ROI配置
                if hasattr(task.config, "roi") and task.config.roi:
                    config["roi"] = task.config.roi

                # 处理检测类别
                if hasattr(task.config, "classes") and task.config.classes:
                    config["detect_classes"] = task.config.classes

                # 处理置信度阈值
                if hasattr(task.config, "confidence") and task.config.confidence is not None:
                    config["confidence"] = task.config.confidence

                # 处理IoU阈值
                if hasattr(task.config, "iou_threshold") and task.config.iou_threshold is not None:
                    config["iou_threshold"] = task.config.iou_threshold

                # 处理嵌套检测
                if hasattr(task.config, "nested_detection"):
                    config["nested_detection"] = task.config.nested_detection

                # 处理图像尺寸
                if hasattr(task.config, "image_size") and task.config.image_size:
                    config["image_size"] = task.config.image_size

                # 处理推理引擎
                if hasattr(task.config, "engine") and task.config.engine is not None:
                    config["engine"] = task.config.engine

                # 处理YOLO版本
                if hasattr(task.config, "yolo_version") and task.config.yolo_version is not None:
                    config["yolo_version"] = task.config.yolo_version

                # 处理自定义权重路径
                if hasattr(task.config, "custom_weights_path") and task.config.custom_weights_path:
                    config["custom_weights_path"] = task.config.custom_weights_path

                # 处理提示类型和提示内容
                if hasattr(task.config, "prompt_type") and task.config.prompt_type is not None:
                    config["prompt_type"] = task.config.prompt_type

                    if task.config.prompt_type == 1 and hasattr(task.config, "text_prompt") and task.config.text_prompt:
                        config["text_prompt"] = task.config.text_prompt
                    elif task.config.prompt_type == 2 and hasattr(task.config, "visual_prompt") and task.config.visual_prompt:
                        config["visual_prompt"] = task.config.visual_prompt

                # 处理分割
                if hasattr(task.config, "segmentation"):
                    config["segmentation"] = task.config.segmentation

                # 处理NMS类型
                if hasattr(task.config, "nms_type") and task.config.nms_type is not None:
                    config["nms_type"] = task.config.nms_type

                # 处理最大检测目标数量
                if hasattr(task.config, "max_detections") and task.config.max_detections is not None:
                    config["max_detections"] = task.config.max_detections

                # 处理设备类型
                if hasattr(task.config, "device") and task.config.device is not None:
                    config["device"] = task.config.device

                # 处理半精度
                if hasattr(task.config, "half_precision"):
                    config["half_precision"] = task.config.half_precision

                # 处理跟踪相关参数
                if hasattr(task.config, "tracking_type") and task.config.tracking_type is not None:
                    config["tracking_type"] = task.config.tracking_type

                if hasattr(task.config, "max_tracks") and task.config.max_tracks is not None:
                    config["max_tracks"] = task.config.max_tracks

                if hasattr(task.config, "max_lost_time") and task.config.max_lost_time is not None:
                    config["max_lost_time"] = task.config.max_lost_time

                if hasattr(task.config, "feature_type") and task.config.feature_type is not None:
                    config["feature_type"] = task.config.feature_type

                if hasattr(task.config, "related_cameras") and task.config.related_cameras:
                    config["related_cameras"] = task.config.related_cameras

                # 处理计数和速度估计
                if hasattr(task.config, "counting_enabled"):
                    config["counting_enabled"] = task.config.counting_enabled

                if hasattr(task.config, "time_threshold") and task.config.time_threshold is not None:
                    config["time_threshold"] = task.config.time_threshold

                if hasattr(task.config, "speed_estimation"):
                    config["speed_estimation"] = task.config.speed_estimation

                if hasattr(task.config, "object_filter") and task.config.object_filter:
                    config["object_filter"] = task.config.object_filter

            # 如果批量任务中有全局配置，合并到每个任务的配置中
            if hasattr(batch_task, "analyze_interval") and batch_task.analyze_interval:
                config["analyze_interval"] = batch_task.analyze_interval

            if hasattr(batch_task, "alarm_interval") and batch_task.alarm_interval:
                config["alarm_interval"] = batch_task.alarm_interval

            if hasattr(batch_task, "push_interval") and batch_task.push_interval:
                config["push_interval"] = batch_task.push_interval

            if hasattr(batch_task, "random_interval") and batch_task.random_interval:
                config["random_interval"] = batch_task.random_interval

            result = await task_service.start_task(
                model_code=task.model_code,
                stream_url=task.stream_url,
                task_name=task.task_name,
                callback_urls=batch_task.callback_urls,
                output_url=task.output_url,
                analysis_type=task.analysis_type,
                config=config,
                enable_callback=task.enable_callback or bool(batch_task.callback_urls),
                save_result=task.save_result,
                save_images=task.save_images,
                frame_rate=task.frame_rate,
                device=task.device,
                enable_alarm_recording=task.enable_alarm_recording,
                alarm_recording_before=task.alarm_recording_before,
                alarm_recording_after=task.alarm_recording_after
            )

            if result["success"]:
                task_ids.append(result["task_id"])
            else:
                failed_tasks.append({
                    "stream_url": task.stream_url,
                    "error": result["message"]
                })

        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/batch/start",
            success=len(failed_tasks) == 0,
            message="批量任务启动完成" if len(failed_tasks) == 0 else f"部分任务启动失败: {len(failed_tasks)}/{len(batch_task.tasks)}",
            code=200 if len(failed_tasks) == 0 else 207,
            data={
                "task_ids": task_ids,
                "failed_tasks": failed_tasks,
                "total": len(batch_task.tasks),
                "success": len(task_ids),
                "failed": len(failed_tasks)
            }
        )

    except Exception as e:
        logger.error(f"批量启动任务失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/batch/start",
            success=False,
            message=f"批量启动任务失败: {str(e)}",
            code=500,
            data=None
        )

@router.post("/stop/{task_id}", response_model=BaseResponse, summary="停止分析任务")
async def stop_task(
    task_id: str = Path(..., description="任务ID"),
    task_service: TaskService = Depends(get_task_service)
) -> BaseResponse:
    """
    停止正在运行的分析任务

    通过任务ID停止一个正在运行的分析任务。任务将首先进入"停止中"状态，然后在资源释放完成后变为"已停止"状态。

    Args:
        task_id: 要停止的任务ID

    Returns:
        BaseResponse: 响应结果，包含操作状态和消息
    """
    request_id = str(uuid.uuid4())

    try:
        # 停止任务
        result = await task_service.stop_task(task_id)

        if not result["success"]:
            return BaseResponse(
                requestId=request_id,
                path=f"/api/v1/tasks/stop/{task_id}",
                success=False,
                message=result["message"],
                code=404 if "不存在" in result["message"] else 500,
                data=None
            )

        return BaseResponse(
            requestId=request_id,
            path=f"/api/v1/tasks/stop/{task_id}",
            success=True,
            message="任务停止指令已发送",
            code=200,
            data={"task_id": task_id}
        )

    except Exception as e:
        logger.error(f"停止任务失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path=f"/api/v1/tasks/stop/{task_id}",
            success=False,
            message=f"停止任务失败: {str(e)}",
            code=500,
            data=None
        )

@router.get("/status/{task_id}", response_model=BaseResponse, summary="获取任务状态")
async def get_task_status(
    task_id: str = Path(..., description="任务ID"),
    task_service: TaskService = Depends(get_task_service)
) -> BaseResponse:
    """
    获取任务状态

    返回指定任务ID的详细状态信息，包括：
    - 任务状态（等待中、处理中、已完成、失败等）
    - 开始时间
    - 结束时间
    - 运行时长
    - 错误信息（如果有）

    Args:
        task_id: 任务ID

    Returns:
        BaseResponse: 响应结果，包含任务状态详情
    """
    request_id = str(uuid.uuid4())

    try:
        # 获取任务状态
        result = await task_service.get_task_status(task_id)

        if not result["success"]:
            return BaseResponse(
                requestId=request_id,
                path=f"/api/v1/tasks/status/{task_id}",
                success=False,
                message=result["message"],
                code=404 if "不存在" in result["message"] else 500,
                data=None
            )

        # 构建任务信息
        task_info = result["task_info"]
        status_map = {
            TaskStatus.WAITING: "等待中",
            TaskStatus.PROCESSING: "处理中",
            TaskStatus.COMPLETED: "已完成",
            TaskStatus.FAILED: "失败",
            TaskStatus.STOPPING: "停止中",
            TaskStatus.STOPPED: "已停止"
        }

        return BaseResponse(
            requestId=request_id,
            path=f"/api/v1/tasks/status/{task_id}",
            success=True,
            message="获取任务状态成功",
            code=200,
            data={
                "task_id": task_id,
                "status": task_info["status"],
                "status_text": status_map.get(task_info["status"], "未知"),
                "start_time": task_info["start_time"],
                "stop_time": task_info["stop_time"],
                "duration": task_info["duration"],
                "error_message": task_info["error_message"]
            }
        )

    except Exception as e:
        logger.error(f"获取任务状态失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path=f"/api/v1/tasks/status/{task_id}",
            success=False,
            message=f"获取任务状态失败: {str(e)}",
            code=500,
            data=None
        )

@router.get("/list", response_model=BaseResponse, summary="获取任务列表")
async def list_tasks(
    status: Optional[int] = Query(None, description="任务状态过滤：0-等待中, 1-处理中, 2-已完成, -1-失败, -4-停止中, -5-已停止"),
    limit: int = Query(100, description="返回数量限制，默认100"),
    task_service: TaskService = Depends(get_task_service)
) -> BaseResponse:
    """
    获取任务列表

    返回系统中的任务列表，支持按状态过滤和限制返回数量。

    每个任务包含以下信息：
    - 任务ID
    - 任务名称
    - 模型代码
    - 流URL
    - 任务状态
    - 开始时间
    - 结束时间
    - 运行时长
    - 创建时间
    - 更新时间
    - 错误信息（如果有）

    Args:
        status: 任务状态过滤：0-等待中, 1-处理中, 2-已完成, -1-失败, -4-停止中, -5-已停止
        limit: 返回数量限制，默认100

    Returns:
        BaseResponse: 响应结果，包含任务列表和总数
    """
    request_id = str(uuid.uuid4())

    try:
        # 获取任务列表
        result = await task_service.list_tasks(status, limit)

        if not result["success"]:
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/list",
                success=False,
                message=result["message"],
                code=500,
                data=None
            )

        # 构建状态文本映射
        status_map = {
            TaskStatus.WAITING: "等待中",
            TaskStatus.PROCESSING: "处理中",
            TaskStatus.COMPLETED: "已完成",
            TaskStatus.FAILED: "失败",
            TaskStatus.STOPPING: "停止中",
            TaskStatus.STOPPED: "已停止"
        }

        # 添加状态文本
        for task in result["tasks"]:
            task["status_text"] = status_map.get(task["status"], "未知")

        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/list",
            success=True,
            message=result["message"],
            code=200,
            data={
                "tasks": result["tasks"],
                "total": result["total"]
            }
        )

    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/list",
            success=False,
            message=f"获取任务列表失败: {str(e)}",
            code=500,
            data=None
        )
