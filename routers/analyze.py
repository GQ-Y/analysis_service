"""
分析路由模块
处理视觉分析请求，包括图片分析、视频分析和流分析
"""
import os
import json
import uuid
import tempfile
from pathlib import Path
from typing import List, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form, BackgroundTasks, Request
from pydantic import BaseModel, Field, validator
from core.detector import YOLODetector
from core.redis_manager import RedisManager
from core.task_queue import TaskQueue, TaskStatus
from core.resource import ResourceMonitor
from core.models import (
    StandardResponse,
    AnalysisType,
    AnalysisStatus,
    DetectionResult,
    SegmentationResult,
    TrackingResult,
    CrossCameraResult
)
from core.exceptions import (
    InvalidInputException,
    ModelLoadException,
    ProcessingException,
    ResourceNotFoundException
)
from models.responses import (
    ImageAnalysisResponse,
    VideoAnalysisResponse,
    StreamAnalysisResponse,
    BaseApiResponse,
    StreamBatchData,
    ImageAnalysisData,
    VideoAnalysisData,
    ResourceStatusResponse
)
from models.requests import (
    ImageAnalysisRequest,
    VideoAnalysisRequest,
    StreamAnalysisRequest,
    StreamTask,
    TaskStatusRequest
)
from shared.utils.logger import setup_logger
from core.config import settings
from datetime import datetime
import base64
import re

logger = setup_logger(__name__)

# 初始化组件
detector = YOLODetector()
resource_monitor = ResourceMonitor()
redis_manager = RedisManager()
task_queue = TaskQueue()

# 创建路由器
router = APIRouter(
    tags=["视觉分析"],
    responses={
        400: {"model": StandardResponse, "description": "请求参数错误"},
        401: {"model": StandardResponse, "description": "未授权访问"},
        403: {"model": StandardResponse, "description": "禁止访问"},
        404: {"model": StandardResponse, "description": "资源未找到"},
        500: {"model": StandardResponse, "description": "服务器内部错误"}
    }
)

# 状态映射
status_map = {
    "waiting": 0,     # 等待中
    "processing": 1,  # 运行中
    "completed": 2,   # 已完成
    "failed": -1      # 失败
}

# 基础请求模型
class BaseAnalysisRequest(BaseModel):
    """基础分析请求模型"""
    model_code: str = Field(..., description="模型代码")
    task_name: Optional[str] = Field(None, description="任务名称")
    callback_urls: Optional[str] = Field(None, description="用户回调地址，多个用逗号分隔。仅当enable_callback=true时生效。")
    enable_callback: bool = Field(False, description="是否启用用户回调。注意：系统级回调始终启用，不受此参数控制。")
    save_result: bool = Field(False, description="是否保存结果")
    config: Optional[dict] = Field(None, description="分析配置")
    
    model_config = {"protected_namespaces": ()}

class ImageAnalysisRequest(BaseAnalysisRequest):
    """图片分析请求"""
    image_urls: List[str] = Field(..., description="图片URL列表，支持以下格式：\n- HTTP/HTTPS URL\n- Base64编码的图片数据（以 'data:image/' 开头）\n- Blob URL（以 'blob:' 开头）")
    is_base64: bool = Field(False, description="是否返回base64编码的结果图片")

    @validator('image_urls')
    def validate_image_urls(cls, v):
        for url in v:
            # 检查是否是有效的 HTTP/HTTPS URL
            if url.startswith(('http://', 'https://')):
                continue
            # 检查是否是有效的 Base64 图片数据
            elif url.startswith('data:image/'):
                try:
                    # 提取实际的 base64 数据
                    base64_data = url.split(',')[1]
                    base64.b64decode(base64_data)
                except:
                    raise ValueError(f"Invalid base64 image data: {url[:50]}...")
            # 检查是否是有效的 Blob URL
            elif url.startswith('blob:'):
                if not re.match(r'^blob:(http[s]?://[^/]+/[a-f0-9-]+)$', url):
                    raise ValueError(f"Invalid blob URL: {url}")
            else:
                raise ValueError(f"Unsupported image URL format: {url}")
        return v

class VideoAnalysisRequest(BaseAnalysisRequest):
    """视频分析请求"""
    video_url: str = Field(..., description="视频URL")

class StreamAnalysisRequest(BaseAnalysisRequest):
    """流分析请求"""
    stream_url: str = Field(..., description="流URL")
    analysis_type: AnalysisType = Field(AnalysisType.DETECTION, description="分析类型")
    task_id: Optional[str] = Field(None, description="任务ID，如果不提供将自动生成")
    callback_url: Optional[str] = Field(None, description="系统回调URL，优先作为系统级回调地址。系统回调始终执行，如果失败会导致任务停止。")

class TaskStatusRequest(BaseModel):
    """任务状态查询请求"""
    task_id: str = Field(..., description="任务ID")

# 依赖注入函数
async def get_detector() -> YOLODetector:
    """获取检测器实例"""
    return detector

async def get_redis() -> RedisManager:
    """获取Redis管理器实例"""
    return redis_manager

async def get_task_queue() -> TaskQueue:
    """获取任务队列实例"""
    return task_queue

@router.post(
    "/image",
    response_model=StandardResponse,
    summary="图片分析",
    description="""
    分析图片中的目标
    
    支持以下功能:
    - 目标检测
    - 实例分割
    - 目标跟踪
    
    请求示例:
    ```json
    {
        "model_code": "yolov8",
        "task_name": "行人检测-1",
        "image_urls": [
            "http://example.com/image.jpg"
        ],
        "callback_urls": "http://callback1,http://callback2",
        "enable_callback": true,
        "is_base64": false,
        "save_result": false,
        "config": {
            "confidence": 0.5,
            "iou": 0.45,
            "classes": [0, 2],
            "roi": {
                "x1": 0.1,
                "y1": 0.1,
                "x2": 0.9,
                "y2": 0.9
            },
            "imgsz": 640,
            "nested_detection": true
        }
    }
    ```
    """
)
async def analyze_image(
    request: Request,
    body: ImageAnalysisRequest,
    detector: YOLODetector = Depends(get_detector)
) -> StandardResponse:
    """图片分析接口"""
    try:
        # 记录请求参数
        logger.info(f"收到图片分析请求: {json.dumps(body.dict(), ensure_ascii=False)}")
        
        # 生成任务ID
        task_id = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 执行分析
        result = await detector.detect_images(
            body.model_code,
            body.image_urls,
            body.callback_urls,
            body.is_base64,
            config=body.config,
            task_name=body.task_name,
            enable_callback=body.enable_callback,
            save_result=body.save_result
        )
        
        # 记录结果
        logger.info(f"图片分析完成: task_id={task_id}, objects={len(result.get('detections', []))}")
        
        return StandardResponse(
            requestId=str(uuid.uuid4()),
            path=str(request.url.path),
            success=True,
            message="图片分析成功",
            code=200,
            data={
                "task_id": task_id,
                "task_name": result.get("task_name"),
                "status": AnalysisStatus.COMPLETED,
                "image_url": body.image_urls[0],
                "saved_path": result.get("saved_path"),
                "objects": result.get("detections", []),
                "result_image": result.get("result_image") if body.is_base64 else None,
                "start_time": result.get("start_time"),
                "end_time": result.get("end_time"),
                "analysis_duration": result.get("analysis_duration")
            }
        )
        
    except Exception as e:
        logger.error(f"图片分析失败: {str(e)}", exc_info=True)
        if isinstance(e, (InvalidInputException, ModelLoadException, ProcessingException)):
            raise
        raise ProcessingException(f"图片分析失败: {str(e)}")

@router.post(
    "/video",
    response_model=StandardResponse,
    summary="视频分析",
    description="""
    分析视频中的目标
    
    支持以下功能:
    - 目标检测
    - 实例分割
    - 目标跟踪
    
    请求示例:
    ```json
    {
        "model_code": "yolov8",
        "task_name": "视频分析-1",
        "video_url": "http://example.com/video.mp4",
        "callback_urls": "http://callback1,http://callback2",
        "enable_callback": true,
        "save_result": false,
        "config": {
            "confidence": 0.5,
            "iou": 0.45,
            "classes": [0, 2],
            "roi": {
                "x1": 0.1,
                "y1": 0.1,
                "x2": 0.9,
                "y2": 0.9
            },
            "imgsz": 640,
            "nested_detection": true
        }
    }
    ```
    """
)
async def analyze_video(
    request: Request,
    body: VideoAnalysisRequest,
    background_tasks: BackgroundTasks,
    detector: YOLODetector = Depends(get_detector)
) -> StandardResponse:
    """视频分析接口"""
    try:
        # 记录请求参数
        logger.info(f"收到视频分析请求: {json.dumps(body.dict(), ensure_ascii=False)}")
        
        # 生成任务ID
        task_id = f"vid_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 启动视频分析任务
        result = await detector.start_video_analysis(
            task_id=task_id,
            model_code=body.model_code,
            video_url=body.video_url,
            callback_urls=body.callback_urls,
            config=body.config,
            task_name=body.task_name,
            enable_callback=body.enable_callback,
            save_result=body.save_result
        )
        
        # 记录结果
        logger.info(f"视频分析任务已启动: task_id={task_id}")
        
        return StandardResponse(
            requestId=str(uuid.uuid4()),
            path=str(request.url.path),
            success=True,
            message="视频分析任务已启动",
            code=200,
            data={
                "task_id": task_id,
                "task_name": body.task_name,
                "status": result.get("status", TaskStatus.PROCESSING),
                "video_url": body.video_url,
                "start_time": result.get("start_time"),
                "progress": result.get("progress", 0)
            }
        )
        
    except Exception as e:
        logger.error(f"启动视频分析任务失败: {str(e)}", exc_info=True)
        if isinstance(e, (InvalidInputException, ModelLoadException, ProcessingException)):
            raise
        raise ProcessingException(f"启动视频分析任务失败: {str(e)}")

@router.post(
    "/task/status",
    response_model=StandardResponse,
    summary="获取任务状态",
    description="获取分析任务的状态信息"
)
async def get_task_status(
    request: Request,
    body: TaskStatusRequest,
    detector: YOLODetector = Depends(get_detector)
) -> StandardResponse:
    """获取任务状态"""
    try:
        logger.info(f"收到任务状态查询请求: task_id={body.task_id}")
        
        # 直接使用detector._get_task_info获取任务信息
        task_info = await detector._get_task_info(body.task_id)
        if not task_info:
            logger.warning(f"任务 {body.task_id} 不存在")
            raise ResourceNotFoundException(f"任务 {body.task_id} 不存在")
            
        logger.info(f"成功获取任务状态: {body.task_id}")
            
        return StandardResponse(
            requestId=str(uuid.uuid4()),
            path=str(request.url.path),
            success=True,
            message="获取任务状态成功",
            code=200,
            data=task_info
        )
        
    except Exception as e:
        logger.error(f"获取任务状态失败: {str(e)}", exc_info=True)
        if isinstance(e, ResourceNotFoundException):
            raise
        raise ProcessingException(f"获取任务状态失败: {str(e)}")

@router.post("/task/stop", response_model=StandardResponse, summary="停止任务", description="强制停止指定的任务。会立即中断任务执行流程，停止所有相关资源和处理线程。")
async def stop_task(
    request: Request,
    body: TaskStatusRequest,
    detector: YOLODetector = Depends(get_detector)
) -> StandardResponse:
    """停止任务"""
    try:
        logger.info(f"收到停止任务请求: {body.task_id}")
        
        # 获取任务信息
        task_info = await detector._get_task_info(body.task_id)
        if not task_info:
            raise ResourceNotFoundException(f"任务 {body.task_id} 不存在")
        
        # 调用强制停止方法   
        await detector.stop_stream_analysis(body.task_id)
        
        return StandardResponse(
            requestId=str(uuid.uuid4()),
            path=str(request.url.path),
            success=True,
            message="任务停止指令已发送，正在强制停止任务",
            code=200,
            data={
                "task_id": body.task_id,
                "status": TaskStatus.STOPPING
            }
        )
        
    except Exception as e:
        logger.error(f"停止任务失败: {str(e)}", exc_info=True)
        if isinstance(e, ResourceNotFoundException):
            raise
        raise ProcessingException(f"停止任务失败: {str(e)}")

@router.post(
    "/resource",
    response_model=StandardResponse,
    summary="获取资源状态",
    description="获取系统资源使用状况"
)
async def get_resource_status(request: Request) -> StandardResponse:
    """获取资源状态"""
    try:
        status = resource_monitor.get_status()
        return StandardResponse(
            requestId=str(uuid.uuid4()),
            path=str(request.url.path),
            success=True,
            message="获取资源状态成功",
            code=200,
            data=status
        )
    except Exception as e:
        logger.error(f"获取资源状态失败: {str(e)}", exc_info=True)
        raise ProcessingException(f"获取资源状态失败: {str(e)}")

# 添加新的请求模型
class VideoStatusRequest(BaseModel):
    """视频状态查询请求"""
    task_id: str = Field(..., description="任务ID")

@router.post(
    "/video/status",
    response_model=StandardResponse,
    summary="获取视频状态",
    description="获取指定视频分析任务的状态"
)
async def get_video_status(
    request: Request,
    body: TaskStatusRequest,
    detector: YOLODetector = Depends(get_detector)
) -> StandardResponse:
    """获取视频分析任务状态"""
    try:
        # 获取任务状态
        status_info = await detector.get_video_task_status(body.task_id)
        if not status_info:
            raise ResourceNotFoundException(f"任务 {body.task_id} 不存在")
            
        return StandardResponse(
            requestId=str(uuid.uuid4()),
            path=str(request.url.path),
            success=True,
            message="获取任务状态成功",
            code=200,
            data=status_info
        )
        
    except Exception as e:
        logger.error(f"获取任务状态失败: {str(e)}", exc_info=True)
        if isinstance(e, ResourceNotFoundException):
            raise
        raise ProcessingException(f"获取任务状态失败: {str(e)}")

@router.post(
    "/stream",
    response_model=StandardResponse,
    summary="流分析",
    description="""
    分析视频流中的目标
    
    支持以下格式:
    - RTSP流 (rtsp://)
    - RTMP流 (rtmp://)
    - HTTP流 (http://, https://)
    
    支持以下功能:
    - 目标检测
    - 实例分割
    - 目标跟踪
    
    回调机制:
    - 系统级回调: 始终启用，不受enable_callback参数控制。如果系统回调失败，任务会自动停止。
    - 用户回调: 由enable_callback参数控制，用户可以选择启用或禁用。
    - 回调优先级: 如果指定了callback_url，它将作为系统回调URL。否则，系统会自动构建一个基于请求来源的系统回调URL。
    - 所有回调都包含图片的base64编码数据。
    
    请求示例:
    ```json
    {
        "model_code": "yolov8",
        "task_name": "流分析-1",
        "stream_url": "rtsp://example.com/stream",
        "callback_urls": "http://user-callback1,http://user-callback2",
        "callback_url": "http://system-callback", 
        "enable_callback": true,
        "save_result": false,
        "task_id": "12345", 
        "config": {
            "confidence": 0.5,
            "iou": 0.45,
            "classes": [0, 2],
            "roi_type": 1,
            "roi": {
                "x1": 0.1,
                "y1": 0.1,
                "x2": 0.9,
                "y2": 0.9
            },
            "imgsz": 640,
            "nested_detection": true
        }
    }
    ```
""")
async def analyze_stream(
    request: Request,
    body: StreamAnalysisRequest,
    background_tasks: BackgroundTasks,
    detector: YOLODetector = Depends(get_detector)
) -> StandardResponse:
    """流分析接口"""
    try:
        # 记录请求参数
        logger.info(f"收到流分析请求: {json.dumps(body.dict(), ensure_ascii=False)}")
        logger.info(f"请求IP: {request.client.host}:{request.client.port}, USER-AGENT: {request.headers.get('user-agent', '未知')}")
        
        # 使用传入的任务ID或生成新的
        task_id = body.task_id if body.task_id else f"str_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        logger.info(f"流分析任务ID: {task_id} (由{'请求指定' if body.task_id else '系统生成'})")
        
        # 获取用户指定的回调URL
        user_callback_urls = body.callback_urls or ""
        
        # 构建系统回调URL，基于请求来源
        client_host = request.client.host if request.client else None
        system_callback = None
        
        if client_host:
            # 如果是直接指定的callback_url，优先使用
            if body.callback_url:
                system_callback = body.callback_url
                logger.info(f"使用请求指定的系统回调URL: {system_callback}")
            # 否则尝试推断
            elif "127.0.0.1" in client_host or "localhost" in client_host:
                # 本地请求，使用本地回调
                system_callback = "http://localhost:8000/api/v1/callback"
                logger.info(f"本地请求，使用本地回调URL: {system_callback}")
            else:
                # 远程请求，基于客户端IP构建回调
                system_callback = f"http://{client_host}/api/v1/callback"
                logger.info(f"远程请求，基于客户端IP构建回调URL: {system_callback}")
            
            logger.info(f"最终确定系统级回调: {system_callback}")
        else:
            logger.warning("无法获取客户端信息，无法自动构建系统回调URL")
        
        # 合并回调URL列表
        combined_callback_urls = user_callback_urls
        
        # 添加系统回调到回调列表（始终添加）
        if system_callback and system_callback not in combined_callback_urls:
            if combined_callback_urls:
                combined_callback_urls = f"{combined_callback_urls},{system_callback}"
            else:
                combined_callback_urls = system_callback
            logger.info(f"最终回调URL列表: {combined_callback_urls}")
        
        # 记录关键请求参数
        logger.info(f"流分析关键参数: model_code={body.model_code}, stream_url={body.stream_url}, analysis_type={body.analysis_type}")
        
        # 启动流分析任务
        logger.info(f"开始启动任务 {task_id} 的流分析...")
        try:
            result = await detector.start_stream_analysis(
                task_id=task_id,
                model_code=body.model_code,
                stream_url=body.stream_url,
                callback_urls=combined_callback_urls,
                system_callback_url=system_callback,  # 新增：传递系统回调URL
                config=body.config,
                task_name=body.task_name,
                enable_callback=body.enable_callback,  # 用户回调是否启用
                save_result=body.save_result,
                analysis_type=body.analysis_type
            )
            
            logger.info(f"任务 {task_id} 创建成功，开始异步处理流分析")
            logger.info(f"任务信息: {result}")
            
        except Exception as e:
            logger.error(f"启动任务 {task_id} 失败: {str(e)}", exc_info=True)
            raise ProcessingException(f"启动流分析任务失败: {str(e)}")
        
        # 构建响应数据
        response_data = {
            "task_id": task_id,
            "task_name": body.task_name,
            "status": result.get("status", TaskStatus.PROCESSING),
            "stream_url": body.stream_url,
            "start_time": result.get("start_time"),
            "progress": result.get("progress", 0),
            "callback_urls": combined_callback_urls,
            "analysis_type": body.analysis_type
        }
        
        # 记录结果
        logger.info(f"流分析任务已启动: task_id={task_id}")
        logger.info(f"响应数据: {response_data}")
        
        return StandardResponse(
            requestId=str(uuid.uuid4()),
            path=str(request.url.path),
            success=True,
            message="流分析任务已启动",
            code=200,
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"启动流分析任务失败: {str(e)}", exc_info=True)
        if isinstance(e, (InvalidInputException, ModelLoadException, ProcessingException)):
            raise
        raise ProcessingException(f"启动流分析任务失败: {str(e)}")