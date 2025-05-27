"""
任务视频编码路由
提供任务的视频直播流功能，支持RTMP、HLS、FLV格式
"""
from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response, Query
from fastapi.responses import FileResponse, HTMLResponse
import uuid
import os
import asyncio
from typing import Dict, Any, Optional

from models.requests import VideoEncodingRequest
from models.responses import BaseResponse
from services.http.task_service import TaskService
from services.video.video_service import VideoService
from core.task_management.utils.status import TaskStatus
from core.config import settings
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

# 创建路由
router = APIRouter(
    prefix="/api/v1/tasks",
    tags=["视频直播流"],
    responses={404: {"description": "Not found"}},
)

# 依赖注入
async def get_task_service(request: Request) -> TaskService:
    """获取任务服务实例"""
    if not hasattr(request.app.state, "task_service"):
        raise HTTPException(status_code=500, detail="任务服务未初始化")
    return request.app.state.task_service

async def get_video_service(request: Request) -> VideoService:
    """获取视频服务实例"""
    if not hasattr(request.app.state, "video_service"):
        # 如果视频服务不存在，创建一个
        request.app.state.video_service = VideoService()
    return request.app.state.video_service

@router.post("/video", response_model=BaseResponse, summary="启动实时分析视频直播流")
async def start_task_video_stream(
    encoding_request: VideoEncodingRequest,
    task_service: TaskService = Depends(get_task_service),
    video_service: VideoService = Depends(get_video_service)
) -> BaseResponse:
    """
    启动实时分析视频直播流
    
    使用ffmpeg将分析结果和识别目标合成，然后通过ZLMediaKit构建直播流播放地址。
    系统将实时分析的结果绘制在视频上，并推送到ZLM服务器作为直播流。
    
    支持的参数包括：
    - task_id: 任务ID，必填
    - enable_encoding: 是否开启编码，默认为true
    - format: 视频格式，支持"rtmp"、"hls"、"flv"，默认为"rtmp"
    - quality: 视频质量(1-100)，默认80
    - width: 视频宽度，为空则使用原始宽度
    - height: 视频高度，为空则使用原始高度
    - fps: 视频帧率，默认15
    
    Args:
        encoding_request: 编码请求参数
        
    Returns:
        BaseResponse: 响应结果，包含直播流播放地址
    """
    request_id = str(uuid.uuid4())
    try:
        # 获取任务管理器
        task_manager = task_service.task_manager
        
        # 检查任务是否存在
        if not task_manager.has_task(encoding_request.task_id):
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video",
                success=False,
                message=f"任务不存在: {encoding_request.task_id}",
                code=404,
                data=None
            )
        
        # 根据请求开启或关闭编码
        if encoding_request.enable_encoding:
            # 开启直播流编码，使用ZLM作为流媒体服务器
            result = await video_service.start_live_stream(
                task_id=encoding_request.task_id,
                task_manager=task_manager,
                format=encoding_request.format or "rtmp",  # 默认使用RTMP格式
                quality=encoding_request.quality,
                width=encoding_request.width,
                height=encoding_request.height,
                fps=encoding_request.fps
            )
            
            if not result["success"]:
                return BaseResponse(
                    requestId=request_id,
                    path="/api/v1/tasks/video",
                    success=False,
                    message=result["message"],
                    code=500,
                    data=None
                )
            
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video",
                success=True,
                message="视频直播流已启动",
                code=200,
                data={
                    "task_id": encoding_request.task_id,
                    "stream_info": result["stream_info"],
                    "play_urls": result["play_urls"]
                }
            )
        else:
            # 关闭直播流编码
            result = await video_service.stop_live_stream(encoding_request.task_id)
            
            if not result["success"]:
                return BaseResponse(
                    requestId=request_id,
                    path="/api/v1/tasks/video",
                    success=False,
                    message=result["message"],
                    code=400,
                    data=None
                )
            
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video",
                success=True,
                message="视频直播流已关闭",
                code=200,
                data=None
            )
    
    except Exception as e:
        exception_logger.exception(f"处理视频直播流请求失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/video",
            success=False,
            message=f"处理视频直播流请求失败: {str(e)}",
            code=500,
            data=None
        )

@router.post("/video/file", response_model=BaseResponse, summary="启动实时分析视频文件编码")
async def start_task_video_file(
    encoding_request: VideoEncodingRequest,
    task_service: TaskService = Depends(get_task_service),
    video_service: VideoService = Depends(get_video_service)
) -> BaseResponse:
    """
    启动实时分析视频文件编码
    
    使用ffmpeg将分析结果和识别目标合成，生成MP4或FLV视频文件。
    系统将实时分析的结果绘制在视频上，并编码为视频文件。
    
    支持的参数包括：
    - task_id: 任务ID，必填
    - enable_encoding: 是否开启编码，默认为true
    - format: 视频格式，支持"mp4"、"flv"，默认为"mp4"
    - quality: 视频质量(1-100)，默认80
    - width: 视频宽度，为空则使用原始宽度
    - height: 视频高度，为空则使用原始高度
    - fps: 视频帧率，默认15
    
    Args:
        encoding_request: 编码请求参数
        
    Returns:
        BaseResponse: 响应结果，包含视频文件URL
    """
    request_id = str(uuid.uuid4())
    try:
        # 获取任务管理器
        task_manager = task_service.task_manager
        
        # 检查任务是否存在
        if not task_manager.has_task(encoding_request.task_id):
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video/file",
                success=False,
                message=f"任务不存在: {encoding_request.task_id}",
                code=404,
                data=None
            )
        
        # 根据请求开启或关闭编码
        if encoding_request.enable_encoding:
            # 开启文件编码
            result = await video_service.start_encoding(
                task_id=encoding_request.task_id,
                task_manager=task_manager,
                format=encoding_request.format or "mp4",  # 默认使用MP4格式
                quality=encoding_request.quality,
                width=encoding_request.width,
                height=encoding_request.height,
                fps=encoding_request.fps
            )
            
            if not result["success"]:
                return BaseResponse(
                    requestId=request_id,
                    path="/api/v1/tasks/video/file",
                    success=False,
                    message=result["message"],
                    code=500,
                    data=None
                )
            
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video/file",
                success=True,
                message="视频文件编码已启动",
                code=200,
                data={
                    "task_id": encoding_request.task_id,
                    "video_url": result["video_url"]
                }
            )
        else:
            # 关闭文件编码
            result = await video_service.stop_encoding(encoding_request.task_id)
            
            if not result["success"]:
                return BaseResponse(
                    requestId=request_id,
                    path="/api/v1/tasks/video/file",
                    success=False,
                    message=result["message"],
                    code=400,
                    data=None
                )
            
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/video/file",
                success=True,
                message="视频文件编码已关闭",
                code=200,
                data=None
            )
    
    except Exception as e:
        exception_logger.exception(f"处理视频文件编码请求失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/video/file",
            success=False,
            message=f"处理视频文件编码请求失败: {str(e)}",
            code=500,
            data=None
        )
