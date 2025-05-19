"""
任务视频编码路由
提供任务的视频编码功能，支持MP4和FLV格式
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
from services.http.video_encoder_service import VideoEncoderService
from core.task_management.utils.status import TaskStatus
from core.config import settings
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

# 创建路由
router = APIRouter(
    prefix="/api/v1/tasks",
    tags=["视频编码"],
    responses={404: {"description": "Not found"}},
)

# 依赖注入
async def get_task_service(request: Request) -> TaskService:
    """获取任务服务实例"""
    if not hasattr(request.app.state, "task_service"):
        raise HTTPException(status_code=500, detail="任务服务未初始化")
    return request.app.state.task_service

async def get_video_encoder_service(request: Request) -> VideoEncoderService:
    """获取视频编码服务实例"""
    if not hasattr(request.app.state, "video_encoder_service"):
        # 如果视频编码服务不存在，创建一个
        request.app.state.video_encoder_service = VideoEncoderService()
    return request.app.state.video_encoder_service

@router.post("/video", response_model=BaseResponse, summary="开启/关闭实时分析视频编码")
async def toggle_task_video_encoding(
    encoding_request: VideoEncodingRequest,
    task_service: TaskService = Depends(get_task_service),
    video_encoder_service: VideoEncoderService = Depends(get_video_encoder_service)
) -> BaseResponse:
    """
    开启或关闭实时分析视频编码
    
    通过任务ID开启或关闭实时分析视频编码功能。开启后，系统将实时分析的结果和视频编码为MP4或FLV格式，
    并返回一个可以直接下载的URL地址。
    
    支持的参数包括：
    - task_id: 任务ID，必填
    - enable_encoding: 是否开启编码，默认为true
    - format: 视频格式，支持"mp4"或"flv"，默认为"mp4"
    - quality: 视频质量(1-100)，默认80
    - width: 视频宽度，为空则使用原始宽度
    - height: 视频高度，为空则使用原始高度
    - fps: 视频帧率，默认15
    
    Args:
        encoding_request: 编码请求参数
        
    Returns:
        BaseResponse: 响应结果，包含视频URL
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
            # 开启编码
            result = await video_encoder_service.start_encoding(
                task_id=encoding_request.task_id,
                task_manager=task_manager,
                format=encoding_request.format,
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
                message="视频编码已开启",
                code=200,
                data={
                    "task_id": encoding_request.task_id,
                    "video_url": result["video_url"]
                }
            )
        else:
            # 关闭编码
            result = await video_encoder_service.stop_encoding(encoding_request.task_id)
            
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
                message="视频编码已关闭",
                code=200,
                data=None
            )
    
    except Exception as e:
        exception_logger.exception(f"处理视频编码请求失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/video",
            success=False,
            message=f"处理视频编码请求失败: {str(e)}",
            code=500,
            data=None
        )

@router.get("/video/{encoding_id}/{file_name}", response_model=None, summary="获取编码视频文件")
async def get_video_file(
    encoding_id: str = Path(..., description="编码ID"),
    file_name: str = Path(..., description="文件名"),
    video_encoder_service: VideoEncoderService = Depends(get_video_encoder_service)
) -> FileResponse:
    """
    获取编码视频文件
    
    返回编码后的视频文件（MP4或FLV）。
    
    Args:
        encoding_id: 编码ID
        file_name: 文件名
        
    Returns:
        FileResponse: 视频文件响应
    """
    try:
        # 构建文件路径
        file_path = os.path.join(video_encoder_service.output_base_dir, encoding_id, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"视频文件不存在: {file_name}")
        
        # 确定内容类型
        content_type = "video/mp4" if file_name.endswith(".mp4") else "video/x-flv"
        
        # 返回文件
        return FileResponse(
            path=file_path,
            media_type=content_type,
            filename=file_name
        )
    
    except HTTPException:
        raise
    except Exception as e:
        exception_logger.exception(f"获取视频文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取视频文件失败: {str(e)}")
