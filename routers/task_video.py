"""
视频任务路由
重构为仅支持零拷贝实时分析
"""

import os
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional

from shared.utils.app_state import app_state_manager
from shared.models.base import BaseResponse
from shared.utils.logger import normal_logger, test_logger

router = APIRouter()

def get_task_manager():
    """获取任务管理器实例"""
    task_manager = app_state_manager.get_task_manager()
    if not task_manager:
        raise HTTPException(status_code=500, detail="任务管理器未初始化")
    return task_manager

class ZeroCopyAnalysisRequest(BaseModel):
    """零拷贝分析请求模型"""
    task_id: str
    stream_url: Optional[str] = ""
    output_path: Optional[str] = ""
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_001",
                "stream_url": "rtsp://example.com/stream",
                "output_path": "output/analysis/"
            }
        }

class TaskRequest(BaseModel):
    """任务请求模型"""
    task_id: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_001"
            }
        }

@router.post("/video", response_model=BaseResponse, summary="启动零拷贝实时分析视频直播流")
async def start_zero_copy_analysis(
    request: ZeroCopyAnalysisRequest,
    task_manager=Depends(get_task_manager)
):
    """
    启动零拷贝实时分析视频直播流
    
    Args:
        request: 零拷贝分析请求
        task_manager: 任务管理器
        
    Returns:
        BaseResponse: 启动结果
    """
    try:
        from services.video.video_service import VideoService
        video_service = VideoService()
        
        normal_logger.info(f"收到零拷贝分析启动请求: {request.task_id}")
        
        result = await video_service.start_zero_copy_analysis(
            task_id=request.task_id,
            task_manager=task_manager,
            stream_url=request.stream_url,
            output_path=request.output_path
        )
        
        test_logger.info("TEST_LOG_MARKER: VIDEO_ZERO_COPY_ANALYSIS_API_SUCCESS")
        
        return BaseResponse(
            status="success" if result["success"] else "error",
            message=result["message"],
            data=result
        )
        
    except Exception as e:
        normal_logger.error(f"启动零拷贝分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动零拷贝分析失败: {str(e)}")

@router.delete("/video/{task_id}", response_model=BaseResponse, summary="停止零拷贝实时分析")
async def stop_zero_copy_analysis(
    task_id: str,
    task_manager=Depends(get_task_manager)
):
    """
    停止零拷贝实时分析
    
    Args:
        task_id: 任务ID
        task_manager: 任务管理器
        
    Returns:
        BaseResponse: 停止结果
    """
    try:
        from services.video.video_service import VideoService
        video_service = VideoService()
        
        normal_logger.info(f"收到零拷贝分析停止请求: {task_id}")
        
        result = await video_service.stop_zero_copy_analysis(task_id)
        
        test_logger.info("TEST_LOG_MARKER: VIDEO_ZERO_COPY_ANALYSIS_STOP_API_SUCCESS")
        
        return BaseResponse(
            status="success" if result["success"] else "error",
            message=result["message"],
            data=result
        )
        
    except Exception as e:
        normal_logger.error(f"停止零拷贝分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"停止零拷贝分析失败: {str(e)}")

@router.get("/video/{task_id}/status", response_model=BaseResponse, summary="获取零拷贝分析状态")
async def get_zero_copy_analysis_status(
    task_id: str,
    task_manager=Depends(get_task_manager)
):
    """
    获取零拷贝分析状态
    
    Args:
        task_id: 任务ID
        task_manager: 任务管理器
        
    Returns:
        BaseResponse: 分析状态
    """
    try:
        from services.video.video_service import VideoService
        video_service = VideoService()
        
        result = await video_service.get_analysis_status(task_id)
        
        return BaseResponse(
            status="success" if result["success"] else "error",
            message=result["message"],
            data=result
        )
        
    except Exception as e:
        normal_logger.error(f"获取零拷贝分析状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取零拷贝分析状态失败: {str(e)}")
