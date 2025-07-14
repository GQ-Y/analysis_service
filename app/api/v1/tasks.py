#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: tasks.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 任务管理API

提供任务的创建、启动、停止、查询等核心功能。

本文件是分析服务项目的一部分。
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body, Depends
from pydantic import BaseModel, Field, validator, ConfigDict

from app.models.response_model import ResponseModel, create_success_response, create_error_response
from app.services.task_service import get_task_service

router = APIRouter(prefix="/tasks", tags=["任务管理"])


# 模型配置
class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    """单个模型配置"""
    model_code: str = Field(..., description="模型代码", example="yolo11n")
    analysis_fps: Optional[float] = Field(None, description="该模型的分析帧率，None表示不限制", example=10.0)
    confidence_threshold: Optional[float] = Field(None, description="该模型的置信度阈值", example=0.5)
    iou_threshold: Optional[float] = Field(None, description="该模型的IOU阈值", example=0.45)

class TaskCreateRequest(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "name": "商场人流检测任务",
                "description": "检测商场入口的人流情况，统计进出人数",
                "analysis_type": 3,
                "model_configs": [
                    {
                        "model_code": "yolo11n",
                        "analysis_fps": 10.0,
                        "confidence_threshold": 0.5,
                        "iou_threshold": 0.45
                    },
                    {
                        "model_code": "yolo11s",
                        "analysis_fps": 5.0,
                        "confidence_threshold": 0.6,
                        "iou_threshold": 0.5
                    }
                ],
                "stream_urls": ["rtsp://admin:password@192.168.1.100:554/stream1"],
                "save_result": True,
                "save_images": True,
                "callback_urls": ["http://your-server.com/api/analysis/callback"],
                "callback_interval": 10,
                "roi_config": {
                    "enabled": True,
                    "regions": [
                        {
                            "name": "entrance",
                            "points": [[100, 100], [500, 100], [500, 400], [100, 400]]
                        }
                    ]
                },
                "target_classes": ["person", "car", "bicycle"],
                "config": {"enable_tracking": True, "max_objects": 100}
            }
        }
    )
    """创建任务请求"""
    name: str = Field(..., description="任务名称", example="商场人流检测任务")
    description: Optional[str] = Field(None, description="任务描述", example="检测商场入口的人流情况，统计进出人数")
    analysis_type: int = Field(..., description="分析类型：1-图片分析，2-视频分析，3-流分析", example=3)
    
    # 方式1：使用模型配置列表（推荐）
    model_configs: Optional[List[ModelConfig]] = Field(
        None, 
        description="模型配置列表（推荐使用）",
        example=[
            {
                "model_code": "yolo11n",
                "analysis_fps": 10.0,
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45
            },
            {
                "model_code": "yolo11s",
                "analysis_fps": 5.0,
                "confidence_threshold": 0.6,
                "iou_threshold": 0.5
            }
        ]
    )
    
    # 方式2：兼容旧版本的全局配置
    model_codes: Optional[List[str]] = Field(
        None, 
        description="模型代码列表（兼容旧版本）",
        example=["yolo11n", "yolo11s"]
    )
    analysis_fps: Optional[float] = Field(
        None, 
        description="全局分析帧率（兼容旧版本）",
        example=8.0
    )
    confidence_threshold: Optional[float] = Field(
        0.5, 
        description="全局置信度阈值（兼容旧版本）",
        example=0.5
    )
    iou_threshold: Optional[float] = Field(
        0.45, 
        description="全局IOU阈值（兼容旧版本）",
        example=0.45
    )
    
    stream_urls: Optional[List[str]] = Field(
        None, 
        description="视频流URL列表（流分析时必填）",
        example=["rtsp://admin:password@192.168.1.100:554/stream1"]
    )
    video_path: Optional[str] = Field(
        None, 
        description="视频文件路径（视频分析时必填）",
        example="/data/videos/test_video.mp4"
    )
    image_paths: Optional[List[str]] = Field(
        None, 
        description="图片路径列表（图片分析时必填）",
        example=["/data/images/img1.jpg", "/data/images/img2.jpg"]
    )
    analysis_interval: Optional[int] = Field(
        5, 
        description="分析间隔（秒）",
        example=5
    )
    
    # 结果保存配置
    save_result: bool = Field(
        True, 
        description="是否保存分析结果元数据",
        example=True
    )
    save_images: bool = Field(
        False, 
        description="是否保存检测图片",
        example=True
    )
    
    # 回调配置
    callback_urls: Optional[List[str]] = Field(
        None, 
        description="结果回调地址列表",
        example=["http://your-server.com/api/analysis/callback"]
    )
    callback_interval: Optional[int] = Field(
        None, 
        description="回调间隔（秒）",
        example=10
    )
    
    # 其他配置
    config: Optional[Dict[str, Any]] = Field(
        None, 
        description="其他配置参数",
        example={"enable_tracking": True, "max_objects": 100}
    )
    playback_duration: Optional[int] = Field(
        None, 
        description="回放持续时间（秒）",
        example=30
    )
    roi_config: Optional[Dict[str, Any]] = Field(
        None, 
        description="ROI配置",
        example={
            "enabled": True,
            "regions": [
                {
                    "name": "entrance",
                    "points": [[100, 100], [500, 100], [500, 400], [100, 400]]
                }
            ]
        }
    )
    target_classes: Optional[List[str]] = Field(
        None, 
        description="目标类别列表",
        example=["person", "car", "bicycle"]
    )

    @validator('target_classes')
    def validate_model_config(cls, v, values):
        """验证必须提供模型配置"""
        model_configs = values.get('model_configs')
        model_codes = values.get('model_codes')
        
        if not model_configs and not model_codes:
            raise ValueError("必须提供 model_configs 或 model_codes 之一")
        
        return v

    


class TaskControlRequest(BaseModel):
    """任务控制请求"""
    task_id: int = Field(..., description="任务ID")


class TaskListRequest(BaseModel):
    """任务列表请求"""
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(20, ge=1, le=100, description="每页大小")
    status: Optional[int] = Field(None, description="状态筛选：0-未启动，1-运行中，2-已停止，3-错误")
    keyword: Optional[str] = Field(None, description="关键词搜索")


# 响应模型
class TaskInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    """任务信息"""
    id: int
    name: str
    description: Optional[str]
    analysis_type: int
    status: int
    progress: float
    created_at: str
    updated_at: str
    model_count: int
    stream_count: int
    result_count: int


# ==================== 核心功能API ====================

@router.post("/create", response_model=ResponseModel)
async def create_task(request: TaskCreateRequest):
    """1. 创建分析任务"""
    try:
        task_service = get_task_service()
        # 处理模型配置，支持两种方式
        if request.model_configs:
            # 新方式：使用模型配置列表
            model_codes = [config.model_code for config in request.model_configs]
            model_fps_config = {config.model_code: config.analysis_fps 
                              for config in request.model_configs 
                              if config.analysis_fps is not None}
            model_confidence_config = {config.model_code: config.confidence_threshold 
                                     for config in request.model_configs 
                                     if config.confidence_threshold is not None}
            model_iou_config = {config.model_code: config.iou_threshold 
                              for config in request.model_configs 
                              if config.iou_threshold is not None}
        else:
            # 旧方式：使用全局配置
            model_codes = request.model_codes
            model_fps_config = {}
            model_confidence_config = {}
            model_iou_config = {}
            if request.analysis_fps:
                for model_code in model_codes:
                    model_fps_config[model_code] = request.analysis_fps

        task_info = await task_service.create_task(
            name=request.name,
            description=request.description,
            analysis_type=request.analysis_type,
            model_codes=model_codes,
            stream_urls=request.stream_urls,
            video_path=request.video_path,
            image_paths=request.image_paths,
            config=request.config or {},
            confidence_threshold=request.confidence_threshold or 0.5,
            iou_threshold=request.iou_threshold or 0.45,
            save_result=request.save_result,
            save_images=request.save_images,
            callback_urls=request.callback_urls,
            callback_interval=request.callback_interval,
            playback_duration=request.playback_duration,
            roi_config=request.roi_config,
            target_classes=request.target_classes,
            analysis_fps=model_fps_config,
            model_confidence_config=model_confidence_config,
            model_iou_config=model_iou_config
        )

        return create_success_response(
            data=task_info.__dict__,
            message="✅ 任务创建成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")


@router.post("/{task_id}/start", response_model=ResponseModel)
async def start_task(task_id: int):
    """启动任务（内部使用，创建后自动启动）"""
    try:
        task_service = get_task_service()
        result = await task_service.start_task(task_id)

        return create_success_response(
            data=result,
            message="🚀 任务启动成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动任务失败: {str(e)}")


@router.post("/{task_id}/stop", response_model=ResponseModel)
async def stop_task(task_id: int):
    """2. 停止分析任务"""
    try:
        task_service = get_task_service()

        # 先检查任务是否存在和状态
        task_info = await task_service.get_task_detail(task_id)
        if not task_info:
            raise HTTPException(status_code=404, detail="任务不存在")

        if task_info.status != 1:  # 不是运行中状态
            return create_success_response(
                data={"task_id": task_id, "status": "already_stopped"},
                message="⏹️ 任务已经是停止状态"
            )

        result = await task_service.stop_task(task_id)

        return create_success_response(
            data=result,
            message="⏹️ 任务停止成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止任务失败: {str(e)}")


@router.get("/{task_id}", response_model=ResponseModel)
async def get_task_detail(task_id: int):
    """3. 查看任务详情"""
    try:
        task_service = get_task_service()
        task_info = await task_service.get_task_detail(task_id)

        if not task_info:
            raise HTTPException(status_code=404, detail="任务不存在")

        return create_success_response(
            data=task_info.__dict__,
            message="📋 获取任务详情成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务详情失败: {str(e)}")


@router.delete("/{task_id}", response_model=ResponseModel)
async def delete_task(task_id: int):
    """4. 删除任务"""
    try:
        task_service = get_task_service()
        result = await task_service.delete_task(task_id)

        return create_success_response(
            data=result,
            message="🗑️ 任务删除成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")


@router.post("/{task_id}/restart", response_model=ResponseModel)
async def restart_task(task_id: int):
    """5. 重启任务"""
    try:
        task_service = get_task_service()
        result = await task_service.restart_task(task_id)

        return create_success_response(
            data=result,
            message="🔄 任务重启成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重启任务失败: {str(e)}")


# ==================== 辅助功能API ====================

@router.get("/list", response_model=ResponseModel)
async def list_tasks(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    status: Optional[int] = Query(None, description="状态筛选：0-未启动，1-运行中，2-已停止，3-错误，4-已完成")
):
    """获取任务列表"""
    try:
        task_service = get_task_service()
        result = await task_service.list_tasks(
            page=page,
            page_size=page_size,
            status=status
        )

        return create_success_response(
            data=result,
            message="📝 获取任务列表成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")
