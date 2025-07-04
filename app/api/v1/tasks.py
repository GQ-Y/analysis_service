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
from pydantic import BaseModel, Field

from app.models.response_model import ResponseModel, create_success_response, create_error_response
from app.services.task_service import get_task_service

router = APIRouter(prefix="/tasks", tags=["任务管理"])


# 请求模型
class TaskCreateRequest(BaseModel):
    """创建任务请求"""
    name: str = Field(..., description="任务名称")
    description: Optional[str] = Field(None, description="任务描述")
    analysis_type: int = Field(..., description="分析类型：1-图片分析，2-视频分析，3-流分析")
    model_codes: List[str] = Field(..., description="模型代码列表")
    stream_urls: Optional[List[str]] = Field(None, description="视频流URL列表（流分析时必填）")
    video_path: Optional[str] = Field(None, description="视频文件路径（视频分析时必填）")
    image_paths: Optional[List[str]] = Field(None, description="图片路径列表（图片分析时必填）")
    analysis_interval: Optional[int] = Field(5, description="分析间隔（秒）")
    save_result: bool = Field(True, description="是否保存分析结果")
    save_images: bool = Field(False, description="是否保存分析图片")
    config: Optional[Dict[str, Any]] = Field(None, description="任务配置")


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


@router.post("/create", response_model=ResponseModel[TaskInfo])
async def create_task(request: TaskCreateRequest):
    """创建分析任务"""
    try:
        task_service = get_task_service()
        task_info = await task_service.create_task(
            name=request.name,
            description=request.description,
            analysis_type=request.analysis_type,
            model_codes=request.model_codes,
            stream_urls=request.stream_urls,
            video_path=request.video_path,
            image_paths=request.image_paths,
            analysis_interval=request.analysis_interval,
            save_result=request.save_result,
            save_images=request.save_images,
            config=request.config or {}
        )
        
        return create_success_response(
            data=task_info,
            message="任务创建成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")


@router.post("/start", response_model=ResponseModel[Dict[str, Any]])
async def start_task(request: TaskControlRequest):
    """启动分析任务"""
    try:
        task_service = get_task_service()
        result = await task_service.start_task(request.task_id)
        
        return create_success_response(
            data=result,
            message="任务启动成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动任务失败: {str(e)}")


@router.post("/stop", response_model=ResponseModel[Dict[str, Any]])
async def stop_task(request: TaskControlRequest):
    """停止分析任务"""
    try:
        task_service = get_task_service()
        result = await task_service.stop_task(request.task_id)
        
        return create_success_response(
            data=result,
            message="任务停止成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止任务失败: {str(e)}")


@router.get("/{task_id}", response_model=ResponseModel[TaskInfo])
async def get_task(task_id: int):
    """获取任务详情"""
    try:
        task_service = get_task_service()
        task_info = await task_service.get_task(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return create_success_response(
            data=task_info,
            message="获取任务详情成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务详情失败: {str(e)}")


@router.post("/list", response_model=ResponseModel[Dict[str, Any]])
async def list_tasks(request: TaskListRequest):
    """获取任务列表"""
    try:
        task_service = get_task_service()
        result = await task_service.list_tasks(
            page=request.page,
            page_size=request.page_size,
            status=request.status,
            keyword=request.keyword
        )
        
        return create_success_response(
            data=result,
            message="获取任务列表成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@router.delete("/{task_id}", response_model=ResponseModel[Dict[str, Any]])
async def delete_task(task_id: int):
    """删除任务"""
    try:
        task_service = get_task_service()
        result = await task_service.delete_task(task_id)
        
        return create_success_response(
            data=result,
            message="任务删除成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")


@router.get("/{task_id}/status", response_model=ResponseModel[Dict[str, Any]])
async def get_task_status(task_id: int):
    """获取任务状态"""
    try:
        task_service = get_task_service()
        status = await task_service.get_task_status(task_id)
        
        return create_success_response(
            data=status,
            message="获取任务状态成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")


@router.get("/{task_id}/results", response_model=ResponseModel[Dict[str, Any]])
async def get_task_results(
    task_id: int,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小")
):
    """获取任务分析结果"""
    try:
        task_service = get_task_service()
        results = await task_service.get_task_results(
            task_id=task_id,
            page=page,
            page_size=page_size
        )
        
        return create_success_response(
            data=results,
            message="获取任务结果成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务结果失败: {str(e)}")
