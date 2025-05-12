"""
任务管理路由
提供任务的创建、查询、停止等API
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, Request
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

from models.requests import StreamTask, BatchStreamTask
from models.responses import BaseResponse, TaskInfo
from services.task_service import TaskService
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

# 创建路由
router = APIRouter(
    prefix="/api/v1/tasks",
    tags=["任务管理"],
    responses={404: {"description": "Not found"}},
)

# 依赖注入
def get_task_service(request: Request) -> TaskService:
    """获取任务服务实例"""
    return request.app.state.task_service

@router.post("/start", response_model=BaseResponse, summary="启动分析任务")
async def start_task(
    task: StreamTask,
    task_service: TaskService = Depends(get_task_service)
) -> BaseResponse:
    """
    启动分析任务
    
    支持的参数包括：
    - model_code: 模型代码，必填
    - stream_url: 流地址，必填
    - task_name: 任务名称，可选
    - callback_urls: 回调地址，可选
    - output_url: 输出地址，可选
    - analysis_type: 分析类型，可选，默认为detection
    - config: 分析配置，可选
    
    Args:
        task: 任务参数
        
    Returns:
        BaseResponse: 响应结果
    """
    request_id = str(uuid.uuid4())
    
    try:
        # 创建并启动任务
        result = await task_service.create_task(task)
        
        if not result["success"]:
            return BaseResponse(
                requestId=request_id,
                path="/api/v1/tasks/start",
                success=False,
                code=400,
                message=result["message"],
                data=None
            )
            
        task_id = result["task_id"]
        
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/start",
            success=True,
            code=200,
            message="任务启动成功",
            data={"task_id": task_id}
        )
        
    except Exception as e:
        logger.error(f"启动任务失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/start",
            success=False,
            code=500,
            message=f"启动任务失败: {str(e)}",
            data=None
        )

@router.post("/stop/{task_id}", response_model=BaseResponse, summary="停止分析任务")
async def stop_task(
    task_id: str = Path(..., description="任务ID"),
    task_service: TaskService = Depends(get_task_service)
) -> BaseResponse:
    """
    停止正在运行的分析任务
    
    Args:
        task_id: 要停止的任务ID
        
    Returns:
        BaseResponse: 响应结果
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
                code=400,
                message=result["message"],
                data=None
            )
            
        return BaseResponse(
            requestId=request_id,
            path=f"/api/v1/tasks/stop/{task_id}",
            success=True,
            code=200,
            message="任务停止成功",
            data={"task_id": task_id}
        )
        
    except Exception as e:
        logger.error(f"停止任务失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path=f"/api/v1/tasks/stop/{task_id}",
            success=False,
            code=500,
            message=f"停止任务失败: {str(e)}",
            data=None
        )

@router.get("/list", response_model=BaseResponse, summary="获取任务列表")
async def list_tasks(
    status: Optional[str] = Query(None, description="任务状态过滤"),
    limit: int = Query(100, description="返回数量限制，默认100"),
    task_service: TaskService = Depends(get_task_service)
) -> BaseResponse:
    """
    获取任务列表
    
    Args:
        status: 任务状态过滤
        limit: 返回数量限制
        
    Returns:
        BaseResponse: 响应结果
    """
    request_id = str(uuid.uuid4())
    
    try:
        # 获取任务列表
        result = await task_service.list_tasks(status, limit)
        
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/list",
            success=True,
            code=200,
            message="获取任务列表成功",
            data=result
        )
        
    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/tasks/list",
            success=False,
            code=500,
            message=f"获取任务列表失败: {str(e)}",
            data=None
        )

@router.get("/status/{task_id}", response_model=BaseResponse, summary="获取任务状态")
async def get_task_status(
    task_id: str = Path(..., description="任务ID"),
    task_service: TaskService = Depends(get_task_service)
) -> BaseResponse:
    """
    获取任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        BaseResponse: 响应结果
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
                code=404,
                message=result["message"],
                data=None
            )
            
        return BaseResponse(
            requestId=request_id,
            path=f"/api/v1/tasks/status/{task_id}",
            success=True,
            code=200,
            message="获取任务状态成功",
            data=result["status"]
        )
        
    except Exception as e:
        logger.error(f"获取任务状态失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path=f"/api/v1/tasks/status/{task_id}",
            success=False,
            code=500,
            message=f"获取任务状态失败: {str(e)}",
            data=None
        )
