"""
任务管理路由
提供任务的创建、查询、停止等API
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

from models.requests import StreamTask, BatchStreamTask
from models.responses import BaseResponse, TaskInfo, SubTaskInfo
from models.task import TaskBase
from services.http.task_service import TaskService
from services.task_store import TaskStore
from crud.task import TaskCRUD
from core.task_management import TaskStatus
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
async def get_task_service() -> TaskService:
    """获取任务服务实例"""
    task_store = TaskStore(settings.REDIS_URL)
    await task_store.connect()
    task_crud = TaskCRUD(task_store)
    return TaskService(task_crud)

@router.post("/start", response_model=BaseResponse)
async def start_task(
    task: StreamTask,
    task_service: TaskService = Depends(get_task_service)
) -> BaseResponse:
    """
    启动单个流分析任务
    
    Args:
        task: 流分析任务参数
        
    Returns:
        BaseResponse: 响应结果
    """
    request_id = str(uuid.uuid4())
    
    try:
        # 启动任务
        result = await task_service.start_task(
            model_code=task.model_code,
            stream_url=task.stream_url,
            task_name=task.task_name,
            callback_urls=None,  # 单任务模式不支持回调
            output_url=task.output_url,
            analysis_type="detection",  # 默认为检测任务
            config=task.config.model_dump() if task.config else None,
            enable_callback=False,
            save_result=task.save_result
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

@router.post("/batch/start", response_model=BaseResponse)
async def start_batch_tasks(
    batch_task: BatchStreamTask,
    task_service: TaskService = Depends(get_task_service)
) -> BaseResponse:
    """
    批量启动流分析任务
    
    Args:
        batch_task: 批量流分析任务参数
        
    Returns:
        BaseResponse: 响应结果
    """
    request_id = str(uuid.uuid4())
    
    try:
        # 批量启动任务
        task_ids = []
        failed_tasks = []
        
        for task in batch_task.tasks:
            result = await task_service.start_task(
                model_code=task.model_code,
                stream_url=task.stream_url,
                task_name=task.task_name,
                callback_urls=batch_task.callback_urls,
                output_url=task.output_url,
                analysis_type="detection",  # 默认为检测任务
                config=task.config.model_dump() if task.config else None,
                enable_callback=bool(batch_task.callback_urls),
                save_result=task.save_result
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

@router.post("/stop/{task_id}", response_model=BaseResponse)
async def stop_task(
    task_id: str = Path(..., description="任务ID"),
    task_service: TaskService = Depends(get_task_service)
) -> BaseResponse:
    """
    停止任务
    
    Args:
        task_id: 任务ID
        
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

@router.get("/status/{task_id}", response_model=BaseResponse)
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

@router.get("/list", response_model=BaseResponse)
async def list_tasks(
    status: Optional[int] = Query(None, description="任务状态过滤"),
    limit: int = Query(100, description="返回数量限制"),
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
    
    # 这个API需要在TaskService中添加list_tasks方法
    # 由于当前实现中没有该方法，这里返回未实现错误
    return BaseResponse(
        requestId=request_id,
        path="/api/v1/tasks/list",
        success=False,
        message="该API尚未实现",
        code=501,
        data=None
    )
