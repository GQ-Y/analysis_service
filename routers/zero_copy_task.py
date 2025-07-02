"""
零拷贝任务路由
提供零拷贝视频流分析任务的API接口
"""
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from typing import Dict, Any, Optional
import uuid

from models.requests import StreamAnalysisRequest
from models.responses import BaseResponse
from services.http.zero_copy_task_service import ZeroCopyTaskService
from core.config import settings
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

# 创建路由
router = APIRouter(
    prefix="/api/v1/zero-copy",
    tags=["零拷贝任务"],
    responses={404: {"description": "Not found"}},
)

# 依赖注入
async def get_zero_copy_task_service(request: Request) -> ZeroCopyTaskService:
    """获取零拷贝任务服务实例"""
    if not hasattr(request.app.state, "task_service"):
        raise HTTPException(status_code=500, detail="零拷贝任务服务未初始化")
    return request.app.state.task_service


@router.post("/tasks", response_model=BaseResponse, summary="创建零拷贝分析任务")
async def create_zero_copy_task(
    request: StreamAnalysisRequest,
    task_service: ZeroCopyTaskService = Depends(get_zero_copy_task_service)
) -> BaseResponse:
    """
    创建零拷贝视频流分析任务
    
    使用零拷贝架构进行高性能视频流处理，支持：
    - 零拷贝帧数据传递
    - 共享内存池管理
    - 批量处理优化
    - 智能内存回收
    
    Args:
        request: 流分析请求参数
        task_service: 零拷贝任务服务
        
    Returns:
        BaseResponse: 创建结果
    """
    try:
        normal_logger.info(f"创建零拷贝任务: model={request.model_code}, stream={request.stream_url}")
        
        # 启动零拷贝任务
        result = await task_service.start_task(
            model_code=request.model_code,
            stream_url=request.stream_url,
            task_name=request.task_name,
            callback_urls=request.callback_urls,
            analysis_type=request.analysis_type,
            config=request.config.dict() if request.config else None,
            enable_callback=request.enable_callback,
            save_result=request.save_result,
            # 零拷贝特定参数
            stream_engine="zero_copy",
            low_latency=True,
            enable_hardware_decode=True
        )
        
        if result["success"]:
            return BaseResponse(
                success=True,
                message="零拷贝任务创建成功",
                data={
                    "task_id": result["task_id"],
                    "zero_copy_enabled": result.get("zero_copy_enabled", True),
                    "memory_pool_status": result.get("memory_pool_status"),
                    "architecture": "zero_copy"
                }
            )
        else:
            return BaseResponse(
                success=False,
                message=result["message"],
                data={"task_id": result.get("task_id")}
            )
            
    except Exception as e:
        exception_logger.exception(f"创建零拷贝任务失败: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"创建零拷贝任务失败: {str(e)}",
            data=None
        )


@router.get("/tasks/{task_id}/status", response_model=BaseResponse, summary="获取零拷贝任务状态")
async def get_zero_copy_task_status(
    task_id: str,
    task_service: ZeroCopyTaskService = Depends(get_zero_copy_task_service)
) -> BaseResponse:
    """
    获取零拷贝任务状态
    
    Args:
        task_id: 任务ID
        task_service: 零拷贝任务服务
        
    Returns:
        BaseResponse: 任务状态信息
    """
    try:
        result = await task_service.get_task_status(task_id)
        
        if result["success"]:
            return BaseResponse(
                success=True,
                message="获取任务状态成功",
                data=result
            )
        else:
            return BaseResponse(
                success=False,
                message=result["message"],
                data={"task_id": task_id}
            )
            
    except Exception as e:
        exception_logger.exception(f"获取零拷贝任务状态失败: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"获取零拷贝任务状态失败: {str(e)}",
            data={"task_id": task_id}
        )


@router.delete("/tasks/{task_id}", response_model=BaseResponse, summary="停止零拷贝任务")
async def stop_zero_copy_task(
    task_id: str,
    task_service: ZeroCopyTaskService = Depends(get_zero_copy_task_service)
) -> BaseResponse:
    """
    停止零拷贝任务
    
    Args:
        task_id: 任务ID
        task_service: 零拷贝任务服务
        
    Returns:
        BaseResponse: 停止结果
    """
    try:
        result = await task_service.stop_task(task_id)
        
        if result["success"]:
            return BaseResponse(
                success=True,
                message="零拷贝任务停止成功",
                data=result
            )
        else:
            return BaseResponse(
                success=False,
                message=result["message"],
                data={"task_id": task_id}
            )
            
    except Exception as e:
        exception_logger.exception(f"停止零拷贝任务失败: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"停止零拷贝任务失败: {str(e)}",
            data={"task_id": task_id}
        )


@router.get("/performance", response_model=BaseResponse, summary="获取零拷贝性能统计")
async def get_zero_copy_performance(
    task_service: ZeroCopyTaskService = Depends(get_zero_copy_task_service)
) -> BaseResponse:
    """
    获取零拷贝架构性能统计信息
    
    Args:
        task_service: 零拷贝任务服务
        
    Returns:
        BaseResponse: 性能统计信息
    """
    try:
        stats = task_service.get_performance_stats()
        
        return BaseResponse(
            success=True,
            message="获取零拷贝性能统计成功",
            data={
                "performance_stats": stats,
                "architecture": "zero_copy",
                "timestamp": stats.get("timestamp")
            }
        )
        
    except Exception as e:
        exception_logger.exception(f"获取零拷贝性能统计失败: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"获取零拷贝性能统计失败: {str(e)}",
            data=None
        )


@router.get("/memory/status", response_model=BaseResponse, summary="获取内存池状态")
async def get_memory_pool_status(
    task_service: ZeroCopyTaskService = Depends(get_zero_copy_task_service)
) -> BaseResponse:
    """
    获取内存池状态信息
    
    Args:
        task_service: 零拷贝任务服务
        
    Returns:
        BaseResponse: 内存池状态信息
    """
    try:
        if not task_service.memory_pool:
            return BaseResponse(
                success=False,
                message="内存池未初始化",
                data=None
            )
        
        stats = task_service.memory_pool.get_stats()
        # 内存池只有get_stats方法，没有get_status方法
        
        return BaseResponse(
            success=True,
            message="获取内存池状态成功",
            data={
                "memory_pool_stats": stats,
                "architecture": "zero_copy"
            }
        )
        
    except Exception as e:
        exception_logger.exception(f"获取内存池状态失败: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"获取内存池状态失败: {str(e)}",
            data=None
        )


@router.post("/memory/cleanup", response_model=BaseResponse, summary="手动触发内存清理")
async def trigger_memory_cleanup(
    task_service: ZeroCopyTaskService = Depends(get_zero_copy_task_service),
    force: bool = Query(False, description="是否强制清理")
) -> BaseResponse:
    """
    手动触发内存池清理
    
    Args:
        task_service: 零拷贝任务服务
        force: 是否强制清理
        
    Returns:
        BaseResponse: 清理结果
    """
    try:
        if not task_service.memory_pool:
            return BaseResponse(
                success=False,
                message="内存池未初始化",
                data=None
            )
        
        # 触发清理
        if force:
            cleaned_count = task_service.memory_pool.force_cleanup()
        else:
            cleaned_count = task_service.memory_pool.cleanup_expired_blocks()
        
        return BaseResponse(
            success=True,
            message=f"内存清理完成，清理了 {cleaned_count} 个内存块",
            data={
                "cleaned_blocks": cleaned_count,
                "force_cleanup": force,
                "memory_pool_stats": task_service.memory_pool.get_stats()
            }
        )
        
    except Exception as e:
        exception_logger.exception(f"内存清理失败: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"内存清理失败: {str(e)}",
            data=None
        )
