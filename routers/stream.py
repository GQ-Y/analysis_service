"""
流管理路由
提供流的查询、状态获取等API
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Request
from typing import List, Dict, Any, Optional
import uuid

from models.responses import BaseResponse
from shared.utils.logger import get_normal_logger, get_exception_logger
from shared.utils.app_state import app_state_manager
from core.task_management.stream import StreamManager

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

# 创建路由
router = APIRouter(
    prefix="/api/v1/streams",
    tags=["流管理"],
    responses={404: {"description": "Not found"}},
)

@router.get("", response_model=BaseResponse, summary="获取所有流信息")
async def get_streams() -> BaseResponse:
    """
    获取所有流信息
    
    返回系统中所有流的基本信息，包括ID、URL、状态等。
    
    Returns:
        BaseResponse: 响应结果，包含所有流信息
    """
    request_id = str(uuid.uuid4())
    
    try:
        # 获取所有流
        stream_manager = app_state_manager.get_stream_manager()
        if not stream_manager:
            raise HTTPException(status_code=503, detail="流管理器未初始化")
            
        streams = await stream_manager.get_all_streams()
        
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/streams",
            success=True,
            message="获取所有流信息成功",
            code=200,
            data={
                "streams": streams,
                "total": len(streams)
            }
        )
    except Exception as e:
        exception_logger.exception(f"获取所有流信息失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path="/api/v1/streams",
            success=False,
            message=f"获取所有流信息失败: {str(e)}",
            code=500,
            data=None
        )

@router.get("/{stream_id}", response_model=BaseResponse, summary="获取流信息")
async def get_stream(stream_id: str = Path(..., description="流ID")) -> BaseResponse:
    """
    获取流信息
    
    返回指定流的详细信息，包括ID、URL、状态、健康状态等。
    
    Args:
        stream_id: 流ID
        
    Returns:
        BaseResponse: 响应结果，包含流信息
    """
    request_id = str(uuid.uuid4())
    
    try:
        # 获取流信息
        stream_manager = app_state_manager.get_stream_manager()
        if not stream_manager:
            raise HTTPException(status_code=503, detail="流管理器未初始化")
            
        stream_info = await stream_manager.get_stream_info(stream_id)
        
        if not stream_info:
            raise HTTPException(status_code=404, detail=f"未找到流: {stream_id}")
        
        return BaseResponse(
            requestId=request_id,
            path=f"/api/v1/streams/{stream_id}",
            success=True,
            message="获取流信息成功",
            code=200,
            data=stream_info
        )
    except Exception as e:
        exception_logger.exception(f"获取流信息失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path=f"/api/v1/streams/{stream_id}",
            success=False,
            message=f"获取流信息失败: {str(e)}",
            code=500,
            data=None
        )

@router.get("/{stream_id}/proxy_url", response_model=BaseResponse, summary="获取流代理URL信息")
async def get_stream_proxy_url(stream_id: str = Path(..., description="流ID")) -> BaseResponse:
    """
    获取流代理URL信息
    
    返回指定流的代理URL信息，包括URL、状态、配置等，用于外部播放器访问。
    
    Args:
        stream_id: 流ID
        
    Returns:
        BaseResponse: 响应结果，包含代理URL信息
    """
    request_id = str(uuid.uuid4())
    
    try:
        # 获取流代理URL信息
        stream_manager = app_state_manager.get_stream_manager()
        if not stream_manager:
            raise HTTPException(status_code=503, detail="流管理器未初始化")
            
        proxy_info = await stream_manager.get_stream_proxy_url(stream_id)
        
        if not proxy_info:
            raise HTTPException(status_code=404, detail=f"未找到流的代理地址: {stream_id}")
        
        # 检查是否有错误
        if "error" in proxy_info:
            return BaseResponse(
                requestId=request_id,
                path=f"/api/v1/streams/{stream_id}/proxy_url",
                success=False,
                message=proxy_info["error"],
                code=400,
                data=proxy_info
            )
        
        return BaseResponse(
            requestId=request_id,
            path=f"/api/v1/streams/{stream_id}/proxy_url",
            success=True,
            message="获取流代理URL信息成功",
            code=200,
            data=proxy_info
        )
    except Exception as e:
        exception_logger.exception(f"获取流代理URL信息失败: {str(e)}")
        return BaseResponse(
            requestId=request_id,
            path=f"/api/v1/streams/{stream_id}/proxy_url",
            success=False,
            message=f"获取流代理URL信息失败: {str(e)}",
            code=500,
            data=None
        )

@router.post("/reconnect/{stream_id}")
async def reconnect_stream(stream_id: str) -> Dict[str, bool]:
    """请求重新连接流"""
    try:
        stream_manager = app_state_manager.get_stream_manager()
        if not stream_manager:
            raise HTTPException(status_code=503, detail="流管理器未初始化")
            
        success = await stream_manager.reconnect_stream(stream_id)
        return {"success": success}
    except Exception as e:
        exception_logger.exception(f"重新连接流时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 