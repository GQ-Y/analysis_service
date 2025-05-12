"""
健康检查路由
提供系统健康状态检查API
"""
from fastapi import APIRouter, Request
import uuid
import psutil
import GPUtil

from core.config import settings
from core.models import StandardResponse
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

# 创建路由
router = APIRouter(
    tags=["系统"],
    responses={404: {"description": "Not found"}},
)

@router.get("/health", response_model=StandardResponse, summary="健康检查")
async def health_check(request: Request) -> StandardResponse:
    """
    健康检查接口
    
    返回系统的健康状态，包括CPU、内存和GPU使用情况
    
    Returns:
        StandardResponse: 响应结果
    """
    # 获取CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # 获取内存使用情况
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # 获取GPU使用情况
    try:
        gpus = GPUtil.getGPUs()
        gpu_usage = f"{gpus[0].load * 100:.1f}%" if gpus else "N/A"
    except:
        gpu_usage = "N/A"
        
    # 获取任务管理器状态
    task_manager = getattr(request.app.state, "task_manager", None)
    task_count = 0
    if task_manager:
        task_count = task_manager.get_task_count() if hasattr(task_manager, "get_task_count") else len(getattr(task_manager, "tasks", {}))
    
    return StandardResponse(
        requestId=getattr(request.state, "request_id", str(uuid.uuid4())),
        path="/health",
        success=True,
        code=200,
        message="服务正常运行",
        data={
            "status": "healthy",
            "name": "analysis",
            "version": settings.VERSION,
            "cpu": f"{cpu_percent:.1f}%",
            "gpu": gpu_usage,
            "memory": f"{memory_percent:.1f}%",
            "task_count": task_count
        }
    )
