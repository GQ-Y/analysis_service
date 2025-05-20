"""
健康检查路由
提供服务健康状态检查API
"""
from fastapi import APIRouter, Request
import psutil
import uuid
import time
from datetime import datetime

from core.config import settings
from core.models import StandardResponse
from shared.utils.logger import get_normal_logger, get_exception_logger, get_test_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()

# 创建路由
router = APIRouter(
    prefix="/api/v1/health",
    tags=["健康检查"],
    responses={404: {"description": "Not found"}},
)

@router.get("", summary="健康检查")
async def health_check(request: Request) -> StandardResponse:
    """
    健康检查接口
    
    返回服务的健康状态信息，包括：
    - 服务名称
    - 版本号
    - CPU使用率
    - 内存使用率
    - GPU使用情况（如果可用）
    - 运行时长
    
    Returns:
        StandardResponse: 响应结果，包含健康状态信息
    """
    request_id = str(uuid.uuid4())
    
    try:
        test_logger.info("TEST_LOG_MARKER: API_HEALTH_CHECK_SUCCESS")
        
        # 获取CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 获取内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 获取GPU使用情况
        gpu_usage = "N/A"
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = f"{gpus[0].load * 100:.1f}%"
        except:
            pass
        
        # 获取启动时间
        if hasattr(request.app.state, "start_time"):
            uptime_seconds = time.time() - request.app.state.start_time
            hours, remainder = divmod(uptime_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        else:
            uptime = "未知"
        
        # 获取任务管理器状态
        task_info = {}
        if hasattr(request.app.state, "task_manager"):
            task_manager = request.app.state.task_manager
            task_info = {
                "total_tasks": task_manager.get_task_count(),
                "active_tasks": len(task_manager.get_active_tasks()),
                "completed_tasks": len(task_manager.get_completed_tasks()),
                "failed_tasks": len(task_manager.get_failed_tasks())
            }
        
        return StandardResponse(
            requestId=request_id,
            path="/api/v1/health",
            success=True,
            code=200,
            message="服务正常运行",
            data={
                "status": "healthy",
                "name": "analysis_service",
                "version": settings.VERSION,
                "environment": settings.ENVIRONMENT,
                "cpu": f"{cpu_percent:.1f}%",
                "memory": f"{memory_percent:.1f}%",
                "gpu": gpu_usage,
                "uptime": uptime,
                "task_info": task_info,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        exception_logger.exception(f"健康检查失败: {str(e)}")
        return StandardResponse(
            requestId=request_id,
            path="/api/v1/health",
            success=False,
            code=500,
            message=f"健康检查失败: {str(e)}",
            data=None
        )
