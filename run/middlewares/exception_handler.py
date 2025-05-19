"""
异常处理中间件
提供全局异常处理功能
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uuid
import time

from core.exceptions import AnalysisException
from core.models import StandardResponse
from shared.utils.logger import get_exception_logger
from core.config import settings

# 初始化日志记录器
exception_logger = get_exception_logger(__name__)

def setup_exception_handlers(app: FastAPI):
    """
    设置全局异常处理器
    
    Args:
        app: FastAPI应用实例
    """
    @app.exception_handler(AnalysisException)
    async def analysis_exception_handler(request: Request, exc: AnalysisException):
        """
        处理分析服务异常
        
        Args:
            request: 请求对象
            exc: 异常对象
            
        Returns:
            JSONResponse: JSON响应
        """
        # 使用异常日志记录器记录业务异常的详细信息
        exception_logger.error(
            f"业务异常: {exc.message}, 代码: {exc.code}, 路径: {request.url.path}, 数据: {exc.data}"
        )
        return JSONResponse(
            status_code=exc.code,
            content=StandardResponse(
                requestId=getattr(request.state, "request_id", str(uuid.uuid4())),
                path=request.url.path,
                success=False,
                code=exc.code,
                message=exc.message,
                data=exc.data
            ).model_dump()
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """
        全局异常处理器
        
        Args:
            request: 请求对象
            exc: 异常对象
            
        Returns:
            JSONResponse: JSON响应
        """
        error_msg = f"请求处理时发生未捕获的内部错误: {str(exc)}"
        
        # 总是使用exception_logger.exception()来记录完整的堆栈跟踪
        exception_logger.exception(f"全局异常捕获于路径 {request.url.path}: {exc}")

        return JSONResponse(
            status_code=500,
            content={
                "requestId": getattr(request.state, "request_id", str(uuid.uuid4())),
                "path": request.url.path,
                "success": False,
                "message": "服务器内部错误，请联系管理员。" if not settings.DEBUG_ENABLED else error_msg, # 生产环境隐藏详细错误
                "code": 500,
                "data": None,
                "timestamp": getattr(request.state, "start_time", int(time.time() * 1000))
            }
        )
