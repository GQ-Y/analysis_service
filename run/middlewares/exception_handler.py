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
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

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
        error_msg = f"请求处理失败: {str(exc)}"
        from core.config import settings
        
        if settings.DEBUG_ENABLED:
            logger.exception(error_msg)
        else:
            logger.error(error_msg)

        return JSONResponse(
            status_code=500,
            content={
                "requestId": getattr(request.state, "request_id", str(uuid.uuid4())),
                "path": request.url.path,
                "success": False,
                "message": error_msg,
                "code": 500,
                "data": None,
                "timestamp": getattr(request.state, "start_time", int(time.time() * 1000))
            }
        )
