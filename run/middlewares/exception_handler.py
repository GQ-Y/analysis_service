"""
FastAPI统一异常处理中间件
"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import traceback
from typing import Callable

from core.exceptions import (
    AnalysisException, 
    ExceptionHandler, 
    ErrorCode,
    ProcessingException
)
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)


class UnifiedExceptionMiddleware(BaseHTTPMiddleware):
    """统一异常处理中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable):
        try:
            # 正常处理请求
            response = await call_next(request)
            return response
        except Exception as exc:
            # 统一异常处理
            return await self.handle_exception(request, exc)
    
    async def handle_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """
        处理异常并返回统一格式的错误响应
        
        Args:
            request: FastAPI请求对象
            exc: 异常对象
        
        Returns:
            JSONResponse: 统一格式的错误响应
        """
        # 获取请求信息用于上下文
        method = request.method
        url = str(request.url)
        context = f"{method} {url}"
        
        # 转换为分析服务异常
        if isinstance(exc, HTTPException):
            # 处理FastAPI的HTTPException
            analysis_exc = AnalysisException(
                message=exc.detail or "HTTP异常",
                error_code=self._http_status_to_error_code(exc.status_code),
                details={"http_status_code": exc.status_code}
            )
            status_code = exc.status_code
        elif isinstance(exc, AnalysisException):
            # 已经是分析服务异常
            analysis_exc = exc
            status_code = analysis_exc.http_status_code
        else:
            # 转换其他异常
            analysis_exc = ExceptionHandler.handle_exception(exc, context, exception_logger)
            status_code = analysis_exc.http_status_code
        
        # 记录异常日志
        error_dict = analysis_exc.to_dict()
        exception_logger.error(f"API异常处理 [{context}]: {error_dict}")
        
        # 创建错误响应
        error_response = ExceptionHandler.create_error_response(analysis_exc)
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )
    
    def _http_status_to_error_code(self, status_code: int) -> ErrorCode:
        """
        将HTTP状态码转换为错误代码
        
        Args:
            status_code: HTTP状态码
        
        Returns:
            ErrorCode: 对应的错误代码
        """
        if status_code == 400:
            return ErrorCode.INVALID_INPUT
        elif status_code == 404:
            return ErrorCode.RESOURCE_NOT_FOUND
        elif status_code == 422:
            return ErrorCode.INVALID_PARAMETER_VALUE
        elif status_code == 500:
            return ErrorCode.INTERNAL_SERVER_ERROR
        elif status_code == 503:
            return ErrorCode.SERVICE_UNAVAILABLE
        else:
            return ErrorCode.UNKNOWN_ERROR


def setup_exception_handlers(app):
    """
    设置全局异常处理器
    
    Args:
        app: FastAPI应用实例
    """
    
    @app.exception_handler(AnalysisException)
    async def analysis_exception_handler(request: Request, exc: AnalysisException):
        """处理分析服务异常"""
        method = request.method
        url = str(request.url)
        context = f"{method} {url}"
        
        # 记录日志
        error_dict = exc.to_dict()
        exception_logger.error(f"分析服务异常 [{context}]: {error_dict}")
        
        # 返回错误响应
        error_response = ExceptionHandler.create_error_response(exc)
        return JSONResponse(
            status_code=exc.http_status_code,
            content=error_response
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """处理HTTP异常"""
        method = request.method
        url = str(request.url)
        context = f"{method} {url}"
        
        # 转换为分析服务异常
        error_code_map = {
            400: ErrorCode.INVALID_INPUT,
            404: ErrorCode.RESOURCE_NOT_FOUND,
            422: ErrorCode.INVALID_PARAMETER_VALUE,
            500: ErrorCode.INTERNAL_SERVER_ERROR,
            503: ErrorCode.SERVICE_UNAVAILABLE
        }
        
        error_code = error_code_map.get(exc.status_code, ErrorCode.UNKNOWN_ERROR)
        analysis_exc = AnalysisException(
            message=exc.detail or "HTTP异常",
            error_code=error_code,
            details={"http_status_code": exc.status_code}
        )
        
        # 记录日志
        error_dict = analysis_exc.to_dict()
        exception_logger.error(f"HTTP异常 [{context}]: {error_dict}")
        
        # 返回错误响应
        error_response = ExceptionHandler.create_error_response(analysis_exc)
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """处理通用异常"""
        method = request.method
        url = str(request.url)
        context = f"{method} {url}"
        
        # 转换为分析服务异常
        analysis_exc = ExceptionHandler.handle_exception(exc, context, exception_logger)
        
        # 记录详细堆栈信息
        exception_logger.exception(f"未处理异常 [{context}]: {str(exc)}")
        
        # 返回错误响应
        error_response = ExceptionHandler.create_error_response(analysis_exc)
        return JSONResponse(
            status_code=analysis_exc.http_status_code,
            content=error_response
        )
    
    normal_logger.info("全局异常处理器设置完成")
