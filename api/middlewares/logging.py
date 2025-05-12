"""
日志中间件
记录请求和响应信息
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid

from core.config import settings
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""
    
    async def dispatch(self, request: Request, call_next):
        """
        处理请求
        
        Args:
            request: 请求对象
            call_next: 下一个中间件
            
        Returns:
            响应对象
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # 添加请求ID和开始时间到请求状态
        request.state.request_id = request_id
        request.state.start_time = int(start_time * 1000)

        try:
            # 只在调试模式下记录请求开始信息
            if settings.DEBUG_ENABLED:
                logger.info(f"请求开始: {request_id} - {request.method} {request.url.path}")

            response = await call_next(request)

            # 只在调试模式下记录响应信息
            if settings.DEBUG_ENABLED:
                process_time = (time.time() - start_time) * 1000
                logger.info(
                    f"请求完成: {request_id} - {request.method} {request.url.path} "
                    f"- 状态: {response.status_code} - 耗时: {process_time:.2f}ms"
                )

            # 添加请求ID到响应头
            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as e:
            # 错误日志仍然需要记录，但使用 ERROR 级别
            process_time = (time.time() - start_time) * 1000
            logger.error(
                f"请求失败: {request_id} - {request.method} {request.url.path} "
                f"- 错误: {str(e)} - 耗时: {process_time:.2f}ms"
            )
            raise
