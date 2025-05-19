"""
请求日志中间件
提供请求日志记录功能
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid

from core.config import settings
# 使用新的日志记录器
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件
    记录请求的开始、结束和处理时间
    """
    async def dispatch(self, request: Request, call_next):
        """
        处理请求
        
        Args:
            request: 请求对象
            call_next: 下一个中间件或路由处理函数
            
        Returns:
            Response: 响应对象
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # 添加请求ID和开始时间到请求状态
        request.state.request_id = request_id
        request.state.start_time = int(start_time * 1000)

        try:
            # 只在调试模式下记录请求开始信息
            if settings.DEBUG_ENABLED:
                normal_logger.info(f"请求开始: ID={request_id}, 方法={request.method}, 路径={request.url.path}, 客户端={request.client.host}:{request.client.port}")

            response = await call_next(request)

            # 只在调试模式下记录响应信息
            if settings.DEBUG_ENABLED:
                process_time = (time.time() - start_time) * 1000
                normal_logger.info(
                    f"请求完成: ID={request_id}, 方法={request.method}, 路径={request.url.path}, "
                    f"状态={response.status_code}, 耗时={process_time:.2f}ms"
                )

            # 添加请求ID到响应头
            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as e:
            # 错误日志仍然需要记录，但使用 ERROR 级别
            process_time = (time.time() - start_time) * 1000
            # 记录异常时使用exception_logger
            exception_logger.exception(
                f"请求处理异常: ID={request_id}, 方法={request.method}, 路径={request.url.path}, "
                f"错误详情={str(e)}, 耗时={process_time:.2f}ms"
            )
            raise
