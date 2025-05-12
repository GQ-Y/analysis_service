"""
API应用模块
创建和配置FastAPI应用
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from core.config import settings
from core.task_management.manager import TaskManager
from core.exceptions import AnalysisException
from core.models import StandardResponse
from shared.utils.logger import setup_logger
import uuid
import time

logger = setup_logger(__name__)

class RequestLoggingMiddleware:
    """请求日志中间件"""
    async def __call__(self, request: Request, call_next):
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

def setup_exception_handlers(app: FastAPI):
    """设置异常处理器"""
    
    @app.exception_handler(AnalysisException)
    async def analysis_exception_handler(request: Request, exc: AnalysisException):
        """处理分析服务异常"""
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
        """全局异常处理器"""
        error_msg = f"请求处理失败: {str(exc)}"
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

def create_app(task_manager: TaskManager = None) -> FastAPI:
    """
    创建FastAPI应用
    
    Args:
        task_manager: 任务管理器实例
        
    Returns:
        FastAPI: FastAPI应用实例
    """
    # 创建应用
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="""
        分析服务模块
        
        提供以下功能:
        - 视觉对象检测和分析
        - 实例分割
        - 目标跟踪
        - 跨摄像头目标跟踪
        - 分析结果存储和查询
        """,
        version=settings.VERSION,
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        openapi_url="/api/v1/openapi.json",
        debug=settings.DEBUG_ENABLED
    )
    
    # 添加中间件
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    if settings.DEBUG_ENABLED:
        app.add_middleware(RequestLoggingMiddleware)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 设置异常处理器
    setup_exception_handlers(app)
    
    # 注册路由
    from api.routes import task_router, health_router
    app.include_router(task_router)
    app.include_router(health_router)
    
    # 保存任务管理器实例
    app.state.task_manager = task_manager
    
    # 创建服务实例
    from services.task_service import TaskService
    from services.analysis_service import AnalysisService
    
    app.state.task_service = TaskService(task_manager)
    app.state.analysis_service = AnalysisService()
    
    return app
