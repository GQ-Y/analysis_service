"""
分析服务入口
提供视觉分析和数据处理功能
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from core.config import settings
from core.exceptions import AnalysisException
from core.models import StandardResponse
from shared.utils.logger import setup_logger
from routers import task_router, health_router
import time
import uuid
import logging
import argparse
import uvicorn

# 设置日志
logger = setup_logger(__name__)

# 关闭 uvicorn 和 fastapi 的访问日志
if not settings.DEBUG_ENABLED:
    logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
    logging.getLogger("fastapi").setLevel(logging.ERROR)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""
    async def dispatch(self, request: Request, call_next):
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

# 从run模块导入应用生命周期管理函数
from run.run import lifespan

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
    debug=settings.DEBUG_ENABLED,
    lifespan=lifespan
)

# 添加Gzip压缩
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 只在调试模式下添加请求日志中间件
if settings.DEBUG_ENABLED:
    app.add_middleware(RequestLoggingMiddleware)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(task_router)
app.include_router(health_router)

# 全局异常处理
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

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="分析服务启动脚本")
    parser.add_argument("--host", type=str, default=settings.SERVICES_HOST, help="服务主机地址")
    parser.add_argument("--port", type=int, default=settings.SERVICES_PORT, help="服务端口")
    parser.add_argument("--reload", action="store_true", help="是否启用热重载")
    parser.add_argument("--debug", action="store_true", help="是否启用调试模式")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数量")
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 使用run模块启动应用
    from run.run import start_app
    start_app(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        debug=args.debug
    )