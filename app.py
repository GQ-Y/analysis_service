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
import time
import uuid
import logging
import uvicorn
import psutil
import GPUtil

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

# 创建FastAPI应用
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

# 添加Gzip压缩
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 只在调试模式下添加请求日志中间件
if settings.DEBUG_ENABLED:
    app.add_middleware(RequestLoggingMiddleware)

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
        ).dict()
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
            "requestId": request.state.request_id,
            "path": request.url.path,
            "success": False,
            "message": error_msg,
            "code": 500,
            "data": None,
            "timestamp": request.state.start_time
        }
    )

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查接口"""
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
    
    return StandardResponse(
        requestId=str(uuid.uuid4()),
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
            "memory": f"{memory_percent:.1f}%"
        }
    )

def show_service_banner(service_name: str):
    """显示服务启动标识"""
    banner = f"""
███╗   ███╗███████╗███████╗██╗  ██╗██╗   ██╗ ██████╗ ██╗      ██████╗     @{service_name}
████╗ ████║██╔════╝██╔════╝██║ ██╔╝╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗
██╔████╔██║█████╗  █████╗  █████╔╝  ╚████╔╝ ██║   ██║██║     ██║   ██║
██║╚██╔╝██║██╔══╝  ██╔══╝  ██╔═██╗   ╚██╔╝  ██║   ██║██║     ██║   ██║
██║ ╚═╝ ██║███████╗███████╗██║  ██╗   ██║   ╚██████╔╝███████╗╚██████╔╝
╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝ 
    """
    print(banner)

@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    show_service_banner("analysis_service")
    logger.info("Starting Analysis Service...")
    
    if settings.DEBUG_ENABLED:
        logger.info("分析服务启动...")
        logger.info(f"环境: {settings.ENVIRONMENT}")
        logger.info(f"调试模式: {settings.DEBUG_ENABLED}")
        logger.info(f"版本: {settings.VERSION}")
        logger.info(f"注册的路由: {[route.path for route in app.routes]}")
    
    # 初始化MQTT通信模式
    logger.info("使用MQTT通信模式，正在初始化MQTT客户端...")
    try:
        from services.mqtt_client import get_mqtt_client
        from services.analyzer import create_analyzer_service
        
        # 记录MQTT配置信息
        mqtt_config = {
            "broker_host": settings.MQTT_BROKER_HOST,
            "broker_port": settings.MQTT_BROKER_PORT,
            "username": settings.MQTT_USERNAME,
            "password": "******" if settings.MQTT_PASSWORD else None,
            "topic_prefix": settings.MQTT_TOPIC_PREFIX
        }
        logger.info(f"MQTT配置: {mqtt_config}")
        
        # 使用工厂函数初始化分析服务
        analyzer_service = create_analyzer_service("mqtt")
        app.state.analyzer_service = analyzer_service
        
        # 确保MQTT分析服务连接并注册处理器
        # 这一步非常关键，确保任务处理器正确注册到MQTT客户端
        connect_result = await analyzer_service.connect()
        if connect_result:
            logger.info("MQTT分析服务已连接并注册处理器")
            
            # 获取MQTT客户端实例以验证处理器是否已注册
            mqtt_client = analyzer_service.mqtt_client
            handler_types = list(mqtt_client.task_handlers.keys())
            logger.info(f"已注册的任务处理器类型: {handler_types}")
            
            if not handler_types:
                logger.error("未找到已注册的任务处理器，MQTT任务处理将无法正常工作!")
        else:
            logger.error("MQTT分析服务连接失败，任务处理器可能未注册!")
    except Exception as e:
        logger.error(f"MQTT客户端初始化失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    logger.info("Shutting down Analysis Service...")
    
    # 关闭分析服务
    if hasattr(app.state, "analyzer_service"):
        await app.state.analyzer_service.disconnect()
        logger.info("分析服务已关闭")
    
    logger.info("Analysis Service stopped.")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.SERVICES_HOST,
        port=settings.SERVICES_PORT,
        reload=settings.DEBUG_ENABLED
    )