"""
分析服务启动模块
负责初始化和启动FastAPI应用
"""
import os
import sys
import uvicorn
import asyncio
import traceback
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.config import settings
from shared.utils.logger import setup_logger
from run.middlewares import setup_exception_handlers, RequestLoggingMiddleware

logger = setup_logger(__name__)

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

def create_app() -> FastAPI:
    """
    创建FastAPI应用

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
        debug=settings.DEBUG_ENABLED,
        lifespan=lifespan
    )

    # 添加Gzip压缩
    from fastapi.middleware.gzip import GZipMiddleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # 只在调试模式下添加请求日志中间件
    if settings.DEBUG_ENABLED:
        app.add_middleware(RequestLoggingMiddleware)

    # 添加CORS中间件
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 设置全局异常处理器
    setup_exception_handlers(app)

    # 注册路由
    from routers import task_router, health_router
    from routers.task_video import router as video_router
    app.include_router(task_router)
    app.include_router(health_router)
    app.include_router(video_router)

    # 添加静态文件支持
    from fastapi.staticfiles import StaticFiles
    static_dir = Path(ROOT_DIR) / "static"
    if not static_dir.exists():
        static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # 静态文件已挂载

    return app

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    show_service_banner("analysis_service")
    logger.info("Starting Analysis Service...")

    # 记录启动时间
    app.state.start_time = time.time()

    if settings.DEBUG_ENABLED:
        logger.info("分析服务启动...")
        logger.info(f"环境: {settings.ENVIRONMENT}")
        logger.info(f"调试模式: {settings.DEBUG_ENABLED}")
        logger.info(f"版本: {settings.VERSION}")
        logger.info(f"注册的路由: {[route.path for route in app.routes]}")

    # 初始化任务管理器
    logger.info("正在初始化任务管理器...")
    try:
        # 导入任务管理器
        from core.task_management.manager import TaskManager

        # 创建任务管理器实例
        task_manager = TaskManager()
        await task_manager.initialize()

        # 保存实例到应用状态
        app.state.task_manager = task_manager

        # 创建服务实例
        from services.http.task_service import TaskService
        from services.analysis_service import AnalysisService
        from services.http.callback_service import CallbackService
        from services.http.video_encoder_service import VideoEncoderService

        app.state.task_service = TaskService(task_manager)
        app.state.analysis_service = AnalysisService()
        app.state.callback_service = CallbackService()
        app.state.video_encoder_service = VideoEncoderService()

        logger.info("任务管理器和服务初始化成功")

    except Exception as e:
        logger.error(f"服务初始化失败: {str(e)}")
        logger.error(traceback.format_exc())

    yield  # 这里是应用运行期间

    # 关闭时执行
    logger.info("Shutting down Analysis Service...")

    # 关闭任务管理器
    if hasattr(app.state, "task_manager") and app.state.task_manager:
        try:
            logger.info("正在停止任务管理器...")
            await app.state.task_manager.shutdown()
            logger.info("任务管理器已停止")
        except Exception as e:
             logger.error(f"停止任务管理器时出错: {e}")
    else:
        logger.info("没有活动的任务管理器需要关闭")

    logger.info("Analysis Service stopped.")

def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description="分析服务启动脚本")
    parser.add_argument("--host", type=str, default=settings.SERVICES_HOST, help="服务主机地址")
    parser.add_argument("--port", type=int, default=settings.SERVICES_PORT, help="服务端口")
    parser.add_argument("--reload", action="store_true", help="是否启用热重载")
    parser.add_argument("--debug", action="store_true", help="是否启用调试模式")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数量")
    return parser.parse_args()

def start_app(host=None, port=None, reload=None, workers=None, debug=None, parse_command_line=False):
    """
    启动FastAPI应用

    Args:
        host: 主机地址，默认使用settings.SERVICES_HOST
        port: 端口号，默认使用settings.SERVICES_PORT
        reload: 是否启用热重载，默认使用settings.DEBUG_ENABLED
        workers: 工作进程数量，默认为1
        debug: 是否启用调试模式，默认为False
        parse_command_line: 是否解析命令行参数，默认为False
    """
    # 如果需要解析命令行参数
    if parse_command_line:
        args = parse_args()
        host = args.host
        port = args.port
        reload = args.reload
        workers = args.workers
        debug = args.debug

    # 设置环境变量
    if debug is not None and debug:
        os.environ["DEBUG_ENABLED"] = "1"

    # 使用参数或默认值
    host = host or settings.SERVICES_HOST
    port = port or settings.SERVICES_PORT
    reload = reload if reload is not None else settings.DEBUG_ENABLED
    workers = workers or 1

    # 打印启动信息
    logger.info(f"启动分析服务 - 版本: {settings.VERSION}")
    logger.info(f"主机: {host}, 端口: {port}")
    logger.info(f"调试模式: {settings.DEBUG_ENABLED}, 热重载: {reload}")
    logger.info(f"工作进程数: {workers}")

    # 创建应用实例
    app = create_app()

    # 如果是直接运行，使用uvicorn启动
    if reload:
        # 热重载模式下，使用uvicorn.run启动
        uvicorn.run(
            "run.run:create_app",
            host=host,
            port=port,
            reload=reload,
            factory=True,
            workers=1,  # 热重载模式下只能使用1个工作进程
            log_level="debug" if settings.DEBUG_ENABLED else "info"
        )
    else:
        # 非热重载模式下，使用Config和Server启动
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            workers=workers,
            log_level="debug" if settings.DEBUG_ENABLED else "info"
        )
        server = uvicorn.Server(config)
        server.run()

def main():
    """
    主函数，作为app.py的入口点
    """
    # 启动应用，并解析命令行参数
    start_app(parse_command_line=True)