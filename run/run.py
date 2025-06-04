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
from shared.utils.logger import get_normal_logger, get_exception_logger
from run.middlewares import setup_exception_handlers, RequestLoggingMiddleware
from run.signal_handler import signal_handler
from run.zlm_exit_handler import zlm_exit_handler
from shared.utils.socket_manager import startup_socket_manager, shutdown_socket_manager

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

def show_service_banner(service_name: str):
    """显示服务启动标识"""
    banner = f"""
███████╗██╗  ██╗██╗   ██╗███████╗██╗   ██╗███████╗     █████╗ ██╗    @{service_name}
██╔════╝██║ ██╔╝╚██╗ ██╔╝██╔════╝╚██╗ ██╔╝██╔════╝    ██╔══██╗██║
███████╗█████╔╝  ╚████╔╝ █████╗   ╚████╔╝ █████╗      ███████║██║
╚════██║██╔═██╗   ╚██╔╝  ██╔══╝    ╚██╔╝  ██╔══╝      ██╔══██║██║
███████║██║  ██╗   ██║   ███████╗   ██║   ███████╗    ██║  ██║██║
╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝   ╚═╝   ╚══════╝    ╚═╝  ╚═╝╚═╝
    """

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
    from routers import task_router, health_router, stream_router, discovery_router
    from routers.task_video import router as video_router
    app.include_router(task_router)
    app.include_router(health_router)
    app.include_router(video_router)
    app.include_router(stream_router)
    app.include_router(discovery_router)

    # 添加静态文件支持
    from fastapi.staticfiles import StaticFiles
    static_dir = Path(ROOT_DIR) / "static"
    if not static_dir.exists():
        static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    return app

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    try:
        # 第一阶段：基础初始化
        show_service_banner("analysis_service")
        normal_logger.info("开始分析服务生命周期管理...")
        signal_handler.setup_signal_handlers()
        app.state.start_time = time.time()

        if settings.DEBUG_ENABLED:
            normal_logger.info("分析服务以调试模式启动。")
            normal_logger.info(f"环境: {settings.ENVIRONMENT}")
            normal_logger.info(f"版本: {settings.VERSION}")
            normal_logger.info(f"注册的路由: {[route.path for route in app.routes]}")

        # 第二阶段：核心服务初始化
        # 1. 全局应用状态管理器（最基础的服务）
        from shared.utils.app_state import app_state_manager
        app_state_manager.initialize()
        normal_logger.info("全局应用状态管理器初始化完成。")

        # 2. ZLMediaKit环境和服务（媒体处理的基础）
        from core.media_kit.zlm_manager import zlm_manager
        await zlm_manager.initialize()
        normal_logger.info("ZLMediaKit环境初始化完成。")

        # 3. 分析器工厂（用于创建分析器实例）
        from core.analyzer.analyzer_factory import analyzer_factory
        analyzer_factory.initialize()

        # 第三阶段：业务管理器初始化
        # 1. 流管理器（依赖ZLMediaKit）
        from core.task_management.stream import StreamManager
        stream_manager = StreamManager()
        await stream_manager.initialize()
        app_state_manager.register_service("stream_manager", stream_manager)
        normal_logger.info("流管理器初始化完成。")

        # 2. 任务管理器（依赖流管理器）
        from core.task_management.manager import TaskManager
        task_manager = TaskManager()
        await task_manager.initialize()
        app_state_manager.register_service("task_manager", task_manager)
        normal_logger.info("任务管理器初始化完成。")

        # 3. 流任务桥接器（依赖任务管理器和流管理器）
        from core.task_management.stream import StreamTaskBridge
        stream_task_bridge = StreamTaskBridge()
        await stream_task_bridge.initialize()
        app_state_manager.register_service("stream_task_bridge", stream_task_bridge)
        normal_logger.info("流任务桥接器初始化完成。")

        # 第四阶段：服务层初始化
        # 1. 回调服务
        from core.task_management.callback_service import callback_service
        callback_service.initialize()
        app_state_manager.register_service("callback_service", callback_service)
        normal_logger.info("回调服务初始化完成。")

        # 等待所有服务就绪
        normal_logger.info("等待所有服务就绪...")
        await asyncio.sleep(2)  # 给予其他服务充分启动时间

        # 第五阶段：测试ZLMediaKit HTTP API连接
        normal_logger.info("正在测试ZLMediaKit HTTP API连接...")
        api_ready = await zlm_manager.test_api_connection()
        if not api_ready:
            normal_logger.warning("ZLMediaKit HTTP API连接失败，部分功能可能不可用")
        else:
            normal_logger.info("ZLMediaKit HTTP API连接成功")

        # 第六阶段：Socket回调管理器启动
        normal_logger.info("正在启动Socket回调管理器...")
        await callback_service.start()
        normal_logger.info("Socket回调管理器启动完成。")

        normal_logger.info("所有服务初始化完成，分析服务已就绪。")
        yield

    except Exception as e:
        exception_logger.exception(f"服务启动失败: {str(e)}")
        raise
    finally:
        # 关闭顺序与启动顺序相反
        normal_logger.info("开始关闭服务...")
        
        # 1. 关闭Socket回调管理器
        try:
            await callback_service.stop()
            normal_logger.info("Socket回调管理器已关闭。")
        except Exception as e:
            exception_logger.error(f"关闭Socket回调管理器失败: {str(e)}")

        # 2. 关闭流任务桥接器
        try:
            await stream_task_bridge.shutdown()
            normal_logger.info("流任务桥接器已关闭。")
        except Exception as e:
            exception_logger.error(f"关闭流任务桥接器失败: {str(e)}")

        # 3. 关闭任务管理器
        try:
            await task_manager.shutdown()
            normal_logger.info("任务管理器已关闭。")
        except Exception as e:
            exception_logger.error(f"关闭任务管理器失败: {str(e)}")

        # 4. 关闭流管理器
        try:
            await stream_manager.shutdown()
            normal_logger.info("流管理器已关闭。")
        except Exception as e:
            exception_logger.error(f"关闭流管理器失败: {str(e)}")

        # 5. 关闭ZLMediaKit环境
        try:
            await zlm_manager.shutdown()
            normal_logger.info("ZLMediaKit环境已关闭。")
        except Exception as e:
            exception_logger.error(f"关闭ZLMediaKit环境失败: {str(e)}")

        normal_logger.info("所有服务已关闭。")

def start_app():
    """
    启动FastAPI应用
    """
    host = settings.SERVICES_HOST
    port = settings.SERVICES_PORT
    
    # 打印启动信息
    normal_logger.info(f"启动分析服务 - 版本: {settings.VERSION}")
    normal_logger.info(f"主机: {host}, 端口: {port}")
    normal_logger.info(f"调试模式: {settings.DEBUG_ENABLED}")
    
    # 创建应用实例
    app = create_app()
    
    # 使用uvicorn启动
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        workers=1,
        log_level="warning"  # 使用warning级别以关闭INFO日志
    )
    server = uvicorn.Server(config)
    server.run()

def main():
    """
    主函数，作为app.py的入口点
    """
    start_app()