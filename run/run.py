"""
分析服务启动模块
负责初始化和启动FastAPI应用
"""
import os
import sys
import uvicorn
import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.config import settings
from shared.utils.logger import get_normal_logger, get_exception_logger
from run.middlewares.exception_handler import setup_exception_handlers, UnifiedExceptionMiddleware
from run.middlewares.request_logging import RequestLoggingMiddleware
from run.signal_handler import signal_handler

# 导入零拷贝架构组件
from core.memory.memory_pool import MemoryPool
from core.config.memory_config import MemoryConfig
from core.initialization.memory_initializer import initialize_memory_system
from core.task_management.zero_copy_processor import ZeroCopyTaskProcessor
from core.task_management.stream.zero_copy_manager import ZeroCopyStreamManager

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

def show_service_banner(service_name: str):
    """显示服务启动标识"""
    global banner
    banner = f"""
███████╗██╗  ██╗██╗   ██╗███████╗██╗   ██╗███████╗     █████╗ ██╗    @{service_name}
██╔════╝██║ ██╔╝╚██╗ ██╔╝██╔════╝╚██╗ ██╔╝██╔════╝    ██╔══██╗██║
███████╗█████╔╝  ╚████╔╝ █████╗   ╚████╔╝ █████╗      ███████║██║
╚════██║██╔═██╗   ╚██╔╝  ██╔══╝    ╚██╔╝  ██╔══╝      ██╔══██║██║
███████║██║  ██╗   ██║   ███████╗   ██║   ███████╗    ██║  ██║██║
╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝   ╚═╝   ╚══════╝    ╚═╝  ╚═╝╚═╝
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
        title=settings.service.project_name,
        description="""
        分析服务模块

        提供以下功能:
        - 视觉对象检测和分析
        - 实例分割
        - 目标跟踪
        - 跨摄像头目标跟踪
        - 分析结果存储和查询
        """,
        version=settings.service.version,
        # 在生产环境下禁用API文档
        docs_url="/api/v1/docs" if settings.service.debug_enabled else None,
        redoc_url="/api/v1/redoc" if settings.service.debug_enabled else None,
        openapi_url="/api/v1/openapi.json" if settings.service.debug_enabled else None,
        debug=settings.service.debug_enabled,
        lifespan=lifespan
    )

    # 添加统一异常处理中间件
    app.add_middleware(UnifiedExceptionMiddleware)
    
    # 添加Gzip压缩
    from fastapi.middleware.gzip import GZipMiddleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # 只在调试模式下添加请求日志中间件
    if settings.service.debug_enabled:
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

    # 添加404错误处理
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return FileResponse(str(static_dir / "404.html"), status_code=404)

    # 添加根路径重定向
    @app.get("/")
    async def root():
        if settings.service.debug_enabled:
            return {"message": "欢迎使用Skyeye AI分析服务", "docs_url": "/api/v1/docs"}
        else:
            return FileResponse(str(static_dir / "404.html"), status_code=404)

    return app

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    try:
        # 显示启动标识
        show_service_banner("analysis_service")
        normal_logger.info("正在启动分析服务...")

        # 设置信号处理器
        signal_handler.setup_signal_handlers()
        app.state.start_time = time.time()

        if settings.service.debug_enabled:
            normal_logger.info(f"调试模式: {settings.service.debug_enabled}")

        # 初始化核心服务
        from shared.utils.app_state import app_state_manager
        from core.media_kit.zlm_manager import zlm_manager
        from core.analyzer.analyzer_factory import analyzer_factory
        from core.task_management.callback_service import callback_service
        from services.http.zero_copy_task_service import ZeroCopyTaskService

        # 1. 初始化基础服务
        app_state_manager.initialize()
        await zlm_manager.initialize()
        analyzer_factory.initialize()

        # 2. 初始化零拷贝内存系统
        normal_logger.info("正在初始化零拷贝内存系统...")
        memory_pool = await initialize_memory_system()
        if not memory_pool:
            raise RuntimeError("零拷贝内存系统初始化失败")
        app_state_manager.register_service("memory_pool", memory_pool)
        normal_logger.info("零拷贝内存系统初始化完成")

        # 3. 初始化零拷贝业务服务
        zero_copy_stream_manager = ZeroCopyStreamManager()
        zero_copy_stream_manager.set_memory_pool(memory_pool)
        await zero_copy_stream_manager.initialize()
        app_state_manager.register_service("stream_manager", zero_copy_stream_manager)

        zero_copy_task_processor = ZeroCopyTaskProcessor(None, memory_pool)
        # 创建零拷贝任务管理器（使用零拷贝处理器）
        from core.task_management.manager import TaskManager
        task_manager = TaskManager()
        task_manager.processor = zero_copy_task_processor  # 修正属性名
        task_manager.memory_pool = memory_pool  # 设置内存池引用
        zero_copy_task_processor.task_manager = task_manager
        await task_manager.initialize()
        app_state_manager.register_service("task_manager", task_manager)

        # 4. 初始化零拷贝流任务桥接器
        from core.task_management.stream import StreamTaskBridge
        stream_task_bridge = StreamTaskBridge()
        # 使用零拷贝流管理器
        stream_task_bridge.stream_manager = zero_copy_stream_manager
        await stream_task_bridge.initialize()
        app_state_manager.register_service("stream_task_bridge", stream_task_bridge)

        callback_service.initialize()
        app_state_manager.register_service("callback_service", callback_service)

        # 5. 初始化零拷贝视频服务并注册到全局状态管理器
        from services.video.video_service import VideoService
        video_service = VideoService()
        # 将内存池注入视频服务以支持零拷贝
        if hasattr(video_service, 'set_memory_pool'):
            video_service.set_memory_pool(memory_pool)
        app_state_manager.register_video_service(video_service)
        app.state.video_service = video_service
        normal_logger.info("零拷贝视频服务已初始化并注册到全局状态管理器")

        # 6. 初始化零拷贝任务服务并注册到app.state
        zero_copy_task_service = ZeroCopyTaskService(task_manager=task_manager, memory_pool=memory_pool)
        # 设置零拷贝组件
        zero_copy_task_service.set_zero_copy_components(zero_copy_task_processor, zero_copy_stream_manager)
        app.state.task_service = zero_copy_task_service
        normal_logger.info("零拷贝任务服务已初始化并注册")

        # 等待服务就绪
        await asyncio.sleep(2)

        # 测试ZLMediaKit连接
        api_ready = await zlm_manager.test_api_connection()
        if not api_ready:
            normal_logger.warning("ZLMediaKit连接失败，部分功能可能不可用")

        # 启动回调服务
        await callback_service.start()

        # 7. 输出零拷贝架构初始化摘要
        normal_logger.info("=== 零拷贝架构初始化完成 ===")
        normal_logger.info(f"内存池状态: {memory_pool.get_stats()}")
        normal_logger.info(f"零拷贝流管理器: {zero_copy_stream_manager.__class__.__name__}")
        normal_logger.info(f"零拷贝任务处理器: {zero_copy_task_processor.__class__.__name__}")
        normal_logger.info("=== 分析服务已就绪（零拷贝模式）===")
        yield

    except Exception as e:
        exception_logger.exception(f"服务启动失败: {str(e)}")
        raise
    finally:
        normal_logger.info("正在关闭服务...")
        
        try:
            normal_logger.info("正在关闭零拷贝架构服务...")
            await callback_service.stop()
            await stream_task_bridge.shutdown()
            await task_manager.shutdown()
            await zero_copy_stream_manager.shutdown()
            await zlm_manager.shutdown()

            # 关闭内存池
            if memory_pool:
                memory_pool.cleanup()
                normal_logger.info("内存池已关闭")
        except Exception as e:
            exception_logger.error(f"关闭服务时发生错误: {str(e)}")

        normal_logger.info("服务已关闭")

def start_app(app=None):
    """
    启动FastAPI应用
    """
    host = settings.service.host
    port = settings.service.port

    # 如果没有传入应用实例，则创建一个新的
    if app is None:
        app = create_app()

    # 使用uvicorn启动
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        workers=1,
        log_level="error"  # 只显示错误日志
    )
    server = uvicorn.Server(config)
    server.run()

def main():
    """
    主函数，作为app.py的入口点
    """
    start_app()