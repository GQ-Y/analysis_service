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
███╗   ███╗███████╗███████╗██╗  ██╗██╗   ██╗ ██████╗ ██╗      ██████╗     @{service_name}
████╗ ████║██╔════╝██╔════╝██║ ██╔╝╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗
██╔████╔██║█████╗  █████╗  █████╔╝  ╚████╔╝ ██║   ██║██║     ██║   ██║
██║╚██╔╝██║██╔══╝  ██╔══╝  ██╔═██╗   ╚██╔╝  ██║   ██║██║     ██║   ██║
██║ ╚═╝ ██║███████╗███████╗██║  ██╗   ██║   ╚██████╔╝███████╗╚██████╔╝
╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝
    """
    # 使用普通日志记录横幅信息
    normal_logger.info(f"服务启动横幅:\n{banner}")

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
        # 启动时执行
        show_service_banner("analysis_service") # Banner会通过normal_logger打印
        normal_logger.info("开始分析服务生命周期管理...")

        # 设置信号处理器
        signal_handler.setup_signal_handlers()

        # 记录启动时间
        app.state.start_time = time.time()

        if settings.DEBUG_ENABLED:
            normal_logger.info("分析服务以调试模式启动。")
            normal_logger.info(f"环境: {settings.ENVIRONMENT}")
            normal_logger.info(f"版本: {settings.VERSION}")
            normal_logger.info(f"注册的路由: {[route.path for route in app.routes]}")

        # 检查并自动安装ZLMediaKit
        from run.middlewares import init_zlm_environment, start_zlm_service
        normal_logger.info("正在检查ZLMediaKit环境...")
        if init_zlm_environment():
            normal_logger.info("ZLMediaKit环境检查完成，库文件已就绪。")

            # 直接启动ZLMediaKit服务，不需要检查
            normal_logger.info("正在启动ZLMediaKit服务进程...")
            try:
                if start_zlm_service():
                    normal_logger.info("ZLMediaKit服务进程已成功启动")
                else:
                    normal_logger.warning("ZLMediaKit服务进程启动失败，将尝试使用现有配置继续")
            except Exception as e:
                exception_logger.exception("启动ZLMediaKit服务进程时出错")
        else:
            normal_logger.warning("ZLMediaKit环境检查失败，将尝试使用现有配置继续。")

        # 初始化ZLMediaKit
        normal_logger.info("正在初始化ZLMediaKit...")
        try:
            from core.media_kit.zlm_manager import zlm_manager
            await zlm_manager.initialize()
            app.state.zlm_manager = zlm_manager

            # 将 ZLMediaKit 管理器注册到信号处理器
            signal_handler.set_zlm_manager(zlm_manager)

            # 注册到退出处理器
            zlm_exit_handler.register_zlm_manager(zlm_manager)

            normal_logger.info("ZLMediaKit初始化成功。")
        except Exception as e:
            exception_logger.exception("ZLMediaKit初始化失败。将继续使用OpenCV进行视频处理。")

        # 初始化任务管理器
        normal_logger.info("正在初始化任务管理器...")
        try:
            from core.task_management.manager import TaskManager
            from core.task_management.stream import StreamManager

            normal_logger.info("正在初始化流管理器...")
            stream_manager = StreamManager()
            await stream_manager.initialize()
            normal_logger.info("流管理器初始化完成。")

            task_manager = TaskManager()
            await task_manager.initialize()

            app.state.task_manager = task_manager
            app.state.stream_manager = stream_manager

            from core.task_management.stream import StreamTaskBridge
            normal_logger.info("正在初始化流任务桥接器...")
            stream_task_bridge = StreamTaskBridge()
            await stream_task_bridge.initialize(task_manager)
            app.state.stream_task_bridge = stream_task_bridge
            normal_logger.info("流任务桥接器初始化完成。")

            # 初始化Socket回调管理器
            normal_logger.info("正在初始化Socket回调管理器...")
            await startup_socket_manager()
            normal_logger.info("Socket回调管理器初始化完成。")

            from services.http.task_service import TaskService
            from services.analysis_service import AnalysisService
            from services.video.video_service import VideoService
            from shared.utils.app_state import app_state_manager

            # 初始化全局应用状态管理器
            app_state_manager.initialize()
            
            # 创建服务实例
            task_service = TaskService(task_manager)
            analysis_service = AnalysisService()
            video_service = VideoService()
            
            # 注册到FastAPI应用状态（向后兼容）
            app.state.task_service = task_service
            app.state.analysis_service = analysis_service
            app.state.video_service = video_service
            
            # 注册到全局应用状态管理器
            app_state_manager.register_task_manager(task_manager)
            app_state_manager.register_video_service(video_service)
            app_state_manager.register_service('analysis_service', analysis_service)
            app_state_manager.register_service('task_service', task_service)

            normal_logger.info("任务管理器和服务初始化成功。")

        except Exception as e:
            exception_logger.exception("服务初始化失败。")

        yield  # 这里是应用运行期间

    except asyncio.CancelledError:
        # 生命周期被取消，直接退出
        normal_logger.info("应用生命周期被取消，正在退出...")
        raise
    except Exception as e:
        exception_logger.exception(f"应用生命周期管理出错: {e}")
        raise
    finally:
        # 关闭时执行
        normal_logger.info("正在关闭分析服务...")
        sys.stdout.flush(); sys.stderr.flush()

        # 优先关闭回调服务
        # normal_logger.info("正在关闭回调服务 (lifespan finally)...") # Old log
        # await shutdown_callback_service() # Old call
        # normal_logger.info("回调服务已关闭 (lifespan finally)。") # Old log

        # 关闭Socket回调管理器
        normal_logger.info("正在关闭Socket回调管理器 (lifespan finally)...")
        await shutdown_socket_manager()
        normal_logger.info("Socket回调管理器已关闭 (lifespan finally)。")

        # 1. TaskManager
        if hasattr(app.state, "task_manager") and app.state.task_manager:
            try:
                normal_logger.info("正在停止任务管理器...")
                try:
                    await asyncio.wait_for(app.state.task_manager.shutdown(), timeout=10.0)
                    normal_logger.info("任务管理器已正常停止。")
                except asyncio.TimeoutError:
                    exception_logger.warning("等待任务管理器关闭超时。")
                except Exception as inner_e:
                    exception_logger.exception("任务管理器关闭过程中出错。")
            except Exception as e:
                 exception_logger.exception("停止任务管理器时出错。")
            finally:
                normal_logger.info("任务管理器关闭流程已完成。")
        else:
            normal_logger.info("没有活动的任务管理器需要关闭。")

        if hasattr(app.state, "stream_manager") and app.state.stream_manager:
            try:
                normal_logger.info("正在关闭流管理器...")
                try:
                    await asyncio.wait_for(app.state.stream_manager.shutdown(), timeout=10.0)
                    normal_logger.info("流管理器已正常关闭。")
                except asyncio.TimeoutError:
                    exception_logger.warning("等待流管理器关闭超时。")
                except Exception as inner_e:
                    exception_logger.exception("流管理器关闭过程中出错。")
            except Exception as e:
                exception_logger.exception("关闭流管理器时出错。")
            finally:
                normal_logger.info("流管理器关闭流程已完成。")
        else:
            normal_logger.info("没有活动的流管理器需要关闭。")

        if hasattr(app.state, "stream_task_bridge") and app.state.stream_task_bridge:
            try:
                normal_logger.info("正在关闭流任务桥接器...")
                try:
                    await asyncio.wait_for(app.state.stream_task_bridge.shutdown(), timeout=5.0)
                    normal_logger.info("流任务桥接器已正常关闭。")
                except asyncio.TimeoutError:
                    exception_logger.warning("等待流任务桥接器关闭超时。")
                except Exception as inner_e:
                    exception_logger.exception("流任务桥接器关闭过程中出错。")
            except Exception as e:
                exception_logger.exception("关闭流任务桥接器时出错。")
            finally:
                normal_logger.info("流任务桥接器关闭流程已完成。")

        if hasattr(app.state, "zlm_manager") and app.state.zlm_manager:
            try:
                normal_logger.info("正在关闭ZLMediaKit...")
                try:
                    await asyncio.wait_for(app.state.zlm_manager.shutdown(), timeout=10.0)
                    normal_logger.info("ZLMediaKit已正常关闭。")
                except asyncio.TimeoutError:
                    exception_logger.warning("等待ZLMediaKit关闭超时。")
                    # 强制标记为关闭状态
                    if hasattr(app.state.zlm_manager, '_is_shutting_down'):
                        app.state.zlm_manager._is_shutting_down = True
                except Exception as inner_e:
                    exception_logger.exception("ZLMediaKit关闭过程中出错。")
            except Exception as e:
                exception_logger.exception("关闭ZLMediaKit时出错。")
            finally:
                # 确保清理库引用
                try:
                    if hasattr(app.state.zlm_manager, '_lib'):
                        app.state.zlm_manager._lib = None
                    # 强制垃圾回收
                    import gc
                    gc.collect()
                except:
                    pass
                normal_logger.info("ZLMediaKit关闭流程已完成。")
        else:
            normal_logger.info("没有ZLMediaKit管理器需要关闭。")

        # 确保ZLMediaKit服务已停止
        try:
            from run.middlewares import stop_zlm_service
            normal_logger.info("正在停止ZLMediaKit服务进程...")
            if stop_zlm_service():
                normal_logger.info("ZLMediaKit服务进程已成功停止")
            else:
                normal_logger.warning("ZLMediaKit服务进程停止失败，请检查进程状态")
        except Exception as e:
            exception_logger.exception("停止ZLMediaKit服务进程时出错")

        # 清理全局应用状态管理器
        try:
            from shared.utils.app_state import app_state_manager
            app_state_manager.shutdown()
            normal_logger.info("全局应用状态管理器已清理。")
        except Exception as e:
            exception_logger.exception("清理全局应用状态管理器时出错")

        # 清理信号处理器
        signal_handler.cleanup()

        normal_logger.info("分析服务已停止。")

        # 强制退出进程以避免 ZLMediaKit 析构错误
        # 这是解决 recursive_mutex lock failed 错误的最有效方法
        normal_logger.info("强制退出进程以避免 ZLMediaKit C++ 析构错误")

        # 给多进程资源一些时间进行正常清理
        time.sleep(0.1)

        # 尝试手动触发多进程资源清理
        try:
            import multiprocessing.util
            # 先让现有的清理器运行
            for finalizer in list(multiprocessing.util._finalizer_registry.values()):
                try:
                    if finalizer.still_active():
                        finalizer()
                except:
                    pass
            # 然后清空注册表
            multiprocessing.util._finalizer_registry.clear()
        except:
            pass

        # 禁用 atexit 处理器以避免额外的清理
        try:
            import atexit
            atexit._clear()
        except:
            pass

        # 给一点时间让日志输出
        time.sleep(0.02)
        sys.stdout.flush(); sys.stderr.flush()

        # 强制退出，跳过所有析构函数和清理过程
        os._exit(0) # <--- 恢复 os._exit(0)

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
    normal_logger.info(f"尝试启动分析服务 - 版本: {settings.VERSION}")
    normal_logger.info(f"主机: {host}, 端口: {port}")
    normal_logger.info(f"调试模式: {settings.DEBUG_ENABLED}, 热重载: {reload}")
    normal_logger.info(f"工作进程数: {workers}")

    # 创建应用实例
    app = create_app()

    # 如果是直接运行，使用uvicorn启动
    if reload:
        normal_logger.info("使用Uvicorn热重载模式启动。")
        uvicorn.run(
            "run.run:create_app",
            host=host,
            port=port,
            reload=reload,
            factory=True,
            workers=1,  # 热重载模式下只能使用1个工作进程
            log_level="warning"  # 使用warning级别以关闭INFO日志
        )
    else:
        normal_logger.info("使用Uvicorn标准服务器模式启动。")
        app_instance = create_app() # 在此模式下，我们需要传递实例
        config = uvicorn.Config(
            app_instance,
            host=host,
            port=port,
            workers=workers,
            log_level="warning"  # 使用warning级别以关闭INFO日志
        )
        server = uvicorn.Server(config)
        server.run()

def main():
    """
    主函数，作为app.py的入口点
    """
    # 启动应用，并解析命令行参数
    start_app(parse_command_line=True)