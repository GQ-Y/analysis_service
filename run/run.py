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
        from services.task_service import TaskService
        from services.analysis_service import AnalysisService

        app.state.task_service = TaskService(task_manager)
        app.state.analysis_service = AnalysisService()

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

def start_app(host=None, port=None, reload=None, workers=None, debug=None):
    """启动FastAPI应用"""
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

    # 启动服务
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="debug" if settings.DEBUG_ENABLED else "info"
    )