"""
中间件初始化模块
"""
from .exception_handler import setup_exception_handlers
from .request_logging import RequestLoggingMiddleware
from .zlm_checker import ZLMChecker

# 导出中间件
__all__ = [
    "setup_exception_handlers",
    "RequestLoggingMiddleware",
    "ZLMChecker",
    "init_zlm_environment",
    "start_zlm_service",
    "stop_zlm_service"
]

# 初始化ZLMediaKit环境检查器
def init_zlm_environment():
    """初始化ZLMediaKit环境
    在应用启动前自动检查ZLMediaKit库并安装
    """
    return ZLMChecker.check_and_install()

# 启动ZLMediaKit服务
def start_zlm_service():
    """启动ZLMediaKit服务
    在应用启动时调用，确保ZLMediaKit服务已启动
    """
    return ZLMChecker.start_zlm_service()

# 停止ZLMediaKit服务
def stop_zlm_service():
    """停止ZLMediaKit服务
    在应用关闭时调用，确保ZLMediaKit服务已停止
    """
    return ZLMChecker.stop_zlm_service()
