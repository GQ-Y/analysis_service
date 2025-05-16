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
    "ZLMChecker"
]

# 初始化ZLMediaKit环境检查器
def init_zlm_environment():
    """初始化ZLMediaKit环境
    在应用启动前自动检查ZLMediaKit库并安装
    """
    return ZLMChecker.check_and_install()
