"""
日志工具模块
"""
from loguru import logger
import sys
from core.config import settings

def setup_logger(name: str):
    """设置日志配置
    
    Args:
        name: 日志记录器名称
    """
    # 移除所有默认处理器
    logger.remove()
    
    # 只在调试模式启用时添加日志处理器
    if settings.DEBUG.enabled:
        # 添加控制台处理器
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=settings.DEBUG.log_level,
            backtrace=True,
            diagnose=True
        )
        
        # 添加文件处理器
        logger.add(
            settings.DEBUG.log_file,
            rotation=settings.DEBUG.log_rotation,
            retention=settings.DEBUG.log_retention,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=settings.DEBUG.log_level,
            backtrace=True,
            diagnose=True
        )
    
    return logger.bind(name=name) 