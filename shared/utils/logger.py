"""
日志工具模块
"""
from loguru import logger
import sys
import os

def setup_logger(name: str):
    """设置日志配置
    
    Args:
        name: 日志记录器名称
    """
    # 移除所有默认处理器
    logger.remove()
    
    # 从环境变量获取调试设置
    debug_enabled = os.getenv("DEBUG_ENABLED", "False").lower() in ["true", "1", "yes"]
    
    # 只在调试模式启用时添加日志处理器
    if debug_enabled:
        # 添加控制台处理器
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=os.getenv("DEBUG_LOG_LEVEL", "DEBUG"),
            backtrace=True,
            diagnose=True
        )
        
        # 添加文件处理器
        logger.add(
            os.getenv("DEBUG_LOG_FILE", "logs/debug.log"),
            rotation=os.getenv("DEBUG_LOG_ROTATION", "1 day"),
            retention=os.getenv("DEBUG_LOG_RETENTION", "7 days"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=os.getenv("DEBUG_LOG_LEVEL", "DEBUG"),
            backtrace=True,
            diagnose=True
        )
    
    return logger.bind(name=name) 