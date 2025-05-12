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

def setup_stream_error_logger():
    """设置视频流错误日志处理器

    Returns:
        配置好的日志记录器
    """
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)

    # 创建专门用于视频流错误的日志处理器
    stream_error_logger = logger.bind(name="stream_error")

    # 移除默认处理器，确保错误不会输出到控制台
    stream_error_logger.remove()

    # 添加文件处理器，只记录到文件，不输出到控制台
    stream_error_logger.add(
        "logs/stream_errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="ERROR",  # 只记录错误级别及以上的信息
        rotation="1 day",
        retention="7 days",
        backtrace=False,
        diagnose=False,
        enqueue=True  # 使用队列，避免多进程写入冲突
    )

    return stream_error_logger

def setup_analysis_logger():
    """设置分析过程日志处理器

    记录分析过程中的详细信息，包括抽帧次数、分析结果、ROI信息等

    Returns:
        配置好的日志记录器
    """
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)

    # 创建专门用于分析过程的日志处理器
    analysis_logger = logger.bind(name="analysis")

    # 移除默认处理器
    analysis_logger.remove()

    # 添加文件处理器，记录到专门的分析日志文件
    analysis_logger.add(
        "logs/analysis.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",  # 记录INFO级别及以上的信息
        rotation="1 day",
        retention="7 days",
        backtrace=False,
        diagnose=False,
        enqueue=True  # 使用队列，避免多进程写入冲突
    )

    # 添加控制台处理器，便于调试
    analysis_logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>分析</cyan> - <level>{message}</level>",
        level="INFO",
        filter=lambda record: record["name"] == "analysis"
    )

    return analysis_logger