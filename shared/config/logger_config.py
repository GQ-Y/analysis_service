"""
日志配置模块
定义各种类型日志的配置参数
"""
import os
from typing import Dict, Any

# 确保日志目录存在
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# 日志文件路径
NORMAL_LOG_FILE = os.path.join(LOGS_DIR, "normal.log")
EXCEPTION_LOG_FILE = os.path.join(LOGS_DIR, "exception.log")
TEST_LOG_FILE = os.path.join(LOGS_DIR, "test.log")
ANALYSIS_LOG_FILE = os.path.join(LOGS_DIR, "analysis.log")

# 日志配置
LOGGER_CONFIG = {
    # 常规日志配置
    "normal": {
        "file": {
            "path": NORMAL_LOG_FILE,
            "level": "INFO",
            "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[name]}:{function}:{line} - {message}",
            "rotation": "1 day",
            "retention": "7 days",
            "enqueue": True,
        },
        "console": {
            "level": "INFO",
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <cyan>常规日志</cyan> | <cyan>{extra[name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        }
    },
    
    # 异常日志配置
    "exception": {
        "file": {
            "path": EXCEPTION_LOG_FILE,
            "level": "ERROR",
            "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[name]}:{function}:{line} - {message}\\n{exception}",
            "rotation": "1 day",
            "retention": "7 days",
            "enqueue": True,
            "backtrace": True,
            "diagnose": True,
        },
        "console": {
            "level": "ERROR",
            "format": "<red>{time:YYYY-MM-DD HH:mm:ss.SSS}</red> | <red>异常日志</red> | <cyan>{extra[name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\\n{exception}",
            "backtrace": True,
            "diagnose": True,
        }
    },
    
    # 测试日志配置
    "test": {
        "file": {
            "path": TEST_LOG_FILE,
            "level": "INFO",
            "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
            "rotation": "1 day",
            "retention": "7 days",
            "enqueue": True,
            "mode": "w"  # 每次启动时覆盖
        },
        "console": {
            "level": "INFO",
            "format": "测试日志 | {message}",
        }
    },
    
    # 分析日志配置
    "analysis": {
        "file": {
            "path": ANALYSIS_LOG_FILE,
            "level": "INFO",
            "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
            "rotation": "1 day",
            "retention": "7 days",
            "enqueue": True,
            "mode": "w"  # 每次启动时覆盖
        },
        "console": {
            "level": "INFO",
            "format": "分析日志 | {message}",
        }
    }
}

# Uvicorn日志配置
UVICORN_LOG_CONFIG = {
    "log_level": "warning",  # 使用warning级别以关闭INFO日志
} 