"""
日志工具模块
"""
from loguru import logger
import sys
import os

# 确保日志目录存在
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# 测试日志标记常量
TEST_LOG_MARKER = "TEST_LOG_MARKER"

# 移除所有默认处理器，以便精细控制
logger.remove()

# --- 常规日志 (Normal Logger) ---
NORMAL_LOG_FILE = os.path.join(LOGS_DIR, "normal.log")
logger.add(
    NORMAL_LOG_FILE,
    level="INFO", 
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[name]}:{function}:{line} - {message}",
    rotation="1 day",
    retention="7 days",
    enqueue=True,
    filter=lambda record: record["extra"].get("log_type") == "normal"
)
# 常规日志也输出到控制台，便于实时查看
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <cyan>常规日志</cyan> | <cyan>{extra[name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    filter=lambda record: record["extra"].get("log_type") == "normal"
)
normal_logger = logger.bind(log_type="normal")

# --- 异常日志 (Exception Logger) ---
EXCEPTION_LOG_FILE = os.path.join(LOGS_DIR, "exception.log")
logger.add(
    EXCEPTION_LOG_FILE,
    level="ERROR", 
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[name]}:{function}:{line} - {message}\\n{exception}",
    rotation="1 day",
    retention="7 days",
    enqueue=True,
    backtrace=True, 
    diagnose=True,
    filter=lambda record: record["extra"].get("log_type") == "exception"
)
# 异常日志也输出到控制台
logger.add(
    sys.stderr,
    level="ERROR",
    format="<red>{time:YYYY-MM-DD HH:mm:ss.SSS}</red> | <red>异常日志</red> | <cyan>{extra[name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\\n{exception}",
    backtrace=True,
    diagnose=True,
    filter=lambda record: record["extra"].get("log_type") == "exception"
)
exception_logger = logger.bind(log_type="exception")

# --- 测试日志 (Test Logger) ---
TEST_LOG_FILE = os.path.join(LOGS_DIR, "test.log")
logger.add(
    TEST_LOG_FILE,
    level="INFO", 
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}", 
    rotation="1 day",
    retention="7 days",
    enqueue=True,
    filter=lambda record: record["extra"].get("log_type") == "test",
    mode="w"  
)
# 测试日志也输出到控制台，以便自动化脚本捕获
logger.add(
    sys.stdout, 
    level="INFO",
    format="测试日志 | {message}", 
    filter=lambda record: record["extra"].get("log_type") == "test"
)
test_logger = logger.bind(log_type="test")

# --- 分析日志 (Analysis Logger) ---
ANALYSIS_LOG_FILE = os.path.join(LOGS_DIR, "analysis.log")
logger.add(
    ANALYSIS_LOG_FILE,
    level="INFO", 
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}", 
    rotation="1 day",
    retention="7 days",
    enqueue=True,
    filter=lambda record: record["extra"].get("log_type") == "analysis",
    mode="w"  
)
# 分析日志也输出到控制台，以便自动化脚本捕获
logger.add(
    sys.stdout, 
    level="INFO",
    format="分析日志 | {message}", 
    filter=lambda record: record["extra"].get("log_type") == "analysis"
)
analysis_logger = logger.bind(log_type="analysis")


# 提供获取logger的函数
def get_normal_logger(name: str):
    # 使用 .patch().bind() 来确保 extra 字段被正确设置，并且 name 能在格式化字符串中通过 record[\"extra\"][\"name\"] 访问
    return normal_logger.patch(lambda record: record["extra"].update(name=name))

def get_exception_logger(name: str):
    return exception_logger.patch(lambda record: record["extra"].update(name=name))

def get_test_logger(): 
    return test_logger

def get_analysis_logger(): 
    return analysis_logger