"""
统一异常处理模块
定义分析服务相关的异常类和错误处理器
"""
from typing import Any, Optional, Dict
from enum import Enum
import traceback
from datetime import datetime

class ErrorCode(Enum):
    """错误代码枚举"""
    # 通用错误
    UNKNOWN_ERROR = (1000, "未知错误")
    INTERNAL_SERVER_ERROR = (1001, "内部服务器错误")
    
    # 输入验证错误 4xx
    INVALID_INPUT = (4000, "无效输入")
    MISSING_PARAMETER = (4001, "缺少必需参数")
    INVALID_PARAMETER_TYPE = (4002, "参数类型无效")
    INVALID_PARAMETER_VALUE = (4003, "参数值无效")
    
    # 资源错误 4xx
    RESOURCE_NOT_FOUND = (4040, "资源未找到")
    FILE_NOT_FOUND = (4041, "文件未找到")
    MODEL_NOT_FOUND = (4042, "模型未找到")
    STREAM_NOT_FOUND = (4043, "视频流未找到")
    TASK_NOT_FOUND = (4044, "任务未找到")
    
    # 业务逻辑错误 4xx
    UNSUPPORTED_OPERATION = (4220, "不支持的操作")
    FEATURE_NOT_IMPLEMENTED = (4221, "功能未实现")
    DEPENDENCY_MISSING = (4222, "依赖缺失")
    
    # 服务错误 5xx
    SERVICE_UNAVAILABLE = (5030, "服务不可用")
    DATABASE_ERROR = (5031, "数据库错误")
    STORAGE_ERROR = (5032, "存储错误")
    NETWORK_ERROR = (5033, "网络错误")
    
    # 模型相关错误 5xx
    MODEL_LOAD_ERROR = (5040, "模型加载失败")
    MODEL_INFERENCE_ERROR = (5041, "模型推理失败")
    MODEL_INITIALIZATION_ERROR = (5042, "模型初始化失败")
    
    # 任务处理错误 5xx
    TASK_CREATION_ERROR = (5050, "任务创建失败")
    TASK_PROCESSING_ERROR = (5051, "任务处理失败")
    TASK_TIMEOUT_ERROR = (5052, "任务超时")
    TASK_CANCELLED_ERROR = (5053, "任务被取消")
    
    # 流处理错误 5xx
    STREAM_CONNECTION_ERROR = (5060, "流连接失败")
    STREAM_PROCESSING_ERROR = (5061, "流处理失败")
    STREAM_CODEC_ERROR = (5062, "流编解码错误")
    
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message

class AnalysisException(Exception):
    """分析服务基础异常类"""
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Any] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details
        self.cause = cause
        self.timestamp = datetime.now()
        super().__init__(self.message)
    
    @property
    def http_status_code(self) -> int:
        """根据错误代码获取HTTP状态码"""
        code = self.error_code.code
        if 4000 <= code < 4040:
            return 400  # Bad Request
        elif 4040 <= code < 4050:
            return 404  # Not Found
        elif 4220 <= code < 4230:
            return 422  # Unprocessable Entity
        elif 5030 <= code < 5040:
            return 503  # Service Unavailable
        else:
            return 500  # Internal Server Error
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于序列化"""
        result = {
            "error_code": self.error_code.code,
            "error_message": self.error_code.message,
            "detail": self.message,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.details:
            result["details"] = self.details
        
        if self.cause:
            result["cause"] = str(self.cause)
        
        return result

# 具体异常类
class InvalidInputException(AnalysisException):
    """无效输入异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.INVALID_INPUT, details, cause)

class ResourceNotFoundException(AnalysisException):
    """资源未找到异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.RESOURCE_NOT_FOUND, details, cause)

class FileNotFoundException(AnalysisException):
    """文件未找到异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.FILE_NOT_FOUND, details, cause)

class ModelNotFoundException(AnalysisException):
    """模型未找到异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.MODEL_NOT_FOUND, details, cause)

class ModelLoadException(AnalysisException):
    """模型加载异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.MODEL_LOAD_ERROR, details, cause)

class ProcessingException(AnalysisException):
    """处理异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.TASK_PROCESSING_ERROR, details, cause)

class DatabaseException(AnalysisException):
    """数据库异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.DATABASE_ERROR, details, cause)

class ValidationException(AnalysisException):
    """验证异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.INVALID_PARAMETER_VALUE, details, cause)

class StorageException(AnalysisException):
    """存储异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.STORAGE_ERROR, details, cause)

class UnsupportedOperationException(AnalysisException):
    """不支持的操作异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.UNSUPPORTED_OPERATION, details, cause)

class FeatureNotImplementedException(AnalysisException):
    """功能未实现异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.FEATURE_NOT_IMPLEMENTED, details, cause)

class ServiceUnavailableException(AnalysisException):
    """服务不可用异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.SERVICE_UNAVAILABLE, details, cause)

class StreamConnectionException(AnalysisException):
    """流连接异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.STREAM_CONNECTION_ERROR, details, cause)

class TaskTimeoutException(AnalysisException):
    """任务超时异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.TASK_TIMEOUT_ERROR, details, cause)

class DependencyMissingException(AnalysisException):
    """依赖缺失异常"""
    def __init__(self, message: str, details: Optional[Any] = None, cause: Optional[Exception] = None):
        super().__init__(message, ErrorCode.DEPENDENCY_MISSING, details, cause)


# 异常转换器和处理器
class ExceptionHandler:
    """统一异常处理器"""
    
    @staticmethod
    def convert_standard_exception(exc: Exception, context: Optional[str] = None) -> AnalysisException:
        """
        将标准异常转换为分析服务异常
        
        Args:
            exc: 标准异常
            context: 上下文信息
        
        Returns:
            AnalysisException: 转换后的分析服务异常
        """
        message = f"{context}: {str(exc)}" if context else str(exc)
        
        # 根据异常类型转换
        if isinstance(exc, ValueError):
            return InvalidInputException(message, cause=exc)
        elif isinstance(exc, FileNotFoundError):
            return FileNotFoundException(message, cause=exc)
        elif isinstance(exc, ImportError):
            return DependencyMissingException(message, cause=exc)
        elif isinstance(exc, NotImplementedError):
            return FeatureNotImplementedException(message, cause=exc)
        elif isinstance(exc, TimeoutError):
            return TaskTimeoutException(message, cause=exc)
        elif isinstance(exc, ConnectionError):
            return StreamConnectionException(message, cause=exc)
        elif isinstance(exc, PermissionError):
            return ValidationException(message, cause=exc)
        elif isinstance(exc, RuntimeError):
            return ProcessingException(message, cause=exc)
        else:
            # 未知异常类型
            return AnalysisException(
                message, 
                ErrorCode.UNKNOWN_ERROR, 
                details={"original_type": type(exc).__name__}, 
                cause=exc
            )
    
    @staticmethod
    def handle_exception(exc: Exception, context: Optional[str] = None, logger=None) -> AnalysisException:
        """
        处理异常，记录日志并转换
        
        Args:
            exc: 异常对象
            context: 上下文信息
            logger: 日志记录器
        
        Returns:
            AnalysisException: 处理后的分析服务异常
        """
        # 如果已经是分析服务异常，直接返回
        if isinstance(exc, AnalysisException):
            analysis_exc = exc
        else:
            # 转换标准异常
            analysis_exc = ExceptionHandler.convert_standard_exception(exc, context)
        
        # 记录日志
        if logger:
            error_dict = analysis_exc.to_dict()
            logger.error(f"异常处理: {error_dict}")
            if analysis_exc.cause:
                logger.exception(f"原始异常: {analysis_exc.cause}")
        
        return analysis_exc
    
    @staticmethod
    def create_error_response(exc: AnalysisException) -> Dict[str, Any]:
        """
        创建错误响应
        
        Args:
            exc: 分析服务异常
        
        Returns:
            Dict: 错误响应字典
        """
        return {
            "success": False,
            "error": exc.to_dict()
        }


# 装饰器：自动异常处理
def handle_exceptions(context: Optional[str] = None, logger=None):
    """
    异常处理装饰器
    
    Args:
        context: 上下文信息
        logger: 日志记录器
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                analysis_exc = ExceptionHandler.handle_exception(e, context, logger)
                raise analysis_exc
        
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                analysis_exc = ExceptionHandler.handle_exception(e, context, logger)
                raise analysis_exc
        
        # 根据函数类型返回相应的wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


# 工具函数
def safe_execute(func, *args, context: Optional[str] = None, logger=None, default=None, **kwargs):
    """
    安全执行函数，捕获异常并返回默认值
    
    Args:
        func: 要执行的函数
        *args: 位置参数
        context: 上下文信息
        logger: 日志记录器
        default: 异常时的默认返回值
        **kwargs: 关键字参数
    
    Returns:
        函数执行结果或默认值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        analysis_exc = ExceptionHandler.handle_exception(e, context, logger)
        if logger:
            logger.warning(f"安全执行失败，返回默认值: {default}")
        return default


async def safe_execute_async(func, *args, context: Optional[str] = None, logger=None, default=None, **kwargs):
    """
    安全执行异步函数，捕获异常并返回默认值
    
    Args:
        func: 要执行的异步函数
        *args: 位置参数
        context: 上下文信息
        logger: 日志记录器
        default: 异常时的默认返回值
        **kwargs: 关键字参数
    
    Returns:
        函数执行结果或默认值
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        analysis_exc = ExceptionHandler.handle_exception(e, context, logger)
        if logger:
            logger.warning(f"安全执行异步函数失败，返回默认值: {default}")
        return default 