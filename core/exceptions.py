"""
自定义异常模块
定义分析服务相关的异常类
"""
from typing import Any, Optional

class AnalysisException(Exception):
    """分析服务基础异常类"""
    def __init__(
        self,
        message: str,
        code: int = 500,
        data: Optional[Any] = None
    ):
        self.message = message
        self.code = code
        self.data = data
        super().__init__(self.message)

class InvalidInputException(AnalysisException):
    """无效输入异常"""
    def __init__(self, message: str, data: Optional[Any] = None):
        super().__init__(message, 400, data)

class ModelLoadException(AnalysisException):
    """模型加载异常"""
    def __init__(self, message: str, data: Optional[Any] = None):
        super().__init__(message, 500, data)

class ProcessingException(AnalysisException):
    """处理异常"""
    def __init__(self, message: str, data: Optional[Any] = None):
        super().__init__(message, 500, data)

class DatabaseException(AnalysisException):
    """数据库异常"""
    def __init__(self, message: str, data: Optional[Any] = None):
        super().__init__(message, 500, data)

class ResourceNotFoundException(AnalysisException):
    """资源未找到异常"""
    def __init__(self, message: str, data: Optional[Any] = None):
        super().__init__(message, 404, data)

class ValidationException(AnalysisException):
    """验证异常"""
    def __init__(self, message: str, data: Optional[Any] = None):
        super().__init__(message, 400, data)

class StorageException(AnalysisException):
    """存储异常"""
    def __init__(self, message: str, data: Optional[Any] = None):
        super().__init__(message, 500, data) 