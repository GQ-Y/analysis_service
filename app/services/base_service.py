#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: base_service.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 基础服务类

定义服务层的基础功能，包括日志记录、异常处理、依赖注入等。
所有业务服务都应该继承此基类。

本文件是分析服务项目的一部分。
"""

import logging
from typing import Any, Dict, Optional, List
from abc import ABC, abstractmethod

from app.exceptions.business_exception import BusinessException
from config.dependencies import get_container


class BaseService(ABC):
    """基础服务类"""
    
    def __init__(self):
        """初始化基础服务"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.container = get_container()
        self._initialized = False
    
    async def initialize(self):
        """初始化服务（异步）"""
        if not self._initialized:
            await self._initialize_service()
            self._initialized = True
    
    async def _initialize_service(self):
        """子类可重写的初始化方法"""
        pass
    
    def get_dependency(self, service_name: str) -> Any:
        """获取依赖服务
        
        Args:
            service_name: 服务名称
            
        Returns:
            Any: 服务实例
        """
        try:
            return self.container.get(service_name)
        except Exception as e:
            self.logger.error(f"获取依赖服务失败 {service_name}: {e}")
            raise BusinessException(f"服务依赖不可用: {service_name}")
    
    def log_info(self, message: str, **kwargs):
        """记录信息日志
        
        Args:
            message: 日志消息
            **kwargs: 额外参数
        """
        self.logger.info(message, extra=kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """记录警告日志
        
        Args:
            message: 日志消息
            **kwargs: 额外参数
        """
        self.logger.warning(message, extra=kwargs)
    
    def log_error(self, message: str, exception: Exception = None, **kwargs):
        """记录错误日志
        
        Args:
            message: 日志消息
            exception: 异常对象
            **kwargs: 额外参数
        """
        if exception:
            kwargs['exception_type'] = type(exception).__name__
            kwargs['exception_message'] = str(exception)
        
        self.logger.error(message, extra=kwargs, exc_info=exception is not None)
    
    def validate_required_params(self, params: Dict[str, Any], required_fields: List[str]):
        """验证必需参数
        
        Args:
            params: 参数字典
            required_fields: 必需字段列表
            
        Raises:
            BusinessException: 参数验证失败
        """
        missing_fields = []
        for field in required_fields:
            if field not in params or params[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            raise BusinessException(f"缺少必需参数: {', '.join(missing_fields)}")
    
    def validate_pagination(self, page: int, page_size: int, max_page_size: int = 100) -> tuple[int, int]:
        """验证分页参数
        
        Args:
            page: 页码
            page_size: 每页大小
            max_page_size: 最大每页大小
            
        Returns:
            tuple[int, int]: 验证后的页码和每页大小
            
        Raises:
            BusinessException: 参数无效
        """
        if page < 1:
            raise BusinessException("页码必须大于0")
        
        if page_size < 1:
            raise BusinessException("每页大小必须大于0")
        
        if page_size > max_page_size:
            raise BusinessException(f"每页大小不能超过{max_page_size}")
        
        return page, page_size
    
    def calculate_pagination(self, total: int, page: int, page_size: int) -> Dict[str, Any]:
        """计算分页信息
        
        Args:
            total: 总数量
            page: 当前页码
            page_size: 每页大小
            
        Returns:
            Dict[str, Any]: 分页信息
        """
        total_pages = (total + page_size - 1) // page_size
        
        return {
            'total': total,
            'page': page,
            'page_size': page_size,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1,
            'start_index': (page - 1) * page_size,
            'end_index': min(page * page_size, total)
        }
    
    async def handle_async_operation(self, operation, *args, operation_name: str = None, **kwargs) -> Any:
        """处理异步操作
        
        Args:
            operation: 异步操作函数
            *args: 位置参数
            operation_name: 操作名称
            **kwargs: 关键字参数
            
        Returns:
            Any: 操作结果
            
        Raises:
            BusinessException: 操作失败
        """
        try:
            self.log_info(f"开始执行操作: {operation_name or operation.__name__}")
            
            if hasattr(operation, '__call__'):
                if hasattr(operation, '__code__') and operation.__code__.co_flags & 0x80:
                    # 异步函数
                    result = await operation(*args, **kwargs)
                else:
                    # 同步函数
                    result = operation(*args, **kwargs)
            else:
                raise ValueError("operation必须是可调用对象")
            
            self.log_info(f"操作执行成功: {operation_name or operation.__name__}")
            return result
            
        except BusinessException:
            # 业务异常直接抛出
            raise
        except Exception as e:
            # 其他异常包装为业务异常
            error_msg = f"操作执行失败: {operation_name or operation.__name__}"
            self.log_error(error_msg, e)
            raise BusinessException(f"{error_msg}: {str(e)}")
    
    def format_error_response(self, error: Exception, context: str = None) -> Dict[str, Any]:
        """格式化错误响应
        
        Args:
            error: 异常对象
            context: 错误上下文
            
        Returns:
            Dict[str, Any]: 错误响应
        """
        return {
            'error': True,
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context,
            'timestamp': self._get_current_timestamp()
        }
    
    def _get_current_timestamp(self) -> int:
        """获取当前时间戳
        
        Returns:
            int: 时间戳
        """
        import time
        return int(time.time())


class CacheableService(BaseService):
    """支持缓存的服务基类"""
    
    def __init__(self, cache_ttl: int = 300):
        """初始化可缓存服务
        
        Args:
            cache_ttl: 缓存过期时间（秒）
        """
        super().__init__()
        self.cache_ttl = cache_ttl
        self._cache_client = None
    
    @property
    def cache_client(self):
        """获取缓存客户端"""
        if self._cache_client is None:
            self._cache_client = self.get_dependency('redis')
        return self._cache_client
    
    async def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """获取缓存数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Optional[Any]: 缓存数据
        """
        try:
            import json
            cached_data = await self.cache_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            self.log_warning(f"获取缓存数据失败: {e}")
            return None
    
    async def set_cached_data(self, cache_key: str, data: Any, ttl: int = None) -> bool:
        """设置缓存数据
        
        Args:
            cache_key: 缓存键
            data: 缓存数据
            ttl: 过期时间
            
        Returns:
            bool: 是否设置成功
        """
        try:
            import json
            ttl = ttl or self.cache_ttl
            await self.cache_client.setex(cache_key, ttl, json.dumps(data, default=str))
            return True
        except Exception as e:
            self.log_warning(f"设置缓存数据失败: {e}")
            return False
    
    async def delete_cached_data(self, cache_key: str) -> bool:
        """删除缓存数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            bool: 是否删除成功
        """
        try:
            await self.cache_client.delete(cache_key)
            return True
        except Exception as e:
            self.log_warning(f"删除缓存数据失败: {e}")
            return False
    
    def generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """生成缓存键
        
        Args:
            prefix: 键前缀
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            str: 缓存键
        """
        import hashlib
        
        # 构建键值字符串
        key_parts = [prefix]
        key_parts.extend(str(arg) for arg in args)
        
        # 添加关键字参数
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}:{value}")
        
        key_string = ":".join(key_parts)
        
        # 如果键太长，使用哈希
        if len(key_string) > 200:
            hash_obj = hashlib.md5(key_string.encode())
            return f"{prefix}:hash:{hash_obj.hexdigest()}"
        
        return key_string
