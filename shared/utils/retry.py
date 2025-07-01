"""
智能重试机制
提供指数退避、条件重试等高级重试功能
"""
import asyncio
import time
import random
from functools import wraps
from typing import Callable, Any, Optional, Union, Type, Tuple
from shared.utils.logger import get_normal_logger, get_exception_logger
from core.config_modules.optimization import optimization_config

normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)


class RetryConfig:
    """重试配置类"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.exceptions = exceptions


def exponential_backoff(
    max_retries: int = None,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    指数退避重试装饰器
    
    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间(秒)
        max_delay: 最大延迟时间(秒)
        backoff_factor: 退避因子
        jitter: 是否添加随机抖动
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
    """
    if max_retries is None:
        max_retries = optimization_config.task_management.max_retries
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        exception_logger.error(f"函数 {func.__name__} 重试 {max_retries} 次后仍然失败: {str(e)}")
                        raise e
                    
                    # 计算延迟时间
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    
                    # 添加随机抖动
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    normal_logger.warning(
                        f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {str(e)}, "
                        f"{delay:.2f}秒后重试"
                    )
                    
                    # 调用重试回调
                    if on_retry:
                        try:
                            if asyncio.iscoroutinefunction(on_retry):
                                await on_retry(attempt, e, delay)
                            else:
                                on_retry(attempt, e, delay)
                        except Exception as callback_error:
                            exception_logger.error(f"重试回调函数执行失败: {str(callback_error)}")
                    
                    await asyncio.sleep(delay)
            
            # 这里不应该到达，但为了类型安全
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                        
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        exception_logger.error(f"函数 {func.__name__} 重试 {max_retries} 次后仍然失败: {str(e)}")
                        raise e
                    
                    # 计算延迟时间
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    
                    # 添加随机抖动
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    normal_logger.warning(
                        f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {str(e)}, "
                        f"{delay:.2f}秒后重试"
                    )
                    
                    # 调用重试回调
                    if on_retry:
                        try:
                            on_retry(attempt, e, delay)
                        except Exception as callback_error:
                            exception_logger.error(f"重试回调函数执行失败: {str(callback_error)}")
                    
                    time.sleep(delay)
            
            # 这里不应该到达，但为了类型安全
            raise last_exception
        
        # 根据函数类型返回对应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception
):
    """
    断路器装饰器
    
    Args:
        failure_threshold: 失败阈值
        recovery_timeout: 恢复超时时间(秒)
        expected_exception: 预期的异常类型
    """
    def decorator(func: Callable) -> Callable:
        func._circuit_breaker_failures = 0
        func._circuit_breaker_last_failure_time = None
        func._circuit_breaker_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            current_time = time.time()
            
            # 检查断路器状态
            if func._circuit_breaker_state == "OPEN":
                if (current_time - func._circuit_breaker_last_failure_time) > recovery_timeout:
                    func._circuit_breaker_state = "HALF_OPEN"
                    normal_logger.info(f"断路器 {func.__name__} 进入半开状态")
                else:
                    raise Exception(f"断路器 {func.__name__} 处于开启状态，拒绝请求")
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # 成功时重置计数器
                if func._circuit_breaker_state == "HALF_OPEN":
                    func._circuit_breaker_state = "CLOSED"
                    func._circuit_breaker_failures = 0
                    normal_logger.info(f"断路器 {func.__name__} 恢复到关闭状态")
                
                return result
                
            except expected_exception as e:
                func._circuit_breaker_failures += 1
                func._circuit_breaker_last_failure_time = current_time
                
                if func._circuit_breaker_failures >= failure_threshold:
                    func._circuit_breaker_state = "OPEN"
                    normal_logger.error(
                        f"断路器 {func.__name__} 开启，失败次数: {func._circuit_breaker_failures}"
                    )
                
                raise e
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            current_time = time.time()
            
            # 检查断路器状态
            if func._circuit_breaker_state == "OPEN":
                if (current_time - func._circuit_breaker_last_failure_time) > recovery_timeout:
                    func._circuit_breaker_state = "HALF_OPEN"
                    normal_logger.info(f"断路器 {func.__name__} 进入半开状态")
                else:
                    raise Exception(f"断路器 {func.__name__} 处于开启状态，拒绝请求")
            
            try:
                result = func(*args, **kwargs)
                
                # 成功时重置计数器
                if func._circuit_breaker_state == "HALF_OPEN":
                    func._circuit_breaker_state = "CLOSED"
                    func._circuit_breaker_failures = 0
                    normal_logger.info(f"断路器 {func.__name__} 恢复到关闭状态")
                
                return result
                
            except expected_exception as e:
                func._circuit_breaker_failures += 1
                func._circuit_breaker_last_failure_time = current_time
                
                if func._circuit_breaker_failures >= failure_threshold:
                    func._circuit_breaker_state = "OPEN"
                    normal_logger.error(
                        f"断路器 {func.__name__} 开启，失败次数: {func._circuit_breaker_failures}"
                    )
                
                raise e
        
        # 根据函数类型返回对应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class RetryableHTTPClient:
    """可重试的HTTP客户端"""
    
    def __init__(self, timeout: int = None, max_retries: int = None):
        self.timeout = timeout or optimization_config.network.http_timeout
        self.max_retries = max_retries or optimization_config.network.http_max_retries
    
    @exponential_backoff(
        exceptions=(Exception,),
        base_delay=1.0,
        max_delay=30.0
    )
    async def post(self, url: str, data: dict = None, headers: dict = None) -> dict:
        """可重试的POST请求"""
        import aiohttp
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, json=data, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
    
    @exponential_backoff(
        exceptions=(Exception,),
        base_delay=1.0,
        max_delay=30.0
    )
    async def get(self, url: str, params: dict = None, headers: dict = None) -> dict:
        """可重试的GET请求"""
        import aiohttp
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                return await response.json()


# 全局可重试HTTP客户端实例
retryable_http_client = RetryableHTTPClient()
