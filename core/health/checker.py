"""
健康检查系统
监控各个组件的健康状态
"""
import asyncio
import time
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from shared.utils.logger import get_normal_logger, get_exception_logger
from shared.utils.retry import exponential_backoff
from core.config import settings

normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    component: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    response_time: float = 0.0


class BaseHealthChecker:
    """基础健康检查器"""
    
    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout
        self.last_check_time = 0
        self.last_result: Optional[HealthCheckResult] = None
    
    async def check(self) -> HealthCheckResult:
        """执行健康检查"""
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(self._do_check(), timeout=self.timeout)
            result.response_time = time.time() - start_time
            result.timestamp = start_time
            
            self.last_check_time = time.time()
            self.last_result = result
            
            return result
            
        except asyncio.TimeoutError:
            result = HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"健康检查超时 ({self.timeout}s)",
                response_time=time.time() - start_time
            )
            self.last_result = result
            return result
            
        except Exception as e:
            exception_logger.error(f"健康检查失败 {self.name}: {str(e)}")
            result = HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"健康检查异常: {str(e)}",
                response_time=time.time() - start_time
            )
            self.last_result = result
            return result
    
    async def _do_check(self) -> HealthCheckResult:
        """子类需要实现的具体检查逻辑"""
        raise NotImplementedError


class ZLMHealthChecker(BaseHealthChecker):
    """ZLMediaKit健康检查器"""
    
    def __init__(self, zlm_manager):
        super().__init__("ZLMediaKit", timeout=optimization_config.zlm.health_check_timeout)
        self.zlm_manager = zlm_manager
    
    async def _do_check(self) -> HealthCheckResult:
        """检查ZLM健康状态"""
        try:
            # 调用getServerConfig API检查服务状态
            result = self.zlm_manager.call_api("getServerConfig")
            
            if result.get("code") == 0:
                # 获取服务器信息
                data = result.get("data", {})
                return HealthCheckResult(
                    component=self.name,
                    status=HealthStatus.HEALTHY,
                    message="ZLM服务正常",
                    details={
                        "version": data.get("version", "unknown"),
                        "uptime": data.get("uptime", 0),
                        "api_response_code": result.get("code")
                    }
                )
            else:
                return HealthCheckResult(
                    component=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"ZLM API返回错误码: {result.get('code')}",
                    details={"api_response": result}
                )
                
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"ZLM连接失败: {str(e)}"
            )


class DatabaseHealthChecker(BaseHealthChecker):
    """数据库健康检查器"""
    
    def __init__(self, db_manager):
        super().__init__("Database", timeout=5.0)
        self.db_manager = db_manager
    
    async def _do_check(self) -> HealthCheckResult:
        """检查数据库健康状态"""
        try:
            # 执行简单查询测试连接
            async with self.db_manager.get_session() as session:
                result = await session.execute("SELECT 1")
                await result.fetchone()
            
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.HEALTHY,
                message="数据库连接正常"
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"数据库连接失败: {str(e)}"
            )


class RedisHealthChecker(BaseHealthChecker):
    """Redis健康检查器"""
    
    def __init__(self, redis_manager):
        super().__init__("Redis", timeout=3.0)
        self.redis_manager = redis_manager
    
    async def _do_check(self) -> HealthCheckResult:
        """检查Redis健康状态"""
        try:
            # 执行ping命令测试连接
            pong = await self.redis_manager.ping()
            
            if pong:
                return HealthCheckResult(
                    component=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Redis连接正常"
                )
            else:
                return HealthCheckResult(
                    component=self.name,
                    status=HealthStatus.DEGRADED,
                    message="Redis ping失败"
                )
                
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Redis连接失败: {str(e)}"
            )


class SystemHealthChecker(BaseHealthChecker):
    """系统资源健康检查器"""
    
    def __init__(self):
        super().__init__("System", timeout=2.0)
    
    async def _do_check(self) -> HealthCheckResult:
        """检查系统资源状态"""
        try:
            import psutil
            
            # 获取系统资源使用情况
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }
            
            # 判断健康状态
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "系统资源使用率过高"
            elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 80:
                status = HealthStatus.DEGRADED
                message = "系统资源使用率较高"
            else:
                status = HealthStatus.HEALTHY
                message = "系统资源正常"
            
            return HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNKNOWN,
                message=f"无法获取系统信息: {str(e)}"
            )


class HealthCheckManager:
    """健康检查管理器"""
    
    def __init__(self):
        self.checkers: Dict[str, BaseHealthChecker] = {}
        self.check_interval = optimization_config.zlm.health_check_interval
        self.running = False
        self._check_task: Optional[asyncio.Task] = None
    
    def register_checker(self, checker: BaseHealthChecker):
        """注册健康检查器"""
        self.checkers[checker.name] = checker
        normal_logger.info(f"注册健康检查器: {checker.name}")
    
    def unregister_checker(self, name: str):
        """注销健康检查器"""
        if name in self.checkers:
            del self.checkers[name]
            normal_logger.info(f"注销健康检查器: {name}")
    
    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """检查所有组件健康状态"""
        results = {}
        
        # 并发执行所有健康检查
        tasks = []
        for name, checker in self.checkers.items():
            tasks.append(checker.check())
        
        if tasks:
            check_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(check_results):
                checker_name = list(self.checkers.keys())[i]
                
                if isinstance(result, Exception):
                    results[checker_name] = HealthCheckResult(
                        component=checker_name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"健康检查异常: {str(result)}"
                    )
                else:
                    results[checker_name] = result
        
        return results
    
    async def get_overall_status(self) -> HealthStatus:
        """获取整体健康状态"""
        results = await self.check_all()
        
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    async def start_monitoring(self):
        """开始健康监控"""
        if self.running:
            return
        
        self.running = True
        self._check_task = asyncio.create_task(self._monitoring_loop())
        normal_logger.info(f"开始健康监控，检查间隔: {self.check_interval}秒")
    
    async def stop_monitoring(self):
        """停止健康监控"""
        self.running = False
        
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        
        normal_logger.info("健康监控已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                results = await self.check_all()
                
                # 记录不健康的组件
                for name, result in results.items():
                    if result.status != HealthStatus.HEALTHY:
                        normal_logger.warning(
                            f"组件 {name} 健康状态: {result.status.value}, "
                            f"消息: {result.message}"
                        )
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                exception_logger.error(f"健康监控循环异常: {str(e)}")
                await asyncio.sleep(self.check_interval)


# 全局健康检查管理器实例
health_manager = HealthCheckManager()
