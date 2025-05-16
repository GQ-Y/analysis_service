"""
节点健康监控模块
负责监控节点的资源使用情况和健康状态
"""
import time
import threading
import asyncio
import json
import uuid
from typing import Dict, Any, Optional
from loguru import logger

from core.redis_manager import RedisManager
from core.resource import ResourceMonitor
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class NodeHealthMonitor:
    """节点健康监控器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NodeHealthMonitor, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化节点健康监控器"""
        if self._initialized:
            return
            
        self._initialized = True
        self.resource_monitor = ResourceMonitor()
        self.node_id = str(uuid.uuid4())
        self.check_interval = 30  # 检查间隔(秒)
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._redis = None
        
        # 状态阈值
        self.cpu_warning_threshold = 0.85
        self.cpu_critical_threshold = 0.95
        self.memory_warning_threshold = 0.85
        self.memory_critical_threshold = 0.95
        self.disk_warning_threshold = 0.85
        self.disk_critical_threshold = 0.95
        
        logger.info("节点健康监控器初始化完成")
        
    def start(self):
        """启动监控线程"""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            logger.info("节点健康监控线程已在运行")
            return
            
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("节点健康监控线程已启动")
        
    def _monitor_loop(self):
        """监控循环"""
        while not self._stop_event.is_set():
            try:
                self._check_node_health()
            except Exception as e:
                logger.error(f"节点健康监控异常: {str(e)}")
                
            time.sleep(self.check_interval)
            
    def _check_node_health(self):
        """检查节点健康状态"""
        # 获取资源使用情况
        usage = self.resource_monitor.get_resource_usage()
        
        # 判断节点健康状态
        health_status = "healthy"
        status_reasons = []
        
        # 检查CPU
        if usage["cpu_percent"] >= self.cpu_critical_threshold:
            health_status = "critical"
            status_reasons.append(f"CPU使用率过高: {usage['cpu_percent']*100:.1f}%")
        elif usage["cpu_percent"] >= self.cpu_warning_threshold:
            if health_status == "healthy":
                health_status = "warning"
            status_reasons.append(f"CPU使用率偏高: {usage['cpu_percent']*100:.1f}%")
            
        # 检查内存
        if usage["memory_percent"] >= self.memory_critical_threshold:
            health_status = "critical"
            status_reasons.append(f"内存使用率过高: {usage['memory_percent']*100:.1f}%")
        elif usage["memory_percent"] >= self.memory_warning_threshold:
            if health_status == "healthy":
                health_status = "warning"
            status_reasons.append(f"内存使用率偏高: {usage['memory_percent']*100:.1f}%")
            
        # 检查磁盘
        if usage["disk_percent"] >= self.disk_critical_threshold:
            health_status = "critical"
            status_reasons.append(f"磁盘使用率过高: {usage['disk_percent']*100:.1f}%")
        elif usage["disk_percent"] >= self.disk_warning_threshold:
            if health_status == "healthy":
                health_status = "warning"
            status_reasons.append(f"磁盘使用率偏高: {usage['disk_percent']*100:.1f}%")
            
        # 记录状态
        if health_status != "healthy":
            logger.warning(f"节点健康状态: {health_status}, 原因: {', '.join(status_reasons)}")
        else:
            logger.debug("节点健康状态: 正常")
            
        # 准备健康数据
        health_data = {
            "node_id": self.node_id,
            "timestamp": time.time(),
            "health_status": health_status,
            "status_reasons": status_reasons,
            "resources": usage
        }
        
        # 发布到Redis
        asyncio.run_coroutine_threadsafe(
            self._publish_health_data(health_data),
            asyncio.get_event_loop()
        )
        
    async def _publish_health_data(self, health_data: Dict[str, Any]):
        """发布健康数据到Redis
        
        Args:
            health_data: 健康数据
        """
        try:
            # 延迟初始化Redis
            if self._redis is None:
                self._redis = RedisManager()
                
            # 发布到节点健康频道
            channel = "node_health"
            await self._redis.publish(channel, json.dumps(health_data))
            
            # 更新Redis中的节点健康记录
            key = f"node:health:{self.node_id}"
            await self._redis.set_value(key, health_data, ex=60)  # 60秒过期
            
            logger.debug(f"节点健康数据已发布: {health_data['health_status']}")
            
        except Exception as e:
            logger.error(f"发布节点健康数据异常: {str(e)}")
            
    def stop(self):
        """停止监控"""
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        logger.info("节点健康监控器已停止")
        
    async def shutdown(self):
        """关闭监控器"""
        self.stop()
        logger.info("节点健康监控器已关闭")
        
    def get_node_health(self) -> Dict[str, Any]:
        """获取节点健康信息
        
        Returns:
            Dict[str, Any]: 节点健康信息
        """
        usage = self.resource_monitor.get_resource_usage()
        
        # 判断节点健康状态
        health_status = "healthy"
        status_reasons = []
        
        # 检查CPU
        if usage["cpu_percent"] >= self.cpu_critical_threshold:
            health_status = "critical"
            status_reasons.append(f"CPU使用率过高: {usage['cpu_percent']*100:.1f}%")
        elif usage["cpu_percent"] >= self.cpu_warning_threshold:
            if health_status == "healthy":
                health_status = "warning"
            status_reasons.append(f"CPU使用率偏高: {usage['cpu_percent']*100:.1f}%")
            
        # 检查内存
        if usage["memory_percent"] >= self.memory_critical_threshold:
            health_status = "critical"
            status_reasons.append(f"内存使用率过高: {usage['memory_percent']*100:.1f}%")
        elif usage["memory_percent"] >= self.memory_warning_threshold:
            if health_status == "healthy":
                health_status = "warning"
            status_reasons.append(f"内存使用率偏高: {usage['memory_percent']*100:.1f}%")
            
        # 检查磁盘
        if usage["disk_percent"] >= self.disk_critical_threshold:
            health_status = "critical"
            status_reasons.append(f"磁盘使用率过高: {usage['disk_percent']*100:.1f}%")
        elif usage["disk_percent"] >= self.disk_warning_threshold:
            if health_status == "healthy":
                health_status = "warning"
            status_reasons.append(f"磁盘使用率偏高: {usage['disk_percent']*100:.1f}%")
            
        return {
            "node_id": self.node_id,
            "timestamp": time.time(),
            "health_status": health_status,
            "status_reasons": status_reasons,
            "resources": usage
        }

# 创建单例实例
node_health_monitor = NodeHealthMonitor() 