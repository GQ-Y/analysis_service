"""
MQTT状态处理器
处理节点状态上报
"""
import logging
import asyncio
import platform
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime
import psutil
import torch

from .base_handler import BaseMQTTHandler
from ..mqtt_topic_manager import TOPIC_TYPE_STATUS
from shared.utils.tools import get_mac_address, get_local_models
from core.config import settings

# 配置日志
logger = logging.getLogger(__name__)

class StatusHandler(BaseMQTTHandler):
    """
    MQTT状态处理器
    处理节点状态上报
    """
    
    def __init__(self):
        """
        初始化状态处理器
        """
        super().__init__()
        self.status_timer = None
        self.is_running = False
        self.report_interval = settings.STATUS_REPORT_INTERVAL if hasattr(settings, 'STATUS_REPORT_INTERVAL') else 60
        self.client_id = get_mac_address()
        self.start_time = datetime.now().timestamp()
        self.total_processed_tasks = 0
        self.error_count = 0
        logger.info(f"MQTT状态处理器已初始化，状态上报间隔: {self.report_interval}秒")
        
    async def start(self):
        """
        启动状态上报
        """
        if not self.is_running:
            self.is_running = True
            self.status_timer = asyncio.create_task(self._status_report_loop())
            logger.info("状态上报已启动")
            
    async def stop(self):
        """
        停止状态上报
        """
        if self.is_running:
            self.is_running = False
            if self.status_timer:
                self.status_timer.cancel()
                try:
                    await self.status_timer
                except asyncio.CancelledError:
                    pass
            logger.info("状态上报已停止")
            
    async def _status_report_loop(self):
        """
        状态上报循环
        """
        while self.is_running:
            try:
                status = await self._get_system_status()
                success = await self._send_status(status)
                if success:
                    logger.debug("状态上报成功")
                else:
                    logger.warning("状态上报失败")
                await asyncio.sleep(self.report_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"状态上报出错: {e}")
                await asyncio.sleep(5)
                
    async def _get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            Dict[str, Any]: 系统状态信息
        """
        try:
            # 基础系统信息
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net = psutil.net_io_counters()
            
            # GPU信息
            gpu_available = False
            gpu_model = "N/A"
            gpu_memory = 0.0
            gpu_usage = 0.0
            cuda_version = "N/A"
            
            try:
                if torch.cuda.is_available():
                    gpu_available = True
                    gpu_model = torch.cuda.get_device_name(0)
                    gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)  # GB
                    gpu_usage = round(torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100, 1)
                    cuda_version = torch.version.cuda
            except Exception as e:
                logger.warning(f"获取GPU信息失败: {e}")
            
            # 获取可用模型列表
            available_models = get_local_models()
            
            # 计算运行时间
            uptime = int(datetime.now().timestamp() - self.start_time)
            
            return {
                "message_type": TOPIC_TYPE_STATUS,
                "mac_address": self.client_id,
                "timestamp": int(datetime.now().timestamp()),
                "resources": {
                    "cpu_usage": round(cpu_percent, 1),
                    "memory_usage": round(memory.percent, 1),
                    "gpu_usage": gpu_usage,
                    "disk_usage": round(disk.percent, 1),
                    "network_rx": net.bytes_recv,
                    "network_tx": net.bytes_sent,
                    "task_count": 0,  # TODO: 从任务管理器获取
                    "image_task_count": 0,
                    "video_task_count": 0,
                    "stream_task_count": 0,
                    "max_tasks": settings.MAX_TASKS if hasattr(settings, 'MAX_TASKS') else 30,
                    "total_processed_tasks": self.total_processed_tasks,
                    "error_count": self.error_count,
                    "uptime": uptime,
                    "avg_processing_time": 0.0,  # TODO: 从任务管理器获取
                    "cpu_cores": psutil.cpu_count(),
                    "memory_total": round(memory.total / (1024**3), 1),  # GB
                    "gpu_model": gpu_model,
                    "gpu_memory": gpu_memory
                },
                "node_configdata": {
                    "capabilities": {
                        "models": available_models,
                        "gpu_available": gpu_available,
                        "supported_services": ["detection", "tracking"]  # TODO: 从配置获取
                    },
                    "system_info": {
                        "os": platform.system(),
                        "kernel": platform.release(),
                        "python_version": platform.python_version(),
                        "cuda_version": cuda_version
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {
                "message_type": TOPIC_TYPE_STATUS,
                "mac_address": self.client_id,
                "timestamp": int(datetime.now().timestamp()),
                "resources": {
                    "error": str(e)
                }
            }
            
    def update_task_stats(self, processed_count: int = 0, error_count: int = 0):
        """
        更新任务统计信息
        
        Args:
            processed_count: 新增处理任务数
            error_count: 新增错误任务数
        """
        self.total_processed_tasks += processed_count
        self.error_count += error_count
            
    async def _send_status(self, status: Dict[str, Any]) -> bool:
        """
        发送状态消息
        
        Args:
            status: 状态信息
            
        Returns:
            bool: 是否发送成功
        """
        try:
            topic = self.mqtt_manager.topic_manager.format_topic(
                TOPIC_TYPE_STATUS,
                mac_address=self.client_id
            )
            if not topic:
                logger.error("获取状态主题失败")
                return False
                
            return await self.mqtt_manager.publish(topic, status, qos=0)
            
        except Exception as e:
            logger.error(f"发送状态消息失败: {e}")
            return False

# 全局状态处理器实例
_status_handler = None

def get_status_handler() -> StatusHandler:
    """
    获取全局状态处理器实例
    
    Returns:
        StatusHandler: 状态处理器实例
    """
    global _status_handler
    if _status_handler is None:
        _status_handler = StatusHandler()
    return _status_handler 