"""
MQTT状态处理器
处理节点状态上报
"""
import logging
import asyncio
from typing import Dict, Any, Optional

from ..mqtt_handler import BaseMQTTHandler, MESSAGE_TYPE_STATUS
from shared.utils.tools import get_mac_address, get_resource_usage
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
        logger.info("MQTT状态处理器已初始化")
        
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
                # 获取系统状态
                status = await self._get_system_status()
                
                # 发送状态消息
                await self._send_status(status)
                
                # 等待60秒
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"状态上报出错: {e}")
                await asyncio.sleep(5)  # 出错后等待5秒再重试
                
    async def _get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            Dict[str, Any]: 系统状态信息
        """
        try:
            # 获取资源使用情况
            resources = get_resource_usage()
            
            # 获取任务状态
            task_count = 0  # TODO: 从任务管理器获取任务数量
            running_tasks = 0  # TODO: 从任务管理器获取运行中的任务数量
            
            return {
                "message_type": MESSAGE_TYPE_STATUS,
                "mac_address": get_mac_address(),
                "client_id": settings.MQTT_CLIENT_ID,
                "status": "running",
                "timestamp": self.get_timestamp(),
                "resources": resources,
                "tasks": {
                    "total": task_count,
                    "running": running_tasks
                }
            }
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {
                "message_type": MESSAGE_TYPE_STATUS,
                "mac_address": get_mac_address(),
                "client_id": settings.MQTT_CLIENT_ID,
                "status": "error",
                "timestamp": self.get_timestamp(),
                "error": str(e)
            }
            
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
                "status",
                mac_address=get_mac_address()
            )
            if not topic:
                logger.error("获取状态主题失败")
                return False
                
            return await self.publish(topic, status, qos=0)
            
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