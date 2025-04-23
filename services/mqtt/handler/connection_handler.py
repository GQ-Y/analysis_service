"""
MQTT连接处理器
处理连接、上线和遗嘱消息
"""
import logging
import socket
import platform
from typing import Dict, Any, Optional

from core.config import settings
from ..mqtt_handler import BaseMQTTHandler, MESSAGE_TYPE_CONNECTION

# 配置日志
logger = logging.getLogger(__name__)

class ConnectionHandler(BaseMQTTHandler):
    """
    MQTT连接处理器
    处理连接、上线和遗嘱消息
    """
    
    def __init__(self):
        """
        初始化连接处理器
        """
        super().__init__()
        logger.info("MQTT连接处理器已初始化")
        
    def get_local_ip(self) -> str:
        """
        获取本地IP地址
        
        Returns:
            str: 本地IP地址
        """
        try:
            # 创建一个临时socket连接来获取本地IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            logger.error(f"获取本地IP地址失败: {e}")
            return "127.0.0.1"
            
    def get_hostname(self) -> str:
        """
        获取主机名
        
        Returns:
            str: 主机名
        """
        return platform.node()
        
    async def send_online_message(self, mac_address: str, client_id: str, compute_type: int) -> bool:
        """
        发送上线消息
        
        Args:
            mac_address: MAC地址
            client_id: 客户端ID
            compute_type: 计算类型
            
        Returns:
            bool: 是否发送成功
        """
        try:
            message = {
                "message_type": MESSAGE_TYPE_CONNECTION,
                "mac_address": mac_address,
                "client_id": client_id,
                "compute_type": compute_type,
                "status": 1,
                "ip": self.get_local_ip(),
                "port": settings.PORT,
                "hostname": self.get_hostname(),
                "version": settings.VERSION,
                "timestamp": self.get_timestamp()
            }
            
            topic = self.mqtt_manager.topic_manager.format_topic("connection")
            if not topic:
                logger.error("获取连接主题失败")
                return False
                
            return await self.publish(topic, message, qos=1)
            
        except Exception as e:
            logger.error(f"发送上线消息失败: {e}")
            return False
            
    def create_will_message(self, mac_address: str, client_id: str, reason: int, resources: Dict[str, int]) -> Dict[str, Any]:
        """
        创建遗嘱消息
        
        Args:
            mac_address: MAC地址
            client_id: 客户端ID
            reason: 离线原因
            resources: 资源信息
            
        Returns:
            Dict[str, Any]: 遗嘱消息
        """
        return {
            "message_type": MESSAGE_TYPE_CONNECTION,
            "mac_address": mac_address,
            "client_id": client_id,
            "status": 0,
            "timestamp": self.get_timestamp(),
            "reason": reason,
            "resources": resources
        }

# 全局连接处理器实例
_connection_handler = None

def get_connection_handler() -> ConnectionHandler:
    """
    获取全局连接处理器实例
    
    Returns:
        ConnectionHandler: 连接处理器实例
    """
    global _connection_handler
    if _connection_handler is None:
        _connection_handler = ConnectionHandler()
    return _connection_handler 