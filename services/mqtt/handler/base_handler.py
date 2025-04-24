"""
MQTT基础处理器
提供基础的消息处理功能
"""
import logging
import time
from typing import Dict, Any

from ..mqtt_printer import MQTTPrinter

# 配置日志
logger = logging.getLogger(__name__)

class BaseMQTTHandler:
    """
    MQTT消息处理器基类
    提供基础的消息处理功能
    """
    def __init__(self):
        self.mqtt_manager = None
        self.printer = MQTTPrinter()
        
    def set_mqtt_manager(self, mqtt_manager):
        """
        设置MQTT管理器
        
        Args:
            mqtt_manager: MQTT管理器实例
        """
        self.mqtt_manager = mqtt_manager
        
    async def publish(self, topic: str, payload: Dict[str, Any], qos: int = 0) -> bool:
        """
        发布消息
        
        Args:
            topic: 主题名称
            payload: 消息内容
            qos: 服务质量等级
            
        Returns:
            bool: 发布是否成功
        """
        if not self.mqtt_manager:
            logger.error("发布消息失败: MQTT管理器未设置")
            return False
            
        # 打印发送的消息
        self.printer.print_message(topic, payload, "发送")
            
        return await self.mqtt_manager.publish(topic, payload, qos)
        
    def get_timestamp(self) -> int:
        """
        获取当前时间戳
        
        Returns:
            int: 当前时间戳
        """
        return int(time.time()) 