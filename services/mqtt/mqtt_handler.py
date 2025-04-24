"""
MQTT消息处理器
负责管理和分发MQTT消息到对应的处理器
"""
import logging
import json
from typing import Dict, Any, Optional, Callable
import asyncio

from .mqtt_printer import MQTTPrinter
from .handler.base_handler import BaseMQTTHandler
from .handler.message_types import *
from .handler.connection_handler import get_connection_handler
from .handler.status_handler import get_status_handler
from .command.command_handler import get_command_handler
from .mqtt_topic_manager import TOPIC_TYPE_REQUEST_SETTING, MQTTTopicManager
from shared.utils.tools import get_mac_address
from core.config import settings

# 配置日志
logger = logging.getLogger(__name__)

class MQTTMessageHandler(BaseMQTTHandler):
    """
    MQTT消息处理器
    负责管理和分发MQTT消息到对应的处理器
    """
    
    def __init__(self):
        """
        初始化MQTT消息处理器
        """
        super().__init__()
        
        # 初始化主题管理器
        self.topic_manager = MQTTTopicManager(topic_prefix=settings.MQTT_TOPIC_PREFIX)
        
        # 初始化其他处理器
        self.connection_handler = get_connection_handler()
        self.status_handler = get_status_handler()
        # 延迟 command_handler 的初始化
        self.command_handler = None 
        
        logger.info("MQTT消息处理器已初始化")
        
    def set_mqtt_manager(self, mqtt_manager):
        """
        设置MQTT管理器，并传递给所有子处理器
        
        Args:
            mqtt_manager: MQTT管理器实例
        """
        super().set_mqtt_manager(mqtt_manager)
        
        # 在这里初始化 command_handler，并传递 mqtt_manager
        if self.command_handler is None:
            self.command_handler = get_command_handler(mqtt_manager)
        
        # 更新所有子处理器的MQTT管理器
        if self.connection_handler:
            self.connection_handler.set_mqtt_manager(mqtt_manager)
        if self.status_handler:
            self.status_handler.set_mqtt_manager(mqtt_manager)
        if self.command_handler:
            self.command_handler.set_mqtt_manager(mqtt_manager)
            
    async def start(self):
        """
        启动消息处理器
        """
        # 启动状态上报
        if self.status_handler:
            await self.status_handler.start()
            logger.info("状态处理器已启动")
            
    async def stop(self):
        """
        停止消息处理器
        """
        # 停止状态上报
        if self.status_handler:
            await self.status_handler.stop()
            logger.info("状态处理器已停止")
    
    async def handle_message(self, topic: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理MQTT消息
        
        Args:
            topic: 消息主题
            payload: 消息内容（已解析的字典）
            
        Returns:
            Optional[Dict[str, Any]]: 处理结果
        """
        try:
            # 打印接收到的消息
            self.printer.print_message(topic, payload, "接收")
            
            # 获取消息类型
            message_type = payload.get("message_type")
            if message_type is None:
                logger.warning(f"消息缺少message_type字段: {payload}")
                return None
            
            # 根据消息类型选择处理器
            if message_type == MESSAGE_TYPE_COMMAND:  # 80002 命令消息
                logger.info(f"收到命令消息，转交给命令处理器处理: {payload}")
                return await self.command_handler.handle_message(topic, payload)
            elif message_type == MESSAGE_TYPE_CONNECTION:  # 80001 连接消息
                return await self.connection_handler.handle_message(topic, payload)
            elif message_type == MESSAGE_TYPE_STATUS:  # 80004 状态消息
                return await self.status_handler.handle_message(topic, payload)
            else:
                logger.warning(f"未知的消息类型: {message_type}")
                return None
                    
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            import traceback
            logger.error(f"错误详情:\n{traceback.format_exc()}")
            return None

# 全局消息处理器实例
_mqtt_message_handler = None

def get_mqtt_message_handler() -> MQTTMessageHandler:
    """
    获取全局MQTT消息处理器实例
    
    Returns:
        MQTTMessageHandler: 消息处理器实例
    """
    global _mqtt_message_handler
    if _mqtt_message_handler is None:
        _mqtt_message_handler = MQTTMessageHandler()
    return _mqtt_message_handler 