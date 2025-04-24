"""
MQTT消息处理器
负责管理和分发MQTT消息到对应的处理器
"""
import logging
import json
from typing import Dict, Any, Optional, Callable

from .mqtt_printer import MQTTPrinter
from .handler.base_handler import BaseMQTTHandler
from .handler.message_types import *
from .handler.connection_handler import get_connection_handler
from .handler.status_handler import get_status_handler

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
        self.handlers = {}
        
        # 初始化各个处理器
        self.connection_handler = get_connection_handler()
        self.status_handler = get_status_handler()
        
        logger.info("MQTT消息处理器已初始化")
        
    def set_mqtt_manager(self, mqtt_manager):
        """
        设置MQTT管理器，并传递给所有子处理器
        
        Args:
            mqtt_manager: MQTT管理器实例
        """
        super().set_mqtt_manager(mqtt_manager)
        
        # 更新所有处理器的MQTT管理器
        if self.connection_handler:
            self.connection_handler.set_mqtt_manager(mqtt_manager)
        if self.status_handler:
            self.status_handler.set_mqtt_manager(mqtt_manager)
            
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
    
    def register_handler(self, topic: str, handler: Callable):
        """
        注册消息处理器
        
        Args:
            topic: 消息主题
            handler: 消息处理函数
        """
        self.handlers[topic] = handler
        logger.info(f"已注册消息处理器: {topic}")
        self.printer.print_subscription(topic, 0, "注册")
    
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
            if not message_type:
                logger.warning(f"消息缺少message_type字段: {payload}")
                return None
            
            # 根据消息类型选择处理器
            if message_type == MESSAGE_TYPE_CONNECTION:
                # 使用连接处理器
                return await self.connection_handler.handle_message(topic, payload)
            elif message_type == MESSAGE_TYPE_STATUS:
                # 使用状态处理器
                return await self.status_handler.handle_message(topic, payload)
            else:
                # 尝试使用注册的处理器
                handler = self.handlers.get(topic)
                if handler:
                    result = await handler(topic, payload)
                    # 打印处理结果
                    if result:
                        self.printer.print_message(topic, result, "处理结果")
                    return result
                else:
                    logger.warning(f"未找到对应的消息处理器: {topic}")
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