"""
MQTT命令处理器
处理来自服务端的命令消息
"""
import logging
from typing import Dict, Any, Optional

from shared.utils.tools import get_mac_address
from ..handler.message_types import (
    MESSAGE_TYPE_REQUEST_SETTING,
    REQUEST_TYPE_NODE_CMD,
    REQUEST_TYPE_TASK_CMD
)
from .node_command_handler import get_node_command_handler
from .task_command_handler import get_task_command_handler
from ..handler.base_handler import BaseMQTTHandler

# 配置日志
logger = logging.getLogger(__name__)

class CommandHandler(BaseMQTTHandler):
    """
    MQTT命令处理器
    处理来自服务端的命令消息
    """
    
    def __init__(self):
        """
        初始化命令处理器
        """
        super().__init__()
        self.node_handler = get_node_command_handler()
        self.task_handler = get_task_command_handler()
        logger.info("MQTT命令处理器已初始化")
        
    def get_mac_address(self) -> str:
        """
        获取MAC地址
        
        Returns:
            str: MAC地址
        """
        return get_mac_address()
        
    async def handle_message(self, topic: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理命令消息
        
        Args:
            topic: 消息主题
            payload: 消息内容
            
        Returns:
            Optional[Dict[str, Any]]: 处理结果
        """
        try:
            # 检查消息类型
            message_type = payload.get("message_type")
            if message_type != MESSAGE_TYPE_REQUEST_SETTING:
                logger.warning(f"非法的消息类型: {message_type}")
                return None
                
            # 获取请求类型
            request_type = payload.get("request_type")
            if not request_type:
                logger.warning(f"消息缺少request_type字段: {payload}")
                return None
                
            # 根据请求类型分发到对应的处理器
            result = None
            if request_type == REQUEST_TYPE_NODE_CMD:
                result = await self.node_handler.handle_command(payload)
            elif request_type == REQUEST_TYPE_TASK_CMD:
                result = await self.task_handler.handle_command(payload)
            else:
                logger.warning(f"未知的请求类型: {request_type}")
                return None
                
            # 发送响应
            if result:
                await self.send_reply(result)
                
            return result
                
        except Exception as e:
            logger.error(f"处理命令时出错: {e}")
            return None
            
    async def send_reply(self, reply: Dict[str, Any]) -> bool:
        """
        发送命令回复
        
        Args:
            reply: 回复消息内容
            
        Returns:
            bool: 是否发送成功
        """
        try:
            topic = self.mqtt_manager.topic_manager.format_topic(
                MESSAGE_TYPE_REQUEST_SETTING,
                mac_address=self.get_mac_address()
            )
            if not topic:
                logger.error("获取回复主题失败")
                return False
                
            return await self.mqtt_manager.publish(topic, reply, qos=1)
            
        except Exception as e:
            logger.error(f"发送命令回复失败: {e}")
            return False

# 全局命令处理器实例
_command_handler = None

def get_command_handler() -> CommandHandler:
    """
    获取全局命令处理器实例
    
    Returns:
        CommandHandler: 命令处理器实例
    """
    global _command_handler
    if _command_handler is None:
        _command_handler = CommandHandler()
    return _command_handler 