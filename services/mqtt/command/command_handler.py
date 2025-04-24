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
    REQUEST_TYPE_TASK_CMD,
    TOPIC_TYPE_DEVICE_CONFIG_REPLY
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
    
    def __init__(self, mqtt_manager):
        """
        初始化命令处理器
        """
        super().__init__()
        self.mqtt_manager = mqtt_manager # 保存mqtt_manager实例
        self.node_handler = get_node_command_handler()
        self.task_handler = get_task_command_handler(self.mqtt_manager) # 传递mqtt_manager
        logger.info("MQTT命令处理器已初始化")
        
    def set_mqtt_manager(self, mqtt_manager):
        """
        设置MQTT管理器，并向下传递
        """
        super().set_mqtt_manager(mqtt_manager)
        # 确保 task_handler 也获得 mqtt_manager
        if self.task_handler and hasattr(self.task_handler, 'mqtt_manager'):
             self.task_handler.mqtt_manager = mqtt_manager
        # 或者如果 task_handler 也有 set_mqtt_manager 方法
        # if self.task_handler and hasattr(self.task_handler, 'set_mqtt_manager'):
        #    self.task_handler.set_mqtt_manager(mqtt_manager)

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
            reply: 回复消息内容 (可能包含 confirmation_topic)
            
        Returns:
            bool: 是否发送成功
        """
        try:
            logger.debug(f"[send_reply] 收到的原始回复字典: {reply}") # 打印收到的完整字典
            
            # 优先使用回复中指定的确认主题
            target_topic = reply.get("confirmation_topic")
            logger.debug(f"[send_reply] 从回复字典获取的 confirmation_topic: {target_topic}") # 打印获取到的值
            
            # 如果没有指定确认主题，则默认回复到设备配置回复主题
            if not target_topic:
                logger.warning(f"[send_reply] 未找到 confirmation_topic 或其值为空，使用默认回复主题")
                target_topic = self.mqtt_manager.topic_manager.format_topic(
                    TOPIC_TYPE_DEVICE_CONFIG_REPLY,
                    mac_address=self.get_mac_address()
                )
                logger.warning(f"[send_reply] 默认回复主题: {target_topic}")
            
            if not target_topic:
                logger.error("无法确定回复主题")
                return False
                
            # 从回复字典中移除临时的 confirmation_topic 键 (如果存在)
            reply_payload = reply.copy()
            reply_payload.pop("confirmation_topic", None)
                
            logger.info(f"准备发送命令回复到主题: {target_topic}")
            return await self.mqtt_manager.publish(target_topic, reply_payload, qos=1)
            
        except Exception as e:
            logger.error(f"发送命令回复失败: {e}")
            return False

# 全局命令处理器实例
_command_handler = None

def get_command_handler(mqtt_manager) -> CommandHandler:
    """
    获取全局命令处理器实例
    
    Returns:
        CommandHandler: 命令处理器实例
    """
    global _command_handler
    if _command_handler is None:
        _command_handler = CommandHandler(mqtt_manager)
    # 可选：更新mqtt_manager引用
    # elif _command_handler.mqtt_manager != mqtt_manager:
    #    _command_handler.mqtt_manager = mqtt_manager
    return _command_handler 