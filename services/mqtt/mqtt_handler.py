"""
MQTT消息处理器
负责管理和分发MQTT消息到对应的处理器
"""
import logging
import json
import time
from typing import Dict, Any, Optional, Callable


# 配置日志
logger = logging.getLogger(__name__)

# 消息类型定义
MESSAGE_TYPE_CONNECTION = 80001  # 连接/上线/遗嘱消息
MESSAGE_TYPE_COMMAND = 80002     # 命令消息
MESSAGE_TYPE_RESULT = 80003      # 分析结果响应
MESSAGE_TYPE_STATUS = 80004      # 状态上报
MESSAGE_TYPE_BROADCAST = 80008   # 系统广播

class BaseMQTTHandler:
    """
    MQTT消息处理器基类
    提供基础的消息处理功能
    """
    def __init__(self):
        self.mqtt_manager = None
        
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
            
        return await self.mqtt_manager.publish_message(topic, payload, qos)
        
    def get_timestamp(self) -> int:
        """
        获取当前时间戳
        
        Returns:
            int: 当前时间戳
        """
        return int(time.time())

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
        # 注册消息处理器
        ## self.connection_handler = get_mqtt_connection_handler()
        logger.info("MQTT消息处理器已初始化")
    
    def register_handler(self, topic: str, handler: Callable):
        """
        注册消息处理器
        
        Args:
            topic: 消息主题
            handler: 消息处理函数
        """
        self.handlers[topic] = handler
        logger.info(f"已注册消息处理器: {topic}")
    
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
            # 获取消息类型
            message_type = payload.get("message_type")
            if not message_type:
                logger.warning(f"消息缺少message_type字段: {payload}")
                return None
            
            # 根据消息类型选择处理器
            if message_type == MESSAGE_TYPE_CONNECTION:
                # return await self.connection_handler.handle_message(topic, payload)
                pass
            else:
                # 尝试使用注册的处理器
                handler = self.handlers.get(topic)
                if handler:
                    return await handler(topic, payload)
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