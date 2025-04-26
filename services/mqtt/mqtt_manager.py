"""
MQTT服务管理器
提供MQTT服务的高层管理功能
"""
import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from core.config import settings
from .mqtt_client import MQTTClient
from .mqtt_topic_manager import MQTTTopicManager
from .mqtt_printer import MQTTPrinter
from .mqtt_middleware import get_mqtt_middleware
from .mqtt_handler import get_mqtt_message_handler

# 配置日志
logger = logging.getLogger(__name__)

class MQTTManager:
    """
    MQTT服务管理器类，负责管理MQTT服务的生命周期和业务功能
    """
    
    def __init__(self):
        """
        初始化MQTT服务管理器
        """
        self.mqtt_client = None
        self.is_running = False
        
        # 初始化主题管理器
        self.topic_manager = MQTTTopicManager()
        
        # 初始化中间件
        self.middleware = get_mqtt_middleware()
        
        # 初始化消息处理器
        self._message_handler = None
        
        logger.info("MQTT服务管理器已初始化")
        
    async def start(self) -> bool:
        """
        启动MQTT服务，使用配置文件中的设置
        
        Returns:
            bool: 服务是否成功启动
        """
        try:
            # 创建MQTT客户端
            self.mqtt_client = MQTTClient(
                client_id=f"{settings.MQTT_SERVICE_TYPE}_{settings.MQTT_NODE_ID}"
            )
            
            # 初始化消息处理器
            self._message_handler = get_mqtt_message_handler()
            
            # 连接到MQTT代理
            if await self.mqtt_client.connect():
                self.is_running = True
                logger.info("MQTT服务已启动")
                return True
                
            return False
        except Exception as e:
            logger.error(f"启动MQTT服务时出错: {e}")
            return False
            
    async def stop(self):
        """
        停止MQTT服务
        """
        if self.mqtt_client and self.is_running:
            await self.mqtt_client.disconnect()
            self.is_running = False
            logger.info("MQTT服务已停止")
            
    async def publish_message(self, topic: str, payload: Any, qos: int = 0, retain: bool = False) -> bool:
        """
        发布消息到指定主题
        
        Args:
            topic: 主题名称
            payload: 消息内容
            qos: 服务质量等级
            retain: 是否保留消息
            
        Returns:
            bool: 发布是否成功
        """
        if not self.mqtt_client: # 检查客户端是否存在
            logger.error(f"发布消息到主题 '{topic}' 失败: MQTT客户端未初始化")
            return False

        # 检查连接状态，添加日志记录
        if not self.mqtt_client.is_connected():
            logger.warning(f"尝试发布消息到主题 '{topic}'，但 MQTT 客户端当前未连接。将继续尝试发布...")
        else:
            logger.debug(f"MQTT 客户端已连接，准备发布到主题 '{topic}'")

        logger.debug(f"调用 self.mqtt_client.publish - 主题: '{topic}', QoS: {qos}, Retain: {retain}")
        try:
            # 调用实际的客户端发布方法
            success = await self.mqtt_client.publish(topic, payload, qos, retain)
            # 根据 gmqtt 的 publish 方法通常返回 PacketIdentifier 或 None，这里我们只记录调用完成
            logger.debug(f"调用 self.mqtt_client.publish 为主题 '{topic}' 完成。")
            # 假设只要没有异常，调用就是尝试成功了，实际发送由 gmqtt 处理
            # 注意：这并不保证消息已成功发送到 Broker
            return True # 或者根据 self.mqtt_client.publish 的实际返回值调整
        except Exception as e:
            # 捕获 self.mqtt_client.publish 可能抛出的同步异常
            logger.error(f"发布消息到主题 '{topic}' 时发生同步异常: {e}", exc_info=True)
            return False
        
    def register_topic_handler(self, topic: str, handler: Callable, qos: int = 0) -> bool:
        """
        注册主题处理函数
        
        Args:
            topic: 主题名称
            handler: 消息处理函数
            qos: 服务质量等级
            
        Returns:
            bool: 注册是否成功
        """
        if not self.mqtt_client:
            logger.error("注册处理函数失败: MQTT客户端未初始化")
            return False
            
        return self.mqtt_client.register_handler(topic, handler, qos)
        
    def get_subscription_info(self, topic: str = None) -> Dict[str, Any]:
        """
        获取订阅信息
        
        Args:
            topic: 主题名称，如果为None则返回所有订阅信息
            
        Returns:
            Dict[str, Any]: 订阅信息
        """
        if topic:
            subscription = self.topic_manager.get_subscription(topic)
            if subscription:
                return {
                    "topic": subscription.topic,
                    "qos": subscription.qos,
                    "is_wildcard": subscription.is_wildcard,
                    "subscribe_time": subscription.subscribe_time,
                    "last_message_time": subscription.last_message_time
                }
            return None
        else:
            subscriptions = self.topic_manager.get_all_subscriptions()
            return {
                sub.topic: {
                    "qos": sub.qos,
                    "is_wildcard": sub.is_wildcard,
                    "subscribe_time": sub.subscribe_time,
                    "last_message_time": sub.last_message_time
                }
                for sub in subscriptions
            }
            
    def is_connected(self) -> bool:
        """
        检查MQTT服务是否已连接
        
        Returns:
            bool: 是否已连接
        """
        return self.mqtt_client and self.mqtt_client.connected

# 创建全局MQTT管理器实例
_mqtt_manager = None

def get_mqtt_manager() -> MQTTManager:
    """
    获取全局MQTT管理器实例
    
    Returns:
        MQTTManager: MQTT管理器实例
    """
    global _mqtt_manager
    if _mqtt_manager is None:
        _mqtt_manager = MQTTManager()
        logger.info("已创建全局MQTT管理器实例")
    return _mqtt_manager 