"""
MQTT客户端
负责与MQTT代理建立连接和通信
"""
import logging
import json
import asyncio
import ssl
from typing import Dict, Any, Optional, Callable
import gmqtt
from gmqtt.mqtt.constants import MQTTv311

from core.config import settings
from .mqtt_topic_manager import MQTTTopicManager, TOPIC_TYPE_CONNECTION
from .handler.connection_handler import get_connection_handler
from shared.utils.tools import get_mac_address

# 配置日志
logger = logging.getLogger(__name__)

class MQTTClient:
    """
    MQTT客户端
    负责与MQTT代理建立连接和通信
    """
    
    def __init__(self, client_id: str):
        """
        初始化MQTT客户端
        
        Args:
            client_id: 客户端ID
        """
        self.client_id = client_id
        self.client = None
        self.connected = False
        self.topic_manager = MQTTTopicManager(topic_prefix=settings.MQTT_TOPIC_PREFIX)
        self.connection_handler = get_connection_handler()
        
        # 连接参数
        self.host = settings.MQTT_BROKER_HOST
        self.port = settings.MQTT_BROKER_PORT
        self.username = settings.MQTT_USERNAME
        self.password = settings.MQTT_PASSWORD
        self.keepalive = settings.MQTT_KEEPALIVE
        self.qos = settings.MQTT_QOS
        self.reconnect_interval = settings.MQTT_RECONNECT_INTERVAL
        
        # SSL配置
        self.use_ssl = getattr(settings, 'MQTT_USE_SSL', False)
        self.ssl_context = None
        if self.use_ssl:
            self.ssl_context = ssl.create_default_context()
            if getattr(settings, 'MQTT_SSL_VERIFY', True):
                self.ssl_context.verify_mode = ssl.CERT_REQUIRED
            else:
                self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # 消息处理器
        self.message_handlers = {}
        
        logger.info(f"MQTT客户端已初始化: {client_id}")
        
    def on_connect(self, client, flags, rc, properties):
        """
        连接回调函数
        
        Args:
            client: MQTT客户端实例
            flags: 连接标志
            rc: 返回码
            properties: 连接属性
        """
        if rc == 0:
            self.connected = True
            logger.info("MQTT连接成功")
            
            # 发送上线消息
            asyncio.create_task(self._send_online_message())
            
            # 重新订阅主题
            asyncio.create_task(self._resubscribe_topics())
        else:
            self.connected = False
            logger.error(f"MQTT连接失败: {rc}")
            
    def on_disconnect(self, client, packet, exc=None):
        """
        断开连接回调函数
        
        Args:
            client: MQTT客户端实例
            packet: 断开连接包
            exc: 异常信息
        """
        self.connected = False
        if exc:
            logger.warning(f"MQTT意外断开连接: {exc}")
            # 尝试重新连接
            asyncio.create_task(self._reconnect())
        else:
            logger.info("MQTT正常断开连接")
            
    def on_message(self, client, topic, payload, qos, properties):
        """
        消息接收回调函数
        
        Args:
            client: MQTT客户端实例
            topic: 主题名称
            payload: 消息内容
            qos: 服务质量等级
            properties: 消息属性
        """
        try:
            # 解析消息
            message = json.loads(payload.decode())
            
            # 更新最后消息时间
            self.topic_manager.update_last_message_time(topic)
            
            # 调用消息处理器
            handler = self.message_handlers.get(topic)
            if handler:
                asyncio.create_task(handler(topic, message))
            else:
                logger.warning(f"未找到消息处理器: {topic}")
                
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            
    async def connect(self) -> bool:
        """
        连接到MQTT代理
        
        Returns:
            bool: 是否连接成功
        """
        try:
            # 创建客户端
            self.client = gmqtt.Client(self.client_id)
            
            # 设置回调函数
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message
            
            # 设置遗嘱消息
            will_message = self.connection_handler.create_will_message(
                mac_address=get_mac_address(),
                client_id=self.client_id,
                reason=403,
                resources={
                    "task_count": 0,
                    "image_task_count": 0,
                    "video_task_count": 0,
                    "stream_task_count": 0
                }
            )
            will_topic = self.topic_manager.format_topic(TOPIC_TYPE_CONNECTION)
            self.client.will_message = {
                'topic': will_topic,
                'payload': json.dumps(will_message),
                'qos': self.qos,
                'retain': True
            }
            
            # 设置认证信息
            if self.username and self.password:
                self.client.set_auth_credentials(self.username, self.password)
                
            # 连接到代理
            await self.client.connect(
                host=self.host,
                port=self.port,
                keepalive=self.keepalive,
                version=MQTTv311,
                ssl=False  # 明确禁用SSL
            )
            
            return self.connected
            
        except Exception as e:
            logger.error(f"连接MQTT代理失败: {e}")
            return False
            
    async def disconnect(self):
        """
        断开MQTT连接
        """
        if self.client:
            await self.client.disconnect()
            self.connected = False
            logger.info("MQTT连接已断开")
            
    async def _reconnect(self):
        """
        重新连接MQTT代理
        """
        while not self.connected:
            try:
                logger.info("尝试重新连接MQTT代理...")
                if await self.connect():
                    break
                await asyncio.sleep(self.reconnect_interval)
            except Exception as e:
                logger.error(f"重新连接失败: {e}")
                await asyncio.sleep(self.reconnect_interval)
                
    async def _resubscribe_topics(self):
        """
        重新订阅所有主题
        """
        for topic, subscription in self.topic_manager.get_all_subscriptions().items():
            await self.client.subscribe(topic, subscription.qos)
            logger.info(f"已重新订阅主题: {topic}")
            
    async def _send_online_message(self):
        """
        发送上线消息
        """
        await self.connection_handler.send_online_message(
            mac_address=get_mac_address(),
            client_id=self.client_id,
            compute_type=settings.MQTT_SERVICE_TYPE
        )
        
    async def register_handler(self, topic: str, handler: Callable, qos: int = None) -> bool:
        """
        注册消息处理器
        
        Args:
            topic: 主题名称
            handler: 消息处理函数
            qos: 服务质量等级，如果为None则使用默认QoS
            
        Returns:
            bool: 是否注册成功
        """
        try:
            # 添加主题订阅
            if self.topic_manager.add_subscription(topic, qos or self.qos):
                # 注册消息处理器
                self.message_handlers[topic] = handler
                
                # 如果已连接，立即订阅主题
                if self.connected:
                    await self.client.subscribe(topic, qos or self.qos)
                    
                logger.info(f"已注册消息处理器: {topic}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"注册消息处理器失败: {e}")
            return False
            
    async def publish(self, topic: str, payload: Any, qos: int = None, retain: bool = False) -> bool:
        """
        发布消息
        
        Args:
            topic: 主题名称
            payload: 消息内容
            qos: 服务质量等级，如果为None则使用默认QoS
            retain: 是否保留消息
            
        Returns:
            bool: 是否发布成功
        """
        try:
            if not self.connected:
                logger.error("发布消息失败: MQTT未连接")
                return False
                
            # 序列化消息
            if isinstance(payload, (dict, list)):
                payload = json.dumps(payload)
            elif not isinstance(payload, str):
                payload = str(payload)
                
            # 发布消息
            await self.client.publish(topic, payload, qos=qos or self.qos, retain=retain)
            return True
                
        except Exception as e:
            logger.error(f"发布消息时出错: {e}")
            return False