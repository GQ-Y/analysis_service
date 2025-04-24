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
from datetime import datetime
import socket

from core.config import settings
from .mqtt_topic_manager import MQTTTopicManager, TOPIC_TYPE_CONNECTION, TOPIC_TYPE_REQUEST_SETTING
from .handler.connection_handler import get_connection_handler
from .mqtt_handler import get_mqtt_message_handler
from .mqtt_printer import MQTTPrinter
from shared.utils.tools import get_mac_address, get_local_ip

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
        self.message_handler = get_mqtt_message_handler()
        self.printer = MQTTPrinter()
        
        # 设置MQTT管理器
        self.connection_handler.set_mqtt_manager(self)
        self.message_handler.set_mqtt_manager(self)
        
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
            self.printer.print_connection_status("成功", "MQTT连接成功")
            
            # 订阅必要的主题
            request_setting_topic = self.topic_manager.format_topic(
                TOPIC_TYPE_REQUEST_SETTING,
                mac_address=get_mac_address()
            )
            if request_setting_topic:
                self.client.subscribe(request_setting_topic, qos=1)
                logger.info(f"已订阅请求设置主题: {request_setting_topic}")
                self.printer.print_subscription(request_setting_topic, 1, "订阅")
                # 添加到主题管理器
                self.topic_manager.add_subscription(request_setting_topic, qos=1)
            
            # 发送上线消息
            asyncio.create_task(self._send_online_message())
            
            # 重新订阅主题
            asyncio.create_task(self._resubscribe_topics())
            
            # 启动消息处理器
            asyncio.create_task(self.message_handler.start())
        else:
            self.connected = False
            logger.error(f"MQTT连接失败: {rc}")
            self.printer.print_connection_status("失败", f"MQTT连接失败: {rc}")
            
    def on_disconnect(self, client, packet, exc=None):
        """
        断开连接回调函数
        
        Args:
            client: MQTT客户端实例
            packet: 断开连接包
            exc: 异常信息
        """
        self.connected = False
        
        # 停止消息处理器
        asyncio.create_task(self.message_handler.stop())
        
        if exc:
            logger.warning(f"MQTT意外断开连接: {exc}")
            self.printer.print_connection_status("失败", f"MQTT意外断开连接: {exc}")
            # 尝试重新连接
            asyncio.create_task(self._reconnect())
        else:
            logger.info("MQTT正常断开连接")
            self.printer.print_connection_status("成功", "MQTT正常断开连接")
            
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
            
            # 打印接收到的消息
            self.printer.print_message(topic, message, "接收")
            
            # 使用消息处理器处理消息
            if self.message_handler:
                asyncio.create_task(self.message_handler.handle_message(topic, message))
            else:
                logger.warning("消息处理器未初始化")
                
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            import traceback
            logger.error(f"错误详情:\n{traceback.format_exc()}")
        
    async def connect(self) -> bool:
        """
        连接到MQTT代理
        
        Returns:
            bool: 是否连接成功
        """
        try:
            # 准备遗嘱消息数据
            will_topic = self.topic_manager.format_topic(TOPIC_TYPE_CONNECTION)
            
            # 使用connection_handler创建遗嘱消息
            will_payload = self.connection_handler.create_will_message(
                mac_address=get_mac_address(),
                client_id=self.client_id,
                reason=0,  # 异常断开状态码
                resources={
                    "task_count": 0,
                    "image_task_count": 0,
                    "video_task_count": 0,
                    "stream_task_count": 0
                }
            )
            
            # 准备遗嘱消息 - 在gmqtt中，遗嘱消息应作为Client的初始化参数
            will_message = gmqtt.Message(
                will_topic,
                payload=json.dumps(will_payload),
                qos=self.qos,
                retain=True
            )
            
            # 创建客户端，将遗嘱消息作为初始化参数
            self.client = gmqtt.Client(self.client_id, will_message=will_message)
            
            # 设置回调函数
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message
            
            # 设置认证信息
            if self.username and self.password:
                self.client.set_auth_credentials(self.username, self.password)
            
            # 连接到代理
            await self.client.connect(
                host=self.host,
                port=self.port,
                keepalive=30,
                version=MQTTv311
            )
            
            return self.connected
            
        except Exception as e:
            logger.error(f"连接MQTT代理失败: {e}")
            self.printer.print_connection_status("失败", f"连接MQTT代理失败: {e}")
            return False
            
    async def disconnect(self, trigger_will=True):
        """
        断开MQTT连接
        
        Args:
            trigger_will: 是否触发遗嘱消息，默认为True
        """
        # 停止消息处理器
        await self.message_handler.stop()
        
        if self.client:
            try:
                if trigger_will:
                    # 强制断开连接，触发遗嘱消息
                    if hasattr(self.client, '_connection') and self.client._connection:
                        self.client._connection.close()
                        logger.info("MQTT连接已强制断开，遗嘱消息将被发送")
                        self.printer.print_connection_status("成功", "MQTT连接已强制断开，遗嘱消息将被发送")
                else:
                    # 正常断开连接（不会触发遗嘱消息）
                    await self.client.disconnect()
                    logger.info("MQTT连接已正常断开，不触发遗嘱消息")
                    self.printer.print_connection_status("成功", "MQTT连接已正常断开，不触发遗嘱消息")
            except Exception as e:
                logger.warning(f"断开MQTT连接时出错: {e}")
                # 尝试强制关闭连接
                if hasattr(self.client, '_connection') and self.client._connection:
                    self.client._connection.close()
            
            self.connected = False
            
    # 重命名force_disconnect为trigger_will_disconnect，使其更加明确
    async def trigger_will_disconnect(self):
        """
        强制断开MQTT连接以触发遗嘱消息
        """
        await self.disconnect(trigger_will=True)
                
    async def _reconnect(self):
        """
        重新连接MQTT代理
        """
        while not self.connected:
            try:
                logger.info("尝试重新连接MQTT代理...")
                self.printer.print_connection_status("重试", "尝试重新连接MQTT代理...")
                if await self.connect():
                    break
                await asyncio.sleep(self.reconnect_interval)
            except Exception as e:
                logger.error(f"重新连接失败: {e}")
                self.printer.print_connection_status("失败", f"重新连接失败: {e}")
                await asyncio.sleep(self.reconnect_interval)
                
    async def _resubscribe_topics(self):
        """
        重新订阅所有主题
        """
        for topic, subscription in self.topic_manager.get_all_subscriptions().items():
            self.client.subscribe(topic, subscription.qos)
            logger.info(f"已重新订阅主题: {topic}")
            self.printer.print_subscription(topic, subscription.qos, "订阅")
            
    async def _send_online_message(self):
        """
        发送上线消息
        """
        try:
            success = await self.connection_handler.send_online_message(
                mac_address=get_mac_address(),
                client_id=self.client_id
            )
            if not success:
                logger.error("发送上线消息失败")
                self.printer.print_connection_status("失败", "发送上线消息失败")
        except Exception as e:
            logger.error(f"发送上线消息时出错: {e}")
            self.printer.print_connection_status("失败", f"发送上线消息时出错: {e}")
            
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
                    self.client.subscribe(topic, qos or self.qos)
                    self.printer.print_subscription(topic, qos or self.qos, "订阅")
                    
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
            self.client.publish(topic, payload, qos=qos or self.qos, retain=retain)
            logger.info(f"消息发布成功: {topic}")
            
            # 打印发送的消息
            self.printer.print_message(topic, payload, "发送")
            
            return True
                
        except Exception as e:
            logger.error(f"发布消息时出错: {e}")
            return False