"""
MQTT连接处理器
处理连接、上线和遗嘱消息
"""
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from core.config import settings
from ..mqtt_handler import BaseMQTTHandler
from ..mqtt_topic_manager import TOPIC_TYPE_CONNECTION
from shared.utils.tools import get_mac_address, get_hostname,get_local_ip
from ..mqtt_printer import MQTTPrinter

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
        self.mqtt_manager = None
        self.printer = MQTTPrinter()
        logger.info("连接处理器已初始化")
        
    def set_mqtt_manager(self, mqtt_manager):
        """
        设置MQTT管理器
        
        Args:
            mqtt_manager: MQTT管理器实例
        """
        self.mqtt_manager = mqtt_manager
        
    async def send_online_message(self, mac_address: str, client_id: str) -> bool:
        """
        发送上线消息
        
        Args:
            mac_address: MAC地址
            client_id: 客户端ID
            
        Returns:
            bool: 是否发送成功
        """
        try:
            # 检测系统计算能力
            compute_type = 0  # 默认为CPU
            try:
                import torch
                if torch.cuda.is_available():
                    compute_type = 2  # CPU和GPU都存在
                else:
                    compute_type = 0  # 只有CPU
            except ImportError:
                compute_type = 0  # 没有安装PyTorch，默认为CPU
                
            # 创建上线消息
            message = {
                "message_type": TOPIC_TYPE_CONNECTION,  # 连接消息
                "mac_address": mac_address,
                "client_id": client_id,
                "status": 1,
                "message": "上线",
                "timestamp": datetime.now().isoformat(),
                "compute_type": compute_type,
                "ip": get_local_ip(),
                "port": settings.SERVICES_PORT,
                "hostname": get_hostname(),
                "version": settings.VERSION,
                "timestamp": datetime.now().isoformat()
            }
            
            # 获取主题
            topic = self.mqtt_manager.topic_manager.format_topic(TOPIC_TYPE_CONNECTION)  # 连接主题
            
            # 发送消息
            success = await self.mqtt_manager.publish(topic, message, retain=True)
            
            # 打印上线消息
            if success:
                self.printer.print_message(topic, message, "发送")
                self.printer.print_connection_status("成功", "发送上线消息成功")
            else:
                self.printer.print_connection_status("失败", "发送上线消息失败")
                
            return success
            
        except Exception as e:
            logger.error(f"发送上线消息时出错: {e}")
            self.printer.print_connection_status("失败", f"发送上线消息时出错: {e}")
            return False
            
    def create_will_message(self, mac_address: str, client_id: str, reason: int = 403, resources: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        创建遗嘱消息
        
        Args:
            mac_address: MAC地址
            client_id: 客户端ID
            reason: 原因代码
            resources: 资源信息
            
        Returns:
            Dict[str, Any]: 遗嘱消息
        """
        try:
            message = {
                "message_type": TOPIC_TYPE_CONNECTION,  # 连接消息
                "mac_address": mac_address,
                "client_id": client_id,
                "status": reason,
                "message": "离线",
                "timestamp": datetime.now().isoformat(),
                "resources": resources or {
                    "task_count": 0,
                    "image_task_count": 0,
                    "video_task_count": 0,
                    "stream_task_count": 0
                }
            }
            
            return message
            
        except Exception as e:
            logger.error(f"创建遗嘱消息时出错: {e}")
            return {}
        
    async def handle_message(self, topic: str, message: Dict[str, Any]) -> None:
        """
        处理消息
        
        Args:
            topic: 消息主题
            message: 消息内容
        """
        try:
            # 打印接收到的消息
            self.printer.print_message(topic, message, "接收")
            
            # 处理消息
            if message.get("type") == TOPIC_TYPE_CONNECTION:  # 连接消息
                await self._handle_connection_message(message)
            elif message.get("type") == TOPIC_TYPE_REQUEST_SETTING:  # 请求设置消息
                await self._handle_request_setting_message(message)
            elif message.get("type") == TOPIC_TYPE_CONFIG_REPLY:  # 配置回复消息
                await self._handle_config_reply_message(message)
            else:
                logger.warning(f"未知的消息类型: {message.get('type')}")
                
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            
    async def _handle_connection_message(self, message: Dict[str, Any]) -> None:
        """
        处理连接消息
        
        Args:
            message: 消息内容
        """
        try:
            # 获取MAC地址
            mac_address = message.get("mac_address")
            if not mac_address:
                logger.error("消息中缺少MAC地址")
                return
                
            # 更新连接状态
            self._update_connection_status(mac_address, message)
            
            # 打印连接状态
            self.printer.print_connection_status(
                "成功" if message.get("status") == 200 else "失败",
                f"处理连接消息: {message.get('message', '')}"
            )
            
        except Exception as e:
            logger.error(f"处理连接消息时出错: {e}")
            
    async def _handle_request_setting_message(self, message: Dict[str, Any]) -> None:
        """
        处理请求设置消息
        
        Args:
            message: 消息内容
        """
        try:
            # 获取MAC地址
            mac_address = message.get("mac_address")
            if not mac_address:
                logger.error("消息中缺少MAC地址")
                return
                
            # 处理请求设置
            self._process_request_setting(mac_address, message)
            
            # 打印请求设置状态
            self.printer.print_connection_status(
                "成功" if message.get("status") == 200 else "失败",
                f"处理请求设置消息: {message.get('message', '')}"
            )
            
        except Exception as e:
            logger.error(f"处理请求设置消息时出错: {e}")
            
    async def _handle_config_reply_message(self, message: Dict[str, Any]) -> None:
        """
        处理配置回复消息
        
        Args:
            message: 消息内容
        """
        try:
            # 获取MAC地址
            mac_address = message.get("mac_address")
            if not mac_address:
                logger.error("消息中缺少MAC地址")
                return
                
            # 处理配置回复
            self._process_config_reply(mac_address, message)
            
            # 打印配置回复状态
            self.printer.print_connection_status(
                "成功" if message.get("status") == 200 else "失败",
                f"处理配置回复消息: {message.get('message', '')}"
            )
            
        except Exception as e:
            logger.error(f"处理配置回复消息时出错: {e}")

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