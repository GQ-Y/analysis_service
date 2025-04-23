"""
MQTT主题管理器
负责管理MQTT主题的订阅和发布
"""
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from shared.utils.tools import get_mac_address

# 配置日志
logger = logging.getLogger(__name__)

# 主题类型常量
TOPIC_TYPE_CONNECTION = 80011      # 连接/上线/遗嘱主题
TOPIC_TYPE_REQUEST_SETTING = 80012  # 命令接收
TOPIC_TYPE_CONFIG_REPLY = 80013    # 命令回复
TOPIC_TYPE_STATUS = 80014          # 状态上报
TOPIC_TYPE_RESULT = 80015          # 分析结果
TOPIC_TYPE_BROADCAST = 80018       # 系统广播

# 节点相关主题类型
NODE_TOPIC_TYPES = [
    TOPIC_TYPE_REQUEST_SETTING,
    TOPIC_TYPE_STATUS,
    TOPIC_TYPE_RESULT
]

@dataclass
class TopicSubscription:
    """
    主题订阅信息
    """
    topic: str
    qos: int
    is_wildcard: bool
    subscribe_time: datetime
    last_message_time: Optional[datetime] = None

class MQTTTopicManager:
    """
    MQTT主题管理器
    负责管理MQTT主题的订阅和发布
    """
    
    def __init__(self, topic_prefix: str = "/meek"):
        """
        初始化MQTT主题管理器
        
        Args:
            topic_prefix: 主题前缀
        """
        self.subscriptions: Dict[str, TopicSubscription] = {}
        self.mac_address = get_mac_address()
        self.topic_prefix = topic_prefix
        logger.info("MQTT主题管理器已初始化")
        
    def set_mac_address(self, mac_address: str):
        """
        设置MAC地址
        
        Args:
            mac_address: MAC地址
        """
        self.mac_address = mac_address
        logger.info(f"已设置MAC地址: {mac_address}")
        
    def format_topic(self, topic_type: int, **kwargs) -> str:
        """
        格式化主题名称
        
        Args:
            topic_type: 主题类型
            **kwargs: 主题参数
            
        Returns:
            str: 格式化后的主题名称
        """
        # 确保mac_address参数存在
        if "mac_address" not in kwargs:
            kwargs["mac_address"] = self.mac_address
            
        # 根据主题类型生成主题
        if topic_type == TOPIC_TYPE_CONNECTION:
            return f"{self.topic_prefix}/connection"
        elif topic_type == TOPIC_TYPE_REQUEST_SETTING:
            return f"{self.topic_prefix}/{kwargs['mac_address']}/request_setting"
        elif topic_type == TOPIC_TYPE_CONFIG_REPLY:
            return f"{self.topic_prefix}/device_config_reply"
        elif topic_type == TOPIC_TYPE_STATUS:
            return f"{self.topic_prefix}/{kwargs['mac_address']}/status"
        elif topic_type == TOPIC_TYPE_RESULT:
            return f"{self.topic_prefix}/{kwargs['mac_address']}/result"
        elif topic_type == TOPIC_TYPE_BROADCAST:
            return f"{self.topic_prefix}/system/broadcast"
        else:
            logger.error(f"未知的主题类型: {topic_type}")
            return None
        
    def get_node_topics(self) -> List[str]:
        """
        获取节点相关的主题列表
        
        Returns:
            List[str]: 节点主题列表
        """
        topics = []
        for topic_type in NODE_TOPIC_TYPES:
            topic = self.format_topic(topic_type)
            if topic:
                topics.append(topic)
        return topics
        
    def add_subscription(self, topic: str, qos: int = 0) -> bool:
        """
        添加主题订阅
        
        Args:
            topic: 主题名称
            qos: 服务质量等级
            
        Returns:
            bool: 是否添加成功
        """
        try:
            # 如果主题包含mac_address变量，替换为实际值
            if "{mac_address}" in topic:
                topic = topic.format(mac_address=self.mac_address)
                
            is_wildcard = "*" in topic or "+" in topic
            subscription = TopicSubscription(
                topic=topic,
                qos=qos,
                is_wildcard=is_wildcard,
                subscribe_time=datetime.now()
            )
            self.subscriptions[topic] = subscription
            logger.info(f"已添加主题订阅: {topic}")
            return True
        except Exception as e:
            logger.error(f"添加主题订阅失败: {e}")
            return False
            
    def remove_subscription(self, topic: str) -> bool:
        """
        移除主题订阅
        
        Args:
            topic: 主题名称
            
        Returns:
            bool: 是否移除成功
        """
        # 如果主题包含mac_address变量，替换为实际值
        if "{mac_address}" in topic:
            topic = topic.format(mac_address=self.mac_address)
            
        if topic in self.subscriptions:
            del self.subscriptions[topic]
            logger.info(f"已移除主题订阅: {topic}")
            return True
        return False
        
    def update_last_message_time(self, topic: str) -> bool:
        """
        更新主题最后消息时间
        
        Args:
            topic: 主题名称
            
        Returns:
            bool: 是否更新成功
        """
        # 如果主题包含mac_address变量，替换为实际值
        if "{mac_address}" in topic:
            topic = topic.format(mac_address=self.mac_address)
            
        if topic in self.subscriptions:
            self.subscriptions[topic].last_message_time = datetime.now()
            return True
        return False
        
    def get_subscription(self, topic: str) -> Optional[TopicSubscription]:
        """
        获取主题订阅信息
        
        Args:
            topic: 主题名称
            
        Returns:
            Optional[TopicSubscription]: 主题订阅信息
        """
        # 如果主题包含mac_address变量，替换为实际值
        if "{mac_address}" in topic:
            topic = topic.format(mac_address=self.mac_address)
            
        return self.subscriptions.get(topic)
        
    def get_all_subscriptions(self) -> Dict[str, TopicSubscription]:
        """
        获取所有主题订阅信息
        
        Returns:
            Dict[str, TopicSubscription]: 所有主题订阅信息
        """
        return self.subscriptions 