"""
分析服务工厂
根据配置创建合适的分析服务实例
"""
import os
import sys
import logging
import socket
from typing import Dict, Any, List, Optional, Union

# 添加父级目录到sys.path以允许导入core模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.config import settings
from shared.utils.logger import setup_logger
from services.base_analyzer import BaseAnalyzerService
from services.http.http_analyzer import HTTPAnalyzerService
from services.mqtt.mqtt_analyzer import MQTTAnalyzerService

logger = setup_logger(__name__)

def create_analyzer_service(service_mode: str = None) -> BaseAnalyzerService:
    """
    创建分析服务实例
    
    Args:
        service_mode: 服务模式，'http'或'mqtt'。如果为None，将从配置中读取
        
    Returns:
        BaseAnalyzerService: 分析服务实例
    """
    # 如果未指定服务模式，从配置中读取
    if service_mode is None:
        service_mode = settings.COMMUNICATION.mode.lower()
    else:
        service_mode = service_mode.lower()
    
    logger.info(f"创建分析服务，模式: {service_mode}")
    
    # 根据服务模式创建对应的服务实例
    if service_mode == "mqtt":
        # 使用MQTT配置创建MQTT分析服务
        device_id = settings.MQTT.node_id if settings.MQTT.node_id else socket.gethostname()
        mqtt_config = {
            "host": settings.MQTT.broker_host,
            "port": settings.MQTT.broker_port,
            "username": settings.MQTT.username,
            "password": settings.MQTT.password,
            "topic_prefix": settings.MQTT.topic_prefix
        }
        logger.info(f"MQTT配置: 设备ID={device_id}, 代理={mqtt_config['host']}:{mqtt_config['port']}, 前缀={mqtt_config['topic_prefix']}")
        return MQTTAnalyzerService(device_id=device_id, mqtt_config=mqtt_config)
    else:
        # 默认使用HTTP模式
        return HTTPAnalyzerService()

# 获取当前服务模式
def get_service_mode() -> str:
    """
    获取当前服务模式
    
    Returns:
        str: 服务模式，'http'或'mqtt'
    """
    return settings.SERVICES.mode.lower()

# 为兼容性保留的类定义
class AnalyzerService(BaseAnalyzerService):
    """
    分析服务类（兼容性保留）
    
    注意：建议使用create_analyzer_service()函数创建分析服务实例
    """
    
    def __init__(self, service_mode: str = None):
        """
        初始化分析服务
        
        Args:
            service_mode: 服务模式，'http'或'mqtt'
        """
        logger.warning("AnalyzerService类已废弃，请使用create_analyzer_service()函数")
        
        # 如果未指定服务模式，从配置中读取
        if service_mode is None:
            service_mode = settings.SERVICES.mode.lower()
        else:
            service_mode = service_mode.lower()
        
        self.service_mode = service_mode
        self.service = create_analyzer_service(service_mode)
    
    def __getattr__(self, name):
        """
        转发所有方法调用到实际的服务实例
        
        Args:
            name: 方法名
            
        Returns:
            任何类型: 方法调用的结果
        """
        return getattr(self.service, name)