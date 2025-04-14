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
        service_mode = settings.COMMUNICATION_MODE.lower()
    else:
        service_mode = service_mode.lower()
    
    logger.info(f"创建分析服务，模式: {service_mode}")
    
    # 根据服务模式创建对应的服务实例
    if service_mode == "mqtt":
        # 使用MQTT配置创建MQTT分析服务
        device_id = settings.MQTT_NODE_ID if settings.MQTT_NODE_ID else socket.gethostname()
        mqtt_config = {
            "host": settings.MQTT_BROKER_HOST,
            "port": settings.MQTT_BROKER_PORT,
            "username": settings.MQTT_USERNAME,
            "password": settings.MQTT_PASSWORD,
            "topic_prefix": settings.MQTT_TOPIC_PREFIX,
            "qos": settings.MQTT_QOS,
            "keepalive": settings.MQTT_KEEPALIVE,
            "reconnect_interval": settings.MQTT_RECONNECT_INTERVAL,
            "node_id": settings.MQTT_NODE_ID,
            "service_type": settings.MQTT_SERVICE_TYPE
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
    return settings.COMMUNICATION_MODE.lower()

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
            service_mode = settings.COMMUNICATION_MODE.lower()
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

class Analyzer:
    def __init__(self):
        """初始化分析器"""
        self.communication_mode = settings.COMMUNICATION_MODE
        self.mqtt_client = None
        self.task_queue = None
        self.task_processor = None
        self.task_manager = None
        self.detector = None
        self.segmentor = None
        self.http_analyzer = None
        self.mqtt_analyzer = None

    async def initialize(self):
        """初始化分析服务"""
        try:
            # 初始化任务队列
            self.task_queue = TaskQueue()
            
            # 初始化任务处理器
            self.task_processor = TaskProcessor()
            
            # 初始化任务管理器
            self.task_manager = TaskManager()
            
            # 初始化检测器和分割器
            self.detector = YOLODetector()
            self.segmentor = YOLOSegmentor()
            
            # 根据通信模式初始化对应的分析器
            if self.communication_mode == "mqtt":
                # 初始化MQTT客户端
                self.mqtt_client = MQTTClient()
                await self.mqtt_client.connect()
                
                # 初始化MQTT分析器
                self.mqtt_analyzer = MQTTAnalyzer(
                    mqtt_client=self.mqtt_client,
                    task_queue=self.task_queue,
                    task_processor=self.task_processor,
                    task_manager=self.task_manager,
                    detector=self.detector,
                    segmentor=self.segmentor
                )
                
            elif self.communication_mode == "http":
                # 初始化HTTP分析器
                self.http_analyzer = HTTPAnalyzer(
                    task_queue=self.task_queue,
                    task_processor=self.task_processor,
                    task_manager=self.task_manager,
                    detector=self.detector,
                    segmentor=self.segmentor
                )
            
            logger.info(f"分析服务初始化完成，通信模式: {self.communication_mode}")
            return True
            
        except Exception as e:
            logger.error(f"初始化分析服务失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False