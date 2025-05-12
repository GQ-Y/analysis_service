"""
服务工厂模块
根据配置创建不同的服务实例
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional, Union

from core.config import settings
from shared.utils.logger import setup_logger
from services.base_analyzer import BaseAnalyzerService

logger = setup_logger(__name__)

def create_analyzer_service() -> BaseAnalyzerService:
    """
    创建分析服务实例
    
    根据配置创建HTTP或MQTT模式的分析服务实例
    
    Returns:
        BaseAnalyzerService: 分析服务实例
    """
    # 获取服务模式
    service_mode = get_service_mode()
    
    if service_mode == "http":
        logger.info("创建HTTP模式分析服务")
        from services.http.http_analyzer import HTTPAnalyzerService
        return HTTPAnalyzerService()
    elif service_mode == "mqtt":
        logger.info("创建MQTT模式分析服务")
        # 未来支持MQTT模式
        # from services.mqtt.mqtt_analyzer import MQTTAnalyzerService
        # return MQTTAnalyzerService()
        raise NotImplementedError("MQTT模式尚未实现")
    else:
        logger.warning(f"未知的服务模式: {service_mode}，使用默认HTTP模式")
        from services.http.http_analyzer import HTTPAnalyzerService
        return HTTPAnalyzerService()

def get_service_mode() -> str:
    """
    获取当前服务模式
    
    从配置中获取服务模式，默认为HTTP模式
    
    Returns:
        str: 服务模式，'http' 或 'mqtt'
    """
    # 从环境变量或配置文件中获取服务模式
    service_mode = os.environ.get("SERVICE_MODE", "http").lower()
    
    # 验证服务模式
    if service_mode not in ["http", "mqtt"]:
        logger.warning(f"无效的服务模式: {service_mode}，使用默认HTTP模式")
        service_mode = "http"
        
    return service_mode

def create_task_service():
    """
    创建任务服务实例
    
    根据服务模式创建相应的任务服务实例
    
    Returns:
        任务服务实例
    """
    # 获取服务模式
    service_mode = get_service_mode()
    
    if service_mode == "http":
        logger.info("创建HTTP模式任务服务")
        from services.http.task_service import TaskService
        return TaskService()
    elif service_mode == "mqtt":
        logger.info("创建MQTT模式任务服务")
        # 未来支持MQTT模式
        # from services.mqtt.task_service import TaskService
        # return TaskService()
        raise NotImplementedError("MQTT模式尚未实现")
    else:
        logger.warning(f"未知的服务模式: {service_mode}，使用默认HTTP模式")
        from services.http.task_service import TaskService
        return TaskService()

def create_analysis_service():
    """
    创建分析服务实例
    
    Returns:
        分析服务实例
    """
    from services.analysis_service import AnalysisService
    return AnalysisService()
