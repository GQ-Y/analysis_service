"""
MQTT消息中间件
用于拦截和处理所有MQTT消息
"""
import json
import logging
from typing import Dict, Any, Callable, List, Awaitable, Optional
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)

class MQTTMiddleware:
    """
    MQTT中间件类，负责消息的预处理和后处理流程
    """
    
    def __init__(self):
        """
        初始化MQTT中间件
        """
        # 预处理器列表
        self.pre_processors: List[Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = []
        
        # 后处理器列表
        self.post_processors: List[Callable[[str, Dict[str, Any], Any], Awaitable[Any]]] = []
        
        logger.info("MQTT中间件已初始化")
    
    def add_pre_processor(self, processor: Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]):
        """
        添加预处理器
        
        Args:
            processor: 预处理器函数，接收topic和data，返回处理后的data
        """
        self.pre_processors.append(processor)
        logger.info(f"已添加MQTT预处理器: {processor.__name__ if hasattr(processor, '__name__') else str(processor)}")
    
    def add_post_processor(self, processor: Callable[[str, Dict[str, Any], Any], Awaitable[Any]]):
        """
        添加后处理器
        
        Args:
            processor: 后处理器函数，接收topic、data和处理结果，返回最终处理结果
        """
        self.post_processors.append(processor)
        logger.info(f"已添加MQTT后处理器: {processor.__name__ if hasattr(processor, '__name__') else str(processor)}")
    
    async def process(self, topic: str, data: Dict[str, Any], handler: Callable[[str, Dict[str, Any]], Awaitable[Any]]) -> Any:
        """
        处理消息
        
        Args:
            topic: 消息主题
            data: 消息数据
            handler: 消息处理函数
            
        Returns:
            Any: 处理结果
        """
        # 应用所有预处理器
        processed_data = data
        for processor in self.pre_processors:
            try:
                processed_data = await processor(topic, processed_data)
            except Exception as e:
                logger.error(f"执行MQTT预处理器时出错: {e}")
        
        # 调用处理函数
        result = None
        try:
            result = await handler(topic, processed_data)
        except Exception as e:
            logger.error(f"执行MQTT消息处理器时出错: {e}")
            import traceback
            logger.error(f"错误详情:\n{traceback.format_exc()}")
        
        # 应用所有后处理器
        for processor in self.post_processors:
            try:
                result = await processor(topic, processed_data, result)
            except Exception as e:
                logger.error(f"执行MQTT后处理器时出错: {e}")
        
        return result

# 默认日志记录中间件
async def log_mqtt_message(topic: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    记录MQTT消息日志的预处理器
    
    Args:
        topic: 消息主题
        data: 消息数据
        
    Returns:
        Dict[str, Any]: 原始消息数据
    """
    # 过滤掉可能的敏感信息
    safe_data = {k: v for k, v in data.items() if not any(sens in k.lower() for sens in ['password', 'secret', 'token', 'key'])}
    
    # 如果数据太大，只记录一部分
    data_str = json.dumps(safe_data, ensure_ascii=False)
    if len(data_str) > 1000:
        logger.info(f"收到MQTT消息: 主题={topic}, 数据长度={len(data_str)}字节, 数据预览={data_str[:997]}...")
    else:
        logger.info(f"收到MQTT消息: 主题={topic}, 数据={data_str}")
    
    return data

# 全局MQTT中间件实例
_mqtt_middleware = None

def get_mqtt_middleware() -> MQTTMiddleware:
    """
    获取全局MQTT中间件实例
    
    Returns:
        MQTTMiddleware: MQTT中间件实例
    """
    global _mqtt_middleware
    if _mqtt_middleware is None:
        _mqtt_middleware = MQTTMiddleware()
    return _mqtt_middleware 