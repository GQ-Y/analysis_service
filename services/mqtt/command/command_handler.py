"""
MQTT命令处理器
处理来自服务端的命令消息
"""
import logging
from typing import Dict, Any, Optional

from ..mqtt_handler import BaseMQTTHandler, MESSAGE_TYPE_COMMAND

# 配置日志
logger = logging.getLogger(__name__)

class CommandHandler(BaseMQTTHandler):
    """
    MQTT命令处理器
    处理来自服务端的命令消息
    """
    
    def __init__(self):
        """
        初始化命令处理器
        """
        super().__init__()
        logger.info("MQTT命令处理器已初始化")
        
    async def handle_command(self, topic: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理命令消息
        
        Args:
            topic: 消息主题
            payload: 消息内容
            
        Returns:
            Optional[Dict[str, Any]]: 处理结果
        """
        try:
            # 获取命令类型
            command = payload.get("command")
            if not command:
                logger.warning(f"消息缺少command字段: {payload}")
                return None
                
            # 根据命令类型处理
            if command == "start_task":
                return await self._handle_start_task(payload)
            elif command == "stop_task":
                return await self._handle_stop_task(payload)
            else:
                logger.warning(f"未知的命令类型: {command}")
                return None
                
        except Exception as e:
            logger.error(f"处理命令时出错: {e}")
            return None
            
    async def _handle_start_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理启动任务命令
        
        Args:
            payload: 命令消息内容
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 获取任务参数
            task_id = payload.get("task_id")
            task_type = payload.get("task_type")
            params = payload.get("params", {})
            
            if not task_id or not task_type:
                logger.warning(f"启动任务命令缺少必要参数: {payload}")
                return None
                
            # TODO: 实现任务启动逻辑
            
            # 返回处理结果
            return {
                "message_type": MESSAGE_TYPE_COMMAND,
                "command": "start_task",
                "task_id": task_id,
                "status": "success",
                "timestamp": self.get_timestamp()
            }
            
        except Exception as e:
            logger.error(f"处理启动任务命令时出错: {e}")
            return None
            
    async def _handle_stop_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理停止任务命令
        
        Args:
            payload: 命令消息内容
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 获取任务ID
            task_id = payload.get("task_id")
            if not task_id:
                logger.warning(f"停止任务命令缺少task_id: {payload}")
                return None
                
            # TODO: 实现任务停止逻辑
            
            # 返回处理结果
            return {
                "message_type": MESSAGE_TYPE_COMMAND,
                "command": "stop_task",
                "task_id": task_id,
                "status": "success",
                "timestamp": self.get_timestamp()
            }
            
        except Exception as e:
            logger.error(f"处理停止任务命令时出错: {e}")
            return None
            
    async def send_reply(self, reply: Dict[str, Any]) -> bool:
        """
        发送命令回复
        
        Args:
            reply: 回复消息内容
            
        Returns:
            bool: 是否发送成功
        """
        try:
            topic = self.mqtt_manager.topic_manager.format_topic("config_reply")
            if not topic:
                logger.error("获取回复主题失败")
                return False
                
            return await self.publish(topic, reply, qos=1)
            
        except Exception as e:
            logger.error(f"发送命令回复失败: {e}")
            return False

# 全局命令处理器实例
_command_handler = None

def get_command_handler() -> CommandHandler:
    """
    获取全局命令处理器实例
    
    Returns:
        CommandHandler: 命令处理器实例
    """
    global _command_handler
    if _command_handler is None:
        _command_handler = CommandHandler()
    return _command_handler 