"""
节点命令处理器
处理节点相关的命令，如同步时间等
"""
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

from ..handler.message_types import (
    MESSAGE_TYPE_REQUEST_SETTING,
    REQUEST_TYPE_NODE_CMD,
    NODE_CMD_SYNC_TIME
)

# 配置日志
logger = logging.getLogger(__name__)

class NodeCommandHandler:
    """
    节点命令处理器
    处理节点相关的命令
    """
    
    def __init__(self):
        """
        初始化节点命令处理器
        """
        logger.info("节点命令处理器已初始化")
        
    async def handle_command(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理节点命令
        
        Args:
            payload: 命令消息内容
            
        Returns:
            Optional[Dict[str, Any]]: 处理结果
        """
        try:
            # 获取命令类型
            data = payload.get("data", {})
            cmd_type = data.get("cmd_type")
            
            if not cmd_type:
                logger.warning(f"节点命令缺少cmd_type字段: {payload}")
                return None
                
            # 根据命令类型处理
            if cmd_type == NODE_CMD_SYNC_TIME:
                return await self._handle_sync_time(data)
            else:
                logger.warning(f"未知的节点命令类型: {cmd_type}")
                return None
                
        except Exception as e:
            logger.error(f"处理节点命令时出错: {e}")
            return None
            
    async def _handle_sync_time(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理同步时间命令
        
        Args:
            data: 命令数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 获取时间戳
            timestamp = data.get("timestamp")
            if not timestamp:
                logger.warning("同步时间命令缺少timestamp字段")
                return self._create_response(False, "缺少时间戳")
                
            # 设置系统时间
            datetime_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            os.system(f'date -s "{datetime_str}"')
            
            logger.info(f"系统时间已同步: {datetime_str}")
            return self._create_response(True, "时间同步成功")
            
        except Exception as e:
            logger.error(f"同步时间失败: {e}")
            return self._create_response(False, f"同步时间失败: {str(e)}")
            
    def _create_response(self, success: bool, message: str) -> Dict[str, Any]:
        """
        创建响应消息
        
        Args:
            success: 是否成功
            message: 响应消息
            
        Returns:
            Dict[str, Any]: 响应消息
        """
        return {
            "message_type": MESSAGE_TYPE_REQUEST_SETTING,
            "request_type": REQUEST_TYPE_NODE_CMD,
            "status": "success" if success else "error",
            "message": message,
            "timestamp": int(datetime.now().timestamp())
        }

# 全局节点命令处理器实例
_node_command_handler = None

def get_node_command_handler():
    """
    获取全局节点命令处理器实例
    
    Returns:
        NodeCommandHandler: 节点命令处理器实例
    """
    global _node_command_handler
    if _node_command_handler is None:
        _node_command_handler = NodeCommandHandler()
    return _node_command_handler 