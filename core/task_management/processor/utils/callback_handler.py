"""
回调处理器
处理分析结果的回调通知
"""
import aiohttp
from typing import Dict, Any
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class CallbackHandler:
    """回调处理器"""
    
    async def handle_callback(
        self,
        task_id: str,
        result: Dict[str, Any],
        task_config: Dict[str, Any]
    ):
        """
        处理回调
        
        Args:
            task_id: 任务ID
            result: 分析结果
            task_config: 任务配置
        """
        try:
            # 检查是否启用回调
            callback_config = task_config.get("subtask", {}).get("callback", {})
            if not callback_config.get("enabled", False):
                return
                
            # 获取回调URL
            callback_url = callback_config.get("url")
            if not callback_url:
                logger.warning(f"未配置回调URL: {task_id}")
                return
                
            # 发送回调请求
            async with aiohttp.ClientSession() as session:
                async with session.post(callback_url, json=result) as response:
                    if response.status != 200:
                        logger.error(f"回调请求失败: {response.status}")
                    else:
                        logger.info(f"回调请求成功: {task_id}")
                        
        except Exception as e:
            logger.error(f"处理回调失败: {str(e)}")
            
    async def send_error_callback(
        self,
        task_id: str,
        error: str,
        task_config: Dict[str, Any]
    ):
        """
        发送错误回调
        
        Args:
            task_id: 任务ID
            error: 错误信息
            task_config: 任务配置
        """
        try:
            callback_config = task_config.get("subtask", {}).get("callback", {})
            if not callback_config.get("enabled", False):
                return
                
            callback_url = callback_config.get("url")
            if not callback_url:
                return
                
            error_result = {
                "task_id": task_id,
                "status": "-1",  # 错误状态
                "error": error
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(callback_url, json=error_result) as response:
                    if response.status != 200:
                        logger.error(f"错误回调请求失败: {response.status}")
                    else:
                        logger.info(f"错误回调请求成功: {task_id}")
                        
        except Exception as e:
            logger.error(f"发送错误回调失败: {str(e)}")
            
    async def send_status_callback(
        self,
        task_id: str,
        status: str,
        task_config: Dict[str, Any]
    ):
        """
        发送状态回调
        
        Args:
            task_id: 任务ID
            status: 状态信息
            task_config: 任务配置
        """
        try:
            callback_config = task_config.get("subtask", {}).get("callback", {})
            if not callback_config.get("enabled", False):
                return
                
            callback_url = callback_config.get("url")
            if not callback_url:
                return
                
            status_result = {
                "task_id": task_id,
                "status": status
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(callback_url, json=status_result) as response:
                    if response.status != 200:
                        logger.error(f"状态回调请求失败: {response.status}")
                    else:
                        logger.info(f"状态回调请求成功: {task_id}")
                        
        except Exception as e:
            logger.error(f"发送状态回调失败: {str(e)}") 