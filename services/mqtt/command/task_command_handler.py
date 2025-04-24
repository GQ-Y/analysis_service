"""
任务命令处理器
处理任务相关的命令，如启动、停止任务等
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from core.task_manager import TaskManager
from core.models import TaskInfo, SubTaskInfo, SourceInfo, ModelConfig, AnalysisConfig, ResultConfig
from ..handler.message_types import (
    MESSAGE_TYPE_REQUEST_SETTING,
    REQUEST_TYPE_TASK_CMD,
    TASK_CMD_START,
    TASK_CMD_STOP
)

# 配置日志
logger = logging.getLogger(__name__)

class TaskCommandHandler:
    """
    任务命令处理器
    处理任务相关的命令
    """
    
    def __init__(self):
        """
        初始化任务命令处理器
        """
        self.task_manager = TaskManager()
        logger.info("任务命令处理器已初始化")
        
    async def handle_command(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理任务命令
        
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
                logger.warning(f"任务命令缺少cmd_type字段: {payload}")
                return None
                
            # 根据命令类型处理
            if cmd_type == TASK_CMD_START:
                return await self._handle_start_task(data)
            elif cmd_type == TASK_CMD_STOP:
                return await self._handle_stop_task(data)
            else:
                logger.warning(f"未知的任务命令类型: {cmd_type}")
                return None
                
        except Exception as e:
            logger.error(f"处理任务命令时出错: {e}")
            return None
            
    async def _handle_start_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理启动任务命令
        
        Args:
            data: 命令数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 获取任务信息
            task_info_data = data.get("task_info")
            subtask_info_data = data.get("subtask_info")
            source_data = data.get("source")
            model_config_data = data.get("model_config")
            analysis_config_data = data.get("analysis_config")
            result_config_data = data.get("result_config")
            
            if not all([task_info_data, subtask_info_data, source_data, 
                       model_config_data, analysis_config_data, result_config_data]):
                logger.warning(f"启动任务命令缺少必要参数: {data}")
                return self._create_response(False, "缺少必要参数")
            
            # 创建任务配置对象
            task_info = TaskInfo(**task_info_data)
            subtask_info = SubTaskInfo(**subtask_info_data)
            source_info = SourceInfo(**source_data)
            model_config = ModelConfig(**model_config_data)
            analysis_config = AnalysisConfig(**analysis_config_data)
            result_config = ResultConfig(**result_config_data)
            
            # 启动任务
            success = await self.task_manager.start_task(
                task_info=task_info,
                subtask_info=subtask_info,
                source_info=source_info,
                model_config=model_config,
                analysis_config=analysis_config,
                result_config=result_config
            )
            
            if success:
                logger.info(f"任务启动成功: {task_info.task_id}")
                return self._create_response(True, "任务启动成功")
            else:
                logger.error(f"任务启动失败: {task_info.task_id}")
                return self._create_response(False, "任务启动失败")
            
        except Exception as e:
            logger.error(f"启动任务失败: {e}")
            import traceback
            logger.error(f"错误详情:\n{traceback.format_exc()}")
            return self._create_response(False, f"启动任务失败: {str(e)}")
            
    async def _handle_stop_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理停止任务命令
        
        Args:
            data: 命令数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 获取任务ID
            task_id = data.get("task_id")
            if not task_id:
                logger.warning(f"停止任务命令缺少task_id: {data}")
                return self._create_response(False, "缺少任务ID")
            
            # 停止任务
            success = await self.task_manager.stop_task(task_id)
            
            if success:
                logger.info(f"任务停止成功: {task_id}")
                return self._create_response(True, "任务停止成功")
            else:
                logger.error(f"任务停止失败: {task_id}")
                return self._create_response(False, "任务停止失败")
            
        except Exception as e:
            logger.error(f"停止任务失败: {e}")
            import traceback
            logger.error(f"错误详情:\n{traceback.format_exc()}")
            return self._create_response(False, f"停止任务失败: {str(e)}")
            
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
            "request_type": REQUEST_TYPE_TASK_CMD,
            "status": "success" if success else "error",
            "message": message,
            "timestamp": int(datetime.now().timestamp())
        }

# 全局任务命令处理器实例
_task_command_handler = None

def get_task_command_handler():
    """
    获取全局任务命令处理器实例
    
    Returns:
        TaskCommandHandler: 任务命令处理器实例
    """
    global _task_command_handler
    if _task_command_handler is None:
        _task_command_handler = TaskCommandHandler()
    return _task_command_handler 