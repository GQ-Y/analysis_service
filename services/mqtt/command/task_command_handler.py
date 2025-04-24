"""
任务命令处理器
处理任务相关的命令，如启动、停止任务等
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from core import TaskManager, TaskStatus, AnalysisType, RoiType
from shared.utils.tools import get_mac_address
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
        
    def get_mac_address(self) -> str:
        """
        获取MAC地址
        
        Returns:
            str: MAC地址
        """
        return get_mac_address()
        
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
            # 1. 提取任务相关信息
            task_info = data.get("task_info")
            subtask_info = data.get("subtask_info")
            source_info = data.get("source")
            model_config = data.get("model_config")
            analysis_config = data.get("analysis_config")
            result_config = data.get("result_config")
            
            if not all([task_info, subtask_info, source_info, model_config, analysis_config, result_config]):
                logger.warning(f"启动任务命令缺少必要参数: {data}")
                return self._create_response(False, "缺少必要参数", data)
            
            # 2. 构建任务配置
            task_config = {
                # 基本信息
                "task_id": task_info["task_id"],
                "name": task_info["name"],
                "analysis_interval": task_info.get("analysis_interval", 1),
                
                # 子任务信息
                "subtask": {
                    "id": subtask_info["subtask_id"],
                    "analysis_task_id": subtask_info["analysis_task_id"],
                    "name": subtask_info["name"],
                    "type": subtask_info["analysis_type"],
                    "callback": {
                        "enabled": subtask_info.get("enable_callback", False),
                        "url": subtask_info.get("callback_url", "")
                    }
                },
                
                # 数据源配置
                "source": {
                    "type": source_info["type"],
                    "urls": source_info["urls"],
                    "stream_info": source_info.get("stream_info", {})
                },
                
                # 模型配置
                "model": {
                    "id": model_config["model_id"],
                    "code": model_config["model_code"],
                    "type": model_config["model_type"],
                    "version": model_config["model_version"],
                    "device": model_config["device"],
                    "batch_size": model_config.get("batch_size", 1),
                    "input_size": model_config.get("input_size", [640, 640])
                },
                
                # 分析配置
                "analysis": {
                    "confidence": analysis_config["confidence"],
                    "iou": analysis_config["iou"],
                    "classes": analysis_config["classes"],
                    "track_config": analysis_config.get("track_config", {}),
                    "count_config": analysis_config.get("count_config", {})
                },
                
                # ROI配置
                "roi": {
                    "type": subtask_info.get("roi_type", RoiType.NONE),
                    "config": subtask_info.get("roi_config", {})
                },
                
                # 结果配置
                "result": {
                    "save_result": result_config["save_result"],
                    "save_images": result_config["save_images"],
                    "return_base64": result_config.get("return_base64", False),
                    "callback_topic": result_config.get("callback_topic", ""),
                    "result_type": result_config.get("result_type", ["detection"]),
                    "image_format": result_config.get("image_format", {
                        "format": "jpg",
                        "quality": 95,
                        "max_size": 1920
                    }),
                    "storage": result_config.get("storage_config", {
                        "save_path": "/data/results",
                        "file_pattern": "{task_id}/{date}/{time}_{frame_id}.jpg"
                    })
                },
                
                # 任务状态
                "status": TaskStatus.WAITING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # 3. 添加任务到管理器
            success = self.task_manager.add_task(task_info["task_id"], task_config)
            if not success:
                logger.error(f"添加任务失败: {task_info['task_id']}")
                return self._create_response(False, "添加任务失败", data)
            
            # 4. 启动任务处理
            try:
                # 更新任务状态为处理中
                self.task_manager.update_task_status(
                    task_info["task_id"],
                    TaskStatus.PROCESSING
                )
                
                # 发送启动中响应
                start_response = self._create_response(True, "启动任务中", data)
                if data.get("confirmation_topic"):
                    await self.mqtt_manager.publish(
                        data["confirmation_topic"],
                        start_response
                    )
                
                # 根据任务类型启动相应的处理器
                if source_info["type"] == "stream":
                    logger.info(f"启动流分析任务: {task_info['task_id']}")
                    # 异步启动流处理
                    success = await self.task_manager.start_stream_task(
                        task_info["task_id"],
                        task_config
                    )
                else:
                    logger.error(f"不支持的数据源类型: {source_info['type']}")
                    return self._create_response(False, f"不支持的数据源类型: {source_info['type']}", data)
                
                if success:
                    logger.info(f"任务启动成功: {task_info['task_id']}")
                    return self._create_response(True, "启动任务成功", data)
                else:
                    logger.error(f"任务启动失败: {task_info['task_id']}")
                    return self._create_response(False, "启动任务失败", data)
                
            except Exception as e:
                logger.error(f"启动任务处理时出错: {e}")
                # 更新任务状态为失败
                self.task_manager.update_task_status(
                    task_info["task_id"],
                    TaskStatus.FAILED,
                    error=str(e)
                )
                return self._create_response(False, f"启动任务处理失败: {str(e)}", data)
            
        except Exception as e:
            logger.error(f"处理启动任务命令失败: {e}")
            import traceback
            logger.error(f"错误详情:\n{traceback.format_exc()}")
            return self._create_response(False, f"处理启动任务命令失败: {str(e)}", data)
            
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
            
    def _create_response(self, success: bool, message: str, task_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        创建响应消息
        
        Args:
            success: 是否成功
            message: 响应消息
            task_info: 任务信息（可选）
            
        Returns:
            Dict[str, Any]: 响应消息
        """
        # 获取当前时间戳
        timestamp = int(datetime.now().timestamp())
        
        # 获取MAC地址
        mac_address = self.get_mac_address()
        
        # 构建响应
        response = {
            "message_id": task_info.get("message_id") if task_info else None,
            "message_type": 80003,  # 响应消息类型
            "message_uuid": task_info.get("message_uuid") if task_info else None,
            "response_type": "cmd_reply",
            "mac_address": mac_address,
            "status": "1" if success else "0",
            "data": {
                "cmd_type": "start_task",
                "task_id": task_info.get("task_info", {}).get("task_id") if task_info else None,
                "subtask_id": task_info.get("subtask_info", {}).get("subtask_id") if task_info else None,
                "message": message,
                "timestamp": timestamp
            }
        }
        
        return response

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