"""
任务命令处理器
处理任务相关的命令，如启动、停止任务等
"""
import json
from datetime import datetime
from typing import Dict, Any, Optional

from loguru import logger

from core.task_management.utils.status import TaskStatus
from core.models import AnalysisType, RoiType
from shared.utils.tools import get_mac_address
from ..handler.message_types import (
    MESSAGE_TYPE_REQUEST_SETTING,
    REQUEST_TYPE_TASK_CMD,
    TASK_CMD_START,
    TASK_CMD_STOP
)

# 配置日志
logger = logger.bind(name=__name__)

class TaskCommandHandler:
    """
    任务命令处理器
    处理任务相关的命令
    """
    
    def __init__(self, mqtt_manager):
        """
        初始化任务命令处理器
        """
        # 延迟导入 TaskManager，避免循环导入
        from core.task_management.manager import TaskManager
        self.task_manager = TaskManager(mqtt_manager=mqtt_manager)
        self.mqtt_manager = mqtt_manager # 保存mqtt_manager实例
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
                return await self._handle_start_task(payload)  # 传递完整的payload
            elif cmd_type == TASK_CMD_STOP:
                return await self._handle_stop_task(payload)  # 传递完整的payload
            else:
                logger.warning(f"未知的任务命令类型: {cmd_type}")
                return None
                
        except Exception as e:
            logger.error(f"处理任务命令时出错: {e}")
            return None
            
    async def _handle_start_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理启动任务命令
        
        Args:
            payload: 完整的命令消息内容
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        # 提前定义 task_id 和 subtask_id，以防某些路径下未定义
        task_id = None
        subtask_id = None
        try:
            # 1. 提取任务相关信息
            data = payload.get("data", {})
            task_info = data.get("task_info")
            subtask_info = data.get("subtask_info")
            source_info = data.get("source")
            model_config = data.get("model_config")
            analysis_config = data.get("analysis_config")
            result_config = data.get("result_config")
            
            if not all([task_info, subtask_info, source_info, model_config, analysis_config, result_config]):
                logger.warning(f"启动任务命令缺少必要参数: {data}")
                # 尝试从 payload 中获取 task_id/subtask_id 用于响应
                task_id = payload.get("data", {}).get("task_info", {}).get("task_id")
                subtask_id = payload.get("data", {}).get("subtask_info", {}).get("subtask_id")
                return self._create_response(
                    success=False,
                    message="缺少必要参数",
                    original_request=payload,
                    task_id=task_id,
                    subtask_id=subtask_id
                )
            
            # 提取 task_id 和 subtask_id 供后续使用
            task_id = task_info.get("task_id")
            subtask_id = subtask_info.get("subtask_id")
            
            # 2. 构建任务配置
            task_config = {
                # 关联原始请求信息
                "message_id": payload.get("message_id"),
                "message_uuid": payload.get("message_uuid"),
                "confirmation_topic": payload.get("confirmation_topic"), # 也保存确认主题

                # 基本信息
                "task_id": task_id,
                "name": task_info["name"],
                "analysis_interval": task_info.get("analysis_interval", 1),
                
                # 子任务信息
                "subtask": {
                    "id": subtask_id,
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
            
            # 3. 添加任务到管理器，使用接收到的任务ID
            # 注意：现在 task_config 包含了 message_id 和 message_uuid
            success_add = self.task_manager.add_task(task_id, task_config)
            if not success_add:
                logger.error(f"添加任务失败: {task_id}")
                return self._create_response(
                    success=False,
                    message="添加任务失败",
                    original_request=payload,
                    task_id=task_id,
                    subtask_id=subtask_id
                )
            
            # 4. 启动任务处理
            try:
                # 更新任务状态为处理中
                self.task_manager.update_task_status(
                    task_id,
                    TaskStatus.PROCESSING
                )
                
                # 发送启动中响应
                start_response = self._create_response(
                    success=True, # 标记为成功，因为我们开始处理了
                    message="启动任务中",
                    original_request=payload,
                    task_id=task_id,
                    subtask_id=subtask_id
                )
                if payload.get("confirmation_topic"):
                    await self.mqtt_manager.publish(
                        payload["confirmation_topic"],
                        start_response
                    )
                
                # 根据任务类型启动相应的处理器
                if source_info["type"] == "stream":
                    logger.info(f"启动流分析任务: {task_id}")
                    # 异步启动流处理，使用接收到的任务ID
                    success_start = await self.task_manager.start_stream_task(
                        task_id,
                        task_config
                    )
                else:
                    logger.error(f"不支持的数据源类型: {source_info['type']}")
                    return self._create_response(
                        success=False,
                        message=f"不支持的数据源类型: {source_info['type']}",
                        original_request=payload,
                        task_id=task_id,
                        subtask_id=subtask_id
                    )
                
                if success_start:
                    logger.info(f"任务启动成功: {task_id}")
                    return self._create_response(
                        success=True,
                        message="启动任务成功",
                        original_request=payload,
                        task_id=task_id,
                        subtask_id=subtask_id
                    )
                else:
                    logger.error(f"任务启动失败: {task_id}")
                    return self._create_response(
                        success=False,
                        message="启动任务失败",
                        original_request=payload,
                        task_id=task_id,
                        subtask_id=subtask_id
                    )
                
            except Exception as e:
                logger.error(f"启动任务时出错: {e}")
                return self._create_response(
                    success=False,
                    message=f"启动任务时出错: {str(e)}",
                    original_request=payload,
                    task_id=task_id,
                    subtask_id=subtask_id
                )
                
        except Exception as e:
            logger.error(f"处理启动任务命令时出错: {e}")
            # 尝试在异常情况下也获取 ID
            task_id = payload.get("data", {}).get("task_info", {}).get("task_id") if not task_id else task_id
            subtask_id = payload.get("data", {}).get("subtask_info", {}).get("subtask_id") if not subtask_id else subtask_id
            return self._create_response(
                success=False,
                message=f"处理启动任务命令时出错: {str(e)}",
                original_request=payload,
                task_id=task_id,
                subtask_id=subtask_id
            )
            
    async def _handle_stop_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理停止任务命令
        
        Args:
            payload: 完整的命令消息内容
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        task_id = None # 初始化以防提前返回
        subtask_id = None
        try:
            # 获取任务ID和子任务ID
            data = payload.get("data", {})
            task_id = data.get("task_id")
            subtask_id = data.get("subtask_id")
            print(f"任务ID: {task_id}, 子任务ID: {subtask_id}") # 调试打印
            
            if not task_id:
                logger.warning("停止任务命令缺少task_id字段")
                return self._create_response(
                    success=False,
                    message="缺少任务ID",
                    original_request=payload,
                    task_id=task_id, # 可能是 None
                    subtask_id=subtask_id # 可能是 None
                )
                
            # 记录停止请求
            logger.info(f"收到停止任务请求: task_id={task_id}, subtask_id={subtask_id}")
            
            # 检查任务是否存在
            task = self.task_manager.get_task(task_id)
            if not task:
                logger.warning(f"要停止的任务不存在: {task_id}")
                return self._create_response(
                    success=False,
                    message="任务不存在",
                    original_request=payload,
                    task_id=task_id,
                    subtask_id=subtask_id
                )
                
            # 检查任务状态
            current_status = task.get("status")
            if current_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED]:
                logger.info(f"任务已经处于终止状态: {task_id}, status={current_status}")
                return self._create_response(
                    success=True, # 认为成功，因为已经是目标状态
                    message=f"任务已经处于{current_status}状态",
                    original_request=payload,
                    task_id=task_id,
                    subtask_id=subtask_id
                )
                
            # 发送停止中响应
            stop_response = self._create_response(
                success=True, # 标记为成功，因为我们开始处理了
                message="正在停止任务",
                original_request=payload,
                task_id=task_id,
                subtask_id=subtask_id
            )
            if payload.get("confirmation_topic"):
                await self.mqtt_manager.publish(
                    payload["confirmation_topic"],
                    stop_response
                )
            
            # 更新状态为 STOPPING
            logger.info(f"尝试将任务状态更新为 STOPPING: task_id={task_id}, subtask_id={subtask_id}")
            self.task_manager.update_task_status(
                task_id,
                TaskStatus.STOPPING # 只传递 task_id 和 status
            )
            
            # 实际停止任务
            success = await self.task_manager.stop_task(task_id, subtask_id)
            
            if success:
                logger.info(f"任务停止成功: task_id={task_id}, subtask_id={subtask_id}")
                self.task_manager.update_task_status(
                    task_id,
                    TaskStatus.STOPPED,
                    result=success # 保留 result/error
                )
                final_response = self._create_response(
                    success=True,
                    message="任务已成功停止",
                    original_request=payload,
                    task_id=task_id,
                    subtask_id=subtask_id
                )
            else:
                logger.error(f"任务停止失败: task_id={task_id}, subtask_id={subtask_id}, error={success}")
                self.task_manager.update_task_status(
                    task_id,
                    TaskStatus.FAILED,
                    error=success # 保留 result/error
                )
                final_response = self._create_response(
                    success=False,
                    message=f"停止任务失败: {success}",
                    original_request=payload,
                    task_id=task_id,
                    subtask_id=subtask_id
                )
            
            return final_response
            
        except Exception as e:
            logger.error(f"处理停止任务命令时出错: {e}")
            # 尝试更新任务状态为失败
            if task_id: # 确保 task_id 不是 None
                try:
                    self.task_manager.update_task_status(
                        task_id,
                        TaskStatus.FAILED,
                        error=str(e)
                    )
                except Exception as status_error:
                    logger.error(f"更新任务状态失败: {status_error}")
            # 停止失败时返回 status_code="2"
            final_response = self._create_response(
                success=False,
                message=f"处理停止任务命令时出错: {str(e)}",
                original_request=payload,
                task_id=task_id, # 使用函数开始处获取的 ID
                subtask_id=subtask_id # 使用函数开始处获取的 ID
            )
            
            return final_response
            
    def _create_response(self,
                         success: bool,
                         message: str,
                         original_request: Optional[Dict] = None,
                         task_id: Optional[str] = None,
                         subtask_id: Optional[str] = None
                         ) -> Dict[str, Any]:
        """
        创建响应消息
        
        Args:
            success: 是否成功
            message: 响应消息
            original_request: 原始请求的完整消息内容（可选），用于获取message_id, uuid, confirmation_topic
            task_id: 任务ID (可选)
            subtask_id: 子任务ID (可选)
            
        Returns:
            Dict[str, Any]: 响应消息
        """
        # 获取当前时间戳
        timestamp = int(datetime.now().timestamp())
        
        # 获取MAC地址
        mac_address = self.get_mac_address()
        
        # 从原始original_request获取关联信息
        request_data = original_request if original_request else {}
        message_id = request_data.get("message_id")
        message_uuid = request_data.get("message_uuid")
        confirmation_topic = request_data.get("confirmation_topic")  # 从根级别获取
        # 尝试从原始请求的 data 中获取 cmd_type
        cmd_type_original = request_data.get("data", {}).get("cmd_type", "unknown_cmd")
        
        # 构建响应
        response = {
            "message_id": message_id,
            "message_type": 80003,  # 响应消息类型
            "message_uuid": message_uuid,
            "response_type": "cmd_reply",
            "mac_address": mac_address,
            "status": "1" if success else "0", # 保持 status 为 "1" 或 "0"
            "data": {
                "cmd_type": cmd_type_original,
                "task_id": task_id, # 直接使用传入的 task_id
                "subtask_id": subtask_id, # 直接使用传入的 subtask_id
                "message": message,
                "timestamp": timestamp
            }
        }
        
        # 如果有确认主题，也添加到响应中，供上层使用
        if confirmation_topic:
            response["confirmation_topic"] = confirmation_topic
            
        return response

# 全局任务命令处理器实例
_task_command_handler = None

def get_task_command_handler(mqtt_manager):
    """
    获取全局任务命令处理器实例
    
    Returns:
        TaskCommandHandler: 任务命令处理器实例
    """
    global _task_command_handler
    if _task_command_handler is None:
        _task_command_handler = TaskCommandHandler(mqtt_manager)
    # 可选：如果需要确保 mqtt_manager 总是最新的
    # elif _task_command_handler.mqtt_manager != mqtt_manager:
    #    _task_command_handler.mqtt_manager = mqtt_manager
    return _task_command_handler 