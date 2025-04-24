"""
流分析任务处理器
处理视频流分析任务
"""
import os
import cv2
import time
import asyncio
import base64
from typing import Dict, Any, Optional
from datetime import datetime

from core.task_management.processor.base_processor import BaseTaskProcessor
from core.task_management.utils.status import TaskStatus
from core.task_management.processor.utils.model_loader import ModelLoader
from core.task_management.processor.utils.result_handler import ResultHandler
from core.task_management.processor.utils.callback_handler import CallbackHandler
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class StreamProcessor(BaseTaskProcessor):
    """流分析任务处理器"""
    
    def __init__(self):
        """初始化流处理器"""
        super().__init__()
        self.model_loader = ModelLoader()
        self.result_handler = ResultHandler()
        self.callback_handler = CallbackHandler()
        
    async def start_task(self, task_id: str, task_config: Dict[str, Any]) -> bool:
        """启动流分析任务"""
        try:
            # 1. 加载模型
            detector = await self.model_loader.load_model(
                task_config["model"]["code"],
                task_config["subtask"]["type"]
            )
            
            # 2. 创建跟踪器（如果需要）
            tracker = await self.model_loader.create_tracker(task_config)
            
            # 3. 启动分析任务
            task = asyncio.create_task(
                self._process_stream(
                    task_id=task_id,
                    detector=detector,
                    tracker=tracker,
                    task_config=task_config
                )
            )
            
            # 4. 保存任务引用
            self.add_active_task(task_id, {"task": task, "config": task_config})
            
            return True
            
        except Exception as e:
            logger.error(f"启动流分析任务失败: {str(e)}")
            return False
            
    async def stop_task(self, task_id: str) -> Dict[str, Any]:
        """停止流分析任务"""
        try:
            if not self.is_task_active(task_id):
                return {
                    "success": False,
                    "error": f"任务不存在或已停止: {task_id}"
                }
                
            # 更新任务状态为停止中
            task_info = self.active_tasks[task_id]
            task_info["status"] = TaskStatus.STOPPING
            
            return {
                "success": True,
                "task_id": task_id,
                "message": "正在停止任务"
            }
            
        except Exception as e:
            logger.error(f"停止任务失败: {str(e)}")
            return {"success": False, "error": str(e)}
            
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        try:
            if not self.is_task_active(task_id):
                return {
                    "success": False,
                    "error": f"任务不存在: {task_id}"
                }
                
            task_info = self.active_tasks[task_id]
            return {
                "success": True,
                "task_id": task_id,
                "status": task_info.get("status"),
                "config": task_info.get("config")
            }
            
        except Exception as e:
            logger.error(f"获取任务状态失败: {str(e)}")
            return {"success": False, "error": str(e)}
            
    async def _process_stream(
        self,
        task_id: str,
        detector: Any,
        tracker: Optional[Any],
        task_config: Dict[str, Any]
    ):
        """处理流分析任务的核心逻辑"""
        try:
            # 1. 获取流URL和配置
            stream_url = task_config["source"]["urls"][0]
            analysis_interval = task_config.get("analysis_interval", 1)
            
            # 2. 打开视频流
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                raise Exception(f"无法打开视频流: {stream_url}")
                
            # 3. 获取流信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            frame_count = 0
            while True:
                # 检查任务是否被停止
                if not self.is_task_active(task_id):
                    break
                    
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"读取视频流失败，尝试重新连接: {stream_url}")
                    cap.release()
                    cap = cv2.VideoCapture(stream_url)
                    continue
                    
                # 按间隔分析
                if frame_count % analysis_interval == 0:
                    # 执行检测
                    result = await detector.detect(frame)
                    
                    # 处理检测结果
                    processed_result = await self.result_handler.process_detection_result(
                        result,
                        task_id,
                        frame,
                        frame_count,
                        task_config,
                        tracker
                    )
                    
                    # 处理回调
                    await self.callback_handler.handle_callback(
                        task_id,
                        processed_result,
                        task_config
                    )
                    
                frame_count += 1
                
            # 清理资源
            cap.release()
            self.remove_active_task(task_id)
            
        except Exception as e:
            logger.error(f"处理流分析任务失败: {str(e)}")
            self.remove_active_task(task_id) 