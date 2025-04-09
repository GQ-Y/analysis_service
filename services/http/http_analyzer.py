"""
HTTP协议的分析服务
处理HTTP协议的图像分析、视频分析和流分析请求
"""
import os
import time
import base64
import uuid
from typing import Dict, List, Any, Optional
from fastapi import UploadFile, HTTPException

from shared.utils.logger import setup_logger
from core.task_manager import TaskManager
from core.task_processor import TaskProcessor
from core.config import settings
from services.base_analyzer import BaseAnalyzerService

logger = setup_logger(__name__)

class HTTPAnalyzerService(BaseAnalyzerService):
    """基于HTTP协议的分析服务类，处理图像和视频分析请求"""
    
    def __init__(self):
        """初始化HTTP分析服务"""
        super().__init__()
        self.task_manager = TaskManager.get_instance()
        self.task_processor = TaskProcessor()
        
        # 确保输出目录存在
        os.makedirs(settings.OUTPUT.save_dir, exist_ok=True)
        os.makedirs(f"{settings.OUTPUT.save_dir}/images", exist_ok=True)
        os.makedirs(f"{settings.OUTPUT.save_dir}/videos", exist_ok=True)
        os.makedirs(f"{settings.OUTPUT.save_dir}/streams", exist_ok=True)
        
        logger.info("HTTP分析服务已初始化")
        
    async def analyze_image(self, file: UploadFile, model_code: str, conf_threshold: float = 0.25, 
                            save_result: bool = True, include_image: bool = False) -> Dict[str, Any]:
        """
        分析上传的图像文件
        
        参数:
            file: 上传的图像文件
            model_code: 模型代码
            conf_threshold: 置信度阈值
            save_result: 是否保存结果图像
            include_image: 是否在响应中包含base64编码的图像
            
        返回:
            Dict: 分析结果
        """
        try:
            # 保存上传的文件
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = f"{settings.OUTPUT.save_dir}/images/{filename}"
            
            # 读取文件内容并保存
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
                
            # 创建任务
            task_id = self.task_manager.create_task(
                task_type="image",
                protocol="http",
                params={
                    "image_path": file_path,
                    "model_code": model_code,
                    "conf_threshold": conf_threshold,
                    "save_result": save_result,
                    "include_image": include_image
                }
            )
            
            # 处理图像分析任务
            result = self.task_processor.process_image(task_id)
            
            return result
            
        except Exception as e:
            logger.error(f"分析图像时出错: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"分析图像时出错: {str(e)}")
            
    async def start_video_analysis(self, file: UploadFile, model_code: str, 
                                   conf_threshold: float = 0.25, save_result: bool = True) -> Dict[str, Any]:
        """
        启动视频分析任务
        
        参数:
            file: 上传的视频文件
            model_code: 模型代码
            conf_threshold: 置信度阈值
            save_result: 是否保存结果视频
            
        返回:
            Dict: 任务信息
        """
        try:
            # 保存上传的文件
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = f"{settings.OUTPUT.save_dir}/videos/{filename}"
            
            # 读取文件内容并保存
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
                
            # 创建任务
            task_id = self.task_manager.create_task(
                task_type="video",
                protocol="http",
                params={
                    "video_path": file_path,
                    "model_code": model_code,
                    "conf_threshold": conf_threshold,
                    "save_result": save_result
                }
            )
            
            # 启动视频分析任务
            result = self.task_processor.start_video_analysis(task_id)
            
            return result
            
        except Exception as e:
            logger.error(f"启动视频分析任务时出错: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"启动视频分析任务时出错: {str(e)}")
            
    async def start_stream_analysis(self, stream_url: str, model_code: str, 
                                   conf_threshold: float = 0.25, save_interval: int = 10,
                                   max_duration: int = 3600) -> Dict[str, Any]:
        """
        启动流分析任务
        
        参数:
            stream_url: 流URL
            model_code: 模型代码
            conf_threshold: 置信度阈值
            save_interval: 保存间隔（秒）
            max_duration: 最大运行时间（秒）
            
        返回:
            Dict: 任务信息
        """
        try:
            # 创建任务
            task_id = self.task_manager.create_task(
                task_type="stream",
                protocol="http",
                params={
                    "stream_url": stream_url,
                    "model_code": model_code,
                    "conf_threshold": conf_threshold,
                    "save_interval": save_interval,
                    "max_duration": max_duration
                }
            )
            
            # 启动流分析任务
            result = self.task_processor.start_stream_analysis(task_id)
            
            return result
            
        except Exception as e:
            logger.error(f"启动流分析任务时出错: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"启动流分析任务时出错: {str(e)}")
            
    async def stop_task(self, task_id: str) -> Dict[str, Any]:
        """
        停止任务
        
        参数:
            task_id: 任务ID
            
        返回:
            Dict: 停止结果
        """
        try:
            result = self.task_processor.stop_task(task_id)
            
            if not result.get("success", False):
                raise HTTPException(status_code=400, detail=result.get("error", "停止任务失败"))
                
            return result
            
        except Exception as e:
            logger.error(f"停止任务 {task_id} 时出错: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"停止任务时出错: {str(e)}")
            
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态
        
        参数:
            task_id: 任务ID
            
        返回:
            Dict: 任务状态
        """
        try:
            result = self.task_processor.get_task_status(task_id)
            
            if not result.get("success", False):
                raise HTTPException(status_code=404, detail=result.get("error", "未找到任务"))
                
            return result
            
        except Exception as e:
            logger.error(f"获取任务 {task_id} 状态时出错: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"获取任务状态时出错: {str(e)}")
            
    async def get_tasks(self) -> Dict[str, Any]:
        """
        获取所有任务
        
        返回:
            Dict: 任务列表
        """
        try:
            tasks = self.task_manager.get_all_tasks()
            
            # 过滤出HTTP协议的任务
            http_tasks = {
                task_id: task for task_id, task in tasks.items()
                if task.get("protocol") == "http"
            }
            
            return {
                "success": True,
                "tasks": http_tasks
            }
            
        except Exception as e:
            logger.error(f"获取任务列表时出错: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"获取任务列表时出错: {str(e)}") 