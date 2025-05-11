"""
HTTP任务服务
提供基于HTTP的任务管理功能
"""
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import httpx
import json

from core.config import settings
from core.task_management import TaskStatus
from models.task import TaskBase, QueueTask
from crud.task import TaskCRUD
from services.task_store import TaskStore
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class TaskService:
    """HTTP任务服务类"""

    def __init__(self, task_crud: TaskCRUD):
        """初始化

        Args:
            task_crud: 任务CRUD操作类
        """
        self.task_crud = task_crud

    async def list_tasks(
        self,
        status: Optional[int] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """获取任务列表

        Args:
            status: 任务状态过滤
            limit: 返回数量限制

        Returns:
            Dict[str, Any]: 任务列表结果
        """
        try:
            # 获取任务列表
            tasks = await self.task_crud.list_tasks(status, limit)

            if not tasks:
                return {
                    "success": True,
                    "message": "没有符合条件的任务",
                    "tasks": [],
                    "total": 0
                }

            # 构建任务信息列表
            task_list = []
            for task in tasks:
                task_info = {
                    "task_id": task.id,
                    "task_name": task.task_name,
                    "model_code": task.model_code,
                    "stream_url": task.stream_url,
                    "status": task.status,
                    "start_time": task.start_time.isoformat() if task.start_time else None,
                    "stop_time": task.stop_time.isoformat() if task.stop_time else None,
                    "duration": task.duration,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat(),
                    "error_message": task.error_message
                }
                task_list.append(task_info)

            return {
                "success": True,
                "message": "获取任务列表成功",
                "tasks": task_list,
                "total": len(task_list)
            }

        except Exception as e:
            logger.error(f"获取任务列表失败: {str(e)}")
            return {
                "success": False,
                "message": f"获取任务列表失败: {str(e)}",
                "tasks": [],
                "total": 0
            }

    async def start_task(
        self,
        model_code: str,
        stream_url: str,
        task_name: Optional[str] = None,
        callback_urls: Optional[str] = None,
        output_url: Optional[str] = None,
        analysis_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enable_callback: bool = False,
        save_result: bool = False
    ) -> Dict[str, Any]:
        """启动任务

        Args:
            model_code: 模型代码
            stream_url: 流URL
            task_name: 任务名称
            callback_urls: 回调地址
            output_url: 输出URL
            analysis_type: 分析类型
            config: 分析配置
            enable_callback: 是否启用回调
            save_result: 是否保存结果

        Returns:
            Dict[str, Any]: 启动结果
        """
        try:
            # 创建任务
            task = await self.task_crud.create_task(
                model_code=model_code,
                stream_url=stream_url,
                task_name=task_name,
                callback_urls=callback_urls,
                output_url=output_url,
                analysis_type=analysis_type,
                config=config,
                enable_callback=enable_callback,
                save_result=save_result
            )

            if not task:
                return {
                    "success": False,
                    "message": "创建任务失败",
                    "task_id": None
                }

            # 创建队列任务
            queue_task = await self.task_crud.create_queue_task(task)

            if not queue_task:
                return {
                    "success": False,
                    "message": "创建队列任务失败",
                    "task_id": task.id
                }

            # 更新任务状态为等待中
            await self.task_crud.update_task_status(task.id, TaskStatus.WAITING)

            return {
                "success": True,
                "message": "任务已创建并加入队列",
                "task_id": task.id
            }

        except Exception as e:
            logger.error(f"启动任务失败: {str(e)}")
            return {
                "success": False,
                "message": f"启动任务失败: {str(e)}",
                "task_id": None
            }

    async def stop_task(self, task_id: str) -> Dict[str, Any]:
        """停止任务

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 停止结果
        """
        try:
            # 获取任务
            task = await self.task_crud.get_task(task_id)

            if not task:
                return {
                    "success": False,
                    "message": f"任务不存在: {task_id}",
                    "task_id": task_id
                }

            # 更新任务状态为停止中
            await self.task_crud.update_task_status(task_id, TaskStatus.STOPPING)

            return {
                "success": True,
                "message": "正在停止任务",
                "task_id": task_id
            }

        except Exception as e:
            logger.error(f"停止任务失败: {str(e)}")
            return {
                "success": False,
                "message": f"停止任务失败: {str(e)}",
                "task_id": task_id
            }

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 任务状态
        """
        try:
            # 获取任务
            task = await self.task_crud.get_task(task_id)

            if not task:
                return {
                    "success": False,
                    "message": f"任务不存在: {task_id}",
                    "task_id": task_id,
                    "status": None
                }

            return {
                "success": True,
                "message": "获取任务状态成功",
                "task_id": task_id,
                "status": task.status,
                "task_info": task.model_dump()
            }

        except Exception as e:
            logger.error(f"获取任务状态失败: {str(e)}")
            return {
                "success": False,
                "message": f"获取任务状态失败: {str(e)}",
                "task_id": task_id,
                "status": None
            }

    async def report_result(
        self,
        task_id: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """上报分析结果

        Args:
            task_id: 任务ID
            result: 分析结果

        Returns:
            Dict[str, Any]: 上报结果
        """
        try:
            # 获取任务
            task = await self.task_crud.get_task(task_id)

            if not task:
                return {
                    "success": False,
                    "message": f"任务不存在: {task_id}",
                    "task_id": task_id
                }

            # 如果启用了回调，发送结果到回调地址
            if task.enable_callback and task.callback_urls:
                await self._send_callback(task.callback_urls, {
                    "task_id": task_id,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })

            return {
                "success": True,
                "message": "分析结果已上报",
                "task_id": task_id
            }

        except Exception as e:
            logger.error(f"上报分析结果失败: {str(e)}")
            return {
                "success": False,
                "message": f"上报分析结果失败: {str(e)}",
                "task_id": task_id
            }

    async def _send_callback(self, callback_urls: str, data: Dict[str, Any]) -> bool:
        """发送回调

        Args:
            callback_urls: 回调地址，多个地址用逗号分隔
            data: 回调数据

        Returns:
            bool: 是否发送成功
        """
        try:
            urls = [url.strip() for url in callback_urls.split(",")]

            for url in urls:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            url,
                            json=data,
                            timeout=10.0
                        )

                        if response.status_code != 200:
                            logger.warning(f"回调请求失败: {url}, 状态码: {response.status_code}")

                except Exception as e:
                    logger.error(f"发送回调失败: {url}, 错误: {str(e)}")

            return True

        except Exception as e:
            logger.error(f"处理回调失败: {str(e)}")
            return False
