"""
任务工作器
负责从队列中获取任务并执行
"""
import asyncio
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime

from core.task_management import TaskStatus
from core.task_management.manager import TaskManager
from core.task_management.processor.task_processor import TaskProcessor
from services.task_store import TaskStore
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class TaskWorker:
    """任务工作器"""

    def __init__(self, task_store: TaskStore):
        """初始化

        Args:
            task_store: 任务存储服务
        """
        self.task_store = task_store
        self.task_manager = TaskManager()
        self.processor = TaskProcessor(self.task_manager)
        self.running = False
        self.poll_interval = 1.0  # 轮询间隔（秒）
        self.max_concurrent = 5  # 最大并发任务数
        self.running_tasks = {}  # 运行中的任务

    async def start(self):
        """启动任务工作器"""
        if self.running:
            logger.warning("任务工作器已经在运行中")
            return

        self.running = True
        logger.info("任务工作器已启动")

        # 启动任务处理循环
        asyncio.create_task(self._process_loop())

    async def stop(self):
        """停止任务工作器"""
        if not self.running:
            logger.warning("任务工作器已经停止")
            return

        self.running = False
        logger.info("任务工作器已停止")

    async def _process_loop(self):
        """任务处理循环"""
        while self.running:
            try:
                # 检查是否可以处理更多任务
                if len(self.running_tasks) < self.max_concurrent:
                    # 获取下一个待处理任务
                    task = await self._get_next_task()
                    if task:
                        # 启动任务
                        asyncio.create_task(self._process_task(task))

                # 等待一段时间再继续
                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"任务处理循环异常: {str(e)}")
                await asyncio.sleep(self.poll_interval * 2)  # 出错后等待更长时间

    async def _get_next_task(self) -> Optional[Dict[str, Any]]:
        """获取下一个待处理任务"""
        try:
            # 从Redis中获取优先级最高的任务
            task_ids = await self.task_store._redis.zrange(
                "task_queue:waiting",
                0,
                0,
                desc=True  # 按优先级降序
            )

            if not task_ids:
                return None

            task_id = task_ids[0]

            # 获取任务详情
            task_data = await self.task_store._redis.get(f"task:{task_id}")
            if not task_data:
                # 任务ID存在但任务数据不存在，从队列中移除
                await self.task_store._redis.zrem("task_queue:waiting", task_id)
                return None

            # 解析任务数据
            task = json.loads(task_data)

            # 从队列中移除任务
            await self.task_store._redis.zrem("task_queue:waiting", task_id)

            # 更新任务状态为处理中
            task["status"] = TaskStatus.PROCESSING
            task["updated_at"] = datetime.now().isoformat()
            await self.task_store._redis.set(f"task:{task_id}", json.dumps(task))

            return task

        except Exception as e:
            logger.error(f"获取下一个任务失败: {str(e)}")
            return None

    async def _process_task(self, task: Dict[str, Any]):
        """处理单个任务"""
        task_id = task["id"]

        try:
            logger.info(f"开始处理任务: {task_id}")

            # 记录到运行中的任务
            self.running_tasks[task_id] = task

            # 构建任务配置
            task_config = self._build_task_config(task)

            # 启动任务
            result = await self.processor.start_stream_analysis(task_id, task_config)

            if not result:
                logger.error(f"启动任务失败: {task_id}")
                # 更新任务状态为失败
                await self._update_task_status(task_id, TaskStatus.FAILED, "启动任务失败")
                return

            logger.info(f"任务启动成功: {task_id}")

        except Exception as e:
            logger.error(f"处理任务异常: {task_id}, {str(e)}")
            # 更新任务状态为失败
            await self._update_task_status(task_id, TaskStatus.FAILED, f"处理任务异常: {str(e)}")

        finally:
            # 从运行中的任务中移除
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

    def _build_task_config(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """构建任务配置"""
        # 获取任务配置
        config = task.get("config", {})

        # 构建模型配置
        model_config = {
            "code": task.get("model_code", "yolov8n"),
            "confidence": config.get("confidence", 0.5),
            "iou_threshold": config.get("iou_threshold", 0.45)
        }

        # 构建子任务配置
        analysis_type = task.get("analysis_type", "detection")
        # 如果启用了跟踪，将分析类型设置为 tracking
        if config.get("tracking_type", 0) > 0:
            analysis_type = "tracking"

        subtask_config = {
            "type": analysis_type,
            "callback": {
                "enabled": task.get("enable_callback", False),
                "url": task.get("callback_urls")
            }
        }

        # 构建分析配置
        analysis_config = {
            "classes": config.get("detect_classes"),
            "roi": config.get("roi"),
            "roi_type": config.get("roi_type", 1),
            "nested_detection": config.get("nested_detection", False),
            "track_config": {
                "enabled": config.get("tracking_type", 0) > 0,
                "tracker_type": "sort",
                "max_age": config.get("max_lost_time", 30),
                "min_hits": 3,
                "iou_threshold": config.get("iou_threshold", 0.45)
            }
        }

        # 构建结果配置
        result_config = {
            "save_images": task.get("save_images", False),
            "return_base64": task.get("return_base64", True),
            "storage": {
                "save_path": "results"
            }
        }

        # 构建完整配置
        task_config = {
            "model": model_config,
            "subtask": subtask_config,
            "analysis": analysis_config,
            "stream_url": task.get("stream_url"),
            "result": result_config,
            "analysis_interval": task.get("analyze_interval", 1),
            "device": task.get("device", "auto")
        }

        return task_config

    async def _update_task_status(self, task_id: str, status: int, error_message: Optional[str] = None):
        """更新任务状态"""
        try:
            # 获取任务
            task_data = await self.task_store._redis.get(f"task:{task_id}")
            if not task_data:
                logger.warning(f"更新任务状态失败，任务不存在: {task_id}")
                return

            # 解析任务数据
            task = json.loads(task_data)

            # 更新状态
            task["status"] = status
            task["updated_at"] = datetime.now().isoformat()

            # 如果是完成或失败状态，记录结束时间
            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                task["stop_time"] = datetime.now().isoformat()
                if task.get("start_time"):
                    start_time = datetime.fromisoformat(task["start_time"])
                    stop_time = datetime.fromisoformat(task["stop_time"])
                    task["duration"] = (stop_time - start_time).total_seconds() / 60

            # 如果是开始运行状态，记录开始时间
            if status == TaskStatus.PROCESSING:
                task["start_time"] = datetime.now().isoformat()

            # 如果有错误信息，记录错误信息
            if error_message:
                task["error_message"] = error_message

            # 保存更新后的任务
            await self.task_store._redis.set(f"task:{task_id}", json.dumps(task))

            logger.info(f"任务状态已更新: {task_id} -> {status}")

        except Exception as e:
            logger.error(f"更新任务状态失败: {task_id}, {str(e)}")
