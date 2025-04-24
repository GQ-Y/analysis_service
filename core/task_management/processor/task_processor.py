"""
任务处理器
处理分析任务，包括视频流分析
"""
import os
import cv2
import time
import asyncio
import base64
import aiohttp
import numpy as np
from typing import Dict, Any, Optional, Union, TYPE_CHECKING
from datetime import datetime
import json
import multiprocessing
from multiprocessing import Process, Queue, Event
import copy

# from core.task_management.manager import TaskManager # 移除顶层导入
from core.task_management.utils.status import TaskStatus
from core.analyzer.detection import YOLODetector
from shared.utils.logger import setup_logger
from core.config import settings

# 条件导入，仅用于类型检查
if TYPE_CHECKING:
    from core.task_management.manager import TaskManager

logger = setup_logger(__name__)

def convert_numpy_types(obj):
    """转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def process_stream_worker(task_id: str, stream_config: Dict, stop_event: Event, result_queue: Queue):
    """
    独立的流处理工作函数
    
    Args:
        task_id: 任务ID
        stream_config: 流配置信息
        stop_event: 停止事件
        result_queue: 结果队列
    """
    try:
        # 1. 初始化检测器
        model_code = stream_config["model"]["code"]
        analysis_type = stream_config["subtask"]["type"]
        
        detector = None
        if analysis_type == "detection":
            from core.analyzer.detection import YOLODetector
            detector = YOLODetector()
            # 使用 asyncio.run() 执行异步加载
            try:
                asyncio.run(detector.load_model(model_code))
            except Exception as load_err:
                 result_queue.put({
                     "type": "error",
                     "message": f"加载模型失败: {load_err}"
                 })
                 return
        elif analysis_type == "segmentation":
            from core.analyzer.segmentation import YOLOSegmentor
            detector = YOLOSegmentor()
             # 使用 asyncio.run() 执行异步加载
            try:
                asyncio.run(detector.load_model(model_code))
            except Exception as load_err:
                 result_queue.put({
                     "type": "error",
                     "message": f"加载模型失败: {load_err}"
                 })
                 return
        else:
            raise ValueError(f"不支持的分析类型: {analysis_type}")
            
        if not detector:
            result_queue.put({
                "type": "error",
                "message": "初始化检测器失败"
            })
            return
            
        # 2. 初始化跟踪器（如果需要）
        tracker = None
        if stream_config.get("analysis", {}).get("track_config"):
            try:
                from core.tracker import create_tracker
                track_config = stream_config["analysis"]["track_config"]
                tracker_type = track_config.pop("tracker_type", "sort")
                tracker = create_tracker(tracker_type, **track_config)
            except Exception as e:
                result_queue.put({
                    "type": "error",
                    "message": f"创建跟踪器失败: {str(e)}"
                })
                return
        
        # 3. 打开视频流
        stream_url = stream_config["source"]["urls"][0]
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            result_queue.put({
                "type": "error",
                "message": f"无法打开视频流: {stream_url}"
            })
            return
            
        # 4. 获取流信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 5. 处理每一帧
        frame_count = 0
        analysis_interval = stream_config.get("analysis_interval", 1)
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"读取视频流失败，尝试重新连接: {stream_url}")
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                continue
                
            # 按间隔分析
            if frame_count % analysis_interval == 0:
                try:
                    # 使用 asyncio.run() 执行异步检测
                    result = asyncio.run(detector.detect(frame))
                    detections = result.get("detections", [])
                    
                    # 过滤指定类别
                    if stream_config.get("analysis", {}).get("classes"):
                        allowed_classes = stream_config["analysis"]["classes"]
                        detections = [
                            det for det in detections 
                            if det["class_id"] in allowed_classes
                        ]
                    
                    # 应用ROI过滤
                    if stream_config.get("roi"):
                        roi_config = stream_config["roi"]["config"]
                        roi_type = stream_config["roi"].get("type", 1)
                        detections = detector._filter_by_roi(
                            detections, 
                            roi_config, 
                            roi_type,
                            height,
                            width
                        )
                    
                    # 执行跟踪
                    if tracker:
                        # 使用 asyncio.run() 执行异步跟踪更新
                        tracked_objects = asyncio.run(tracker.update(detections))
                        detections = [obj.to_dict() for obj in tracked_objects]
                        
                    # 发送结果
                    result_queue.put({
                        "type": "result",
                        "frame_id": frame_count,
                        "detections": convert_numpy_types(detections),
                        "timestamp": time.time()
                    })
                    
                except Exception as e:
                    logger.error(f"处理帧时出错: {str(e)}")
                    continue
                    
            frame_count += 1
            
        # 6. 清理资源
        cap.release()
        
        # 7. 发送完成消息
        result_queue.put({
            "type": "complete",
            "total_frames": frame_count,
            "processed_frames": frame_count // analysis_interval
        })
        
    except Exception as e:
        result_queue.put({
            "type": "error",
            "message": str(e)
        })

class TaskProcessor:
    """任务处理器，处理分析任务"""
    
    def __init__(self, task_manager: 'TaskManager'):
        """
        初始化任务处理器
        
        Args:
            task_manager: 任务管理器实例
        """
        self.task_manager = task_manager
        self.active_tasks = {}  # 存储活动任务的进程
        self.stop_events = {}  # 存储任务停止事件
        self.result_queues = {}  # 存储任务结果队列
        
        logger.info("任务处理器已初始化")
        
    async def start_stream_analysis(self, task_id: str) -> bool:
        """
        启动流分析任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否启动成功
        """
        try:
            # 1. 获取任务信息
            task = self.task_manager.get_task(task_id)
            if not task:
                logger.error(f"任务不存在: {task_id}")
                return False

            task_data = task.get("data", {})
            if not task_data:
                 logger.error(f"任务配置数据为空: {task_id}")
                 # 将任务状态更新为失败
                 self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error="任务配置数据为空")
                 return False

            # DEBUG: 打印原始 task_data 和关键的中间配置字典
            logger.debug(f"任务 {task_id} - 原始 task_data: {task_data}")
            # 使用 task_data 中的实际键名
            model_intermediate = task_data.get('model')
            subtask_intermediate = task_data.get('subtask')
            source_intermediate = task_data.get('source')
            analysis_intermediate = task_data.get('analysis')
            roi_intermediate = task_data.get('roi')
            # task_info 的内容在 task_data 根级别
            logger.debug(f"任务 {task_id} - 中间 model: {model_intermediate}")
            logger.debug(f"任务 {task_id} - 中间 subtask: {subtask_intermediate}")
            logger.debug(f"任务 {task_id} - 中间 source: {source_intermediate}")
            logger.debug(f"任务 {task_id} - 中间 analysis: {analysis_intermediate}")
            logger.debug(f"任务 {task_id} - 中间 roi: {roi_intermediate}")

            # 2. 构建一个明确可序列化的配置字典给工作进程
            try:
                worker_config = {
                    "model": {
                        # 从 model 获取 code
                        "code": model_intermediate.get("code") if isinstance(model_intermediate, dict) else None
                    },
                    "subtask": {
                        # 从 subtask 获取 type
                        "type": subtask_intermediate.get("type") if isinstance(subtask_intermediate, dict) else None
                    },
                    "analysis": {
                        # 从 analysis 获取 track_config 和 classes
                        "track_config": analysis_intermediate.get("track_config") if isinstance(analysis_intermediate, dict) else None,
                        "classes": analysis_intermediate.get("classes") if isinstance(analysis_intermediate, dict) else None
                    },
                    "source": {
                        "urls": source_intermediate.get("urls", []) if isinstance(source_intermediate, dict) else []
                    },
                    # 直接从 task_data 获取 analysis_interval
                    "analysis_interval": task_data.get("analysis_interval", 1),
                    # 从 roi 获取 config 和 type
                    "roi": {
                        "config": roi_intermediate.get("config") if isinstance(roi_intermediate, dict) else None,
                        "type": roi_intermediate.get("type", 1) if isinstance(roi_intermediate, dict) else 1
                    } if roi_intermediate and isinstance(roi_intermediate, dict) and roi_intermediate.get("config") else None
                }

                # 验证关键配置是否存在
                model_code_val = worker_config.get("model", {}).get("code")
                analysis_type_val = worker_config.get("subtask", {}).get("type")
                source_urls_val = worker_config.get("source", {}).get("urls")
                logger.debug(f"任务 {task_id} 配置验证前的值: model_code='{model_code_val}', analysis_type='{analysis_type_val}', source_urls={source_urls_val}")

                if not model_code_val or not analysis_type_val or not source_urls_val:
                     error_msg = f"工作进程配置不完整: model_code='{model_code_val}', analysis_type='{analysis_type_val}', source_urls={source_urls_val}"
                     logger.error(f"{error_msg}, 任务ID: {task_id}")
                     self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                     return False

                # 确保配置是JSON可序列化的 (进一步检查)
                try:
                    json.dumps(worker_config)
                except TypeError as json_err:
                    error_msg = f"工作进程配置无法JSON序列化: {json_err}"
                    logger.error(f"{error_msg}, 任务ID: {task_id}")
                    self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                    return False

            except KeyError as e:
                error_msg = f"构建工作进程配置时缺少键: {e}"
                logger.error(f"{error_msg}, 任务ID: {task_id}")
                self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                return False
            except Exception as e:
                error_msg = f"构建工作进程配置时发生意外错误: {str(e)}"
                logger.error(f"{error_msg}, 任务ID: {task_id}")
                self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                return False

            # 3. 检查任务是否已经在运行
            if task_id in self.active_tasks and self.active_tasks[task_id].is_alive():
                logger.warning(f"任务已经在运行: {task_id}")
                # 注意：这里不更新状态，因为任务实际上在运行
                return False # 或者根据需求返回 True

            # 4. 创建停止事件和结果队列
            stop_event = Event()
            result_queue = Queue()

            # 5. 创建并启动任务进程 (传递干净的 worker_config)
            process = Process(
                target=process_stream_worker,
                args=(task_id, worker_config, stop_event, result_queue)
            )
            process.start() # 序列化错误应该在这里解决

            # 6. 保存进程引用和控制对象
            self.active_tasks[task_id] = process
            self.stop_events[task_id] = stop_event
            self.result_queues[task_id] = result_queue

            # 7. 启动结果处理协程
            asyncio.create_task(self._handle_results(task_id, result_queue))

            logger.info(f"流分析任务启动成功: {task_id}")
            # 注意：任务状态应由 _handle_results 更新，这里不显式设置 PENDING 或 RUNNING
            return True

        except Exception as e:
            error_msg = f"启动流分析任务时发生意外错误: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"错误详情:\n{traceback.format_exc()}")
            # 确保在启动过程中发生任何异常时都将任务标记为失败
            if 'task_id' in locals(): # 仅当 task_id 已定义时尝试更新状态
                 self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
            return False
            
    async def _handle_results(self, task_id: str, result_queue: Queue):
        """
        处理任务结果
        
        Args:
            task_id: 任务ID
            result_queue: 结果队列
        """
        try:
            while True:
                # 检查任务是否已停止
                if task_id not in self.active_tasks:
                    break
                    
                # 非阻塞方式获取结果
                try:
                    result = result_queue.get_nowait()
                except:
                    await asyncio.sleep(0.01)
                    continue
                    
                # 处理不同类型的结果
                if result["type"] == "result":
                    # 更新任务状态
                    self.task_manager.update_task_status(
                        task_id,
                        TaskStatus.PROCESSING,
                        result=result
                    )
                elif result["type"] == "error":
                    # 更新任务状态为失败
                    self.task_manager.update_task_status(
                        task_id,
                        TaskStatus.FAILED,
                        error=result["message"]
                    )
                    break
                elif result["type"] == "complete":
                    # 更新任务状态为完成
                    self.task_manager.update_task_status(
                        task_id,
                        TaskStatus.COMPLETED,
                        result=result
                    )
                    break
                    
        except Exception as e:
            logger.error(f"处理结果时出错: {str(e)}")
            
        finally:
            # 清理资源
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            if task_id in self.stop_events:
                del self.stop_events[task_id]
            if task_id in self.result_queues:
                del self.result_queues[task_id]
                
    async def stop_task(self, task_id: str) -> Dict[str, Any]:
        """
        停止任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 停止结果
        """
        try:
            # 1. 检查任务是否存在
            if task_id not in self.active_tasks:
                return {
                    "success": False,
                    "error": f"任务不存在或已停止: {task_id}"
                }
                
            # 2. 设置停止事件
            if task_id in self.stop_events:
                self.stop_events[task_id].set()
                
            # 3. 等待进程结束
            process = self.active_tasks[task_id]
            process.join(timeout=5.0)
            
            # 4. 如果进程仍在运行，强制终止
            if process.is_alive():
                process.terminate()
                process.join()
                
            # 5. 清理资源
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            if task_id in self.stop_events:
                del self.stop_events[task_id]
            if task_id in self.result_queues:
                del self.result_queues[task_id]
                
            return {
                "success": True,
                "message": "任务已停止"
            }
            
        except Exception as e:
            logger.error(f"停止任务失败: {str(e)}")
            return {
                "success": False,
                "error": f"停止任务失败: {str(e)}"
            }
            
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 任务状态
        """
        try:
            # 获取任务信息
            task = self.task_manager.get_task(task_id)
            if not task:
                return {"success": False, "error": f"任务不存在: {task_id}"}
                
            return {
                "success": True,
                "task_id": task_id,
                "status": task.get("status"),
                "progress": task.get("progress", 0),
                "created_at": task.get("created_at"),
                "updated_at": task.get("updated_at"),
                "result": task.get("result")
            }
            
        except Exception as e:
            logger.error(f"获取任务状态失败: {str(e)}")
            return {"success": False, "error": f"获取任务状态失败: {str(e)}"}
            
    async def get_tasks(self, protocol: Optional[str] = None) -> Dict[str, Any]:
        """
        获取任务列表
        
        Args:
            protocol: 协议过滤
            
        Returns:
            Dict: 任务列表
        """
        try:
            # 获取所有任务
            tasks = self.task_manager.get_all_tasks()
            
            # 如果指定了协议，过滤任务
            if protocol:
                filtered_tasks = {
                    task_id: task for task_id, task in tasks.items()
                    if task.get("protocol") == protocol
                }
                return {"success": True, "tasks": filtered_tasks}
            
            return {"success": True, "tasks": tasks}
            
        except Exception as e:
            logger.error(f"获取任务列表失败: {str(e)}")
            return {"success": False, "error": f"获取任务列表失败: {str(e)}"} 