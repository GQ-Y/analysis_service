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
from typing import Dict, Any, Optional, Union
from datetime import datetime
from scipy.optimize import linear_sum_assignment
import json

from core.task_management.manager import TaskManager
from core.task_management.utils.status import TaskStatus
from core.analyzer.detection import YOLODetector
from shared.utils.logger import setup_logger
from core.config import settings

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

class TaskProcessor:
    """任务处理器，处理分析任务"""
    
    def __init__(self, task_manager: TaskManager):
        """
        初始化任务处理器
        
        Args:
            task_manager: 任务管理器实例
        """
        self.task_manager = task_manager
        self.detectors = {}  # 存储已加载的检测器
        self.active_tasks = {}  # 存储活动任务的线程或协程
        
        logger.info("任务处理器已初始化")
        
    async def get_detector(self, model_code: str, analysis_type: str = "detection") -> Union[YOLODetector, Any]:
        """
        获取或创建检测器
        
        Args:
            model_code: 模型代码
            analysis_type: 分析类型（detection/segmentation）
            
        Returns:
            Union[YOLODetector, Any]: 检测器实例
        """
        detector_key = f"{model_code}_{analysis_type}"
        if detector_key not in self.detectors:
            try:
                logger.info(f"加载{analysis_type}模型: {model_code}")
                
                if analysis_type == "detection":
                    from core.analyzer.detection import YOLODetector
                    self.detectors[detector_key] = YOLODetector()
                    await self.detectors[detector_key].load_model(model_code)
                elif analysis_type == "segmentation":
                    from core.analyzer.segmentation import YOLOSegmentor
                    self.detectors[detector_key] = YOLOSegmentor()
                    await self.detectors[detector_key].load_model(model_code)
                else:
                    raise ValueError(f"不支持的分析类型: {analysis_type}")
                    
            except Exception as e:
                logger.error(f"加载模型失败: {model_code}, 错误: {str(e)}")
                raise ValueError(f"加载模型失败: {model_code}, 错误: {str(e)}")
                
        return self.detectors[detector_key]
        
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
            
            task_config = task.get("data", {})
            if not task_config:
                logger.error(f"任务配置为空: {task_id}")
                return False
            
            # 2. 检查任务是否已经在运行
            if task_id in self.active_tasks:
                logger.warning(f"任务已经在运行: {task_id}")
                return False
            
            # 3. 获取检测器
            try:
                model_code = task_config["model"]["code"]
                analysis_type = task_config["subtask"]["type"]
                detector = await self.get_detector(model_code, analysis_type)
            except Exception as e:
                logger.error(f"加载检测器失败: {str(e)}")
                self.task_manager.update_task_status(
                    task_id,
                    TaskStatus.FAILED,
                    error=f"加载检测器失败: {str(e)}"
                )
                return False
            
            # 4. 创建跟踪器（如果需要）
            tracker = None
            if task_config["analysis"].get("track_config"):
                try:
                    from core.tracker import create_tracker
                    track_config = task_config["analysis"]["track_config"]
                    tracker_type = track_config.pop("tracker_type", "sort")  # 获取并移除tracker_type
                    tracker = create_tracker(tracker_type, **track_config)
                    logger.info(f"创建跟踪器成功: {tracker_type}")
                except Exception as e:
                    logger.error(f"创建跟踪器失败: {str(e)}")
                    self.task_manager.update_task_status(
                        task_id,
                        TaskStatus.FAILED,
                        error=f"创建跟踪器失败: {str(e)}"
                    )
                    return False
            
            # 5. 启动后台任务处理流
            try:
                # 更新任务状态为处理中
                self.task_manager.update_task_status(
                    task_id,
                    TaskStatus.PROCESSING
                )
                
                # 创建异步任务
                task = asyncio.create_task(
                    self._process_stream(
                        task_id=task_id,
                        detector=detector,
                        tracker=tracker,
                        task_config=task_config
                    )
                )
                
                # 保存任务引用
                self.active_tasks[task_id] = task
                
                logger.info(f"流分析任务启动成功: {task_id}")
                return True
                
            except Exception as e:
                logger.error(f"启动任务处理失败: {str(e)}")
                self.task_manager.update_task_status(
                    task_id,
                    TaskStatus.FAILED,
                    error=f"启动任务处理失败: {str(e)}"
                )
                return False
                
        except Exception as e:
            logger.error(f"启动流分析任务失败: {str(e)}")
            import traceback
            logger.error(f"错误详情:\n{traceback.format_exc()}")
            
            # 更新任务状态为失败
            self.task_manager.update_task_status(
                task_id,
                TaskStatus.FAILED,
                error=str(e)
            )
            return False
            
    async def _process_stream(
        self,
        task_id: str,
        detector: YOLODetector,
        tracker: Optional[Any],
        task_config: Dict[str, Any]
    ):
        """
        处理流分析任务
        
        Args:
            task_id: 任务ID
            detector: 检测器实例
            tracker: 跟踪器实例（可选）
            task_config: 任务配置
        """
        try:
            # 检查必要的配置
            if not task_config.get("result"):
                raise ValueError("缺少result配置")
            
            result_config = task_config["result"]
            save_images = result_config.get("save_images", False)
            save_result = result_config.get("save_result", False)
            return_base64 = result_config.get("return_base64", False)
            
            if not task_config.get("subtask"):
                raise ValueError("缺少subtask配置")
                
            subtask_info = task_config["subtask"]
            enable_callback = subtask_info.get("callback", {}).get("enabled", False)
            callback_url = subtask_info.get("callback", {}).get("url", "")
            
            # 1. 获取流URL
            if not task_config.get("source") or not task_config["source"].get("urls"):
                raise ValueError("缺少视频流URL配置")
            stream_url = task_config["source"]["urls"][0]
            
            # 2. 打开视频流
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                raise Exception(f"无法打开视频流: {stream_url}")
                
            # 3. 获取流信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"视频流信息 - FPS: {fps}, 分辨率: {width}x{height}")
            
            # 设置检测器参数
            if task_config.get("analysis"):
                detector.model.conf = task_config["analysis"].get("confidence", detector.default_confidence)
                detector.model.iou = task_config["analysis"].get("iou", detector.default_iou)
            
            # 4. 处理每一帧
            frame_count = 0
            analysis_interval = task_config.get("analysis_interval", 1)
            
            # 准备保存目录
            save_dir = None
            if save_images:
                try:
                    # 获取项目根目录
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                    
                    # 使用配置中的相对路径
                    storage_config = result_config.get("storage", {})
                    relative_path = storage_config.get("save_path", "results").lstrip('/')
                    
                    # 完整的保存目录路径
                    base_save_dir = os.path.join(project_root, relative_path)
                    os.makedirs(base_save_dir, exist_ok=True)
                    logger.info(f"结果保存根目录: {base_save_dir}")
                    
                    # 获取文件命名模式
                    file_pattern = storage_config.get("file_pattern", "{task_id}/{date}/{time}_{frame_id}.jpg")
                    
                    # 预创建任务目录
                    task_dir = os.path.join(base_save_dir, task_id)
                    os.makedirs(task_dir, exist_ok=True)
            
                except Exception as e:
                    logger.error(f"创建保存目录失败: {str(e)}")
                    save_images = False  # 如果创建目录失败,禁用图片保存
            
            while True:
                # 检查任务是否被取消
                task = self.task_manager.get_task(task_id)
                if not task or task["status"] == TaskStatus.STOPPING:
                    logger.info(f"任务已停止: {task_id}")
                    break
                    
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"读取视频流失败，尝试重新连接: {stream_url}")
                    # 重新连接
                    cap.release()
                    cap = cv2.VideoCapture(stream_url)
                    continue
                
                # 记录开始时间
                start_time = time.time()
                
                # 按间隔分析
                if frame_count % analysis_interval == 0:
                    try:
                        # 执行检测，只在检测到目标时打印日志
                        result = await detector.detect(frame, verbose=False)
                        detections = result.get("detections", [])
                
                        # 过滤指定类别
                        if task_config.get("analysis", {}).get("classes"):
                            allowed_classes = task_config["analysis"]["classes"]
                            detections = [
                                det for det in detections 
                                if det["class_id"] in allowed_classes
                            ]
                        
                        # 应用ROI过滤
                        if task_config.get("roi"):
                            roi_config = task_config["roi"]["config"]
                            roi_type = task_config["roi"].get("type", 1)
                            detections = detector._filter_by_roi(
                                detections, 
                                roi_config, 
                                roi_type,
                                height,
                                width
                            )
                        
                        # 执行跟踪
                        if tracker:
                            tracked_objects = await tracker.update(detections)
                            # 将TrackingObject对象转换为字典
                            detections = [obj.to_dict() for obj in tracked_objects]
                        
                        # 计算处理时间
                        process_time = time.time() - start_time
                        
                        # 准备分析结果
                        objects = []
                        for det in detections:
                            # 检查是否是跟踪对象
                            if "track_id" in det:
                                # 跟踪对象的处理
                                obj = {
                                    "class_id": det["class_id"],
                                    "class_name": det.get("class_name", ""),  # 可能需要从类别映射中获取
                                    "confidence": det["confidence"],
                                    "track_id": det["track_id"],
                                    "bbox": det["bbox"],
                                    "track_info": det.get("track_info", {})
                                }
                            else:
                                # 普通检测对象的处理
                                obj = {
                                    "class_id": det["class_id"],
                                    "class_name": det["class_name"],
                                    "confidence": det["confidence"],
                                    "bbox": [
                                        det["bbox"]["x1"],
                                        det["bbox"]["y1"],
                                        det["bbox"]["x2"],
                                        det["bbox"]["y2"]
                                    ]
                                }
                            objects.append(obj)
                        
                        # 只在检测到目标时保存结果
                        image_results = {}
                        if detections and save_images:
                            try:
                                # 生成文件名
                                timestamp = datetime.now()
                                filename = file_pattern.format(
                                    task_id=task_id,
                                    date=timestamp.strftime("%Y%m%d"),
                                    time=timestamp.strftime("%H%M%S"),
                                    frame_id=frame_count
                                )
                                
                                # 完整保存路径
                                save_path = os.path.join(base_save_dir, filename)
                                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                
                                # 转换检测结果为绘制格式
                                draw_detections = []
                                for det in detections:
                                    draw_det = {
                                        "bbox": det["bbox"] if isinstance(det["bbox"], dict) else {
                                            "x1": float(det["bbox"][0]),
                                            "y1": float(det["bbox"][1]),
                                            "x2": float(det["bbox"][2]),
                                            "y2": float(det["bbox"][3])
                                        },
                                        "class_id": det["class_id"],
                                        "class_name": det.get("class_name", ""),
                                        "confidence": det["confidence"],
                                        "label": f"{det.get('class_name', '')} {det.get('track_id', '')}" if "track_id" in det else None
                                    }
                                    draw_detections.append(draw_det)
                                
                                # 保存图片
                                result_image = await detector.draw_detections(frame, draw_detections)
                                cv2.imwrite(save_path, result_image)
                                logger.debug(f"检测到 {len(detections)} 个目标，已保存结果图片: {save_path}")
                                
                                # 如果需要返回base64
                                if return_base64:
                                    image_format = result_config.get("image_format", {})
                                    _, buffer = cv2.imencode(
                                        f".{image_format.get('format', 'jpg')}", 
                                        result_image,
                                        [cv2.IMWRITE_JPEG_QUALITY, image_format.get('quality', 95)]
                                    )
                                    image_results = {
                                        "annotated": {
                                            "format": image_format.get('format', 'jpg'),
                                            "base64": base64.b64encode(buffer).decode('utf-8'),
                                            "save_path": save_path
                                        }
                                    }
                            except Exception as e:
                                logger.error(f"保存图片失败: {str(e)}")
                        
                        # 准备结果数据
                        result_data = {
                            "task_id": task_id,
                            "subtask_id": subtask_info["id"],
                            "status": "0",  # 进行中
                            "progress": 1,  # 流分析永远是1
                            "timestamp": int(time.time()),
                            "result": {
                                "frame_id": frame_count,
                                "objects": objects,
                                "frame_info": {
                                    "width": width,
                                    "height": height,
                                    "processed_time": process_time,
                                    "frame_index": frame_count,
                                    "timestamp": int(time.time()),
                                    "source_info": {
                                        "type": "stream",
                                        "path": stream_url,
                                        "frame_rate": fps
                                    }
                                },
                                "image_results": image_results,
                                "analysis_info": {
                                    "model_name": task_config["model"]["code"],
                                    "model_version": task_config["model"]["version"],
                                    "inference_time": process_time,
                                    "device": task_config["model"]["device"],
                                    "batch_size": task_config["model"]["batch_size"]
                                }
                            }
                        }
                        
                        # 更新任务状态
                        self.task_manager.update_task_status(
                            task_id,
                            TaskStatus.PROCESSING,
                            result=result_data
                        )
                        
                        # 如果启用了回调,发送结果到回调地址
                        if enable_callback and callback_url:
                            try:
                                # 转换numpy类型为Python原生类型
                                result_data = convert_numpy_types(result_data)
                                async with aiohttp.ClientSession() as session:
                                    async with session.post(callback_url, json=result_data) as response:
                                        if response.status != 200:
                                            logger.error(f"回调请求失败: {response.status}")
                            except Exception as e:
                                logger.error(f"发送回调请求失败: {str(e)}")
                        
                    except Exception as e:
                        logger.error(f"处理帧时出错: {str(e)}")
                        continue
                
                frame_count += 1
            
            # 5. 清理资源
            cap.release()
            
            # 6. 更新任务状态
            final_status = TaskStatus.STOPPED if task["status"] == TaskStatus.STOPPING else TaskStatus.COMPLETED
            self.task_manager.update_task_status(
                task_id,
                final_status,
                result={
                    "total_frames": frame_count,
                    "processed_frames": frame_count // analysis_interval,
                    "save_dir": base_save_dir if save_images else None
                }
            )
            
        except Exception as e:
            logger.error(f"处理流分析任务失败: {str(e)}")
            import traceback
            logger.error(f"错误详情:\n{traceback.format_exc()}")
            
            # 更新任务状态为失败
            self.task_manager.update_task_status(
                task_id,
                TaskStatus.FAILED,
                error=str(e)
            )
            
        finally:
            # 从活动任务中移除
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
                
    async def stop_task(self, task_id: str) -> Dict[str, Any]:
        """
        停止任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 停止结果
        """
        try:
            # 获取任务信息
            task = self.task_manager.get_task(task_id)
            if not task:
                return {"success": False, "error": f"任务不存在: {task_id}"}
                
            # 获取任务状态
            status = task.get("status")
            
            # 如果任务已经完成或失败，无需停止
            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED]:
                return {
                    "success": True,
                    "task_id": task_id,
                    "message": f"任务已经处于 {status} 状态，无需停止"
                }
                
            # 更新任务状态为正在停止
            self.task_manager.update_task_status(task_id, TaskStatus.STOPPING)
            logger.info(f"正在停止任务: {task_id}")
            
            # 等待任务停止（由处理线程自行停止）
            return {
                "success": True,
                "task_id": task_id,
                "message": "正在停止任务"
            }
            
        except Exception as e:
            logger.error(f"停止任务失败: {str(e)}")
            return {"success": False, "error": f"停止任务失败: {str(e)}"}
            
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