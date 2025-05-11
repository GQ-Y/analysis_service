"""
任务处理器
负责执行和监控分析任务的执行
"""
import time
import json
import asyncio
import base64
import datetime
import logging
import traceback
import os
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING, Union
from datetime import datetime
from threading import Thread, Event
from queue import Queue
from core.config import settings
from core.models import AnalysisResult
from core.types import BoundingBox
from shared.utils.logger import setup_logger

# 移除MQTT导入
# 导入 json
from shared.utils.tools import get_mac_address

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
    start_time_process = time.time() # 记录进程启动时间
    detector = None # 初始化 detector
    cap = None # 初始化 cap
    try:
        # 1. 初始化检测器
        model_code = stream_config["model"]["code"]
        analysis_type = stream_config["subtask"]["type"]
        
        logger.info(f"工作进程 {task_id}: 开始初始化 {analysis_type} 检测器: {model_code}")
        
        # 根据分析类型选择并初始化检测器
        if analysis_type == "detection":
            from core.analyzer.detection import YOLODetector
            detector = YOLODetector()
        elif analysis_type == "segmentation":
            from core.analyzer.segmentation import YOLOSegmentor
            detector = YOLOSegmentor()
        else:
             # 直接放入队列并返回
             result_queue.put({
                 "type": "error",
                 "message": f"不支持的分析类型: {analysis_type}"
             })
             return

        # 加载模型
        try:
            asyncio.run(detector.load_model(model_code))
            logger.info(f"工作进程 {task_id}: 模型 {model_code} 加载成功")
        except Exception as load_err:
             logger.error(f"工作进程 {task_id}: 加载模型失败: {load_err}")
             result_queue.put({
                 "type": "error",
                 "message": f"加载模型失败: {load_err}"
             })
             return
            
        # 2. 初始化跟踪器（如果需要）
        tracker = None
        track_config = stream_config.get("analysis", {}).get("track_config")
        if track_config and isinstance(track_config, dict) and track_config.get("enabled", True): # 检查 track_config 是否存在且为字典
            try:
                from core.tracker import create_tracker
                tracker_type = track_config.pop("tracker_type", "sort") # pop 会修改字典，后续不再包含 tracker_type
                tracker = create_tracker(tracker_type, **track_config) # 使用剩余配置
                logger.info(f"工作进程 {task_id}: 跟踪器 {tracker_type} 创建成功")
            except Exception as e:
                logger.error(f"工作进程 {task_id}: 创建跟踪器失败: {str(e)}")
                result_queue.put({
                    "type": "error",
                    "message": f"创建跟踪器失败: {str(e)}"
                })
                return
        
        # 3. 打开视频流
        stream_url = stream_config["source"]["urls"][0]
        logger.info(f"工作进程 {task_id}: 尝试打开视频流: {stream_url}")
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            logger.error(f"工作进程 {task_id}: 无法打开视频流: {stream_url}")
            result_queue.put({
                "type": "error",
                "message": f"无法打开视频流: {stream_url}"
            })
            return
        logger.info(f"工作进程 {task_id}: 视频流打开成功")
            
        # 4. 获取流信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"工作进程 {task_id}: 流信息 - FPS: {fps}, 宽度: {width}, 高度: {height}")
        
        # 提取结果配置
        result_config = stream_config.get("result", {})
        save_images = result_config.get("save_images", False)
        return_base64 = result_config.get("return_base64", False)
        
        # 5. 处理每一帧
        frame_count = 0
        analysis_interval = stream_config.get("analysis_interval", 1)
        if not isinstance(analysis_interval, int) or analysis_interval <= 0:
            logger.warning(f"工作进程 {task_id}: 无效的 analysis_interval ({analysis_interval})，使用默认值 1")
            analysis_interval = 1
        
        logger.info(f"工作进程 {task_id}: 开始处理帧，分析间隔: {analysis_interval}")
        
        while not stop_event.is_set():
            frame_start_time = time.time() # 记录帧处理开始时间
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"工作进程 {task_id}: 读取视频流失败，尝试重新连接: {stream_url}")
                cap.release()
                time.sleep(1) # 使用 time.sleep 替代 await asyncio.sleep
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    logger.error(f"工作进程 {task_id}: 重新连接视频流失败: {stream_url}")
                    result_queue.put({
                        "type": "error",
                        "message": f"无法重新连接视频流: {stream_url}"
                    })
                    break # 退出循环
                logger.info(f"工作进程 {task_id}: 视频流重新连接成功")
                continue # 继续下一次循环尝试读取
                
            # 按间隔分析
            if frame_count % analysis_interval == 0:
                frame_process_start_time = time.time()
                inference_time = 0
                pre_process_time = 0
                post_process_time = 0
                detections = []
                mask_data = None # 初始化 mask_data

                try:
                    # 调用 detect 方法，不再需要传递图像标志
                    analysis_result = asyncio.run(detector.detect(frame))

                    # 提取结果和可能的计时信息
                    detections = analysis_result.get("detections", [])
                    inference_time = analysis_result.get("inference_time", 0)
                    pre_process_time = analysis_result.get("pre_process_time", 0)
                    post_process_time = analysis_result.get("post_process_time", 0)
                    # 始终提取标注图像的字节流 (可能为 None)
                    annotated_image_bytes = analysis_result.get("annotated_image_bytes")

                    # 如果是分割任务，尝试获取掩码
                    if analysis_type == "segmentation":
                         mask_data = analysis_result.get("masks") # 假设 detector 返回 'masks' 键

                    # --- 保存标注图像 (仅当 save_images=True) ---
                    annotated_image_save_path = None
                    if save_images and annotated_image_bytes:
                        storage_config = stream_config.get("result", {}).get("storage", {})
                        save_dir_base_config = storage_config.get("save_path", "results")
                        save_dir_base = save_dir_base_config.lstrip('/').lstrip('\\\\')
                        
                        now = datetime.now()
                        date_str = now.strftime("%Y%m%d")
                        time_str = now.strftime("%H%M%S_%f")[:-3] # 到毫秒
                        save_dir_task = os.path.join(save_dir_base, str(task_id), date_str)
                        
                        try:
                            os.makedirs(save_dir_task, exist_ok=True)
                            filename_only = f"{time_str}_{frame_count}"
                            annotated_image_save_path = os.path.join(save_dir_task, f"{filename_only}_annotated.jpg")
                            with open(annotated_image_save_path, "wb") as f:
                                f.write(annotated_image_bytes)
                        except OSError as e:
                             logger.error(f"创建目录或保存标注图像失败: {save_dir_task}, 错误: {e}")
                             annotated_image_save_path = None # 保存失败
                        except Exception as e:
                             logger.error(f"保存标注图像时发生未知错误: {e}")
                             annotated_image_save_path = None # 保存失败
                    # --- 结束保存标注图像 ---

                    # --- 始终进行 Base64 编码 (标注图) ---
                    annotated_image_base64 = None
                    if annotated_image_bytes:
                        try:
                             annotated_image_base64 = base64.b64encode(annotated_image_bytes).decode('utf-8')
                        except Exception as e:
                             logger.error(f"Base64 编码标注图像失败: {e}")
                    # --- 结束 Base64 编码 ---

                    # 过滤指定类别 (确保 analysis_config 和 classes 存在且 classes 是列表)
                    analysis_config = stream_config.get("analysis", {})
                    allowed_classes = analysis_config.get("classes") if isinstance(analysis_config, dict) else None
                    if allowed_classes and isinstance(allowed_classes, list):
                        detections = [
                            det for det in detections 
                            if det.get("class_id") in allowed_classes # 使用 .get() 更安全
                        ]
                    
                    # 应用ROI过滤 (确保 roi_config 存在)
                    roi = stream_config.get("roi")
                    if roi and isinstance(roi, dict):
                         roi_config = roi.get("config")
                         roi_type = roi.get("type", 1) # 默认为1 (假设1代表某种ROI类型)
                         if roi_config: # 仅当 config 存在时过滤
                            detections = detector._filter_by_roi(
                                detections, 
                                roi_config, 
                                roi_type,
                                height, # 使用之前获取的高度
                                width  # 使用之前获取的宽度
                            )
                    
                    # 执行跟踪 (如果 tracker 已初始化)
                    if tracker:
                        tracked_objects = asyncio.run(tracker.update(detections)) # 假设 update 返回对象列表
                        detections = [obj.to_dict() for obj in tracked_objects] # 假设跟踪对象有 to_dict 方法
                    
                    # 准备发送的结果字典
                    frame_process_end_time = time.time()
                    processed_time = frame_process_end_time - frame_process_start_time

                    # 转换 NumPy 类型
                    serializable_detections = convert_numpy_types(detections)
                    serializable_mask_data = convert_numpy_types(mask_data) if mask_data else None

                    # 获取类别名称 (从 detector.model.names 获取)
                    class_names_map = detector.model.names if hasattr(detector, 'model') and hasattr(detector.model, 'names') else {}
                    for det in serializable_detections:
                         class_id = det.get('class_id')
                         det['class_name'] = class_names_map.get(class_id, 'unknown') 

                    # 构建 image_results 字典 (只包含 annotated)
                    image_results_payload = {
                         "annotated": {
                             "format": "jpg",
                             "base64": annotated_image_base64, # 始终包含，可能为 None
                             "save_path": annotated_image_save_path # 可能为 None
                         }
                         # TODO: Add mask image handling if needed
                    }
                    # 如果 base64 和 save_path 都为 None，则不发送 image_results
                    if annotated_image_base64 is None and annotated_image_save_path is None:
                         image_results_payload = None 

                    result_payload = {
                        "type": "result",
                        "frame_id": frame_count,
                        "timestamp": time.time(),
                        "detections": serializable_detections,
                        "frame_info": {
                            "width": width,
                            "height": height,
                            "fps": fps,
                            "processed_time": processed_time # 单帧处理时间
                        },
                        "analysis_info": {
                             "inference_time": inference_time,
                             "pre_process_time": pre_process_time,
                             "post_process_time": post_process_time
                        },
                        "mask_data": serializable_mask_data, # 添加掩码数据
                        "image_results": image_results_payload # 添加图像结果 (可能为 None)
                    }
                    result_queue.put(result_payload)
                    
                except Exception as e:
                    logger.error(f"工作进程 {task_id}: 处理帧 {frame_count} 时出错: {str(e)}")
                    import traceback
                    logger.error(f"详细错误: {traceback.format_exc()}")
                    # 不因为单帧错误而停止整个进程，继续处理下一帧
                    continue # 跳过当前帧的处理
                    
            frame_count += 1
            # 短暂休眠避免CPU占用过高（可选）
            time.sleep(0.001) # 使用 time.sleep
            
        # 6. 清理资源
        if cap:
             cap.release()
        logger.info(f"工作进程 {task_id}: 视频流已释放")
        
        # 7. 发送完成消息
        total_processed_frames = frame_count // analysis_interval
        end_time_process = time.time()
        total_duration = end_time_process - start_time_process
        logger.info(f"工作进程 {task_id}: 处理完成，总帧数: {frame_count}, 已分析帧数: {total_processed_frames}, 总耗时: {total_duration:.2f}s")
        result_queue.put({
            "type": "complete",
            "total_frames": frame_count,
            "processed_frames": total_processed_frames,
            "duration": total_duration
        })
        
    except Exception as e:
        logger.error(f"工作进程 {task_id}: 发生未处理的严重错误: {str(e)}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        # 发送错误消息到主进程
        result_queue.put({
            "type": "error",
            "message": f"工作进程发生严重错误: {str(e)}"
        })
    finally:
        # 确保资源被释放
        if cap and cap.isOpened():
            cap.release()
        logger.info(f"工作进程 {task_id}: 退出")

class TaskProcessor:
    """
    任务处理器
    负责处理各种类型的分析任务
    """
    
    def __init__(self, task_manager: 'TaskManager'):
        """
        初始化任务处理器
        
        Args:
            task_manager: 任务管理器实例
        """
        self.task_manager = task_manager
        # 移除MQTT管理器
        
        # 任务进程映射
        self.task_processes = {}
        # 任务结果队列映射
        self.task_result_queues = {}
        # 任务停止事件映射
        self.task_stop_events = {}
        
        # 为每个任务创建一个异步处理协程
        self.task_handlers = {}
        
        logger.info("任务处理器初始化完成")
        
    async def start_stream_analysis(self, task_id: str, task_config: Dict[str, Any]) -> bool:
        """
        启动流分析任务
        
        Args:
            task_id: 任务ID
            task_config: 任务配置
            
        Returns:
            bool: 是否启动成功
        """
        try:
            # 检查任务是否已经存在
            if task_id in self.task_processes:
                logger.warning(f"任务已经在运行中: {task_id}")
                return False
                
            # 创建结果队列和停止事件
            result_queue = Queue()
            stop_event = Event()
            
            # 存储队列和事件
            self.task_result_queues[task_id] = result_queue
            self.task_stop_events[task_id] = stop_event
            
            # 创建处理进程
            logger.info(f"创建流处理进程: {task_id}")
            
            # 启动处理线程
            process = Thread(
                target=process_stream_worker,
                args=(task_id, task_config, stop_event, result_queue),
                daemon=True
            )
            
            # 存储进程对象
            self.task_processes[task_id] = process
            
            # 启动进程
            process.start()
            logger.info(f"流处理进程已启动: {task_id}")
            
            # 启动结果处理器
            result_handler = asyncio.create_task(self._handle_results(task_id, result_queue))
            self.task_handlers[task_id] = result_handler
            
            return True
            
        except Exception as e:
            logger.error(f"启动流分析任务失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    async def _handle_results(self, task_id: str, result_queue: Queue):
        """
        处理结果队列中的消息
        
        Args:
            task_id: 任务ID
            result_queue: 结果队列
        """
        try:
            logger.info(f"启动结果处理器: {task_id}")
            
            while task_id in self.task_processes:
                # 检查队列是否有结果
                if not result_queue.empty():
                    # 获取结果
                    result = result_queue.get()
                    
                    # 处理结果类型
                    result_type = result.get("type")
                    
                    if result_type == "error":
                        # 处理错误
                        error_message = result.get("message", "未知错误")
                        logger.error(f"任务 {task_id} 发生错误: {error_message}")
                        
                        # 更新任务状态
                        self.task_manager.update_task_status(
                            task_id, 
                            "failed",
                            error=error_message
                        )
                        
                        # 发送错误通知（日志记录代替MQTT通知）
                        logger.info(f"任务 {task_id} 发送错误通知: {error_message}")
                        
                    elif result_type == "result":
                        # 处理分析结果
                        detections = result.get("detections", [])
                        metadata = result.get("metadata", {})
                        frame_number = metadata.get("frame_number", 0)
                        timestamp = metadata.get("timestamp", time.time())
                        
                        # 日志记录检测结果数量
                        logger.debug(f"任务 {task_id} 处理结果: {len(detections)} 个检测结果, 帧 #{frame_number}")
                        
                        # 构建结果对象
                        analysis_result = {
                            "task_id": task_id,
                            "timestamp": timestamp,
                            "frame_number": frame_number,
                            "detections": detections,
                            "metadata": metadata
                        }
                        
                        # 更新任务状态
                        self.task_manager.update_task_status(
                            task_id, 
                            "processing",
                            result=analysis_result
                        )
                        
                        # 发送结果通知（日志记录代替MQTT通知）
                        if "image_base64" in result:
                            logger.debug(f"任务 {task_id} 发送分析结果: 帧 #{frame_number}, 附带图像数据")
                        else:
                            logger.debug(f"任务 {task_id} 发送分析结果: 帧 #{frame_number}")
                            
                    elif result_type == "complete":
                        # 处理完成通知
                        logger.info(f"任务 {task_id} 完成")
                        
                        # 更新任务状态
                        self.task_manager.update_task_status(
                            task_id, 
                            "completed"
                        )
                        
                        # 发送完成通知（日志记录代替MQTT通知）
                        logger.info(f"任务 {task_id} 发送完成通知")
                        
                        # 清理资源
                        if task_id in self.task_processes:
                            del self.task_processes[task_id]
                        if task_id in self.task_stop_events:
                            del self.task_stop_events[task_id]
                        if task_id in self.task_result_queues:
                            del self.task_result_queues[task_id]
                            
                        break
                        
                    else:
                        # 未知结果类型
                        logger.warning(f"任务 {task_id} 收到未知结果类型: {result_type}")
                        
                # 等待一段时间再检查队列
                await asyncio.sleep(0.01)
                
            logger.info(f"结果处理器结束: {task_id}")
            
        except Exception as e:
            logger.error(f"处理结果时出错: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 更新任务状态
            self.task_manager.update_task_status(
                task_id, 
                "failed",
                error=f"处理结果时出错: {str(e)}"
            )
            
        finally:
            # 确保清理资源
            if task_id in self.task_handlers:
                del self.task_handlers[task_id]
                
    async def stop_task(self, task_id: str) -> Dict[str, Any]:
        """
        停止任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 停止结果
        """
        try:
            # 检查任务是否存在
            if task_id not in self.task_processes:
                logger.warning(f"要停止的任务不存在或已经停止: {task_id}")
                return {"success": False, "error": "任务不存在或已经停止"}
                
            # 设置停止事件
            if task_id in self.task_stop_events:
                logger.info(f"设置任务停止事件: {task_id}")
                self.task_stop_events[task_id].set()
                
            # 等待进程结束
            if task_id in self.task_processes:
                logger.info(f"等待任务进程结束: {task_id}")
                self.task_processes[task_id].join(timeout=5)
                
                # 检查进程是否仍然活动
                if self.task_processes[task_id].is_alive():
                    logger.warning(f"任务进程未能在超时时间内结束: {task_id}")
                    
                # 清理任务
                del self.task_processes[task_id]
                
            # 清理其他资源
            if task_id in self.task_stop_events:
                del self.task_stop_events[task_id]
            if task_id in self.task_result_queues:
                del self.task_result_queues[task_id]
                
            # 取消结果处理器
            if task_id in self.task_handlers:
                logger.info(f"取消结果处理器: {task_id}")
                self.task_handlers[task_id].cancel()
                del self.task_handlers[task_id]
                
            # 发送停止通知（日志记录代替MQTT通知）
            logger.info(f"任务 {task_id} 发送停止通知")
                
            return {"success": True}
            
        except Exception as e:
            logger.error(f"停止任务时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
            
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 任务状态
        """
        try:
            # 获取任务信息
            task = self.task_manager.get_task(task_id)
            if not task:
                logger.warning(f"任务不存在: {task_id}")
                return {
                    "success": False,
                    "error": "任务不存在"
                }
                
            # 检查任务是否在运行中
            is_running = task_id in self.task_processes
            
            # 构建状态信息
            status_info = {
                "task_id": task_id,
                "status": task.get("status", "unknown"),
                "is_running": is_running,
                "create_time": task.get("created_at", 0),
                "update_time": task.get("updated_at", 0),
                "result": task.get("result"),
                "error": task.get("error")
            }
            
            return {
                "success": True,
                "status": status_info
            }
            
        except Exception as e:
            logger.error(f"获取任务状态时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
            
    async def get_tasks(self, protocol: Optional[str] = None) -> Dict[str, Any]:
        """
        获取所有任务状态
        
        Args:
            protocol: 协议类型过滤(http)
            
        Returns:
            Dict[str, Any]: 任务状态列表
        """
        try:
            # 获取所有任务
            tasks = self.task_manager.get_all_tasks()
            
            # 构建任务状态列表
            task_list = []
            for task in tasks:
                task_id = task.get("id")
                is_running = task_id in self.task_processes
                
                # 构建任务状态信息
                task_info = {
                    "task_id": task_id,
                    "status": task.get("status", "unknown"),
                    "is_running": is_running,
                    "create_time": task.get("created_at", 0),
                    "update_time": task.get("updated_at", 0),
                }
                
                task_list.append(task_info)
                
            return {
                "success": True,
                "tasks": task_list
            }
            
        except Exception as e:
            logger.error(f"获取任务列表时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            } 