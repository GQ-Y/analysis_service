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
import shutil # 导入 shutil 模块

# from core.task_management.manager import TaskManager # 移除顶层导入
from core.task_management.utils.status import TaskStatus
from core.analyzer.detection import YOLODetector
from shared.utils.logger import setup_logger
from core.config import settings
# 导入 MqttManager 用于类型提示
from services.mqtt import MQTTManager
# 导入 get_mac_address 和 json
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
    """任务处理器，处理分析任务"""
    
    def __init__(self, task_manager: 'TaskManager', mqtt_manager: MQTTManager):
        """
        初始化任务处理器
        
        Args:
            task_manager: 任务管理器实例
            mqtt_manager: MQTT管理器实例
        """
        self.task_manager = task_manager
        self.mqtt_manager = mqtt_manager # 存储 MQTT 管理器实例
        self.active_tasks = {}  # 存储活动任务的进程 {task_id: Process}
        self.stop_events = {}  # 存储任务停止事件 {task_id: Event}
        self.result_queues = {}  # 存储任务结果队列 {task_id: Queue}
        self.result_handlers = {} # 存储结果处理协程句柄 {task_id: asyncio.Task}
        
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
                    } if roi_intermediate and isinstance(roi_intermediate, dict) and roi_intermediate.get("config") else None,
                    # 添加 result 配置
                    "result": task_data.get("result") if isinstance(task_data.get("result"), dict) else {}
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
                return False
            
            # 4. 创建停止事件和结果队列
            stop_event = Event()
            result_queue = Queue()

            # 5. 创建并启动任务进程 (传递干净的 worker_config)
            process = Process(
                target=process_stream_worker,
                args=(task_id, worker_config, stop_event, result_queue)
            )
            process.start()

            # 6. 保存进程引用和控制对象
            self.active_tasks[task_id] = process
            self.stop_events[task_id] = stop_event
            self.result_queues[task_id] = result_queue

            # 7. 启动并保存结果处理协程句柄
            result_handler_task = asyncio.create_task(self._handle_results(task_id, result_queue))
            self.result_handlers[task_id] = result_handler_task
                
            logger.info(f"流分析任务启动成功: {task_id}")
            return True
                
        except Exception as e:
            error_msg = f"启动流分析任务时发生意外错误: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"错误详情:\n{traceback.format_exc()}")
            if 'task_id' in locals():
                 self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
            return False
            
    async def _handle_results(self, task_id: str, result_queue: Queue):
        """
        处理来自工作进程的任务结果
        
        Args:
            task_id: 任务ID
            result_queue: 结果队列
        """
        task_info = None # 初始化 task_info
        try:
            # 提前获取一次任务信息，主要为了获取配置
            task_info = self.task_manager.get_task(task_id)
            if not task_info:
                logger.error(f"处理结果时未找到任务信息: {task_id}")
                return # 任务信息是后续处理的基础，找不到则直接退出

            task_data = task_info.get("data", {})
            if not task_data:
                logger.error(f"任务 {task_id} 配置数据为空，无法处理结果")
                return

            result_config = task_data.get("result", {})
            subtask_info = task_data.get("subtask", {})
            model_config = task_data.get("model", {})
            source_config = task_data.get("source", {})

            mac_address = get_mac_address()

            # 确定目标主题
            callback_topic = result_config.get("callback_topic")
            if not callback_topic:
                target_topic = f"/meek/{mac_address}/result"
            else:
                target_topic = callback_topic
            
            while True:
                # 检查任务是否已停止 (主进程中控制)
                current_status = self.task_manager.get_task_status(task_id)
                if current_status in [TaskStatus.STOPPED, TaskStatus.FAILED, TaskStatus.COMPLETED]:
                     logger.info(f"任务 {task_id} 已停止，结果处理循环退出")
                     break
                # 检查任务是否还在 active_tasks 中 (以防万一)
                if task_id not in self.active_tasks:
                    logger.warning(f"任务 {task_id} 不在 active_tasks 中，结果处理循环退出")
                    break
                    
                # 非阻塞方式获取结果
                try:
                    worker_result = result_queue.get_nowait()
                except:
                    await asyncio.sleep(0.01) # 短暂等待
                    continue
                
                # 处理不同类型的结果
                if worker_result["type"] == "result":
                    # 1. 更新任务状态 (标记为处理中，并存储最新结果)
                    # 注意：存储的 result 现在是 worker_result 的原始结构
                    self.task_manager.update_task_status(
                        task_id,
                        TaskStatus.PROCESSING,
                        result=worker_result 
                    )

                    # 2. 构建符合目标格式的 MQTT 消息体
                    # 注意：这里的 result_mqtt 可能需要根据实际需求调整
                    # 特别是分析信息和帧信息的字段结构可能会有所不同
                    # 3. 发布结果到 MQTT 主题
                    # 注意：这里的发布逻辑需要确保消息格式符合目标主题的要求

                    # 调试输出 worker_result 的结构
                    logger.debug(f"任务 {task_id} - worker_result: {worker_result}")

                    # 如果是单帧结果，处理单帧数据
                    try:
                        # 提取关联信息
                        message_id = task_data.get("message_id")
                        message_uuid = task_data.get("message_uuid")
                        subtask_id = subtask_info.get("id")

                        # 提取帧和分析信息
                        frame_info_worker = worker_result.get("frame_info", {})
                        analysis_info_worker = worker_result.get("analysis_info", {})
                        mask_data_worker = worker_result.get("mask_data")

                        # --- 构建 objects 列表 --- 
                        objects = []
                        for det in worker_result.get("detections", []):
                            obj_data = {
                                "class_id": det.get("class_id"),
                                "class_name": det.get("class_name", "unknown"), # 使用从 worker 传来的 class_name
                                "confidence": det.get("confidence"),
                                "bbox": det.get("bbox")
                                # ---- 暂未实现字段 ----
                                # "attributes": {},
                                # "keypoints": [],
                                # "nested_objects": []
                            }
                            if "track_id" in det:
                                obj_data["track_id"] = det.get("track_id")
                            # 如果有掩码数据 (假设分割任务的 detection 也包含掩码引用或数据)
                            if mask_data_worker and det.get('mask_ref'): # 假设 det 有 mask_ref 指向 mask_data
                                 obj_data["mask"] = {
                                     "format": "rle", # 假设是 RLE 格式
                                     "data": mask_data_worker.get(det['mask_ref']), # 通过引用获取
                                     "size": [frame_info_worker.get("width"), frame_info_worker.get("height")]
                                 }
                            elif mask_data_worker and isinstance(mask_data_worker, dict) and analysis_info_worker.get("type") == "segmentation": # 另一种可能是掩码直接在顶层
                                 # 这个逻辑取决于分割器如何返回掩码，需要适配
                                 # 假设 mask_data_worker 是 {'format': 'rle', 'data': '...', 'size': [w,h]}
                                 obj_data["mask"] = mask_data_worker 

                            objects.append(obj_data)

                        # --- 构建 frame_info --- 
                        frame_info_mqtt = {
                            "width": frame_info_worker.get("width"),
                            "height": frame_info_worker.get("height"),
                            "processed_time": frame_info_worker.get("processed_time"),
                            "frame_index": worker_result.get("frame_id"),
                            "timestamp": int(worker_result.get("timestamp", time.time())),
                            "source_info": {
                                "type": source_config.get("type"),
                                "path": source_config.get("urls", [None])[0], # 取第一个 URL 作为路径
                                "frame_rate": frame_info_worker.get("fps")
                            }
                        }
                        
                        # --- 构建 analysis_info --- 
                        analysis_info_mqtt = {
                            "model_name": model_config.get("code"),
                            "model_version": model_config.get("version"),
                            "inference_time": analysis_info_worker.get("inference_time"),
                            "pre_process_time": analysis_info_worker.get("pre_process_time"),
                            "post_process_time": analysis_info_worker.get("post_process_time"),
                            "device": model_config.get("device"),
                            "batch_size": model_config.get("batch_size", 1)
                        }

                        # --- 构建完整的 result --- 
                        result_mqtt = {
                            "frame_id": worker_result.get("frame_id"),
                                "objects": objects,
                            "frame_info": frame_info_mqtt,
                            "analysis_info": analysis_info_mqtt
                            # ---- 暂未实现字段 ----
                            # "scene_understanding": {}
                        }
                        # 添加 image_results (如果存在)
                        image_results_payload = worker_result.get("image_results")
                        if image_results_payload:
                            result_mqtt["image_results"] = image_results_payload

                        # 构建最终 MQTT 消息体
                        mqtt_message = {
                            "message_id": message_id,
                            "message_uuid": message_uuid,
                            "message_type": 80003, # 响应消息类型
                            "mac_address": mac_address,
                            "data": {
                                "task_id": task_id,
                                "subtask_id": subtask_id,
                                "status": "0", # 0: 进行中 (因为这是单帧结果)
                                "progress": 0, # 流式处理的进度不好确定，暂设为0
                                "timestamp": int(worker_result.get("timestamp", time.time())), # 使用 worker 的时间戳
                                "result": result_mqtt
                            }
                        }

                        # 序列化并发布
                        json_payload = json.dumps(mqtt_message)
                        asyncio.create_task(self.mqtt_manager.publish(target_topic, json_payload))
                        logger.debug(f"任务 {task_id} 结果已按新格式发布到主题: {target_topic}")

                    except Exception as build_err:
                        logger.error(f"任务 {task_id} 构建或发布 MQTT 结果时出错: {build_err}")
                        import traceback
                        logger.error(f"详细错误: {traceback.format_exc()}")
                        # 可选：标记任务失败
                        # self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=f"构建结果失败: {build_err}")

                elif worker_result["type"] == "error":
                    logger.error(f"工作进程 {task_id} 报告错误: {worker_result['message']}")
                    # 更新任务状态为失败
                    self.task_manager.update_task_status(
                        task_id,
                        TaskStatus.FAILED,
                        error=worker_result["message"]
                    )
                    # 发送失败状态的MQTT消息
                    try:
                         error_message = {
                            "message_id": task_data.get("message_id"),
                            "message_uuid": task_data.get("message_uuid"),
                            "message_type": 80003, # 响应消息类型
                            "mac_address": mac_address,
                            "data": {
                                "task_id": task_id,
                                "subtask_id": subtask_info.get("id"),
                                "status": "-1", # -1: 分析失败
                                "progress": 0,
                                "timestamp": int(time.time()),
                                "result": { # 包含错误信息
                                    "error_message": worker_result['message']
                                }
                            }
                         }
                         json_payload = json.dumps(error_message)
                         asyncio.create_task(self.mqtt_manager.publish(target_topic, json_payload))
                         logger.info(f"任务 {task_id} 的失败状态已发布到主题: {target_topic}")
                    except Exception as pub_err:
                         logger.error(f"任务 {task_id} 发布失败状态时出错: {pub_err}")

                    break # 工作进程出错，退出处理循环

                elif worker_result["type"] == "complete":
                    logger.info(f"工作进程 {task_id} 处理完成: {worker_result}")
                    # 更新任务状态为完成
                    self.task_manager.update_task_status(
                        task_id,
                        TaskStatus.COMPLETED,
                        result=worker_result # 存储完成信息
                    )
                     # 发送完成状态的MQTT消息
                    try:
                         complete_message = {
                            "message_id": task_data.get("message_id"),
                            "message_uuid": task_data.get("message_uuid"),
                            "message_type": 80003, # 响应消息类型
                            "mac_address": mac_address,
                            "data": {
                                "task_id": task_id,
                                "subtask_id": subtask_info.get("id"),
                                "status": "1", # 1: 已完成
                                "progress": 100,
                                "timestamp": int(time.time()),
                                "result": { # 可以包含完成时的统计信息
                                    "total_frames": worker_result.get("total_frames"),
                                    "processed_frames": worker_result.get("processed_frames"),
                                    "duration": worker_result.get("duration")
                                }
                            }
                         }
                         json_payload = json.dumps(complete_message)
                         asyncio.create_task(self.mqtt_manager.publish(target_topic, json_payload))
                         logger.info(f"任务 {task_id} 的完成状态已发布到主题: {target_topic}")
                    except Exception as pub_err:
                         logger.error(f"任务 {task_id} 发布完成状态时出错: {pub_err}")

                    break # 任务完成，退出处理循环
                    
        except asyncio.CancelledError:
            logger.info(f"结果处理协程被取消: {task_id}")
            # 协程被取消时，不需要再更新任务状态，由 stop_task 统一处理
            # 可以在这里添加额外的清理逻辑（如果需要）
        except Exception as e:
            logger.error(f"处理结果循环 ({task_id}) 发生意外错误: {str(e)}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            if task_id: 
                 try:
                      if self.task_manager.get_task(task_id):
                           self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error=f"结果处理循环异常: {e}")
                      else:
                           logger.warning(f"尝试更新已不存在的任务 {task_id} 的状态为失败")
                 except Exception as update_err:
                      logger.error(f"尝试更新任务 {task_id} 状态为失败时出错: {update_err}")
            
        finally:
            logger.info(f"结果处理循环退出: {task_id}")
            # 清理工作移到 stop_task 或 TaskManager 中
                
    async def stop_task(self, task_id: str) -> Dict[str, Any]:
        """
        停止任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 停止结果
        """
        logger.info(f"开始停止任务: {task_id}")
        process = self.active_tasks.get(task_id)
        stop_event = self.stop_events.get(task_id)
        result_handler = self.result_handlers.get(task_id)
        
        # 1. 检查进程是否存在或是否存活
        if not process or not process.is_alive():
            logger.warning(f"尝试停止的任务进程不存在或已结束: {task_id}")
            # 即使进程已结束，也尝试清理可能残留的图片目录
            try:
                task = self.task_manager.get_task(task_id)
                if task and task.get("data"):
                    storage_config = task["data"].get("result", {}).get("storage", {})
                    save_dir_base_config = storage_config.get("save_path", "results")
                    save_dir_base = save_dir_base_config.lstrip('/').lstrip('\\\\')
                    task_image_dir = os.path.join(save_dir_base, str(task_id))
                    
                    if os.path.isdir(task_image_dir):
                        logger.info(f"尝试删除任务图片目录: {task_image_dir}")
                        shutil.rmtree(task_image_dir)
                        logger.info(f"任务图片目录已删除: {task_image_dir}")
                    else:
                        logger.info(f"任务图片目录不存在，无需删除: {task_image_dir}")
                else:
                    logger.warning(f"无法获取任务 {task_id} 的配置信息，无法删除图片目录")
            except Exception as e:
                logger.error(f"删除任务 {task_id} 图片目录时出错: {e}")
            # 清理句柄
            if task_id in self.active_tasks: del self.active_tasks[task_id]
            if task_id in self.stop_events: del self.stop_events[task_id]
            if task_id in self.result_queues: del self.result_queues[task_id]
            if task_id in self.result_handlers: del self.result_handlers[task_id]
            return {
                "success": True, # 任务已处于停止状态
                "message": "任务不存在或已停止"
            }
            
        try:
            # 1. 设置停止事件，通知工作进程
            if stop_event:
                logger.info(f"设置任务停止事件: {task_id}")
                stop_event.set()
                
            # 2. 取消结果处理协程
            if result_handler and not result_handler.done():
                logger.info(f"取消结果处理协程: {task_id}")
                result_handler.cancel()
                try:
                    # 等待协程实际取消完成 (可选，带超时)
                    await asyncio.wait_for(result_handler, timeout=1.0)
                except asyncio.CancelledError:
                    logger.info(f"结果处理协程 {task_id} 已成功取消")
                except asyncio.TimeoutError:
                    logger.warning(f"等待结果处理协程 {task_id} 取消超时")
                except Exception as cancel_err:
                     logger.error(f"取消结果协程 {task_id} 时出错: {cancel_err}")

            # 3. 等待进程结束 (带超时)
            logger.info(f"等待工作进程结束: {task_id}")
            process.join(timeout=5.0)
            
            # 4. 如果进程仍在运行，强制终止
            if process.is_alive():
                logger.warning(f"工作进程 {task_id} 未在5秒内正常退出，强制终止")
                process.terminate()
                process.join(timeout=1.0) # 等待终止完成
                if process.is_alive():
                     logger.error(f"强制终止进程 {task_id} 失败")
            else:
                 logger.info(f"工作进程 {task_id} 已正常结束")
                
            # 5. 进程停止后，删除图片目录
            try:
                task = self.task_manager.get_task(task_id)
                if task and task.get("data"):
                    storage_config = task["data"].get("result", {}).get("storage", {})
                    save_dir_base_config = storage_config.get("save_path", "results")
                    save_dir_base = save_dir_base_config.lstrip('/').lstrip('\\\\')
                    task_image_dir = os.path.join(save_dir_base, str(task_id))
                    
                    if os.path.isdir(task_image_dir):
                        logger.info(f"尝试删除任务图片目录: {task_image_dir}")
                        shutil.rmtree(task_image_dir)
                        logger.info(f"任务图片目录已删除: {task_image_dir}")
                    else:
                        logger.info(f"任务图片目录不存在，无需删除: {task_image_dir}")
                else:
                    logger.warning(f"无法获取任务 {task_id} 的配置信息，无法删除图片目录")
            except Exception as e:
                logger.error(f"删除任务 {task_id} 图片目录时出错: {e}")
                # 删除失败不应阻止后续清理

            # 6. 清理资源 (确保所有相关句柄都被移除)
            logger.info(f"清理任务资源: {task_id}")
            if task_id in self.active_tasks: del self.active_tasks[task_id]
            if task_id in self.stop_events: del self.stop_events[task_id]
            if task_id in self.result_queues: del self.result_queues[task_id]
            if task_id in self.result_handlers: del self.result_handlers[task_id]

            logger.info(f"任务停止成功: {task_id}")
            return {
                "success": True,
                "message": "任务已成功停止"
            }
            
        except Exception as e:
            logger.error(f"停止任务 {task_id} 过程中发生错误(进程停止部分): {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 即使停止过程中出错，也尝试清理资源
            if task_id in self.active_tasks: del self.active_tasks[task_id]
            if task_id in self.stop_events: del self.stop_events[task_id]
            if task_id in self.result_queues: del self.result_queues[task_id]
            if task_id in self.result_handlers: del self.result_handlers[task_id]
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