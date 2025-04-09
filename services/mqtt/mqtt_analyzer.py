"""
MQTT协议的分析服务
处理MQTT协议的图像分析、视频分析和流分析请求
"""
import os
import json
import uuid
import base64
from typing import Dict, Any, List, Optional, Callable
import asyncio
from asyncio import Queue
from paho.mqtt import client as mqtt_client
import time
import threading
import cv2
import numpy as np
import torch

from shared.utils.logger import setup_logger
from core.task_manager import TaskManager
from core.task_processor import TaskProcessor
from core.config import settings
from core.detection.yolo_detector import YOLODetector
from core.segmentation.yolo_segmentor import YOLOSegmentor
from services.base_analyzer import BaseAnalyzerService
from services.mqtt_client import MQTTClient

logger = setup_logger(__name__)

# 添加StreamDetectionTask类定义
class StreamDetectionTask:
    """流检测任务类，用于处理流视频的实时检测"""
    
    def __init__(self, task_id, subtask_id, stream_url, model_config, result_config, mqtt_client, should_stop):
        """
        初始化流检测任务
        
        Args:
            task_id: 任务ID
            subtask_id: 子任务ID
            stream_url: 流URL
            model_config: 模型配置
            result_config: 结果配置
            mqtt_client: MQTT客户端
            should_stop: 停止检查函数
        """
        self.task_id = task_id
        self.subtask_id = subtask_id
        self.stream_url = stream_url
        self.model_config = model_config
        self.result_config = result_config
        self.mqtt_client = mqtt_client
        self.should_stop = should_stop
        self.running = False
        self.thread = None
        
        # 检测相关参数
        self.model_code = model_config.get("model_code", "yolo11n")
        self.confidence = model_config.get("confidence", 0.5)
        self.iou = model_config.get("iou", 0.5)
        self.classes = model_config.get("classes", None)
        self.imgsz = model_config.get("imgsz", 640)
        self.nested_detection = model_config.get("nested_detection", False)
        
        # 处理ROI参数
        self.roi = None
        if "roi" in model_config:
            roi_data = model_config.get("roi", {})
            roi_type = model_config.get("roi_type", 0)
            
            # 矩形ROI (roi_type=1)
            if roi_type == 1 and all(k in roi_data for k in ["x1", "y1", "x2", "y2"]):
                # 基于固定尺寸464x261
                base_width = 464
                base_height = 261
                # 归一化坐标（转为0-1范围）
                self.roi = {
                    "x1": roi_data.get("x1") / base_width,
                    "y1": roi_data.get("y1") / base_height,
                    "x2": roi_data.get("x2") / base_width,
                    "y2": roi_data.get("y2") / base_height,
                    "normalized": True,  # 标记这是归一化坐标
                    "roi_type": roi_type  # 保存ROI类型
                }
            # 多边形ROI (roi_type=2) 或 线段ROI (roi_type=3)
            elif (roi_type == 2 or roi_type == 3) and "points" in roi_data:
                points = roi_data.get("points", [])
                if points:
                    # 检查点的格式，支持两种格式：[x, y] 或 {'x': x, 'y': y}
                    if isinstance(points[0], dict) and 'x' in points[0] and 'y' in points[0]:
                        # 字典格式的点 - ROI坐标基于固定尺寸464x261
                        base_width = 464
                        base_height = 261
                        # 归一化坐标（转为0-1范围）
                        x_coords = [p['x'] / base_width for p in points]
                        y_coords = [p['y'] / base_height for p in points]
                        
                        # 创建归一化的点列表
                        normalized_points = []
                        for i in range(len(points)):
                            normalized_points.append({
                                'x': x_coords[i],
                                'y': y_coords[i]
                            })
                    else:
                        # 数组格式的点 - 假设已经是归一化坐标
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        
                        # 创建归一化的点列表
                        normalized_points = []
                        for i in range(len(points)):
                            normalized_points.append([x_coords[i], y_coords[i]])
                    
                    # 存储归一化的ROI坐标（0-1范围）
                    self.roi = {
                        "x1": min(x_coords),
                        "y1": min(y_coords),
                        "x2": max(x_coords),
                        "y2": max(y_coords),
                        "normalized": True,  # 标记这是归一化坐标
                        "roi_type": roi_type,  # 保存ROI类型
                        "points": normalized_points  # 添加点列表
                    }
            # 圆形ROI (roi_type=4)
            elif roi_type == 4 and "center" in roi_data and "radius" in roi_data:
                center = roi_data.get("center")
                radius = roi_data.get("radius")
                # 基于固定尺寸464x261
                base_width = 464
                base_height = 261
                # 归一化中心点和半径
                center_x = center[0] / base_width if isinstance(center, list) else center.get("x") / base_width
                center_y = center[1] / base_height if isinstance(center, list) else center.get("y") / base_height
                radius_x = radius / base_width  # 在x方向的归一化半径
                radius_y = radius / base_height  # 在y方向的归一化半径
                
                self.roi = {
                    "x1": max(0, center_x - radius_x),
                    "y1": max(0, center_y - radius_y),
                    "x2": min(1, center_x + radius_x),
                    "y2": min(1, center_y + radius_y),
                    "normalized": True,  # 标记这是归一化坐标
                    "roi_type": roi_type,  # 保存ROI类型
                    "is_circle": True,  # 标记这是圆形ROI
                    "center": [center_x, center_y],
                    "radius": (radius_x + radius_y) / 2  # 平均半径
                }
        
        # 结果相关参数
        self.save_result = result_config.get("save_result", True)
        self.callback_topic = result_config.get("callback_topic", "")
        self.callback_interval = model_config.get("callback", {}).get("interval", 5)
        
        # 创建检测器实例
        self.detector = None
        
        # 打印完整的配置信息，包括嵌套检测参数
        logger.info(f"流检测任务初始化 - 任务ID: {task_id}, 子任务ID: {subtask_id}")
        logger.info(f"模型配置: {json.dumps(model_config, ensure_ascii=False)}")
        logger.info(f"嵌套检测参数: {self.nested_detection}")
        logger.info(f"ROI参数: {json.dumps(self.roi, ensure_ascii=False) if self.roi else 'None'}")
        logger.info(f"结果配置: {json.dumps(result_config, ensure_ascii=False)}")
        
    def start(self):
        """启动任务"""
        if self.running:
            logger.warning(f"任务已在运行中: {self.task_id}/{self.subtask_id}")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_task)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"已启动流检测任务: {self.task_id}/{self.subtask_id}")
        
    def stop(self):
        """停止任务"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
            logger.info(f"已停止流检测任务: {self.task_id}/{self.subtask_id}")
            
    def _run_task(self):
        """执行任务"""
        try:
            logger.info(f"开始执行流检测任务: {self.task_id}/{self.subtask_id}")
            
            # 更新任务状态为处理中
            self._send_status("processing")
            
            # 初始化检测器
            self._init_detector()
            
            # 打开视频流
            try:
                cap = cv2.VideoCapture(self.stream_url)
                if not cap.isOpened():
                    error_msg = f"无法打开流 {self.stream_url}"
                    logger.error(error_msg)
                    self._send_status("error", error=error_msg)
                    return
            except Exception as e:
                error_msg = f"打开流时出错: {self.stream_url}, 错误: {str(e)}"
                logger.error(error_msg)
                logger.exception(e)
                self._send_status("error", error=error_msg)
                return
                
            # 创建输出目录（如果需要保存结果）
            output_dir = None
            if self.save_result or self.result_config.get("save_images", False):
                output_dir = os.path.join(settings.OUTPUT.save_dir, "streams", f"{self.task_id}_{self.subtask_id}")
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"已创建输出目录: {output_dir}")
                
            # 处理参数
            last_callback_time = 0
            frame_count = 0
            
            # 获取回调配置
            callback_enabled = self.model_config.get("callback", {}).get("enabled", True)
            logger.info(f"回调状态: {'已启用' if callback_enabled else '已禁用'}")
            
            # 处理视频流
            while self.running and not self.should_stop():
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"无法从流中读取帧，尝试重新连接: {self.stream_url}")
                    # 尝试重新连接
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(self.stream_url)
                    if not cap.isOpened():
                        logger.error(f"无法重新连接到流: {self.stream_url}")
                        break
                    continue
                    
                # 检测当前帧
                detections = self._detect_frame(frame)
                frame_count += 1
                
                # 保存结果图片（如果需要）
                saved_file_path = None
                if output_dir and self.result_config.get("save_images", False) and frame_count % 30 == 0:  # 每30帧保存一次
                    # 检查是否有检测结果，只有有目标时才保存图片
                    if detections and len(detections) > 0:
                        timestamp = int(time.time())
                        filename = f"{timestamp}_{frame_count}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        # 绘制检测结果到图像
                        try:
                            result_image = self.detector.draw_detections(frame.copy(), detections)
                            cv2.imwrite(filepath, result_image)
                            saved_file_path = filepath
                            logger.debug(f"已保存检测结果图片: {filepath}")
                        except Exception as e:
                            logger.error(f"保存检测结果图片失败: {str(e)}")
                            # 保存原始帧作为备用
                            cv2.imwrite(filepath, frame)
                            saved_file_path = filepath
                    else:
                        logger.debug(f"跳过保存图片：当前帧未检测到目标")
                
                # 发送回调（如果需要）
                current_time = time.time()
                if callback_enabled and self.callback_topic and (current_time - last_callback_time) >= self.callback_interval:
                    self._send_result(frame, detections, saved_file_path)
                    last_callback_time = current_time
                    
                # 显示进度
                if frame_count % 100 == 0:
                    logger.info(f"任务 {self.task_id}/{self.subtask_id} 已处理 {frame_count} 帧")
                    
            # 关闭视频流
            cap.release()
            
            # 任务结束
            if not self.should_stop():
                self._send_status("completed")
                logger.info(f"流检测任务完成: {self.task_id}/{self.subtask_id}, 共处理 {frame_count} 帧")
            else:
                self._send_status("stopped")
                logger.info(f"流检测任务已停止: {self.task_id}/{self.subtask_id}, 共处理 {frame_count} 帧")
                
        except Exception as e:
            logger.error(f"流检测任务执行出错: {str(e)}")
            logger.exception(e)
            self._send_status("error", error=str(e))
            
    def _init_detector(self):
        """初始化检测器"""
        try:
            # 创建检测器实例
            self.detector = YOLODetector()
            
            # 加载模型 - 由于YOLODetector.load_model是异步的，我们需要使用同步方式调用
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.detector.load_model(self.model_code))
            loop.close()
            
            logger.info(f"流检测任务 {self.task_id}/{self.subtask_id} 已加载模型: {self.model_code}")
            
        except Exception as e:
            logger.error(f"初始化检测器失败: {str(e)}")
            raise
            
    def _detect_frame(self, frame):
        """检测当前帧"""
        try:
            # 检测参数
            config = {
                "confidence": self.confidence,
                "iou": self.iou,
                "classes": self.classes,
                "nested_detection": self.nested_detection,
                "roi": self.roi,
                "roi_type": self.model_config.get("roi_type", 1),  # 添加roi_type参数
                "imgsz": self.imgsz
            }
            
            # 记录详细的检测参数
            logger.debug(f"检测参数: {json.dumps(config, ensure_ascii=False)}")
            
            # 调用检测器 - 由于detector.detect是异步的，我们需要使用同步方式调用
            loop = asyncio.new_event_loop()
            detections = loop.run_until_complete(self.detector.detect(frame, config))
            loop.close()
            
            # 直接返回检测结果列表，YOLODetector.detect已经返回detections列表
            return detections
            
        except Exception as e:
            logger.error(f"检测帧时出错: {str(e)}")
            return []
            
    def _send_result(self, frame, detections, saved_file_path=None):
        """发送检测结果"""
        # 检查回调是否启用和主题是否存在
        callback_enabled = self.model_config.get("callback", {}).get("enabled", True)
        if not callback_enabled or not self.callback_topic:
            logger.debug("不发送结果：回调未启用或未指定回调主题")
            return
        
        # 检查是否有检测结果，没有检测到目标时不发送回调
        if not detections or len(detections) == 0:
            logger.debug("不发送结果：未检测到目标")
            return
        
        try:
            # 准备结果数据
            result = {
                "task_id": self.task_id,
                "subtask_id": self.subtask_id,
                "timestamp": int(time.time()),
                "detections": detections,
                "frame_size": {
                    "width": frame.shape[1],
                    "height": frame.shape[0]
                }
            }
            
            # 添加保存的图片路径
            if saved_file_path:
                result["image_path"] = saved_file_path
            
            # 如果需要，添加帧图像
            if self.result_config.get("include_image", True):
                # 压缩图像并编码为base64
                compression_quality = self.result_config.get("compression_quality", 90)  # 默认90%质量
                max_width = self.result_config.get("max_width", 1280)  # 默认最大宽度
                
                # 调整图像大小（如果需要）
                h, w = frame.shape[:2]
                if w > max_width:
                    scale = max_width / w
                    new_w = max_width
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # 压缩并转换为base64
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                result["image"] = image_base64
                result["frame_base64"] = image_base64  # 添加标准字段名
                
            # 发送到MQTT主题 - 使用_send_task_result而不是直接_publish_message
            if self.mqtt_client:
                # 如果callback_topic存在于result_config中，_send_task_result会使用它
                self.mqtt_client._send_task_result(
                    task_id=self.task_id,
                    subtask_id=self.subtask_id,
                    status="processing",
                    result=result
                )
                logger.debug(f"已发送检测结果到主题: {self.callback_topic}")
            
        except Exception as e:
            logger.error(f"发送检测结果时出错: {str(e)}")
            logger.exception(e)
            
    def _send_status(self, status, error=None):
        """发送任务状态"""
        if self.mqtt_client:
            # 发送MQTT状态消息
            self.mqtt_client._send_task_status(self.task_id, self.subtask_id, status, error=error)
            
            # 移除TaskManager中更新任务状态的代码
            # 我们遵循无状态原则，仅发送MQTT消息，不更新TaskManager

# 添加StreamSegmentationTask类定义
class StreamSegmentationTask:
    """流分割任务类，用于处理流视频的实时分割"""
    
    def __init__(self, task_id, subtask_id, stream_url, model_config, result_config, mqtt_client, should_stop):
        """
        初始化流分割任务
        
        Args:
            task_id: 任务ID
            subtask_id: 子任务ID
            stream_url: 流URL
            model_config: 模型配置
            result_config: 结果配置
            mqtt_client: MQTT客户端
            should_stop: 停止检查函数
        """
        self.task_id = task_id
        self.subtask_id = subtask_id
        self.stream_url = stream_url
        self.model_config = model_config
        self.result_config = result_config
        self.mqtt_client = mqtt_client
        self.should_stop = should_stop
        self.running = False
        self.thread = None
        
        # 分割相关参数
        self.model_code = model_config.get("model_code", "yolov8n-seg")
        self.confidence = model_config.get("confidence", 0.5)
        self.iou = model_config.get("iou", 0.5)
        self.classes = model_config.get("classes", None)
        self.imgsz = model_config.get("imgsz", 640)
        self.retina_masks = model_config.get("retina_masks", True)  # 精细掩码默认开启
        
        # 处理ROI参数
        self.roi = None
        if "roi" in model_config:
            roi_data = model_config.get("roi", {})
            roi_type = model_config.get("roi_type", 0)
            
            # 矩形ROI (roi_type=1)
            if roi_type == 1 and all(k in roi_data for k in ["x1", "y1", "x2", "y2"]):
                # 基于固定尺寸464x261
                base_width = 464
                base_height = 261
                # 归一化坐标（转为0-1范围）
                self.roi = {
                    "x1": roi_data.get("x1") / base_width,
                    "y1": roi_data.get("y1") / base_height,
                    "x2": roi_data.get("x2") / base_width,
                    "y2": roi_data.get("y2") / base_height,
                    "normalized": True,  # 标记这是归一化坐标
                    "roi_type": roi_type  # 保存ROI类型
                }
            # 多边形ROI (roi_type=2) 或 线段ROI (roi_type=3)
            elif (roi_type == 2 or roi_type == 3) and "points" in roi_data:
                points = roi_data.get("points", [])
                if points:
                    # 检查点的格式，支持两种格式：[x, y] 或 {'x': x, 'y': y}
                    if isinstance(points[0], dict) and 'x' in points[0] and 'y' in points[0]:
                        # 字典格式的点 - ROI坐标基于固定尺寸464x261
                        base_width = 464
                        base_height = 261
                        # 归一化坐标（转为0-1范围）
                        x_coords = [p['x'] / base_width for p in points]
                        y_coords = [p['y'] / base_height for p in points]
                        
                        # 创建归一化的点列表
                        normalized_points = []
                        for i in range(len(points)):
                            normalized_points.append({
                                'x': x_coords[i],
                                'y': y_coords[i]
                            })
                    else:
                        # 数组格式的点 - 假设已经是归一化坐标
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        
                        # 创建归一化的点列表
                        normalized_points = []
                        for i in range(len(points)):
                            normalized_points.append([x_coords[i], y_coords[i]])
                    
                    # 存储归一化的ROI坐标（0-1范围）
                    self.roi = {
                        "x1": min(x_coords),
                        "y1": min(y_coords),
                        "x2": max(x_coords),
                        "y2": max(y_coords),
                        "normalized": True,  # 标记这是归一化坐标
                        "roi_type": roi_type,  # 保存ROI类型
                        "points": normalized_points  # 添加点列表
                    }
            # 圆形ROI (roi_type=4)
            elif roi_type == 4 and "center" in roi_data and "radius" in roi_data:
                center = roi_data.get("center")
                radius = roi_data.get("radius")
                # 基于固定尺寸464x261
                base_width = 464
                base_height = 261
                # 归一化中心点和半径
                center_x = center[0] / base_width if isinstance(center, list) else center.get("x") / base_width
                center_y = center[1] / base_height if isinstance(center, list) else center.get("y") / base_height
                radius_x = radius / base_width  # 在x方向的归一化半径
                radius_y = radius / base_height  # 在y方向的归一化半径
                
                self.roi = {
                    "x1": max(0, center_x - radius_x),
                    "y1": max(0, center_y - radius_y),
                    "x2": min(1, center_x + radius_x),
                    "y2": min(1, center_y + radius_y),
                    "normalized": True,  # 标记这是归一化坐标
                    "roi_type": roi_type,  # 保存ROI类型
                    "is_circle": True,  # 标记这是圆形ROI
                    "center": [center_x, center_y],
                    "radius": (radius_x + radius_y) / 2  # 平均半径
                }
        
        # 结果相关参数
        self.save_result = result_config.get("save_result", True)
        self.callback_topic = result_config.get("callback_topic", "")
        self.callback_interval = model_config.get("callback", {}).get("interval", 5)
        
        # 创建分割器实例
        self.segmentor = None
        
        # 打印完整的配置信息
        logger.info(f"流分割任务初始化 - 任务ID: {task_id}, 子任务ID: {subtask_id}")
        logger.info(f"模型配置: {json.dumps(model_config, ensure_ascii=False)}")
        logger.info(f"ROI参数: {json.dumps(self.roi, ensure_ascii=False) if self.roi else 'None'}")
        logger.info(f"结果配置: {json.dumps(result_config, ensure_ascii=False)}")
        
    def start(self):
        """启动任务"""
        if self.running:
            logger.warning(f"任务已在运行中: {self.task_id}/{self.subtask_id}")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_task)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"已启动流分割任务: {self.task_id}/{self.subtask_id}")
        
    def stop(self):
        """停止任务"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
            logger.info(f"已停止流分割任务: {self.task_id}/{self.subtask_id}")
            
    def _run_task(self):
        """执行任务"""
        try:
            logger.info(f"开始执行流分割任务: {self.task_id}/{self.subtask_id}")
            
            # 更新任务状态为处理中
            self._send_status("processing")
            
            # 初始化分割器
            self._init_segmentor()
            
            # 打开视频流
            try:
                cap = cv2.VideoCapture(self.stream_url)
                if not cap.isOpened():
                    error_msg = f"无法打开流 {self.stream_url}"
                    logger.error(error_msg)
                    self._send_status("error", error=error_msg)
                    return
            except Exception as e:
                error_msg = f"打开流时出错: {self.stream_url}, 错误: {str(e)}"
                logger.error(error_msg)
                logger.exception(e)
                self._send_status("error", error=error_msg)
                return
                
            # 创建输出目录（如果需要保存结果）
            output_dir = None
            if self.save_result or self.result_config.get("save_images", False):
                output_dir = os.path.join(settings.OUTPUT.save_dir, "segmentation", f"{self.task_id}_{self.subtask_id}")
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"已创建输出目录: {output_dir}")
                
            # 处理参数
            last_callback_time = 0
            frame_count = 0
            
            # 获取回调配置
            callback_enabled = self.model_config.get("callback", {}).get("enabled", True)
            logger.info(f"回调状态: {'已启用' if callback_enabled else '已禁用'}")
            
            # 处理视频流
            while self.running and not self.should_stop():
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"无法从流中读取帧，尝试重新连接: {self.stream_url}")
                    # 尝试重新连接
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(self.stream_url)
                    if not cap.isOpened():
                        logger.error(f"无法重新连接到流: {self.stream_url}")
                        break
                    continue
                    
                # 对当前帧进行分割
                segmentation_results = self._segment_frame(frame)
                frame_count += 1
                
                # 保存结果图片（如果需要）
                saved_file_path = None
                if output_dir and self.result_config.get("save_images", False) and frame_count % 30 == 0:  # 每30帧保存一次
                    # 检查是否有分割结果，只有有目标时才保存图片
                    if segmentation_results and len(segmentation_results) > 0:
                        timestamp = int(time.time())
                        filename = f"{timestamp}_{frame_count}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        # 绘制分割结果到图像
                        try:
                            result_image = self.segmentor.draw_segmentations(frame.copy(), segmentation_results)
                            cv2.imwrite(filepath, result_image)
                            saved_file_path = filepath
                            logger.debug(f"已保存分割结果图片: {filepath}")
                        except Exception as e:
                            logger.error(f"保存分割结果图片失败: {str(e)}")
                            # 保存原始帧作为备用
                            cv2.imwrite(filepath, frame)
                            saved_file_path = filepath
                    else:
                        logger.debug(f"跳过保存图片：当前帧未检测到分割目标")
                
                # 发送回调（如果需要）
                current_time = time.time()
                if callback_enabled and self.callback_topic and (current_time - last_callback_time) >= self.callback_interval:
                    self._send_result(frame, segmentation_results, saved_file_path)
                    last_callback_time = current_time
                    
                # 显示进度
                if frame_count % 100 == 0:
                    logger.info(f"任务 {self.task_id}/{self.subtask_id} 已处理 {frame_count} 帧")
                    
            # 关闭视频流
            cap.release()
            
            # 任务结束
            if not self.should_stop():
                self._send_status("completed")
                logger.info(f"流分割任务完成: {self.task_id}/{self.subtask_id}, 共处理 {frame_count} 帧")
            else:
                self._send_status("stopped")
                logger.info(f"流分割任务已停止: {self.task_id}/{self.subtask_id}, 共处理 {frame_count} 帧")
                
        except Exception as e:
            logger.error(f"流分割任务执行出错: {str(e)}")
            logger.exception(e)
            self._send_status("error", error=str(e))
            
    def _init_segmentor(self):
        """初始化分割器"""
        try:
            # 创建分割器实例
            self.segmentor = YOLOSegmentor()
            
            # 加载模型 - 由于YOLOSegmentor.load_model是异步的，我们需要使用同步方式调用
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.segmentor.load_model(self.model_code))
            loop.close()
            
            logger.info(f"流分割任务 {self.task_id}/{self.subtask_id} 已加载模型: {self.model_code}")
            
        except Exception as e:
            logger.error(f"初始化分割器失败: {str(e)}")
            raise
            
    def _segment_frame(self, frame):
        """分割当前帧"""
        try:
            # 分割参数
            config = {
                "confidence": self.confidence,
                "iou": self.iou,
                "classes": self.classes,
                "roi": self.roi,
                "roi_type": self.model_config.get("roi_type", 1),  # 添加roi_type参数
                "imgsz": self.imgsz,
                "retina_masks": self.retina_masks
            }
            
            # 记录详细的分割参数
            logger.debug(f"分割参数: {json.dumps(config, ensure_ascii=False)}")
            
            # 调用分割器 - 由于segmentor.segment是异步的，我们需要使用同步方式调用
            loop = asyncio.new_event_loop()
            segmentation_results = loop.run_until_complete(self.segmentor.segment(frame, config))
            loop.close()
            
            # 返回分割结果列表
            return segmentation_results
            
        except Exception as e:
            logger.error(f"分割帧时出错: {str(e)}")
            return []
            
    def _send_result(self, frame, segmentation_results, saved_file_path=None):
        """发送分割结果"""
        # 检查回调是否启用和主题是否存在
        callback_enabled = self.model_config.get("callback", {}).get("enabled", True)
        if not callback_enabled or not self.callback_topic:
            logger.debug("不发送结果：回调未启用或未指定回调主题")
            return
        
        # 检查是否有分割结果，没有检测到分割目标时不发送回调
        if not segmentation_results or len(segmentation_results) == 0:
            logger.debug("不发送结果：未检测到分割目标")
            return
        
        try:
            # 准备结果数据
            result = {
                "task_id": self.task_id,
                "subtask_id": self.subtask_id,
                "timestamp": int(time.time()),
                "segmentation_results": segmentation_results,
                "frame_size": {
                    "width": frame.shape[1],
                    "height": frame.shape[0]
                }
            }
            
            # 添加保存的图片路径
            if saved_file_path:
                result["image_path"] = saved_file_path
            
            # 如果需要，添加帧图像
            if self.result_config.get("include_image", True):
                # 压缩图像并编码为base64
                compression_quality = self.result_config.get("compression_quality", 90)  # 默认90%质量
                max_width = self.result_config.get("max_width", 1280)  # 默认最大宽度
                
                # 调整图像大小（如果需要）
                h, w = frame.shape[:2]
                if w > max_width:
                    scale = max_width / w
                    new_w = max_width
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # 压缩并转换为base64
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                result["image"] = image_base64
                result["frame_base64"] = image_base64  # 添加标准字段名
                
            # 发送到MQTT主题 - 使用_send_task_result而不是直接_publish_message
            if self.mqtt_client:
                # 如果callback_topic存在于result_config中，_send_task_result会使用它
                self.mqtt_client._send_task_result(
                    task_id=self.task_id,
                    subtask_id=self.subtask_id,
                    status="processing",
                    result=result
                )
                logger.debug(f"已发送分割结果到主题: {self.callback_topic}")
            
        except Exception as e:
            logger.error(f"发送分割结果时出错: {str(e)}")
            logger.exception(e)
            
    def _send_status(self, status, error=None):
        """发送任务状态"""
        if self.mqtt_client:
            # 发送MQTT状态消息
            self.mqtt_client._send_task_status(self.task_id, self.subtask_id, status, error=error)
            
            # 移除TaskManager中更新任务状态的代码
            # 我们遵循无状态原则，仅发送MQTT消息，不更新TaskManager

class MQTTAnalyzerService(BaseAnalyzerService):
    """MQTT分析服务"""

    def __init__(self, device_id, mqtt_config, model_configs=None):
        """
        初始化MQTT分析服务
        
        Args:
            device_id: 设备ID
            mqtt_config: MQTT配置
            model_configs: 模型配置
        """
        # 调用父类的__init__方法，不传递额外参数
        super().__init__()
        self.device_id = device_id
        self.mqtt_config = mqtt_config
        self.model_configs = model_configs or {}
        
        # 初始化任务管理器
        self.task_manager = TaskManager()
        
        # 初始化MQTT客户端
        self.mqtt_client = MQTTClient(
            device_id=device_id,
            broker_host=mqtt_config.get("host", "localhost"),
            broker_port=mqtt_config.get("port", 1883),
            username=mqtt_config.get("username"),
            password=mqtt_config.get("password"),
            command_topic=mqtt_config.get("command_topic"),
            response_topic=mqtt_config.get("response_topic"),
            status_topic=mqtt_config.get("status_topic")
        )
        
        logger.info(f"MQTT分析服务已初始化: 设备ID={device_id}")

    async def connect(self):
        """连接到MQTT服务器并注册任务处理器"""
        # 连接前确保注册所有任务处理器
        self._register_task_handlers()
        
        # 启动MQTT客户端
        if not self.mqtt_client.start():
            logger.error("MQTT客户端启动失败")
            return False
            
        logger.info("MQTT客户端启动成功")
        
        # 记录状态信息
        logger.info(f"MQTT客户端节点ID: {self.mqtt_client.node_id}")
        logger.info(f"MQTT客户端MAC地址: {self.mqtt_client.mac_address}")
        request_setting_topic = f"{self.mqtt_client.topic_prefix}{self.mqtt_client.node_id}/request_setting"
        logger.info(f"订阅的请求主题: {request_setting_topic}")
        
        # 所有处理器已注册，连接成功
        logger.info("成功连接到MQTT服务器")
        return True
            
    def _register_task_handlers(self):
        """注册任务处理器"""
        # 注册图像分析任务处理器
        self.mqtt_client.register_task_handler("image", self._handle_image_task)
        logger.info("已注册图像分析任务处理器")
        
        # 注册视频分析任务处理器
        self.mqtt_client.register_task_handler("video", self._handle_video_task)
        logger.info("已注册视频分析任务处理器")
        
        # 注册流分析任务处理器
        self.mqtt_client.register_task_handler("stream", self._handle_stream_task)
        logger.info("已注册流分析任务处理器")
        
        # 注册分割任务处理器
        self.mqtt_client.register_task_handler("segmentation", self._handle_segmentation_task)
        logger.info("已注册分割任务处理器")
        
        # 根据模型配置注册特定类型的处理器
        if self.model_configs:
            for model_code, config in self.model_configs.items():
                analysis_type = config.get("analysis_type")
                if analysis_type and analysis_type not in ["image", "video", "stream", "segmentation"]:
                    # 只为不同于基本类型的分析类型注册处理器
                    self.mqtt_client.register_task_handler(analysis_type, self._handle_detection_task)
                    logger.info(f"已注册{analysis_type}分析任务处理器")
                    
        # 注册通用检测任务处理器作为默认处理器
        self.mqtt_client.register_task_handler("detection", self._handle_detection_task)
        logger.info("已注册通用检测任务处理器")
        
        # 记录所有已注册的处理器
        available_handlers = list(self.mqtt_client.task_handlers.keys())
        logger.info(f"所有已注册的任务处理器: {available_handlers}")
            
    def _handle_stream_task(self, task_id, subtask_id, source, config, result_config, message_id=None, message_uuid=None, confirmation_topic=None):
        """
        处理流分析任务
        
        Args:
            task_id: 任务ID
            subtask_id: 子任务ID
            source: 源配置
            config: 任务配置
            result_config: 结果配置
            message_id: 消息ID
            message_uuid: 消息UUID
            confirmation_topic: 确认主题
        
        Returns:
            bool: 处理结果
        """
        logger.info(f"处理流分析任务: {task_id}/{subtask_id}")
        
        # 获取流URL
        url = source.get("url", "")
        if not url and "urls" in source and source["urls"]:
            url = source["urls"][0]
            
        if not url:
            error_msg = "未指定流URL"
            logger.error(error_msg)
            if confirmation_topic:
                self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", 
                                                data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
            return False
            
        logger.info(f"流URL: {url}")
        
        # 获取模型配置
        model_code = config.get("model_code", "default")
        model_config = self.model_configs.get(model_code, {})
        
        # 创建分析任务
        try:
            # 创建停止检查函数 - 使用subtask_id作为唯一标识符，不再使用组合ID
            def should_stop():
                return subtask_id not in self.mqtt_client.active_tasks or self.mqtt_client.stop_event.is_set()
                
            # 记录任务 - 使用subtask_id作为任务键（这是必要的运行时状态管理）
            with self.mqtt_client.active_tasks_lock:
                self.mqtt_client.active_tasks[subtask_id] = {
                    "start_time": time.time(),
                    "source": source,
                    "config": config,
                    "result_config": result_config
                }
                
            # 通知MQTT客户端已接受任务
            if confirmation_topic:
                self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "success", 
                                               data={"message": "任务已接受", "task_id": task_id, "subtask_id": subtask_id})
                
            # 创建任务
            detection_task = StreamDetectionTask(
                task_id=task_id,
                subtask_id=subtask_id,
                stream_url=url,
                model_config=config,  # 使用完整配置
                result_config=result_config,
                mqtt_client=self.mqtt_client,
                should_stop=should_stop
            )
            
            # 启动任务
            detection_task.start()
            
            logger.info(f"流分析任务已启动: {task_id}/{subtask_id}")
            return True
            
        except Exception as e:
            error_msg = f"启动流分析任务时出错: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            
            # 移除任务
            with self.mqtt_client.active_tasks_lock:
                if subtask_id in self.mqtt_client.active_tasks:
                    del self.mqtt_client.active_tasks[subtask_id]
            
            if confirmation_topic:
                self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", 
                                               data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
            return False
            
    def _handle_image_task(self, task_id, subtask_id, source, config, result_config, message_id=None, message_uuid=None, confirmation_topic=None):
        """处理图像分析任务"""
        logger.info(f"处理图像分析任务: {task_id}/{subtask_id}")
        # TODO: 实现图像分析逻辑
        return self._handle_detection_task(task_id, subtask_id, source, config, result_config, message_id, message_uuid, confirmation_topic)
        
    def _handle_video_task(self, task_id, subtask_id, source, config, result_config, message_id=None, message_uuid=None, confirmation_topic=None):
        """处理视频分析任务"""
        logger.info(f"处理视频分析任务: {task_id}/{subtask_id}")
        # TODO: 实现视频分析逻辑
        return self._handle_detection_task(task_id, subtask_id, source, config, result_config, message_id, message_uuid, confirmation_topic)
        
    def _handle_detection_task(self, task_id, subtask_id, source, config, result_config, message_id=None, message_uuid=None, confirmation_topic=None):
        """处理通用检测任务"""
        logger.info(f"处理通用检测任务: {task_id}/{subtask_id}")
        # 根据source类型选择合适的处理方法
        source_type = source.get("type", "")
        
        if source_type == "image":
            return self._handle_image_task(task_id, subtask_id, source, config, result_config, message_id, message_uuid, confirmation_topic)
        elif source_type == "video":
            return self._handle_video_task(task_id, subtask_id, source, config, result_config, message_id, message_uuid, confirmation_topic)
        elif source_type == "stream":
            return self._handle_stream_task(task_id, subtask_id, source, config, result_config, message_id, message_uuid, confirmation_topic)
        else:
            error_msg = f"不支持的数据源类型: {source_type}"
            logger.error(error_msg)
            if confirmation_topic:
                self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", 
                                               data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
            return False

    def disconnect(self):
        """断开MQTT连接"""
        if self.mqtt_client and self.connected:
            try:
                self.mqtt_client.disconnect()
                self.mqtt_client.loop_stop()
                self.connected = False
                logger.info("已断开与MQTT代理服务器的连接")
                return True
            except Exception as e:
                logger.error(f"断开MQTT连接时出错: {str(e)}", exc_info=True)
                return False
        return True
        
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT连接回调"""
        if rc == 0:
            self.connected = True
            logger.info(f"已连接到MQTT代理服务器: {self.broker}:{self.port}")
            
            # 订阅命令主题
            client.subscribe(self.command_topic)
            logger.info(f"已订阅命令主题: {self.command_topic}")
        else:
            logger.error(f"连接MQTT代理服务器失败，返回码: {rc}")
            
    def _on_message(self, client, userdata, msg):
        """MQTT消息回调"""
        try:
            # 解析消息
            payload = msg.payload.decode("utf-8")
            logger.debug(f"收到MQTT消息: {payload}")
            
            # 解析JSON命令
            command = json.loads(payload)
            
            # 将命令添加到队列
            asyncio.run_coroutine_threadsafe(
                self.command_queue.put(command),
                asyncio.get_event_loop()
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "error": f"无效的JSON格式: {str(e)}"
            })
        except Exception as e:
            logger.error(f"处理MQTT消息时出错: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "error": f"处理消息时出错: {str(e)}"
            })
            
    def _on_disconnect(self, client, userdata, rc):
        """MQTT断开连接回调"""
        self.connected = False
        if rc != 0:
            logger.warning(f"与MQTT代理服务器的连接意外断开，返回码: {rc}")
            # 尝试重新连接
            asyncio.create_task(self._reconnect())
        else:
            logger.info("已主动断开与MQTT代理服务器的连接")
            
    async def _reconnect(self, max_retries=5, retry_interval=5):
        """重新连接到MQTT代理服务器"""
        retries = 0
        while not self.connected and retries < max_retries:
            logger.info(f"尝试重新连接MQTT代理服务器，第 {retries+1} 次...")
            try:
                self.mqtt_client.reconnect()
                return
            except Exception as e:
                logger.error(f"重新连接失败: {str(e)}", exc_info=True)
                retries += 1
                await asyncio.sleep(retry_interval)
                
        if not self.connected:
            logger.error(f"重新连接MQTT代理服务器失败，已达到最大重试次数: {max_retries}")
            
    def _publish_response(self, response):
        """发布响应到MQTT主题"""
        if not self.connected:
            logger.error("未连接到MQTT代理服务器，无法发布响应")
            return False
            
        try:
            # 将响应对象转换为JSON字符串
            response_json = json.dumps(response)
            
            # 发布响应
            self.mqtt_client.publish(self.response_topic, response_json)
            logger.debug(f"已发布响应到主题 {self.response_topic}")
            return True
            
        except Exception as e:
            logger.error(f"发布MQTT响应时出错: {str(e)}", exc_info=True)
            return False
            
    async def _process_commands(self):
        """处理命令队列中的命令"""
        logger.info("开始处理MQTT命令队列")
        while True:
            try:
                # 从队列获取命令
                command = await self.command_queue.get()
                logger.info(f"正在处理命令: {command.get('action', 'unknown')}")
                
                # 解析命令类型
                action = command.get("action")
                if not action:
                    self._publish_response({
                        "success": False,
                        "command_id": command.get("command_id"),
                        "error": "无效的命令，缺少 'action' 字段"
                    })
                    continue
                    
                # 处理不同类型的命令
                if action == "analyze_image":
                    await self._handle_analyze_image(command)
                elif action == "start_video_analysis":
                    await self._handle_start_video_analysis(command)
                elif action == "start_stream_analysis":
                    await self._handle_start_stream_analysis(command)
                elif action == "stop_task":
                    await self._handle_stop_task(command)
                elif action == "get_task_status":
                    await self._handle_get_task_status(command)
                elif action == "get_tasks":
                    await self._handle_get_tasks(command)
                else:
                    self._publish_response({
                        "success": False,
                        "command_id": command.get("command_id"),
                        "error": f"未知的命令类型: {action}"
                    })
                    
            except Exception as e:
                logger.error(f"处理MQTT命令时出错: {str(e)}", exc_info=True)
                try:
                    self._publish_response({
                        "success": False,
                        "command_id": command.get("command_id") if "command" in locals() else None,
                        "error": f"处理命令时出错: {str(e)}"
                    })
                except Exception as e2:
                    logger.error(f"发送错误响应时出错: {str(e2)}", exc_info=True)
                    
    async def _handle_analyze_image(self, command):
        """处理图像分析命令"""
        try:
            # 获取命令参数
            command_id = command.get("command_id")
            model_code = command.get("model_code")
            image_data = command.get("image_data")  # Base64编码的图像数据
            image_path = command.get("image_path")  # 或者图像路径
            conf_threshold = command.get("conf_threshold", 0.25)
            save_result = command.get("save_result", True)
            include_image = command.get("include_image", False)
            
            # 验证参数
            if not model_code:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: model_code"
                })
                return
                
            if not image_data and not image_path:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: image_data 或 image_path"
                })
                return
                
            # 如果提供了图像数据，保存为文件
            if image_data:
                try:
                    # 解码Base64图像数据
                    image_bytes = base64.b64decode(image_data)
                    
                    # 保存为文件
                    filename = f"{uuid.uuid4().hex}.jpg"
                    file_path = f"{settings.OUTPUT.save_dir}/images/{filename}"
                    
                    with open(file_path, "wb") as f:
                        f.write(image_bytes)
                        
                    image_path = file_path
                except Exception as e:
                    self._publish_response({
                        "success": False,
                        "command_id": command_id,
                        "error": f"图像数据解码失败: {str(e)}"
                    })
                    return
                    
            # 创建任务
            task_id = self.task_manager.create_task(
                task_type="image",
                protocol="mqtt",
                params={
                    "image_path": image_path,
                    "model_code": model_code,
                    "conf_threshold": conf_threshold,
                    "save_result": save_result,
                    "include_image": include_image,
                    "command_id": command_id
                }
            )
            
            # 处理图像分析任务
            result = self.task_processor.process_image(task_id)
            
            # 添加命令ID到响应中
            result["command_id"] = command_id
            
            # 发布响应
            self._publish_response(result)
            
        except Exception as e:
            logger.error(f"处理图像分析命令时出错: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "command_id": command.get("command_id"),
                "error": f"处理图像分析命令时出错: {str(e)}"
            })
            
    async def _handle_start_video_analysis(self, command):
        """处理视频分析命令"""
        try:
            # 获取命令参数
            command_id = command.get("command_id")
            model_code = command.get("model_code")
            video_path = command.get("video_path")
            conf_threshold = command.get("conf_threshold", 0.25)
            save_result = command.get("save_result", True)
            
            # 验证参数
            if not model_code:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: model_code"
                })
                return
                
            if not video_path:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: video_path"
                })
                return
                
            # 验证视频文件是否存在
            if not os.path.exists(video_path):
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": f"视频文件不存在: {video_path}"
                })
                return
                
            # 创建任务
            task_id = self.task_manager.create_task(
                task_type="video",
                protocol="mqtt",
                params={
                    "video_path": video_path,
                    "model_code": model_code,
                    "conf_threshold": conf_threshold,
                    "save_result": save_result,
                    "command_id": command_id
                }
            )
            
            # 启动视频分析任务
            result = self.task_processor.start_video_analysis(task_id)
            
            # 添加命令ID到响应中
            result["command_id"] = command_id
            
            # 发布响应
            self._publish_response(result)
            
        except Exception as e:
            logger.error(f"处理视频分析命令时出错: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "command_id": command.get("command_id"),
                "error": f"处理视频分析命令时出错: {str(e)}"
            })
            
    async def _handle_start_stream_analysis(self, command):
        logger.info(f"打印全部命令数据: {command}")
        """处理流分析命令"""
        try:
            # 获取命令参数
            command_id = command.get("command_id")
            model_code = command.get("model_code")
            stream_url = command.get("stream_url")
            conf_threshold = command.get("conf_threshold", 0.25)
            save_interval = command.get("save_interval", 10)
            max_duration = command.get("max_duration", 3600)
            
            # 验证参数
            if not model_code:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: model_code"
                })
                return
                
            if not stream_url:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: stream_url"
                })
                return
                
            # 创建任务
            task_id = self.task_manager.create_task(
                task_type="stream",
                protocol="mqtt",
                params={
                    "stream_url": stream_url,
                    "model_code": model_code,
                    "conf_threshold": conf_threshold,
                    "save_interval": save_interval,
                    "max_duration": max_duration,
                    "command_id": command_id
                }
            )
            
            # 启动流分析任务
            result = self.task_processor.start_stream_analysis(task_id)
            
            # 添加命令ID到响应中
            result["command_id"] = command_id
            
            # 发布响应
            self._publish_response(result)
            
        except Exception as e:
            logger.error(f"处理流分析命令时出错: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "command_id": command.get("command_id"),
                "error": f"处理流分析命令时出错: {str(e)}"
            })
            
    async def _handle_stop_task(self, command):
        """处理停止任务命令"""
        try:
            # 获取命令参数
            command_id = command.get("command_id")
            task_id = command.get("task_id")
            
            # 验证参数
            if not task_id:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: task_id"
                })
                return
                
            # 停止任务
            result = self.task_processor.stop_task(task_id)
            
            # 添加命令ID到响应中
            result["command_id"] = command_id
            
            # 发布响应
            self._publish_response(result)
            
        except Exception as e:
            logger.error(f"处理停止任务命令时出错: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "command_id": command.get("command_id"),
                "error": f"处理停止任务命令时出错: {str(e)}"
            })
            
    async def _handle_get_task_status(self, command):
        """处理获取任务状态命令"""
        try:
            # 获取命令参数
            command_id = command.get("command_id")
            task_id = command.get("task_id")
            
            # 验证参数
            if not task_id:
                self._publish_response({
                    "success": False,
                    "command_id": command_id,
                    "error": "缺少必需的参数: task_id"
                })
                return
                
            # 获取任务状态
            result = self.task_processor.get_task_status(task_id)
            
            # 添加命令ID到响应中
            result["command_id"] = command_id
            
            # 发布响应
            self._publish_response(result)
            
        except Exception as e:
            logger.error(f"处理获取任务状态命令时出错: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "command_id": command.get("command_id"),
                "error": f"处理获取任务状态命令时出错: {str(e)}"
            })
            
    async def _handle_get_tasks(self, command):
        """处理获取任务列表命令"""
        try:
            # 获取命令参数
            command_id = command.get("command_id")
            
            # 获取所有任务
            tasks = self.task_manager.get_all_tasks()
            
            # 过滤出MQTT协议的任务
            mqtt_tasks = {
                task_id: task for task_id, task in tasks.items()
                if task.get("protocol") == "mqtt"
            }
            
            # 构建响应
            result = {
                "success": True,
                "command_id": command_id,
                "tasks": mqtt_tasks
            }
            
            # 发布响应
            self._publish_response(result)
            
        except Exception as e:
            logger.error(f"处理获取任务列表命令时出错: {str(e)}", exc_info=True)
            self._publish_response({
                "success": False,
                "command_id": command.get("command_id"),
                "error": f"处理获取任务列表命令时出错: {str(e)}"
            })

    def _handle_segmentation_task(self, task_id, subtask_id, source, config, result_config, message_id=None, message_uuid=None, confirmation_topic=None):
        """
        处理分割任务
        
        Args:
            task_id: 任务ID
            subtask_id: 子任务ID
            source: 源配置
            config: 任务配置
            result_config: 结果配置
            message_id: 消息ID
            message_uuid: 消息UUID
            confirmation_topic: 确认主题
        
        Returns:
            bool: 处理结果
        """
        logger.info(f"处理分割任务: {task_id}/{subtask_id}")
        
        # 获取数据源URL
        url = ""
        source_type = source.get("type", "")
        
        if source_type == "stream":
            # 从流源获取URL
            url = source.get("url", "")
            if not url and "urls" in source and source["urls"]:
                url = source["urls"][0]
        elif source_type == "video":
            # 从视频源获取URL
            url = source.get("path", "")
        elif source_type == "image":
            # 对于图像源，暂不支持，返回错误
            error_msg = "分割任务暂不支持图像源类型"
            logger.error(error_msg)
            if confirmation_topic:
                self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", 
                                               data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
            return False
        else:
            error_msg = f"不支持的数据源类型: {source_type}"
            logger.error(error_msg)
            if confirmation_topic:
                self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", 
                                               data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
            return False
        
        if not url:
            error_msg = "未指定数据源URL"
            logger.error(error_msg)
            if confirmation_topic:
                self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", 
                                               data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
            return False
        
        logger.info(f"数据源URL: {url}")
        
        # 验证是否指定了分割模型
        model_code = config.get("model_code", "")
        if not model_code:
            error_msg = "未指定分割模型代码"
            logger.error(error_msg)
            if confirmation_topic:
                self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", 
                                               data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
            return False
        
        # 验证模型代码是否为分割模型
        if "seg" not in model_code.lower():
            logger.warning(f"模型代码 {model_code} 可能不是分割模型，请确保使用正确的分割模型")
        
        # 创建分析任务
        try:
            # 创建停止检查函数 - 使用subtask_id作为唯一标识符
            def should_stop():
                return subtask_id not in self.mqtt_client.active_tasks or self.mqtt_client.stop_event.is_set()
            
            # 记录任务 - 使用subtask_id作为任务键（这是必要的运行时状态管理）
            with self.mqtt_client.active_tasks_lock:
                self.mqtt_client.active_tasks[subtask_id] = {
                    "start_time": time.time(),
                    "source": source,
                    "config": config,
                    "result_config": result_config
                }
            
            # 通知MQTT客户端已接受任务
            if confirmation_topic:
                self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "success", 
                                               data={"message": "分割任务已接受", "task_id": task_id, "subtask_id": subtask_id})
            
            # 创建任务
            segmentation_task = StreamSegmentationTask(
                task_id=task_id,
                subtask_id=subtask_id,
                stream_url=url,
                model_config=config,  # 使用完整配置
                result_config=result_config,
                mqtt_client=self.mqtt_client,
                should_stop=should_stop
            )
            
            # 启动任务
            segmentation_task.start()
            
            logger.info(f"分割任务已启动: {task_id}/{subtask_id}")
            return True
            
        except Exception as e:
            error_msg = f"启动分割任务时出错: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            
            # 移除任务
            with self.mqtt_client.active_tasks_lock:
                if subtask_id in self.mqtt_client.active_tasks:
                    del self.mqtt_client.active_tasks[subtask_id]
            
            if confirmation_topic:
                self.mqtt_client._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", 
                                               data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
            return False 