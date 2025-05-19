"""
越界检测器模块
实现目标越界检测功能
"""
import os
import time
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

from shared.utils.logger import get_normal_logger, get_exception_logger
from core.analyzer.base_analyzer import LineCrossingAnalyzer
from core.analyzer.tracking import YOLOTracker

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class LineCrossingDetector(LineCrossingAnalyzer):
    """越界检测器实现"""
    
    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0, 
                 yolo_version: int = 0, device: str = "auto", 
                 tracker_type: int = 0, **kwargs):
        """
        初始化越界检测器
        
        Args:
            model_code: 模型代码
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            device: 推理设备
            tracker_type: 跟踪器类型
            **kwargs: 其他参数
        """
        super().__init__(model_code, engine_type, yolo_version, device, tracker_type, **kwargs)
        
        # 创建YOLO跟踪器
        self.tracker = YOLOTracker(model_code, engine_type, yolo_version, device, tracker_type, **kwargs)
        
        # 越界检测相关参数
        self.counting_line = kwargs.get("counting_line", [(0.1, 0.5), (0.9, 0.5)])
        self.alarm_threshold = kwargs.get("alarm_threshold", 1)
        self.alarm_cooldown = kwargs.get("alarm_cooldown", 10)  # 秒
        self.alarm_classes = kwargs.get("alarm_classes", [])  # 空列表表示所有类别
        
        # 越界记录
        self.crossing_records = []
        self.last_alarm_time = datetime.now()
        
        normal_logger.info(f"初始化越界检测器: 越界线={self.counting_line}, 报警阈值={self.alarm_threshold}")
    
    async def load_model(self, model_code: str) -> bool:
        """
        加载模型
        
        Args:
            model_code: 模型代码
            
        Returns:
            bool: 是否成功加载模型
        """
        # 加载跟踪器模型
        result = await self.tracker.load_model(model_code)
        
        # 更新当前模型代码
        if result:
            self.current_model_code = model_code
            self.model = self.tracker.model
        
        return result
    
    async def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        对输入图像进行越界检测
        
        Args:
            image: BGR格式的输入图像
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 检测结果
        """
        start_time = time.time()
        
        try:
            # 执行跟踪
            tracking_result = await self.tracker.detect(image, **kwargs)
            
            # 获取跟踪结果
            tracked_objects = tracking_result.get("tracked_objects", [])
            
            # 检测越界
            crossing_events = self._detect_line_crossing(image, tracked_objects)
            
            # 检查是否需要报警
            alarm_triggered = self._check_alarm_condition(crossing_events)
            
            # 计算总时间
            total_time = (time.time() - start_time) * 1000
            
            # 构建返回结果
            result = {
                "detections": tracking_result.get("detections", []),
                "tracked_objects": tracked_objects,
                "crossing_events": crossing_events,
                "alarm_triggered": alarm_triggered,
                "pre_process_time": tracking_result.get("pre_process_time", 0),
                "inference_time": tracking_result.get("inference_time", 0),
                "post_process_time": tracking_result.get("post_process_time", 0),
                "tracking_time": tracking_result.get("tracking_time", 0),
                "line_crossing_time": total_time - tracking_result.get("pre_process_time", 0) - tracking_result.get("inference_time", 0) - tracking_result.get("post_process_time", 0) - tracking_result.get("tracking_time", 0),
                "annotated_image_bytes": tracking_result.get("annotated_image_bytes")
            }
            
            return result
            
        except Exception as e:
            exception_logger.exception(f"越界检测失败: {str(e)}")
            return {
                "detections": [],
                "tracked_objects": [],
                "crossing_events": [],
                "alarm_triggered": False,
                "pre_process_time": 0,
                "inference_time": 0,
                "post_process_time": 0,
                "tracking_time": 0,
                "line_crossing_time": 0,
                "annotated_image_bytes": None
            }
    
    def _detect_line_crossing(self, image: np.ndarray, tracked_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        检测目标越界
        
        Args:
            image: 输入图像
            tracked_objects: 跟踪结果
            
        Returns:
            List[Dict[str, Any]]: 越界事件列表
        """
        height, width = image.shape[:2]
        crossing_events = []
        
        # 转换越界线坐标为像素坐标
        line_p1 = (int(self.counting_line[0][0] * width), int(self.counting_line[0][1] * height))
        line_p2 = (int(self.counting_line[1][0] * width), int(self.counting_line[1][1] * height))
        
        # 检查每个跟踪目标
        for obj in tracked_objects:
            # 检查类别过滤
            if self.alarm_classes and obj["class_id"] not in self.alarm_classes:
                continue
                
            # 获取目标中心点
            bbox = obj["bbox"]
            center_x = (bbox["x1"] + bbox["x2"]) / 2
            center_y = (bbox["y1"] + bbox["y2"]) / 2
            
            # 检查是否越界
            if self._check_point_line_position(center_x, center_y, line_p1, line_p2) > 0:
                # 创建越界事件
                event = {
                    "track_id": obj["track_id"],
                    "class_id": obj["class_id"],
                    "timestamp": datetime.now().isoformat(),
                    "position": {
                        "x": center_x,
                        "y": center_y
                    },
                    "direction": "positive"  # 正向越界
                }
                crossing_events.append(event)
                
                # 添加到越界记录
                self.crossing_records.append(event)
            
        # 清理过期记录
        self._clean_old_records()
        
        return crossing_events
    
    def _check_point_line_position(self, x: float, y: float, line_p1: Tuple[int, int], line_p2: Tuple[int, int]) -> float:
        """
        检查点相对于线的位置
        
        Args:
            x: 点的x坐标
            y: 点的y坐标
            line_p1: 线的第一个端点
            line_p2: 线的第二个端点
            
        Returns:
            float: 点到线的有符号距离 (正值表示在线的一侧，负值表示在线的另一侧)
        """
        # 计算线的向量
        line_vec_x = line_p2[0] - line_p1[0]
        line_vec_y = line_p2[1] - line_p1[1]
        
        # 计算点到线的向量
        point_vec_x = x - line_p1[0]
        point_vec_y = y - line_p1[1]
        
        # 计算叉积
        cross_product = line_vec_x * point_vec_y - line_vec_y * point_vec_x
        
        return cross_product
    
    def _check_alarm_condition(self, crossing_events: List[Dict[str, Any]]) -> bool:
        """
        检查是否需要报警
        
        Args:
            crossing_events: 越界事件列表
            
        Returns:
            bool: 是否触发报警
        """
        # 检查是否有越界事件
        if not crossing_events:
            return False
            
        # 检查报警冷却时间
        now = datetime.now()
        if (now - self.last_alarm_time).total_seconds() < self.alarm_cooldown:
            return False
            
        # 检查越界数量是否达到阈值
        recent_events = [event for event in self.crossing_records 
                        if (now - datetime.fromisoformat(event["timestamp"])).total_seconds() < 60]
        
        if len(recent_events) >= self.alarm_threshold:
            # 更新最后报警时间
            self.last_alarm_time = now
            return True
            
        return False
    
    def _clean_old_records(self):
        """清理过期记录"""
        now = datetime.now()
        self.crossing_records = [event for event in self.crossing_records 
                               if (now - datetime.fromisoformat(event["timestamp"])).total_seconds() < 3600]
    
    async def process_video_frame(self, frame: np.ndarray, frame_index: int, **kwargs) -> Dict[str, Any]:
        """
        处理视频帧
        
        Args:
            frame: 视频帧
            frame_index: 帧索引
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        # 调用detect方法处理帧
        result = await self.detect(frame, **kwargs)
        
        # 添加帧索引
        result["frame_index"] = frame_index
        
        return result
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        if not hasattr(self, "tracker") or not self.tracker or not self.tracker.model:
            return {
                "loaded": False,
                "model_code": None
            }
            
        tracker_info = self.tracker.model_info
        
        return {
            "loaded": tracker_info.get("loaded", False),
            "model_code": self.current_model_code,
            "engine_type": self.engine_type,
            "yolo_version": self.yolo_version,
            "device": self.device,
            "tracker_type": self.tracker_type if hasattr(self, "tracker_type") else None,
            "counting_line": self.counting_line,
            "alarm_threshold": self.alarm_threshold
        }
    
    def release(self) -> None:
        """释放资源"""
        if hasattr(self, "tracker") and self.tracker:
            self.tracker.release()
            self.tracker = None
            
        self.model = None
        self.crossing_records = []
        normal_logger.info("越界检测器资源已释放")
