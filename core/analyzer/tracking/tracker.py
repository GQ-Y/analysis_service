"""
目标跟踪器模块
实现基于SORT、ByteTrack、DeepSORT等算法的目标跟踪功能
"""
import os
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from shared.utils.logger import setup_logger
from core.config import settings
import time
from collections import defaultdict
import math

logger = setup_logger(__name__)

class Tracker:
    """基础跟踪器类"""
    
    def __init__(self, tracker_type: int = 0, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        初始化跟踪器
        
        Args:
            tracker_type: 跟踪器类型 (0=SORT, 1=ByteTrack, 2=DeepSORT)
            max_age: 最大丢失帧数
            min_hits: 最小命中次数
            iou_threshold: IoU阈值
        """
        self.tracker_type = tracker_type
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        
        # 用于计数的线
        self.counting_line = None
        self.counting_enabled = False
        self.counts = {"up": 0, "down": 0, "left": 0, "right": 0}
        self.counted_ids = set()
        
        # 用于速度估计
        self.speed_estimation_enabled = False
        self.track_history = defaultdict(list)
        self.speed_estimates = {}
        self.pixels_per_meter = 100  # 默认值，可以通过标定更新
        self.fps = 25  # 默认帧率
        
        logger.info(f"初始化跟踪器: 类型={tracker_type}, 最大丢失帧数={max_age}, 最小命中次数={min_hits}, IoU阈值={iou_threshold}")
    
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        更新跟踪器
        
        Args:
            detections: 检测结果列表
            
        Returns:
            List[Dict[str, Any]]: 跟踪结果列表
        """
        self.frame_count += 1
        
        # 将检测结果转换为跟踪器所需的格式
        detection_boxes = []
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            confidence = det["confidence"]
            class_id = det["class_id"]
            detection_boxes.append([x1, y1, x2, y2, confidence, class_id])
        
        # 如果没有检测结果，返回空列表
        if not detection_boxes:
            # 更新现有轨迹（标记为丢失）
            self._update_lost_tracks()
            return []
        
        # 将检测结果转换为numpy数组
        detection_boxes = np.array(detection_boxes)
        
        # 根据跟踪器类型选择不同的更新方法
        if self.tracker_type == 0:  # SORT
            tracked_objects = self._update_sort(detection_boxes)
        elif self.tracker_type == 1:  # ByteTrack
            tracked_objects = self._update_bytetrack(detection_boxes)
        elif self.tracker_type == 2:  # DeepSORT
            tracked_objects = self._update_deepsort(detection_boxes)
        else:
            # 默认使用SORT
            tracked_objects = self._update_sort(detection_boxes)
        
        # 更新计数和速度估计
        if self.counting_enabled and self.counting_line:
            self._update_counting(tracked_objects)
        
        if self.speed_estimation_enabled:
            self._update_speed_estimation(tracked_objects)
        
        return tracked_objects
    
    def _update_sort(self, detection_boxes: np.ndarray) -> List[Dict[str, Any]]:
        """
        使用SORT算法更新跟踪器
        
        Args:
            detection_boxes: 检测框数组 [x1, y1, x2, y2, confidence, class_id]
            
        Returns:
            List[Dict[str, Any]]: 跟踪结果列表
        """
        # 提取边界框和置信度
        boxes = detection_boxes[:, :4]
        scores = detection_boxes[:, 4]
        class_ids = detection_boxes[:, 5]
        
        # 预测现有轨迹的新位置
        predicted_tracks = []
        for track in self.tracks:
            if track["time_since_update"] <= self.max_age:
                # 使用简单的线性运动模型预测
                new_bbox = self._predict_next_bbox(track["bboxes"][-1], track["velocity"])
                track["predicted_bbox"] = new_bbox
                predicted_tracks.append(track)
        
        # 计算IoU矩阵
        iou_matrix = np.zeros((len(predicted_tracks), len(boxes)))
        for i, track in enumerate(predicted_tracks):
            for j, box in enumerate(boxes):
                iou_matrix[i, j] = self._calculate_iou(track["predicted_bbox"], box)
        
        # 匹配检测结果和轨迹
        matched_indices = self._hungarian_matching(iou_matrix)
        
        # 更新匹配的轨迹
        for track_idx, detection_idx in matched_indices:
            if iou_matrix[track_idx, detection_idx] >= self.iou_threshold:
                track = predicted_tracks[track_idx]
                box = boxes[detection_idx]
                score = scores[detection_idx]
                class_id = int(class_ids[detection_idx])
                
                # 更新轨迹
                track["bboxes"].append(box)
                track["scores"].append(score)
                track["class_ids"].append(class_id)
                track["time_since_update"] = 0
                track["hits"] += 1
                
                # 计算速度
                if len(track["bboxes"]) >= 2:
                    prev_box = track["bboxes"][-2]
                    curr_box = track["bboxes"][-1]
                    track["velocity"] = self._calculate_velocity(prev_box, curr_box)
                
                # 更新状态
                if track["hits"] >= self.min_hits:
                    track["state"] = "confirmed"
            else:
                # IoU太低，视为未匹配
                predicted_tracks[track_idx]["time_since_update"] += 1
        
        # 找出未匹配的检测结果和轨迹
        unmatched_detections = [i for i in range(len(boxes)) if i not in [detection_idx for _, detection_idx in matched_indices]]
        unmatched_tracks = [i for i in range(len(predicted_tracks)) if i not in [track_idx for track_idx, _ in matched_indices]]
        
        # 更新未匹配的轨迹
        for track_idx in unmatched_tracks:
            predicted_tracks[track_idx]["time_since_update"] += 1
        
        # 创建新轨迹
        for det_idx in unmatched_detections:
            box = boxes[det_idx]
            score = scores[det_idx]
            class_id = int(class_ids[det_idx])
            
            # 创建新轨迹
            new_track = {
                "id": self.next_id,
                "bboxes": [box],
                "scores": [score],
                "class_ids": [class_id],
                "time_since_update": 0,
                "hits": 1,
                "state": "tentative",
                "velocity": [0, 0],
                "predicted_bbox": box
            }
            self.tracks.append(new_track)
            self.next_id += 1
        
        # 移除丢失太久的轨迹
        self.tracks = [track for track in self.tracks if track["time_since_update"] <= self.max_age]
        
        # 构建返回结果
        tracked_objects = []
        for track in self.tracks:
            if track["state"] == "confirmed":
                # 获取最新的边界框
                x1, y1, x2, y2 = track["bboxes"][-1]
                
                tracked_obj = {
                    "track_id": track["id"],
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    },
                    "confidence": float(track["scores"][-1]),
                    "class_id": int(track["class_ids"][-1]),
                    "velocity": {
                        "x": float(track["velocity"][0]),
                        "y": float(track["velocity"][1])
                    },
                    "state": track["state"],
                    "time_since_update": track["time_since_update"]
                }
                
                # 添加速度估计（如果启用）
                if self.speed_estimation_enabled and track["id"] in self.speed_estimates:
                    tracked_obj["speed"] = self.speed_estimates[track["id"]]
                
                tracked_objects.append(tracked_obj)
        
        return tracked_objects
