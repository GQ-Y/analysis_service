"""
目标跟踪模块
实现多种跟踪算法的封装
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
import cv2
from loguru import logger
from scipy.optimize import linear_sum_assignment

@dataclass
class TrackingObject:
    """跟踪对象"""
    track_id: int                  # 跟踪ID
    bbox: np.ndarray              # 边界框 [x1, y1, x2, y2]
    class_id: int                 # 类别ID
    confidence: float             # 置信度
    trajectory: List[np.ndarray]  # 轨迹
    age: int                      # 跟踪持续帧数
    time_since_update: int        # 最后更新后经过的帧数
    velocity: np.ndarray          # 速度向量 [dx, dy]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.tolist(),
            "class_id": self.class_id,
            "confidence": self.confidence,
            "track_info": {
                "trajectory": [point.tolist() for point in self.trajectory],
                "velocity": self.velocity.tolist(),
                "age": self.age,
                "time_since_update": self.time_since_update
            }
        }

class BaseTracker(ABC):
    """跟踪器基类"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """初始化跟踪器
        
        Args:
            max_age: 目标消失后保持跟踪的最大帧数
            min_hits: 确认为有效目标所需的最小检测次数
            iou_threshold: 跟踪器的IOU阈值
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count = 0
        self.tracks: List[TrackingObject] = []
        
    @abstractmethod
    async def update(self, detections: List[Dict[str, Any]]) -> List[TrackingObject]:
        """更新跟踪状态
        
        Args:
            detections: 检测结果列表，每个检测结果包含bbox、class_id和confidence
            
        Returns:
            List[TrackingObject]: 更新后的跟踪对象列表
        """
        pass
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """计算两个边界框的IOU
        
        Args:
            bbox1: 第一个边界框 [x1, y1, x2, y2]
            bbox2: 第二个边界框 [x1, y1, x2, y2]
            
        Returns:
            float: IOU值
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_velocity(self, track: TrackingObject) -> np.ndarray:
        """计算跟踪对象的速度
        
        Args:
            track: 跟踪对象
            
        Returns:
            np.ndarray: 速度向量 [dx, dy]
        """
        if len(track.trajectory) < 2:
            return np.zeros(2)
            
        current = track.trajectory[-1]
        previous = track.trajectory[-2]
        center_current = np.array([(current[0] + current[2]) / 2, (current[1] + current[3]) / 2])
        center_previous = np.array([(previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2])
        
        return center_current - center_previous

class SORTTracker(BaseTracker):
    """简单在线实时跟踪 (SORT) 实现"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """初始化SORT跟踪器"""
        super().__init__(max_age, min_hits, iou_threshold)
        self.next_track_id = 1
    
    async def update(self, detections: List[Dict[str, Any]]) -> List[TrackingObject]:
        """更新跟踪状态"""
        self.frame_count += 1
        
        # 将检测结果转换为numpy数组
        if not detections:
            # 如果没有检测结果，更新所有跟踪对象的状态
            for track in self.tracks:
                track.time_since_update += 1
            
            # 移除过期的跟踪对象
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return self.tracks
            
        detection_bboxes = []
        detection_scores = []
        detection_classes = []
        
        for det in detections:
            bbox = det["bbox"]
            detection_bboxes.append([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
            detection_scores.append(det["confidence"])
            detection_classes.append(det.get("class_id", 0))
            
        detection_bboxes = np.array(detection_bboxes)
        detection_scores = np.array(detection_scores)
        detection_classes = np.array(detection_classes)
        
        # 匹配现有跟踪对象和新的检测结果
        if self.tracks:
            # 计算IoU矩阵
            iou_matrix = np.zeros((len(self.tracks), len(detection_bboxes)))
            for t, track in enumerate(self.tracks):
                for d, det_bbox in enumerate(detection_bboxes):
                    iou_matrix[t, d] = self._calculate_iou(track.bbox, det_bbox)
            
            # 使用匈牙利算法进行匹配
            track_indices, detection_indices = linear_sum_assignment(-iou_matrix)
            
            # 更新匹配的跟踪对象
            for track_idx, det_idx in zip(track_indices, detection_indices):
                if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                    track = self.tracks[track_idx]
                    bbox = detection_bboxes[det_idx]
                    track.bbox = bbox
                    track.class_id = detection_classes[det_idx]
                    track.confidence = detection_scores[det_idx]
                    track.trajectory.append(bbox)
                    track.time_since_update = 0
                    track.age += 1
                    track.velocity = self._calculate_velocity(track)
                else:
                    detection_indices = np.delete(detection_indices, np.where(detection_indices == det_idx))
            
            # 更新未匹配的跟踪对象
            unmatched_tracks = [t for i, t in enumerate(self.tracks) if i not in track_indices]
            for track in unmatched_tracks:
                track.time_since_update += 1
        
        # 为未匹配的检测创建新的跟踪对象
        unmatched_detections = [i for i in range(len(detection_bboxes)) if i not in detection_indices]
        for det_idx in unmatched_detections:
            bbox = detection_bboxes[det_idx]
            new_track = TrackingObject(
                track_id=self.next_track_id,
                bbox=bbox,
                class_id=detection_classes[det_idx],
                confidence=detection_scores[det_idx],
                trajectory=[bbox],
                age=1,
                time_since_update=0,
                velocity=np.zeros(2)
            )
            self.tracks.append(new_track)
            self.next_track_id += 1
        
        # 移除过期的跟踪对象
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        return self.tracks

class ByteTracker(BaseTracker):
    """ByteTrack跟踪器实现"""
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        track_buffer: int = 30,     # 跟踪缓冲区大小
        match_thresh: float = 0.8,  # 匹配阈值
        track_high_thresh: float = 0.6,  # 高置信度阈值
        track_low_thresh: float = 0.1,   # 低置信度阈值
        new_track_thresh: float = 0.7,   # 新轨迹阈值
        **kwargs  # 添加kwargs来处理其他未使用的参数
    ):
        """初始化ByteTrack跟踪器
        
        Args:
            max_age: 目标消失后保持跟踪的最大帧数
            min_hits: 确认为有效目标所需的最小检测次数
            iou_threshold: 跟踪器的IOU阈值
            track_buffer: 跟踪缓冲区大小
            match_thresh: 匹配阈值
            track_high_thresh: 高置信度阈值
            track_low_thresh: 低置信度阈值
            new_track_thresh: 新轨迹阈值
            **kwargs: 其他参数
        """
        super().__init__(max_age, min_hits, iou_threshold)
        self.next_track_id = 1
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.high_thresh = track_high_thresh
        self.low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        
        logger.info(f"初始化ByteTrack跟踪器 - 参数: max_age={max_age}, min_hits={min_hits}, "
                   f"iou_threshold={iou_threshold}, track_buffer={track_buffer}, "
                   f"match_thresh={match_thresh}, high_thresh={track_high_thresh}, "
                   f"low_thresh={track_low_thresh}, new_track_thresh={new_track_thresh}")
        
    async def update(self, detections: List[Dict[str, Any]]) -> List[TrackingObject]:
        """更新跟踪状态"""
        self.frame_count += 1
        
        # 如果没有检测结果
        if not detections:
            for track in self.tracks:
                track.time_since_update += 1
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return self.tracks
            
        # 将检测结果分为高置信度和低置信度两组
        high_dets = []
        low_dets = []
        for det in detections:
            if det["confidence"] >= self.high_thresh:
                high_dets.append(det)
            elif det["confidence"] >= self.low_thresh:
                low_dets.append(det)
                
        # 先处理高置信度检测结果
        high_bboxes = []
        high_scores = []
        high_classes = []
        detection_indices = []  
        
        for det in high_dets:
            bbox = det["bbox"]
            high_bboxes.append([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
            high_scores.append(det["confidence"])
            high_classes.append(det.get("class_id", 0))
            
        high_bboxes = np.array(high_bboxes) if high_bboxes else np.empty((0, 4))
        high_scores = np.array(high_scores)
        high_classes = np.array(high_classes)
        
        # 匹配高置信度检测结果
        if self.tracks and len(high_bboxes) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(high_bboxes)))
            for t, track in enumerate(self.tracks):
                for d, det_bbox in enumerate(high_bboxes):
                    iou_matrix[t, d] = self._calculate_iou(track.bbox, det_bbox)
                    
            track_indices, detection_indices = linear_sum_assignment(-iou_matrix)
            
            # 更新匹配的跟踪对象
            matched_detection_indices = []  # 记录成功匹配的检测索引
            for track_idx, det_idx in zip(track_indices, detection_indices):
                if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                    track = self.tracks[track_idx]
                    bbox = high_bboxes[det_idx]
                    track.bbox = bbox
                    track.class_id = high_classes[det_idx]
                    track.confidence = high_scores[det_idx]
                    track.trajectory.append(bbox)
                    track.time_since_update = 0
                    track.age += 1
                    track.velocity = self._calculate_velocity(track)
                    matched_detection_indices.append(det_idx)
                    
            # 更新未匹配的检测索引
            detection_indices = matched_detection_indices
                    
            # 处理未匹配的跟踪对象
            unmatched_tracks = [t for i, t in enumerate(self.tracks) if i not in track_indices]
            for track in unmatched_tracks:
                track.time_since_update += 1
                
        # 处理低置信度检测结果
        if low_dets:
            low_bboxes = []
            low_scores = []
            low_classes = []
            
            for det in low_dets:
                bbox = det["bbox"]
                low_bboxes.append([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
                low_scores.append(det["confidence"])
                low_classes.append(det.get("class_id", 0))
                
            low_bboxes = np.array(low_bboxes)
            low_scores = np.array(low_scores)
            low_classes = np.array(low_classes)
            
            # 只将低置信度检测结果与未匹配的轨迹进行关联
            unmatched_tracks = [t for t in self.tracks if t.time_since_update > 0]
            if unmatched_tracks and len(low_bboxes) > 0:
                iou_matrix = np.zeros((len(unmatched_tracks), len(low_bboxes)))
                for t, track in enumerate(unmatched_tracks):
                    for d, det_bbox in enumerate(low_bboxes):
                        iou_matrix[t, d] = self._calculate_iou(track.bbox, det_bbox)
                        
                track_indices, detection_indices = linear_sum_assignment(-iou_matrix)
                
                # 更新匹配的跟踪对象
                for track_idx, det_idx in zip(track_indices, detection_indices):
                    if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                        track = unmatched_tracks[track_idx]
                        bbox = low_bboxes[det_idx]
                        track.class_id = low_classes[det_idx]
                        track.confidence = low_scores[det_idx]
                        track.trajectory.append(bbox)
                        track.time_since_update = 0
                        track.age += 1
                        track.velocity = self._calculate_velocity(track)
                        
        # 为未匹配的高置信度检测创建新的跟踪对象
        if len(high_bboxes) > 0:
            unmatched_detections = [i for i in range(len(high_bboxes)) if i not in detection_indices]
            for det_idx in unmatched_detections:
                bbox = high_bboxes[det_idx]
                new_track = TrackingObject(
                    track_id=self.next_track_id,
                    bbox=bbox,
                    class_id=high_classes[det_idx],
                    confidence=high_scores[det_idx],
                    trajectory=[bbox],
                    age=1,
                    time_since_update=0,
                    velocity=np.zeros(2)
                )
                self.tracks.append(new_track)
                self.next_track_id += 1
                
        # 移除过期的跟踪对象
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        return self.tracks

def create_tracker(tracker_type: str, **kwargs) -> BaseTracker:
    """创建跟踪器实例
    
    Args:
        tracker_type: 跟踪器类型，支持 'sort'、'bytetrack'
        **kwargs: 跟踪器参数
        
    Returns:
        BaseTracker: 跟踪器实例
    """
    tracker_map = {
        "sort": SORTTracker,
        "bytetrack": ByteTracker
    }
    
    if tracker_type.lower() not in tracker_map:
        raise ValueError(f"不支持的跟踪器类型: {tracker_type}，支持的类型有: {list(tracker_map.keys())}")
        
    return tracker_map[tracker_type.lower()](**kwargs) 