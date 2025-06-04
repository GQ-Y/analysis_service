"""
智能帧跳过器
当检测结果稳定时，跳过部分帧的渲染以提高性能
"""
from typing import Dict, Any, Optional, List
import threading
import time
from collections import deque
import numpy as np

from shared.utils.logger import get_normal_logger

normal_logger = get_normal_logger(__name__)


class SmartFrameSkipper:
    """智能帧跳过器 - 在检测结果稳定时跳过渲染"""
    
    def __init__(self, stability_threshold: int = 5, skip_ratio: float = 0.5, 
                 confidence_threshold: float = 0.1):
        """
        初始化智能帧跳过器
        
        Args:
            stability_threshold: 连续稳定帧数阈值
            skip_ratio: 稳定时的跳过比例 (0.0-1.0)
            confidence_threshold: 检测结果变化的置信度阈值
        """
        self.stability_threshold = stability_threshold
        self.skip_ratio = skip_ratio
        self.confidence_threshold = confidence_threshold
        
        # 历史检测结果
        self.detection_history = deque(maxlen=stability_threshold * 2)
        self.frame_counter = 0
        self.stable_counter = 0
        self.last_rendered_frame = 0
        
        # 线程安全
        self.lock = threading.Lock()
        
        # 统计信息
        self.total_frames = 0
        self.skipped_frames = 0
        
        normal_logger.info(f"智能帧跳过器初始化完成，稳定阈值: {stability_threshold}, "
                          f"跳过比例: {skip_ratio}")
    
    def _calculate_detection_similarity(self, detections1: List[Dict], 
                                      detections2: List[Dict]) -> float:
        """
        计算两组检测结果的相似度
        
        Args:
            detections1: 第一组检测结果
            detections2: 第二组检测结果
            
        Returns:
            float: 相似度 (0.0-1.0)
        """
        if not detections1 and not detections2:
            return 1.0
        
        if len(detections1) != len(detections2):
            return 0.0
        
        if not detections1:  # 都为空
            return 1.0
        
        # 简化相似度计算：比较检测数量和平均置信度
        avg_conf1 = np.mean([float(det.get("confidence", 0)) for det in detections1])
        avg_conf2 = np.mean([float(det.get("confidence", 0)) for det in detections2])
        
        conf_similarity = 1.0 - abs(avg_conf1 - avg_conf2)
        
        # 比较类别分布
        classes1 = set(det.get("class_name", "") for det in detections1)
        classes2 = set(det.get("class_name", "") for det in detections2)
        
        if classes1 and classes2:
            class_similarity = len(classes1 & classes2) / len(classes1 | classes2)
        else:
            class_similarity = 1.0 if classes1 == classes2 else 0.0
        
        # 加权平均
        return (conf_similarity * 0.6 + class_similarity * 0.4)
    
    def _is_detection_stable(self) -> bool:
        """
        检查检测结果是否稳定
        
        Returns:
            bool: 是否稳定
        """
        if len(self.detection_history) < self.stability_threshold:
            return False
        
        # 取最近的检测结果
        recent_detections = list(self.detection_history)[-self.stability_threshold:]
        
        # 计算相邻帧之间的相似度
        similarities = []
        for i in range(1, len(recent_detections)):
            similarity = self._calculate_detection_similarity(
                recent_detections[i-1], recent_detections[i]
            )
            similarities.append(similarity)
        
        # 如果平均相似度高于阈值，认为稳定
        avg_similarity = np.mean(similarities) if similarities else 0.0
        stable = avg_similarity > (1.0 - self.confidence_threshold)
        
        return stable
    
    def should_skip_frame(self, analysis_result: Optional[Dict[str, Any]]) -> bool:
        """
        判断是否应该跳过当前帧的渲染
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            bool: 是否跳过渲染
        """
        with self.lock:
            self.frame_counter += 1
            self.total_frames += 1
            
            # 如果没有分析结果，不跳过
            if not analysis_result:
                return False
            
            detections = analysis_result.get("detections", [])
            
            # 记录检测历史
            self.detection_history.append(detections)
            
            # 检查是否稳定
            is_stable = self._is_detection_stable()
            
            if is_stable:
                self.stable_counter += 1
                
                # 在稳定状态下，根据跳过比例决定是否跳过
                frames_since_last_render = self.frame_counter - self.last_rendered_frame
                skip_interval = max(1, int(1 / (1 - self.skip_ratio)))
                
                should_skip = frames_since_last_render < skip_interval
                
                if should_skip:
                    self.skipped_frames += 1
                    return True
                else:
                    # 渲染这一帧
                    self.last_rendered_frame = self.frame_counter
                    return False
            else:
                # 不稳定时，重置稳定计数器，不跳过
                self.stable_counter = 0
                self.last_rendered_frame = self.frame_counter
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.lock:
            skip_rate = (self.skipped_frames / self.total_frames * 100) if self.total_frames > 0 else 0
            
            return {
                "total_frames": self.total_frames,
                "skipped_frames": self.skipped_frames,
                "skip_rate_percent": round(skip_rate, 2),
                "stable_counter": self.stable_counter,
                "current_stability": self._is_detection_stable(),
                "detection_history_size": len(self.detection_history)
            }
    
    def reset(self):
        """重置跳过器状态"""
        with self.lock:
            self.detection_history.clear()
            self.frame_counter = 0
            self.stable_counter = 0
            self.last_rendered_frame = 0
            self.total_frames = 0
            self.skipped_frames = 0
            normal_logger.info("智能帧跳过器状态已重置")
    
    def set_skip_ratio(self, skip_ratio: float):
        """
        设置跳过比例
        
        Args:
            skip_ratio: 新的跳过比例 (0.0-1.0)
        """
        if 0.0 <= skip_ratio <= 1.0:
            self.skip_ratio = skip_ratio
            normal_logger.info(f"帧跳过比例设置为: {skip_ratio}")
        else:
            normal_logger.warning(f"无效的跳过比例: {skip_ratio}, 必须在0.0-1.0之间")
    
    def set_stability_threshold(self, threshold: int):
        """
        设置稳定性阈值
        
        Args:
            threshold: 新的稳定性阈值
        """
        if threshold > 0:
            self.stability_threshold = threshold
            self.detection_history = deque(maxlen=threshold * 2)
            normal_logger.info(f"稳定性阈值设置为: {threshold}")
        else:
            normal_logger.warning(f"无效的稳定性阈值: {threshold}, 必须大于0")


# 全局智能帧跳过器实例
smart_frame_skipper = SmartFrameSkipper(
    stability_threshold=3,  # 3帧稳定
    skip_ratio=0.3,         # 稳定时跳过30%的帧
    confidence_threshold=0.1
) 