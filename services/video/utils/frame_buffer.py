"""
智能帧缓冲系统
用于处理视频流卡顿，保持FFmpeg输出的稳定性和流畅性
"""
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np
import time
import threading
from collections import deque
from datetime import datetime, timedelta

from shared.utils.logger import get_normal_logger

normal_logger = get_normal_logger(__name__)


class FrameBuffer:
    """智能帧缓冲器 - 处理视频流卡顿和丢帧"""
    
    def __init__(self, buffer_size: int = None, target_fps: int = None):
        """
        初始化帧缓冲器

        Args:
            buffer_size: 缓冲区大小（帧数）
            target_fps: 目标帧率
        """
        # 延迟导入优化配置以避免循环导入
        try:
            from core.config_modules.optimization import optimization_config
            self.buffer_size = buffer_size or optimization_config.frame_processing.buffer_size
            self.target_fps = target_fps or optimization_config.frame_processing.target_fps
        except ImportError:
            self.buffer_size = buffer_size or 30  # 默认值
            self.target_fps = target_fps or 15  # 默认值
        self.frame_interval = 1.0 / target_fps
        
        # 帧缓冲区 - 存储 (frame, timestamp, analysis_result) 元组
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # 卡顿检测参数
        try:
            from core.config_modules.optimization import optimization_config
            self.stall_threshold = optimization_config.frame_processing.stall_threshold
        except ImportError:
            self.stall_threshold = 3.0  # 默认值
        self.last_frame_time = time.time()
        
        # 统计信息
        self.stats = {
            "total_frames_received": 0,
            "stall_events": 0,
            "cache_hits": 0,
            "interpolated_frames": 0,
            "last_stall_time": None,
            "current_stall_duration": 0.0
        }
        
        # 线程安全锁
        self.lock = threading.RLock()
        
        # 当前状态
        self.is_stalled = False
        self.last_valid_frame = None
        self.last_analysis_result = None
        
        normal_logger.info(f"帧缓冲器初始化完成 - 缓冲区大小: {buffer_size}, 目标FPS: {target_fps}")
    
    def add_frame(self, frame: np.ndarray, analysis_result: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加新帧到缓冲区
        
        Args:
            frame: 视频帧
            analysis_result: 分析结果
            
        Returns:
            bool: 是否成功添加
        """
        with self.lock:
            try:
                current_time = time.time()
                
                # 更新统计
                self.stats["total_frames_received"] += 1
                self.last_frame_time = current_time
                
                # 如果之前是卡顿状态，现在恢复了
                if self.is_stalled:
                    stall_duration = current_time - self.stats["last_stall_time"]
                    self.stats["current_stall_duration"] = stall_duration
                    normal_logger.info(f"视频流恢复，卡顿持续时间: {stall_duration:.2f}秒")
                    self.is_stalled = False
                
                # 添加到缓冲区
                frame_entry = (frame.copy(), current_time, analysis_result)
                self.frame_buffer.append(frame_entry)
                
                # 更新最后有效帧
                self.last_valid_frame = frame.copy()
                if analysis_result:
                    self.last_analysis_result = analysis_result.copy()
                
                return True
                
            except Exception as e:
                normal_logger.error(f"添加帧到缓冲区失败: {str(e)}")
                return False
    
    def get_frame_for_encoding(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], str]:
        """
        获取用于编码的帧 - 直播优化版本
        
        Returns:
            Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], str]: 
            (帧, 分析结果, 帧来源类型)
            帧来源类型: "live", "cached", "repeated", "interpolated"
        """
        with self.lock:
            current_time = time.time()
            
            # 优化：提高卡顿检测阈值，减少误判
            time_since_last_frame = current_time - self.last_frame_time
            
            # 将卡顿阈值从1.0秒增加到3.0秒，减少对短暂延迟的敏感度
            stall_threshold_live = 3.0
            
            if time_since_last_frame > stall_threshold_live:
                # 检测到真正的卡顿
                if not self.is_stalled:
                    self.is_stalled = True
                    self.stats["stall_events"] += 1
                    self.stats["last_stall_time"] = current_time
                    normal_logger.warning(f"检测到视频流卡顿，距离上次帧: {time_since_last_frame:.2f}秒")
                
                # 卡顿状态下，使用缓存策略
                return self._get_stall_frame()
            
            # 正常状态，优先获取最新帧
            if self.frame_buffer:
                frame, timestamp, analysis_result = self.frame_buffer[-1]
                return frame.copy(), analysis_result, "live"
            
            # 没有可用帧，使用最后的有效帧
            if self.last_valid_frame is not None:
                return self.last_valid_frame.copy(), self.last_analysis_result, "repeated"
            
            return None, None, "none"
    
    def _get_stall_frame(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], str]:
        """
        在卡顿状态下获取帧
        
        Returns:
            Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], str]: 
            (帧, 分析结果, 帧来源类型)
        """
        self.stats["cache_hits"] += 1
        
        # 策略1: 使用缓冲区中的最新帧
        if self.frame_buffer:
            frame, timestamp, analysis_result = self.frame_buffer[-1]
            return frame.copy(), analysis_result, "cached"
        
        # 策略2: 使用最后的有效帧
        if self.last_valid_frame is not None:
            return self.last_valid_frame.copy(), self.last_analysis_result, "repeated"
        
        # 策略3: 创建默认帧
        default_frame = self._create_stall_indicator_frame()
        return default_frame, None, "generated"
    
    def _create_stall_indicator_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """
        创建表示卡顿的指示帧
        
        Args:
            width: 帧宽度
            height: 帧高度
            
        Returns:
            np.ndarray: 指示帧
        """
        # 创建黑色背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加卡顿提示文字
        text = "Video Stream Buffering..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 255, 255)  # 黄色
        thickness = 2
        
        # 计算文字位置（居中）
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # 添加时间戳
        timestamp_text = f"Stall Duration: {self.stats['current_stall_duration']:.1f}s"
        timestamp_size = cv2.getTextSize(timestamp_text, font, 0.6, 1)[0]
        timestamp_x = (width - timestamp_size[0]) // 2
        timestamp_y = text_y + 40
        
        cv2.putText(frame, timestamp_text, (timestamp_x, timestamp_y), 
                   font, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def get_interpolated_frame(self, frame1: np.ndarray, frame2: np.ndarray, 
                             alpha: float = 0.5) -> np.ndarray:
        """
        生成两帧之间的插值帧
        
        Args:
            frame1: 第一帧
            frame2: 第二帧
            alpha: 插值权重 (0.0-1.0)
            
        Returns:
            np.ndarray: 插值帧
        """
        try:
            # 确保两帧尺寸相同
            if frame1.shape != frame2.shape:
                frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
            
            # 线性插值
            interpolated = cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0)
            
            self.stats["interpolated_frames"] += 1
            return interpolated
            
        except Exception as e:
            normal_logger.error(f"帧插值失败: {str(e)}")
            return frame1  # 返回第一帧作为备选
    
    def get_smooth_transition_frames(self, target_count: int = 3) -> List[np.ndarray]:
        """
        获取平滑过渡帧序列
        
        Args:
            target_count: 目标帧数
            
        Returns:
            List[np.ndarray]: 过渡帧列表
        """
        with self.lock:
            if len(self.frame_buffer) < 2:
                return []
            
            # 获取最近的两帧
            frame1, _, _ = self.frame_buffer[-2]
            frame2, _, _ = self.frame_buffer[-1]
            
            # 生成插值帧
            transition_frames = []
            for i in range(target_count):
                alpha = (i + 1) / (target_count + 1)
                interpolated = self.get_interpolated_frame(frame1, frame2, alpha)
                transition_frames.append(interpolated)
            
            return transition_frames
    
    def maintain_fps_with_cache(self) -> bool:
        """
        使用缓存维持目标帧率
        
        Returns:
            bool: 是否成功维持帧率
        """
        current_time = time.time()
        time_since_last = current_time - self.last_frame_time
        
        # 如果距离上次帧时间超过帧间隔，需要填补
        if time_since_last > self.frame_interval:
            missing_frames = int(time_since_last / self.frame_interval)
            
            if missing_frames > 1:
                normal_logger.debug(f"需要填补 {missing_frames} 帧以维持帧率")
                return True
        
        return False
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """
        获取缓冲区状态信息
        
        Returns:
            Dict[str, Any]: 状态信息
        """
        with self.lock:
            current_time = time.time()
            buffer_fill_ratio = len(self.frame_buffer) / self.buffer_size
            
            return {
                "buffer_size": len(self.frame_buffer),
                "max_buffer_size": self.buffer_size,
                "buffer_fill_ratio": buffer_fill_ratio,
                "is_stalled": self.is_stalled,
                "time_since_last_frame": current_time - self.last_frame_time,
                "stall_threshold": self.stall_threshold,
                "stats": self.stats.copy()
            }
    
    def set_stall_threshold(self, threshold: float):
        """设置卡顿检测阈值"""
        with self.lock:
            self.stall_threshold = threshold
            normal_logger.info(f"卡顿检测阈值已设置为: {threshold}秒")
    
    def clear_buffer(self):
        """清空缓冲区"""
        with self.lock:
            self.frame_buffer.clear()
            self.last_valid_frame = None
            self.last_analysis_result = None
            self.is_stalled = False
            normal_logger.info("帧缓冲区已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            stats = self.stats.copy()
            stats["current_buffer_size"] = len(self.frame_buffer)
            stats["buffer_fill_ratio"] = len(self.frame_buffer) / self.buffer_size
            stats["is_stalled"] = self.is_stalled
            return stats


# 全局帧缓冲器实例
frame_buffer = FrameBuffer() 