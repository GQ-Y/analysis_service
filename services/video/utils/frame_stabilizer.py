"""
帧稳定器
专门用于保证FFmpeg输出流的稳定性和连续性
"""
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np
import time
import threading
import asyncio
from collections import deque

from shared.utils.logger import get_normal_logger
from services.video.utils.frame_buffer import FrameBuffer

normal_logger = get_normal_logger(__name__)


class FrameStabilizer:
    """帧稳定器 - 确保FFmpeg输出的连续性和稳定性"""
    
    def __init__(self, target_fps: int = 15, buffer_size: int = 30):
        """
        初始化帧稳定器
        
        Args:
            target_fps: 目标帧率
            buffer_size: 缓冲区大小
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # 初始化帧缓冲器
        self.frame_buffer = FrameBuffer(buffer_size, target_fps)
        
        # 输出控制
        self.last_output_time = time.time()
        self.frame_count = 0
        
        # 稳定性策略配置
        self.stabilization_config = {
            "enable_interpolation": True,      # 启用帧插值
            "enable_frame_repeat": True,       # 启用帧重复
            "max_repeat_count": 5,             # 最大重复次数
            "interpolation_threshold": 2.0,    # 插值触发阈值(秒)
            "stall_recovery_frames": 3,        # 恢复时的过渡帧数
        }
        
        # 统计信息
        self.stats = {
            "total_output_frames": 0,
            "repeated_frames": 0,
            "interpolated_frames": 0,
            "stall_recovery_events": 0,
            "average_output_fps": 0.0,
            "last_fps_calculation": time.time()
        }
        
        # 状态管理
        self.is_running = False
        self.last_frame_source = "none"
        self.consecutive_repeats = 0
        
        # 线程安全
        self.lock = threading.RLock()
        
        normal_logger.info(f"帧稳定器初始化完成 - 目标FPS: {target_fps}, 缓冲区: {buffer_size}")
    
    def add_source_frame(self, frame: np.ndarray, analysis_result: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加源帧到稳定器
        
        Args:
            frame: 源帧
            analysis_result: 分析结果
            
        Returns:
            bool: 是否成功添加
        """
        return self.frame_buffer.add_frame(frame, analysis_result)
    
    def get_stable_frame(self) -> Tuple[Optional[np.ndarray], str, Dict[str, Any]]:
        """
        获取稳定的输出帧 - 直播优化版本
        
        Returns:
            Tuple[Optional[np.ndarray], str, Dict[str, Any]]: 
            (帧, 帧来源类型, 元数据)
        """
        with self.lock:
            current_time = time.time()
            
            # 直播模式优化：移除严格的帧率限制，优先实时性
            # 注释掉原来的rate_limited检查
            # if time_since_last_output < self.frame_interval:
            #     return None, "rate_limited", {}
            
            # 从缓冲器获取帧
            frame, analysis_result, source_type = self.frame_buffer.get_frame_for_encoding()
            
            if frame is None:
                return None, "no_frame", {}
            
            # 处理不同的帧来源
            output_frame, final_source_type, metadata = self._process_frame_by_source(
                frame, analysis_result, source_type, current_time
            )
            
            # 更新输出统计
            self._update_output_stats(final_source_type, current_time)
            
            # 记录最后输出时间
            self.last_output_time = current_time
            
            return output_frame, final_source_type, metadata
    
    def _process_frame_by_source(self, frame: np.ndarray, analysis_result: Optional[Dict[str, Any]], 
                               source_type: str, current_time: float) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        """
        根据帧来源类型处理帧
        
        Args:
            frame: 输入帧
            analysis_result: 分析结果
            source_type: 帧来源类型
            current_time: 当前时间
            
        Returns:
            Tuple[np.ndarray, str, Dict[str, Any]]: (处理后的帧, 最终类型, 元数据)
        """
        metadata = {
            "original_source": source_type,
            "timestamp": current_time,
            "has_analysis": analysis_result is not None
        }
        
        if source_type == "live":
            # 实时帧，重置重复计数
            self.consecutive_repeats = 0
            self.last_frame_source = "live"
            return frame, "live", metadata
        
        elif source_type == "cached":
            # 缓存帧，检查是否需要插值
            if (self.stabilization_config["enable_interpolation"] and 
                self.consecutive_repeats > 0):
                
                # 尝试生成插值帧
                interpolated_frames = self.frame_buffer.get_smooth_transition_frames(1)
                if interpolated_frames:
                    self.stats["interpolated_frames"] += 1
                    metadata["interpolation_applied"] = True
                    return interpolated_frames[0], "interpolated", metadata
            
            self.consecutive_repeats += 1
            return frame, "cached", metadata
        
        elif source_type == "repeated":
            # 重复帧处理
            if self.consecutive_repeats >= self.stabilization_config["max_repeat_count"]:
                # 超过最大重复次数，尝试其他策略
                return self._handle_excessive_repeats(frame, metadata)
            
            self.consecutive_repeats += 1
            self.stats["repeated_frames"] += 1
            metadata["repeat_count"] = self.consecutive_repeats
            return frame, "repeated", metadata
        
        else:
            # 其他类型，直接返回
            return frame, source_type, metadata
    
    def _handle_excessive_repeats(self, frame: np.ndarray, 
                                metadata: Dict[str, Any]) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        """
        处理过度重复的情况
        
        Args:
            frame: 当前帧
            metadata: 元数据
            
        Returns:
            Tuple[np.ndarray, str, Dict[str, Any]]: 处理结果
        """
        # 添加视觉指示器表示连接不稳定
        stabilized_frame = self._add_stability_indicator(frame.copy())
        
        metadata["stability_indicator"] = True
        metadata["excessive_repeats"] = self.consecutive_repeats
        
        # 重置重复计数
        self.consecutive_repeats = 0
        
        return stabilized_frame, "stabilized", metadata
    
    def _add_stability_indicator(self, frame: np.ndarray) -> np.ndarray:
        """
        在帧上添加稳定性指示器
        
        Args:
            frame: 输入帧
            
        Returns:
            np.ndarray: 添加指示器的帧
        """
        height, width = frame.shape[:2]
        
        # 在右上角添加小的不稳定指示器
        indicator_size = 10
        x_pos = width - 20
        y_pos = 10
        
        # 绘制黄色小圆点
        cv2.circle(frame, (x_pos, y_pos), indicator_size, (0, 255, 255), -1)
        cv2.circle(frame, (x_pos, y_pos), indicator_size, (0, 0, 0), 2)
        
        return frame
    
    def _update_output_stats(self, source_type: str, current_time: float):
        """
        更新输出统计信息
        
        Args:
            source_type: 帧来源类型
            current_time: 当前时间
        """
        self.stats["total_output_frames"] += 1
        self.frame_count += 1
        
        # 每10秒计算一次平均FPS
        time_since_fps_calc = current_time - self.stats["last_fps_calculation"]
        if time_since_fps_calc >= 10.0:
            self.stats["average_output_fps"] = self.frame_count / time_since_fps_calc
            self.frame_count = 0
            self.stats["last_fps_calculation"] = current_time
        
        # 根据来源类型更新相应统计
        if source_type == "repeated":
            self.stats["repeated_frames"] += 1
        elif source_type == "interpolated":
            self.stats["interpolated_frames"] += 1
    
    async def start_stable_output_stream(self, ffmpeg_process, frame_provider_callback):
        """
        启动稳定输出流
        
        Args:
            ffmpeg_process: FFmpeg进程
            frame_provider_callback: 帧提供回调函数
        """
        self.is_running = True
        normal_logger.info("帧稳定器开始稳定输出流")
        
        try:
            while self.is_running and ffmpeg_process.poll() is None:
                # 获取稳定帧
                stable_frame, source_type, metadata = self.get_stable_frame()
                
                if stable_frame is not None:
                    try:
                        # 写入FFmpeg
                        ffmpeg_process.stdin.write(stable_frame.tobytes())
                        
                        # 记录详细信息（可选）
                        if source_type in ["cached", "repeated", "interpolated"]:
                            normal_logger.debug(f"输出稳定帧: {source_type}, 元数据: {metadata}")
                            
                    except BrokenPipeError:
                        normal_logger.error("FFmpeg管道已断开")
                        break
                    except Exception as e:
                        normal_logger.error(f"写入FFmpeg失败: {str(e)}")
                        break
                
                # 调用帧提供者获取新帧
                if frame_provider_callback:
                    try:
                        await frame_provider_callback(self)
                    except Exception as e:
                        normal_logger.error(f"帧提供回调失败: {str(e)}")
                
                # 控制循环频率
                await asyncio.sleep(0.001)  # 1ms
                
        except Exception as e:
            normal_logger.error(f"稳定输出流异常: {str(e)}")
        finally:
            self.is_running = False
            normal_logger.info("帧稳定器停止输出流")
    
    def stop_stable_output(self):
        """停止稳定输出"""
        self.is_running = False
        normal_logger.info("帧稳定器收到停止信号")
    
    def get_stabilizer_stats(self) -> Dict[str, Any]:
        """
        获取稳定器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.lock:
            # 合并帧缓冲器和稳定器的统计
            buffer_stats = self.frame_buffer.get_stats()
            stabilizer_stats = self.stats.copy()
            
            return {
                "stabilizer": stabilizer_stats,
                "buffer": buffer_stats,
                "config": self.stabilization_config.copy(),
                "status": {
                    "is_running": self.is_running,
                    "last_frame_source": self.last_frame_source,
                    "consecutive_repeats": self.consecutive_repeats
                }
            }
    
    def configure_stabilization(self, config: Dict[str, Any]):
        """
        配置稳定化参数
        
        Args:
            config: 配置参数
        """
        with self.lock:
            for key, value in config.items():
                if key in self.stabilization_config:
                    self.stabilization_config[key] = value
                    normal_logger.info(f"稳定器配置更新: {key} = {value}")
    
    def reset_stats(self):
        """重置统计信息"""
        with self.lock:
            self.stats = {
                "total_output_frames": 0,
                "repeated_frames": 0,
                "interpolated_frames": 0,
                "stall_recovery_events": 0,
                "average_output_fps": 0.0,
                "last_fps_calculation": time.time()
            }
            self.frame_count = 0
            self.frame_buffer.clear_buffer()
            normal_logger.info("帧稳定器统计信息已重置")


# 全局帧稳定器实例
frame_stabilizer = FrameStabilizer() 