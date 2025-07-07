#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视频回放处理器 - 负责处理视频播放相关的分析结果
"""

import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import cv2
import numpy as np

from app.core.zero_copy.frame_buffer import FrameBuffer
from app.models.analysis_result import AnalysisResult
from app.core.memory.time_axis_manager import TimeAxisManager
from .base_processor import BaseResultProcessor


class VideoProcessor(BaseResultProcessor):
    """视频回放处理器"""
    
    def __init__(
        self,
        output_dir: str,
        playback_duration: int,
        time_axis_manager: TimeAxisManager,
        analysis_type: int,
        task_id: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化视频回放处理器
        
        Args:
            output_dir: 输出目录
            playback_duration: 回放视频时长（秒）
            time_axis_manager: 时间轴管理器
            analysis_type: 分析类型（2-视频分析，3-流分析）
            task_id: 任务ID
            logger: 日志记录器
        """
        super().__init__(logger)
        self.output_dir = Path(output_dir)
        self.playback_duration = playback_duration
        self.time_axis_manager = time_axis_manager
        self.analysis_type = analysis_type
        
        # 检查是否支持视频回放
        self.enabled = (
            analysis_type in [2, 3] and  # 只支持视频和流分析
            playback_duration >= 5       # 最小回放时长5秒
        )
        
        if self.enabled:
            # 创建输出目录
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 视频编码设置
            self.fps = 25  # 目标FPS
            self.codec = cv2.VideoWriter_fourcc(*'mp4v')
            
            # 统计信息
            self.videos_created = 0
            self.total_processed = 0
            
            self.logger.info(
                f"🎬 视频处理器已启用: 回放{playback_duration}秒, 输出到{output_dir}"
            )
        else:
            self.logger.info(
                f"🎬 视频处理器已禁用: 分析类型{analysis_type}, 回放时长{playback_duration}秒"
            )
    
    async def process_result(self, result: AnalysisResult):
        """
        处理分析结果，生成视频回放
        
        Args:
            result: 分析结果
        """
        if not self.enabled or not result or not result.detections:
            return
        
        self.total_processed += 1
        
        try:
            # 异步生成视频回放
            await self._generate_playback_video(result)
            
        except Exception as e:
            self.logger.error(f"❌ 视频回放生成失败 (帧{result.frame_id}): {e}")
    
    async def _generate_playback_video(self, result: AnalysisResult):
        """生成回放视频"""
        try:
            # 计算回放时间范围
            detection_timestamp = result.timestamp
            half_duration = self.playback_duration / 2
            start_time = detection_timestamp - half_duration
            end_time = detection_timestamp + half_duration
            
            self.logger.debug(
                f"🎬 生成回放视频: 帧{result.frame_id}, "
                f"时间范围{start_time:.2f}-{end_time:.2f}秒"
            )
            
            # 从时间轴获取帧序列
            frames = await self._get_frames_for_playback(start_time, end_time)
            
            if not frames:
                self.logger.warning(f"⚠️ 未获取到回放帧: 帧{result.frame_id}")
                return
            
            # 生成视频文件
            video_path = await self._create_video_file(result, frames)
            
            if video_path:
                self.videos_created += 1
                self.logger.info(
                    f"🎬 回放视频已生成: {video_path.name} "
                    f"({len(frames)}帧, {self.playback_duration}秒)"
                )
            
        except Exception as e:
            self.logger.error(f"❌ 生成回放视频失败: {e}")
    
    async def _get_frames_for_playback(self, start_time: float, end_time: float) -> List[np.ndarray]:
        """从时间轴获取回放帧"""
        try:
            # 计算需要的帧数
            target_frame_count = int(self.playback_duration * self.fps)
            
            # 从时间轴管理器获取帧
            frames = []
            
            # 计算时间步长
            time_step = (end_time - start_time) / target_frame_count
            
            for i in range(target_frame_count):
                frame_time = start_time + i * time_step
                
                # 从时间轴获取最近的帧
                frame_buffer = await self._get_frame_at_time(frame_time)
                
                if frame_buffer and frame_buffer.image_data is not None:
                    frames.append(frame_buffer.image_data.copy())
                else:
                    # 如果没有帧，使用前一帧或创建黑帧
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        # 创建黑帧（假设标准分辨率）
                        black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                        frames.append(black_frame)
            
            return frames
            
        except Exception as e:
            self.logger.error(f"❌ 获取回放帧失败: {e}")
            return []
    
    async def _get_frame_at_time(self, timestamp: float):
        """获取指定时间的帧"""
        try:
            # 从时间轴管理器获取最近的帧
            # 这里需要根据实际的时间轴管理器接口调用
            
            # 如果时间轴管理器有异步接口
            if hasattr(self.time_axis_manager, 'get_frame_at_timestamp_async'):
                return await self.time_axis_manager.get_frame_at_timestamp_async(timestamp)
            else:
                # 使用同步接口（在异步上下文中运行）
                return await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.time_axis_manager.get_frame_at_timestamp, 
                    timestamp
                )
                
        except Exception as e:
            self.logger.error(f"❌ 获取时间戳{timestamp}的帧失败: {e}")
            return None
    
    async def _create_video_file(self, result: AnalysisResult, frames: List[np.ndarray]) -> Optional[Path]:
        """创建视频文件"""
        try:
            if not frames:
                return None
            
            # 生成文件名
            timestamp_str = str(int(result.timestamp * 1000))
            filename = f"playback_frame_{result.frame_id}_{timestamp_str}.mp4"
            video_path = self.output_dir / filename
            
            # 获取视频尺寸
            height, width = frames[0].shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(video_path), 
                fourcc, 
                self.fps, 
                (width, height)
            )
            
            if not video_writer.isOpened():
                self.logger.error(f"❌ 无法创建视频写入器: {video_path}")
                return None
            
            # 写入帧
            for i, frame in enumerate(frames):
                # 在检测帧上绘制标注
                if i == len(frames) // 2:  # 中间帧是检测帧
                    frame = self._draw_detection_on_frame(frame, result)
                
                video_writer.write(frame)
            
            # 释放写入器
            video_writer.release()
            
            return video_path
            
        except Exception as e:
            self.logger.error(f"❌ 创建视频文件失败: {e}")
            return None
    
    def _draw_detection_on_frame(self, frame: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """在帧上绘制检测结果"""
        try:
            frame_copy = frame.copy()
            
            # 颜色映射（BGR格式）
            colors = [
                (0, 255, 0),    # 绿色
                (255, 0, 0),    # 蓝色
                (0, 0, 255),    # 红色
                (255, 255, 0),  # 青色
                (255, 0, 255),  # 品红
                (0, 255, 255),  # 黄色
            ]
            
            for i, detection in enumerate(result.detections):
                x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
                color = colors[i % len(colors)]
                
                # 绘制检测框（较粗的线条）
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
                
                # 绘制标签
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                # 标签背景
                cv2.rectangle(
                    frame_copy, 
                    (x1, y1 - label_size[1] - 15), 
                    (x1 + label_size[0] + 10, y1), 
                    color, 
                    -1
                )
                
                # 标签文字
                cv2.putText(
                    frame_copy, 
                    label, 
                    (x1 + 5, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (255, 255, 255), 
                    2
                )
            
            # 在左上角绘制时间戳
            timestamp_text = f"Detection Time: {result.timestamp:.2f}s"
            cv2.putText(
                frame_copy,
                timestamp_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            # 绘制检测统计
            stats_text = f"Detections: {len(result.detections)}"
            cv2.putText(
                frame_copy,
                stats_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            return frame_copy
            
        except Exception as e:
            self.logger.error(f"❌ 绘制检测结果失败: {e}")
            return frame
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "enabled": self.enabled,
            "total_processed": self.total_processed,
            "videos_created": self.videos_created,
            "playback_duration": self.playback_duration,
            "analysis_type": self.analysis_type,
            "output_dir": str(self.output_dir)
        }
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.enabled:
                self.logger.info(
                    f"🎬 视频处理器已清理 - 总处理{self.total_processed} "
                    f"视频{self.videos_created}"
                )
        except Exception as e:
            self.logger.error(f"❌ 清理视频处理器失败: {e}") 