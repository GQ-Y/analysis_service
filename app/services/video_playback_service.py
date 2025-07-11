#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视频回放服务 - 队列机制版本
接收分析结果，从视频缓存中查询帧数据，生成回放视频
"""

import os
import time
import threading
import queue
import cv2
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging
import numpy as np
from collections import deque

from ..services.video_cache_service import VideoCacheService


class VideoPlaybackService:
    """视频回放服务
    
    功能：
    - 队列机制接收分析结果
    - 从视频缓存服务查询指定时间范围的帧
    - 合成回放视频文件
    - 添加分析结果叠加层
    """
    
    def __init__(
        self,
        video_cache_service: VideoCacheService,
        output_dir: str = "./storage/playback_videos",
        playback_duration: float = 10.0,  # 前后各playback_duration秒
        fps: float = 25.0,  # 视频输出帧率（将根据实际缓存帧率自动调整）
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化视频回放服务
        
        Args:
            video_cache_service: 视频缓存服务实例
            output_dir: 回放视频输出目录
            playback_duration: 回放视频时长（秒）
            fps: 输出视频帧率
            logger: 日志记录器
        """
        self.video_cache_service = video_cache_service
        self.playback_duration = playback_duration
        self.fps = fps
        
        # 设置输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志设置
        self.logger = logger or self._setup_logger()
        
        # 队列和线程控制
        self.analysis_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        self.lock = threading.Lock()
        
        # 【新增】时间段去重和追踪机制
        self.processed_time_segments = set()  # 已处理的时间段集合
        self.segment_lock = threading.Lock()  # 时间段操作锁
        self.time_segment_precision = 1.0  # 时间段精度（秒），用于去重
        
        # 统计信息
        self._stats = {
            "total_requests": 0,
            "successful_videos": 0,
            "failed_videos": 0,
            "duplicate_segments": 0,  # 新增：重复时间段计数
            "queue_size": 0,
            "processing_time_avg": 0.0,
            "start_time": 0
        }
        
        self.logger.info(f"🎬 视频回放服务初始化完成:")
        self.logger.info(f"   - 回放时长: {playback_duration}秒")
        self.logger.info(f"   - 输出帧率: {fps}fps")
        self.logger.info(f"   - 输出目录: {self.output_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("VideoPlaybackService")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 控制台输出
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(name)s: %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # 文件输出
            log_file = self.output_dir / "video_playback.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def start(self):
        """启动视频回放服务"""
        if self.is_running:
            self.logger.warning("⚠️ 视频回放服务已经在运行")
            return False
        
        self.logger.info("🚀 启动视频回放服务...")
        
        self.is_running = True
        self._stats["start_time"] = time.time()
        
        # 启动工作线程
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        self.logger.info("✅ 视频回放服务启动成功")
        return True
    
    def stop(self):
        """停止视频回放服务"""
        if not self.is_running:
            return
        
        self.logger.info("⏹️ 停止视频回放服务...")
        
        self.is_running = False
        
        # 等待工作线程结束
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10.0)
        
        self.logger.info("✅ 视频回放服务已停止")
    
    def add_analysis_result(self, analysis_data: Dict[str, Any]) -> bool:
        """
        添加分析结果到队列（兼容接口）
        
        Args:
            analysis_data: 分析结果数据，包含:
                - frame_id: 帧ID
                - timestamp: 时间戳
                - detections: 检测结果列表
                - model_name: 模型名称
                - confidence: 置信度
                - stream_id: 流ID
                
        Returns:
            bool: 是否成功添加
        """
        # 转换为标准格式
        analysis_result = {
            "timestamp": analysis_data.get("timestamp", 0.0),
            "detection_id": f"frame_{analysis_data.get('frame_id', 0)}_t{int(analysis_data.get('timestamp', 0))}",
            "analysis_data": analysis_data,
            "playback_duration": self.playback_duration
        }
        
        return self.submit_analysis_result(analysis_result)
    
    def submit_analysis_result(self, analysis_result: Dict[str, Any]) -> bool:
        """
        提交分析结果到队列
        
        Args:
            analysis_result: 分析结果字典，必须包含:
                - timestamp: 分析时间戳
                - detection_id: 检测ID  
                - analysis_data: 分析数据
                - playback_duration: 可选，自定义回放时长
                
        Returns:
            bool: 是否成功提交
        """
        if not self.is_running:
            self.logger.error("❌ 视频回放服务未运行，无法提交分析结果")
            return False
        
        # 验证必要字段
        required_fields = ["timestamp", "detection_id", "analysis_data"]
        for field in required_fields:
            if field not in analysis_result:
                self.logger.error(f"❌ 分析结果缺少必要字段: {field}")
                return False
        
        try:
            # 添加处理时间戳
            analysis_result["submitted_at"] = time.time()
            
            # 设置默认回放时长
            if "playback_duration" not in analysis_result:
                analysis_result["playback_duration"] = self.playback_duration
            
            # 提交到队列
            self.analysis_queue.put(analysis_result, timeout=1.0)
            
            with self.lock:
                self._stats["total_requests"] += 1
                self._stats["queue_size"] = self.analysis_queue.qsize()
            
            self.logger.info(f"📥 提交分析结果到队列: 检测ID={analysis_result['detection_id']}, "
                           f"时间戳={analysis_result['timestamp']:.3f}, 队列大小={self._stats['queue_size']}")
            
            return True
            
        except queue.Full:
            self.logger.error("❌ 分析结果队列已满，无法提交")
            return False
        except Exception as e:
            self.logger.error(f"❌ 提交分析结果失败: {e}")
            return False
    
    def _worker_loop(self):
        """工作线程主循环"""
        self.logger.info("🔄 视频回放工作线程启动")
        
        try:
            while self.is_running:
                try:
                    # 从队列获取分析结果（带超时）
                    analysis_result = self.analysis_queue.get(timeout=1.0)
                    
                    # 处理分析结果
                    self._process_analysis_result(analysis_result)
                    
                    # 标记任务完成
                    self.analysis_queue.task_done()
                    
                    # 更新队列大小统计
                    with self.lock:
                        self._stats["queue_size"] = self.analysis_queue.qsize()
                
                except queue.Empty:
                    # 队列为空，继续循环
                    continue
                except Exception as e:
                    self.logger.error(f"❌ 处理分析结果异常: {e}")
                    with self.lock:
                        self._stats["failed_videos"] += 1
                    continue
                    
        except Exception as e:
            self.logger.error(f"❌ 工作线程异常: {e}")
        finally:
            self.logger.info("🏁 视频回放工作线程结束")
    
    def _process_analysis_result(self, analysis_result: Dict[str, Any]):
        """
        处理单个分析结果，生成回放视频（支持时间段去重）
        
        Args:
            analysis_result: 分析结果数据
        """
        start_time = time.time()
        
        detection_id = analysis_result["detection_id"]
        timestamp = analysis_result["timestamp"]
        playback_duration = analysis_result.get("playback_duration", self.playback_duration)
        analysis_data = analysis_result["analysis_data"]
        
        # playback_duration 表示前后各N秒，所以总时长是 2 * playback_duration
        total_duration = 2 * playback_duration
        
        # 【关键新增】检查时间段是否已经被处理过
        if not self._check_and_mark_time_segment(timestamp, total_duration):
            self.logger.info(f"⏭️ 跳过重复时间段: 检测ID={detection_id}, 时间戳={timestamp:.3f}, "
                           f"时间段=[{timestamp-playback_duration:.1f}, {timestamp+playback_duration:.1f}]")
            with self.lock:
                self._stats["duplicate_segments"] += 1
            return
        
        self.logger.info(f"🎬 开始处理视频回放: 检测ID={detection_id}, 时间戳={timestamp:.3f}, 前后各{playback_duration}s (总时长{total_duration}s)")
        
        try:
            # 1. 【关键修复】等待未来帧被缓存
            # 检测事件发生时，只有过去的帧存在，需要等待未来的帧被录制
            current_time = time.time()
            wait_time = playback_duration  # 等待后半段时间的帧被缓存
            
            if timestamp + playback_duration > current_time:
                # 如果需要的结束时间还没到，计算需要等待的时间
                actual_wait_time = (timestamp + playback_duration) - current_time + 2.0  # 额外等待2秒确保缓存完成
                self.logger.info(f"⏳ 等待未来帧被缓存: 需要等待 {actual_wait_time:.1f}秒 (检测时间={timestamp:.3f}, 当前时间={current_time:.3f})")
                time.sleep(actual_wait_time)
            else:
                self.logger.info(f"✅ 所需帧应该已存在: 检测时间={timestamp:.3f}, 当前时间={current_time:.3f}")
            
            # 2. 从视频缓存查询帧数据和分析结果
            frames_with_metadata = self.video_cache_service.get_frames_with_metadata_for_time_range(timestamp, total_duration)
            frame_analysis_results = self.video_cache_service.get_analysis_results_for_time_range(timestamp, total_duration)
            
            if not frames_with_metadata:
                self.logger.error(f"❌ 未找到对应时间范围的视频帧: {timestamp:.3f} ± {playback_duration:.1f}s (总时长{total_duration}s)")
                with self.lock:
                    self._stats["failed_videos"] += 1
                return
            
            self.logger.info(f"📹 获取到 {len(frames_with_metadata)} 帧数据 + {len(frame_analysis_results)} 个分析结果，开始合成视频")
            
            # 3. 生成输出文件路径
            output_filename = f"playback_{detection_id}_{int(timestamp)}_{int(total_duration)}s.mp4"
            output_path = self.output_dir / output_filename
            
            # 4. 合成视频
            success = self._create_playback_video(
                frames_with_metadata=frames_with_metadata,
                output_path=output_path,
                analysis_data=analysis_data,
                frame_analysis_results=frame_analysis_results,  # 传递每帧的分析结果
                detection_id=detection_id,
                timestamp=timestamp,
                total_duration=total_duration  # 传递实际的总时长
            )
            
            # 5. 更新统计
            processing_time = time.time() - start_time
            
            with self.lock:
                if success:
                    self._stats["successful_videos"] += 1
                    self.logger.info(f"✅ 视频回放生成成功: {output_path} (耗时: {processing_time:.2f}s)")
                else:
                    self._stats["failed_videos"] += 1
                    self.logger.error(f"❌ 视频回放生成失败: {output_path}")
                
                # 更新平均处理时间
                total_videos = self._stats["successful_videos"] + self._stats["failed_videos"]
                if total_videos > 0:
                    self._stats["processing_time_avg"] = (
                        self._stats["processing_time_avg"] * (total_videos - 1) + processing_time
                    ) / total_videos
            
        except Exception as e:
            self.logger.error(f"❌ 处理分析结果异常: {e}")
            with self.lock:
                self._stats["failed_videos"] += 1
    
    def _create_playback_video(
        self,
        frames_with_metadata: List[Dict[str, Any]],
        output_path: Path,
        analysis_data: Dict[str, Any],
        frame_analysis_results: List[Dict[str, Any]],
        detection_id: str,
        timestamp: float,
        total_duration: Optional[float] = None
    ) -> bool:
        """
        创建回放视频
        
        Args:
            frames_with_metadata: 带元数据的帧列表
            output_path: 输出路径
            analysis_data: 触发分析数据
            frame_analysis_results: 帧分析结果列表
            detection_id: 检测ID
            timestamp: 时间戳
            total_duration: 总时长
            
        Returns:
            bool: 是否成功
        """
        if not frames_with_metadata:
            self.logger.error("❌ 没有帧数据用于创建视频")
            return False
        
        video_writer = None
        
        try:
            # 【关键修复1】确定分析时的分辨率
            # 防止frame_analysis_results为空的情况
            if frame_analysis_results:
                analysis_resolution = self._determine_analysis_resolution(frame_analysis_results[0].get('frame_buffer'), analysis_data)
            else:
                self.logger.warning("⚠️ 无分析结果数据，使用默认分析分辨率")
                analysis_resolution = self._determine_analysis_resolution(None, analysis_data)
            
            # 获取缓存帧的尺寸
            first_frame = frames_with_metadata[0]["frame"]
            cached_height, cached_width = first_frame.shape[:2]
            
            # 【关键修复2】决定输出视频的分辨率
            # 优先使用分析时的分辨率，如果无法确定则使用缓存帧的分辨率
            if analysis_resolution:
                output_width, output_height = analysis_resolution
                self.logger.info(f"🎯 使用分析时的分辨率作为输出分辨率: {output_width}x{output_height}")
                
                # 检查是否需要缩放
                need_resize = (cached_width != output_width or cached_height != output_height)
                if need_resize:
                    self.logger.info(f"🔧 需要将缓存帧从 {cached_width}x{cached_height} 缩放到 {output_width}x{output_height}")
                else:
                    self.logger.info(f"✅ 缓存帧分辨率与分析分辨率一致: {cached_width}x{cached_height}")
            else:
                # 无法确定分析分辨率，使用缓存帧分辨率
                output_width, output_height = cached_width, cached_height
                need_resize = False
                self.logger.warning(f"⚠️ 无法确定分析时的分辨率，使用缓存帧分辨率: {output_width}x{output_height}")
            
            # 【关键修复】智能计算有效帧率，确保时间连续性
            expected_duration = total_duration if total_duration is not None else (2 * self.playback_duration)
            
            # 计算帧之间的平均时间间隔
            if len(frames_with_metadata) >= 2:
                time_intervals = []
                for i in range(1, len(frames_with_metadata)):
                    interval = frames_with_metadata[i]["timestamp"] - frames_with_metadata[i-1]["timestamp"]
                    if 0 < interval <= 1.0:  # 过滤异常间隔
                        time_intervals.append(interval)
                
                if time_intervals:
                    avg_interval = sum(time_intervals) / len(time_intervals)
                    calculated_fps = 1.0 / avg_interval if avg_interval > 0 else self.fps
                    self.logger.info(f"📊 基于帧间隔计算FPS: 平均间隔={avg_interval:.3f}s, FPS={calculated_fps:.1f}")
                else:
                    # 回退到基于总时长的计算
                    calculated_fps = len(frames_with_metadata) / expected_duration if expected_duration > 0 else self.fps
                    self.logger.warning(f"⚠️ 无有效帧间隔，使用总时长计算FPS: {calculated_fps:.1f}")
            else:
                # 帧数不足，使用默认FPS
                calculated_fps = self.fps
                self.logger.warning(f"⚠️ 帧数不足，使用默认FPS: {calculated_fps}")
            
            # 使用计算出的FPS，但限制在合理范围内
            effective_fps = max(5.0, min(60.0, calculated_fps))  # 限制在5-60 FPS之间
            
            self.logger.info(f"🎬 最终使用FPS: {effective_fps:.1f} (原始计算: {calculated_fps:.1f})")
            
            # 设置视频编码器和输出格式
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, effective_fps, (output_width, output_height))
            
            if not video_writer.isOpened():
                self.logger.error(f"❌ 无法打开视频写入器: {output_path}")
                return False
            
            self.logger.info(f"📹 开始写入视频: {output_width}x{output_height}@{effective_fps:.1f}fps, 帧数={len(frames_with_metadata)}, 预期时长={expected_duration}s")
            
            # 【关键修复3】创建增强的帧-分析结果映射，包含分辨率信息和智能填充
            frame_analysis_map = {}
            
            # 首先按时间戳排序分析结果
            sorted_analysis_results = sorted(frame_analysis_results, key=lambda x: x.get("timestamp", 0))
            
            for analysis in sorted_analysis_results:
                frame_id = analysis.get("frame_id")
                if frame_id is not None:
                    analysis_data_content = analysis.get("analysis_data", {})
                    
                    # 增强分析数据，确保包含正确的分辨率信息
                    enhanced_analysis_data = analysis_data_content.copy()
                    
                    # 设置分析时的分辨率（用于坐标缩放计算）
                    if analysis_resolution:
                        enhanced_analysis_data["analysis_resolution"] = analysis_resolution
                    
                    frame_analysis_map[frame_id] = enhanced_analysis_data
            
            # 【新增】智能填充缺失的分析结果：使用最近的分析结果（间隔不超过1秒）
            filled_frame_analysis_map = {}
            last_valid_analysis = None
            last_valid_timestamp = None
            
            for i, frame_metadata in enumerate(frames_with_metadata):
                frame_id = frame_metadata["frame_id"]
                frame_timestamp = frame_metadata["timestamp"]
                
                if frame_id in frame_analysis_map:
                    # 当前帧有分析结果，直接使用
                    filled_frame_analysis_map[frame_id] = frame_analysis_map[frame_id]
                    last_valid_analysis = frame_analysis_map[frame_id]
                    last_valid_timestamp = frame_timestamp
                    self.logger.debug(f"✅ 帧{frame_id}使用原始分析结果")
                else:
                    # 当前帧无分析结果，尝试使用最近的分析结果
                    if (last_valid_analysis is not None and 
                        last_valid_timestamp is not None and 
                        abs(frame_timestamp - last_valid_timestamp) <= 1.0):  # 间隔不超过1秒
                        filled_frame_analysis_map[frame_id] = last_valid_analysis.copy()
                        self.logger.debug(f"🔄 帧{frame_id}使用最近的分析结果（时间差: {abs(frame_timestamp - last_valid_timestamp):.2f}s）")
                    else:
                        # 间隔太长或没有有效分析结果，使用空结果
                        filled_frame_analysis_map[frame_id] = {}
                        self.logger.debug(f"❌ 帧{frame_id}无可用分析结果（时间差: {abs(frame_timestamp - last_valid_timestamp) if last_valid_timestamp else 'N/A'}）")
            
            self.logger.info(f"📊 创建帧-分析映射: 原始{len(frame_analysis_map)}个，填充后{len(filled_frame_analysis_map)}个分析结果可用")
            
            # 写入每一帧
            for i, frame_metadata in enumerate(frames_with_metadata):
                frame = frame_metadata["frame"]
                frame_id = frame_metadata["frame_id"]
                frame_timestamp = frame_metadata["timestamp"]
                is_compensation = frame_metadata.get("is_compensation", False)  # 新增：检查是否为补偿帧
                
                # 【关键修复4】调整帧分辨率到输出分辨率
                if need_resize:
                    frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
                
                # 【新增】补偿帧处理日志
                if is_compensation:
                    self.logger.debug(f"📋 处理补偿帧: frame_id={frame_id}, timestamp={frame_timestamp:.3f}")
                
                # 查找该帧对应的分析结果（使用填充后的映射）
                frame_analysis = filled_frame_analysis_map.get(frame_id, {})
                
                # 添加分析结果叠加层
                # 【修复】使用基于帧时间戳的检测ID，确保ID和时间显示一致
                frame_detection_id = f"frame_{frame_id}_t{int(frame_timestamp)}"
                frame_with_overlay = self._add_analysis_overlay(
                    frame=frame,
                    analysis_data=frame_analysis,
                    detection_id=frame_detection_id,  # 使用帧时间戳生成的ID
                    timestamp=frame_timestamp,         # 使用帧的真实时间戳
                    frame_index=i,
                    total_frames=len(frames_with_metadata),
                    output_resolution=(output_width, output_height)  # 新增：传递输出分辨率
                )
                
                # 写入视频
                video_writer.write(frame_with_overlay)
            
            self.logger.info(f"✅ 视频写入完成: {len(frames_with_metadata)} 帧")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 创建视频失败: {e}")
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return False
        finally:
            if video_writer:
                video_writer.release()
    
    def _determine_analysis_resolution(self, frame_buffer, analysis_data: Dict[str, Any]) -> Tuple[int, int]:
        """
        确定分析时使用的分辨率 - 关键修复：YOLO模型实际输入分辨率
        
        Args:
            frame_buffer: 帧缓冲区
            analysis_data: 分析数据
            
        Returns:
            Tuple[int, int]: 分析分辨率 (width, height)
        """
        try:
            # 🔧 优先级1: 从analysis_data中获取模型的实际输入尺寸
            if "model_results" in analysis_data:
                for model_code, model_result in analysis_data["model_results"].items():
                    # 检查是否包含模型输入尺寸信息
                    if isinstance(model_result, dict):
                        # 方法1: 直接从模型结果中获取input_size
                        if "input_size" in model_result:
                            input_size = model_result["input_size"]
                            if isinstance(input_size, (list, tuple)) and len(input_size) >= 2:
                                width, height = input_size[0], input_size[1]
                                self.logger.info(f"📏 从模型结果获取分析分辨率: {width}x{height} (模型: {model_code})")
                                return (width, height)
                        
                        # 方法2: 从image_shape获取(YOLO模型推理时的实际输入尺寸)
                        if "image_shape" in model_result:
                            image_shape = model_result["image_shape"]
                            if isinstance(image_shape, (list, tuple)) and len(image_shape) >= 2:
                                # image_shape格式通常是 (height, width, channels)
                                height, width = image_shape[0], image_shape[1]
                                self.logger.info(f"📏 从image_shape获取分析分辨率: {width}x{height} (模型: {model_code})")
                                return (width, height)
                        
                        # 方法3: 检查是否是YOLO模型，使用默认640x640
                        if "yolo" in model_code.lower() or "detection" in str(model_result.get("analyzer_info", {})):
                            self.logger.info(f"📏 检测到YOLO模型，使用默认分析分辨率: 640x640 (模型: {model_code})")
                            return (640, 640)
           
            # 🔧 优先级2: 从frame_buffer.analysis_resolution获取
            if hasattr(frame_buffer, 'analysis_resolution') and frame_buffer.analysis_resolution:
                analysis_resolution = frame_buffer.analysis_resolution
                if isinstance(analysis_resolution, (list, tuple)) and len(analysis_resolution) >= 2:
                    width, height = analysis_resolution[0], analysis_resolution[1]
                    self.logger.info(f"📏 从frame_buffer获取分析分辨率: {width}x{height}")
                    return (width, height)
           
            # 🔧 优先级3: 从analysis_result.image_shape获取  
            if hasattr(frame_buffer, 'analysis_result') and frame_buffer.analysis_result:
                analysis_result = frame_buffer.analysis_result
                if hasattr(analysis_result, 'image_shape') and analysis_result.image_shape:
                    image_shape = analysis_result.image_shape
                    if isinstance(image_shape, (list, tuple)) and len(image_shape) >= 2:
                        # 处理OpenCV格式 (height, width, channels) -> (width, height)
                        if len(image_shape) >= 3:
                            height, width = image_shape[0], image_shape[1]
                        else:
                            width, height = image_shape[0], image_shape[1]
                        self.logger.info(f"📏 从analysis_result.image_shape获取分析分辨率: {width}x{height}")
                        return (width, height)
           
            # 🔧 优先级4: 从video_cache_service获取目标分辨率(注意：这可能不是分析分辨率！)
            if self.video_cache_service and hasattr(self.video_cache_service, 'get_target_resolution'):
                target_resolution = self.video_cache_service.get_target_resolution()
                if target_resolution and len(target_resolution) >= 2:
                    width, height = target_resolution[0], target_resolution[1]
                    self.logger.warning(f"⚠️ 使用缓存目标分辨率作为分析分辨率: {width}x{height} (可能不准确)")
                    return (width, height)
           
            # 🔧 优先级5: 从analysis_data中查找任何分辨率信息
            if "analysis_resolution" in analysis_data:
                analysis_resolution = analysis_data["analysis_resolution"]
                if isinstance(analysis_resolution, (list, tuple)) and len(analysis_resolution) >= 2:
                    width, height = analysis_resolution[0], analysis_resolution[1]
                    self.logger.info(f"📏 从analysis_data获取分析分辨率: {width}x{height}")
                    return (width, height)
           
            # 🔧 优先级6: 最后的默认值 - YOLO常用的640x640
            self.logger.warning("⚠️ 无法确定分析分辨率，使用YOLO默认分辨率: 640x640")
            return (640, 640)
           
        except Exception as e:
            self.logger.error(f"❌ 确定分析分辨率时出错: {e}")
            # 发生错误时使用安全的默认值
            self.logger.warning("⚠️ 发生错误，使用默认分析分辨率: 640x640")
            return (640, 640)
    
    def _add_analysis_overlay(
        self,
        frame: np.ndarray,
        analysis_data: Dict[str, Any],
        detection_id: str,
        timestamp: float,
        frame_index: int,
        total_frames: int,
        output_resolution: Optional[tuple] = None # 新增：接收输出分辨率
    ) -> np.ndarray:
        """
        为帧添加分析结果叠加层
        
        Args:
            frame: 原始帧
            analysis_data: 分析数据
            detection_id: 检测ID
            timestamp: 时间戳
            frame_index: 当前帧索引
            total_frames: 总帧数
            output_resolution: 输出视频的分辨率，用于坐标缩放
            
        Returns:
            np.ndarray: 带叠加层的帧
        """
        # 复制帧避免修改原始数据
        overlay_frame = frame.copy()
        
        try:
            # 基本信息文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (255, 255, 255)  # 白色
            thickness = 2
            
            # 【已移除】不再显示顶部时间和检测ID信息栏
            # 注释理由：用户要求去除视频左上角的时间和分析ID显示
            height, width = overlay_frame.shape[:2]
            
            # # 顶部信息栏（已禁用）
            # cv2.rectangle(overlay_frame, (0, 0), (width, 80), (0, 0, 0), -1)  # 黑色背景
            # overlay = overlay_frame.copy()
            # cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
            # cv2.addWeighted(overlay_frame, 0.7, overlay, 0.3, 0, overlay_frame)  # 半透明效果
            
            # # 添加文本信息（已禁用）
            # y_offset = 25
            
            # # 检测ID和时间戳（已禁用）
            # text1 = f"Detection ID: {detection_id}"
            # cv2.putText(overlay_frame, text1, (10, y_offset), font, font_scale, font_color, thickness)
            
            # # 时间戳（格式化为可读时间）（已禁用）
            # import datetime
            # readable_time = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            # text2 = f"Time: {readable_time}"
            # cv2.putText(overlay_frame, text2, (10, y_offset + 25), font, font_scale, font_color, thickness)
            
            # # 进度条（已禁用）
            # progress = frame_index / max(total_frames - 1, 1)
            # progress_width = int(300 * progress)
            # cv2.rectangle(overlay_frame, (10, y_offset + 35), (310, y_offset + 45), (100, 100, 100), -1)  # 灰色背景
            # cv2.rectangle(overlay_frame, (10, y_offset + 35), (10 + progress_width, y_offset + 45), (0, 255, 0), -1)  # 绿色进度
            
            # 【关键修复】添加分析结果特定信息 - 处理不同的数据结构
            detections_to_draw = []
            
            # 情况1：直接包含detections字段
            if "detections" in analysis_data:
                detections_to_draw = analysis_data["detections"]
                self.logger.debug(f"📊 使用直接detections字段: {len(detections_to_draw)} 个检测")
            
            # 情况2：包含model_results结构
            elif "model_results" in analysis_data:
                for model_name, model_result in analysis_data["model_results"].items():
                    if "detections" in model_result:
                        detections_to_draw.extend(model_result["detections"])
                self.logger.debug(f"📊 从model_results提取: {len(detections_to_draw)} 个检测")
            
            # 情况3：嵌套的analysis_data结构（从视频缓存服务可能获取的格式）
            elif "analysis_data" in analysis_data:
                nested_data = analysis_data["analysis_data"]
                if "detections" in nested_data:
                    detections_to_draw = nested_data["detections"]
                elif "model_results" in nested_data:
                    for model_result in nested_data["model_results"].values():
                        if "detections" in model_result:
                            detections_to_draw.extend(model_result["detections"])
                self.logger.debug(f"📊 从嵌套analysis_data提取: {len(detections_to_draw)} 个检测")
            
            # 绘制检测框
            if detections_to_draw:
                self._add_detection_boxes(overlay_frame, detections_to_draw, analysis_data, output_resolution)
                self.logger.debug(f"✅ 已绘制 {len(detections_to_draw)} 个检测框")
            else:
                self.logger.debug(f"🔍 该帧无检测结果")
            
            # 添加告警信息（如果有）
            if "alert_info" in analysis_data:
                self._add_alert_info(overlay_frame, analysis_data["alert_info"])
            
        except Exception as e:
            self.logger.warning(f"⚠️ 添加叠加层失败: {e}")
            import traceback
            self.logger.debug(f"详细错误: {traceback.format_exc()}")
            return frame  # 返回原始帧
        
        return overlay_frame
    
    def _add_detection_boxes(self, frame: np.ndarray, detections: List[Dict[str, Any]], analysis_data: Dict[str, Any] = None, output_resolution: Optional[tuple] = None):
        """添加检测框"""
        # 获取当前帧尺寸
        frame_height, frame_width = frame.shape[:2]
        
        # 【关键修复】获取分析时的分辨率信息
        analysis_resolution = None
        
        self.logger.info(f"🔍 开始绘制检测框: 帧分辨率={frame_width}x{frame_height}, 检测数量={len(detections)}")
        
        # 优先从传入的analysis_data中获取分析分辨率
        if analysis_data and "analysis_resolution" in analysis_data:
            analysis_resolution = analysis_data["analysis_resolution"]
            self.logger.info(f"📊 使用传入的分析分辨率: {analysis_resolution}")
        
        # 回退：尝试从analysis_data中的其他字段获取
        elif analysis_data:
            self.logger.debug(f"🔍 analysis_data包含以下字段: {list(analysis_data.keys())}")
            
            # 直接从分析数据获取image_shape
            if "image_shape" in analysis_data:
                image_shape = analysis_data["image_shape"]
                if image_shape and len(image_shape) >= 2:
                    # 如果是(height, width, channels)格式，转换为(width, height)
                    if len(image_shape) == 3:
                        height, width, _ = image_shape
                        analysis_resolution = (width, height)
                    else:
                        analysis_resolution = tuple(image_shape[:2])
                    self.logger.info(f"📊 从analysis_data的image_shape获取分析分辨率: {analysis_resolution}")
            
            # 从model_results中尝试获取
            elif "model_results" in analysis_data:
                for model_name, model_result in analysis_data["model_results"].items():
                    if isinstance(model_result, dict) and "image_shape" in model_result:
                        image_shape = model_result["image_shape"]
                        if image_shape and len(image_shape) >= 2:
                            if len(image_shape) == 3:
                                height, width, _ = image_shape
                                analysis_resolution = (width, height)
                            else:
                                analysis_resolution = tuple(image_shape[:2])
                            self.logger.info(f"📊 从model_results({model_name})获取分析分辨率: {analysis_resolution}")
                            break
        
        # 【新增】如果仍未获取到分析分辨率，尝试从视频缓存服务获取
        if not analysis_resolution and self.video_cache_service:
            if hasattr(self.video_cache_service, 'target_resolution') and self.video_cache_service.target_resolution:
                analysis_resolution = self.video_cache_service.target_resolution
                self.logger.info(f"📊 从视频缓存服务获取目标分辨率: {analysis_resolution}")
            elif hasattr(self.video_cache_service, 'original_resolution') and self.video_cache_service.original_resolution:
                analysis_resolution = self.video_cache_service.original_resolution
                self.logger.warning(f"⚠️ 使用视频缓存服务的原始分辨率: {analysis_resolution}")
        
        # 默认分辨率：1080P
        if not analysis_resolution:
            analysis_resolution = (1920, 1080)
            self.logger.warning(f"⚠️ 使用默认1080P分辨率: {analysis_resolution}")
        
        # 计算坐标缩放比例
        scale_x = 1.0
        scale_y = 1.0
        
        if analysis_resolution:
            analysis_width, analysis_height = analysis_resolution
            scale_x = frame_width / analysis_width
            scale_y = frame_height / analysis_height
            
            self.logger.info(f"🔧 坐标缩放计算: 分析分辨率{analysis_width}x{analysis_height} -> 输出分辨率{frame_width}x{frame_height}")
            self.logger.info(f"🔧 缩放比例: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")
            
            if scale_x != 1.0 or scale_y != 1.0:
                self.logger.info(f"✅ 需要进行坐标缩放")
            else:
                self.logger.info(f"✅ 分辨率一致，无需坐标缩放")
        else:
            self.logger.error(f"❌ 无法获取分析分辨率，使用原始坐标（可能导致位置错误）")
        
        # 绘制每个检测框
        for i, detection in enumerate(detections):
            try:
                if "bbox" not in detection:
                    self.logger.warning(f"⚠️ 检测结果{i}缺少bbox字段")
                    continue
                
                bbox = detection["bbox"]
                
                # 处理不同的bbox格式
                if isinstance(bbox, dict):
                    # 字典格式：{"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    x1 = float(bbox.get("x1", 0))
                    y1 = float(bbox.get("y1", 0)) 
                    x2 = float(bbox.get("x2", 0))
                    y2 = float(bbox.get("y2", 0))
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    # 列表格式：[x1, y1, x2, y2]
                    x1, y1, x2, y2 = [float(x) for x in bbox[:4]]
                else:
                    self.logger.warning(f"⚠️ 检测结果{i}的bbox格式无效: {type(bbox)}, 值: {bbox}")
                    continue
                
                self.logger.debug(f"🎯 检测框{i}原始坐标: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                
                # 【关键修复】应用坐标缩放
                if scale_x != 1.0 or scale_y != 1.0:
                    # 缩放坐标
                    x1_scaled = x1 * scale_x
                    y1_scaled = y1 * scale_y
                    x2_scaled = x2 * scale_x
                    y2_scaled = y2 * scale_y
                    
                    self.logger.debug(f"🔧 检测框{i}缩放后坐标: ({x1_scaled:.1f}, {y1_scaled:.1f}, {x2_scaled:.1f}, {y2_scaled:.1f})")
                    
                    x1, y1, x2, y2 = x1_scaled, y1_scaled, x2_scaled, y2_scaled
                
                # 确保坐标为整数并在图像范围内
                x1 = max(0, min(int(x1), frame_width - 1))
                y1 = max(0, min(int(y1), frame_height - 1))
                x2 = max(0, min(int(x2), frame_width - 1))
                y2 = max(0, min(int(y2), frame_height - 1))
                
                # 确保x2 > x1, y2 > y1
                if x2 <= x1 or y2 <= y1:
                    self.logger.warning(f"⚠️ 检测框{i}坐标无效: ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                self.logger.debug(f"✅ 检测框{i}最终坐标: ({x1}, {y1}, {x2}, {y2})")
                
                # 绘制检测框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加标签（优先使用class_name，fallback到label）
                class_name = detection.get("class_name") or detection.get("label", "unknown")
                confidence = detection.get("confidence", 0.0)
                text = f"{class_name}: {confidence:.2f}"
                
                # 计算文本尺寸
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # 绘制文本背景
                cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), (0, 255, 0), -1)
                
                # 绘制文本
                cv2.putText(frame, text, (x1, y1 - 5), 
                          font, font_scale, (0, 0, 0), thickness)
                    
            except Exception as e:
                self.logger.error(f"❌ 绘制检测框{i}失败: {e}")
                import traceback
                self.logger.debug(f"详细错误: {traceback.format_exc()}")
                continue
        
        self.logger.info(f"✅ 检测框绘制完成，共处理 {len(detections)} 个检测结果")
    
    def _add_alert_info(self, frame: np.ndarray, alert_info: Dict[str, Any]):
        """添加警报信息"""
        try:
            if alert_info.get("triggered", False):
                # 在右上角显示警报
                height, width = frame.shape[:2]
                alert_text = f"ALERT: {alert_info.get('type', 'Unknown')}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                color = (0, 0, 255)  # 红色
                
                # 计算文本位置
                (text_width, text_height), _ = cv2.getTextSize(alert_text, font, font_scale, thickness)
                x = width - text_width - 10
                y = text_height + 10
                
                # 绘制警报文本
                cv2.putText(frame, alert_text, (x, y), font, font_scale, color, thickness)
                
        except Exception as e:
            self.logger.warning(f"⚠️ 添加警报信息失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            current_time = time.time()
            start_time = self._stats.get("start_time", current_time)
            runtime = current_time - start_time
            
            stats = self._stats.copy()
            stats.update({
                "runtime_seconds": runtime,
                "current_queue_size": self.analysis_queue.qsize(),
                "success_rate": (
                    self._stats["successful_videos"] / max(self._stats["total_requests"], 1) * 100
                ),
                "is_running": self.is_running,
                "output_dir": str(self.output_dir),
                "playback_duration": self.playback_duration,
                "fps": self.fps
            })
        
        # 添加时间段统计（在lock外部获取，避免死锁）
        with self.segment_lock:
            stats["processed_segments_count"] = len(self.processed_time_segments)
            
            return stats
    
    def _check_and_mark_time_segment(self, timestamp: float, duration: float) -> bool:
        """
        检查时间段是否已处理过，如果没有则标记为已处理
        
        Args:
            timestamp: 中心时间戳
            duration: 总时长
            
        Returns:
            bool: True=可以处理（首次），False=已处理过（跳过）
        """
        # 计算时间段的标准化键值（基于精度对齐）
        half_duration = duration / 2.0
        start_time = timestamp - half_duration
        end_time = timestamp + half_duration
        
        # 使用精度对齐，避免浮点数误差
        aligned_start = int(start_time / self.time_segment_precision) * self.time_segment_precision
        aligned_end = int(end_time / self.time_segment_precision) * self.time_segment_precision
        
        # 生成时间段键值
        segment_key = f"{aligned_start:.1f}-{aligned_end:.1f}"
        
        with self.segment_lock:
            # 检查是否重复
            if segment_key in self.processed_time_segments:
                self.logger.debug(f"🔍 时间段重复检测: {segment_key} 已存在")
                return False
            
            # 标记为已处理
            self.processed_time_segments.add(segment_key)
            self.logger.debug(f"✅ 标记时间段: {segment_key} 已处理 (总段数: {len(self.processed_time_segments)})")
            
            return True
    
    def get_processed_segments_info(self) -> Dict[str, Any]:
        """获取已处理时间段信息"""
        with self.segment_lock:
            return {
                "total_segments": len(self.processed_time_segments),
                "segments": sorted(list(self.processed_time_segments)),
                "precision": self.time_segment_precision
            }
    
    def clear_processed_segments(self) -> int:
        """
        清空已处理时间段记录（用于重新开始或测试）
        
        Returns:
            int: 清空的段数
        """
        with self.segment_lock:
            count = len(self.processed_time_segments)
            self.processed_time_segments.clear()
            self.logger.info(f"🧹 清空已处理时间段记录: {count} 个段")
            return count

    def clear_queue(self):
        """清空队列"""
        cleared_count = 0
        while not self.analysis_queue.empty():
            try:
                self.analysis_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
        
        self.logger.info(f"🧹 清空队列: 移除了 {cleared_count} 个待处理项目")
        return cleared_count


# 异步接口封装
class AsyncVideoPlaybackService:
    """异步视频回放服务封装"""
    
    def __init__(self, video_playback_service: VideoPlaybackService):
        self.service = video_playback_service
    
    async def submit_analysis_result_async(self, analysis_result: Dict[str, Any]) -> bool:
        """异步提交分析结果"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.service.submit_analysis_result, analysis_result)
    
    async def get_stats_async(self) -> Dict[str, Any]:
        """异步获取统计信息"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.service.get_stats)


if __name__ == "__main__":
    # 测试代码
    import logging
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 模拟视频缓存服务
    cache_service = VideoCacheService(
        stream_url="test.mp4",
        cache_dir="./test_cache"
    )
    
    # 创建回放服务
    playback_service = VideoPlaybackService(
        video_cache_service=cache_service,
        output_dir="./test_output"
    )
    
    try:
        # 启动服务
        playback_service.start()
        
        # 模拟分析结果
        test_result = {
            "timestamp": time.time(),
            "detection_id": "test_001",
            "analysis_data": {
                "detections": [
                    {
                        "bbox": [100, 100, 200, 200],
                        "label": "person",
                        "confidence": 0.95
                    }
                ],
                "alert_info": {
                    "triggered": True,
                    "type": "intrusion"
                }
            }
        }
        
        # 提交测试
        playback_service.submit_analysis_result(test_result)
        
        # 等待处理
        time.sleep(5)
        
        # 获取统计
        stats = playback_service.get_stats()
        print(f"统计信息: {stats}")
        
    finally:
        playback_service.stop() 