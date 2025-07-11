#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视频缓存服务 - 独立进程版本
直接从拉流获取帧数据并缓存到磁盘，不依赖零拷贝架构
"""

import os
import time
import json
import cv2
import threading
import multiprocessing
import queue
import signal
import sys
from pathlib import Path
from collections import deque
from typing import Optional, List, Dict, Any
import logging
import numpy as np


class VideoCacheService:
    """独立的视频缓存服务
    
    功能：
    - 独立进程运行，直接从RTSP/视频源获取帧
    - 持续缓存固定时长的视频帧到磁盘  
    - 提供基于时间的帧查询接口
    - 自动清理过期缓存
    - 不依赖零拷贝内存架构
    """
    
    def __init__(
        self,
        stream_url: str,
        cache_dir: str = "./storage/video_cache",
        cache_duration: int = 60,   # 缓存时长（秒）
        target_fps: Optional[float] = None,   # 目标帧率，None表示使用原始帧率
        max_cache_size_gb: float = 5.0,  # 最大缓存大小（GB）
        target_resolution: tuple = (1920, 1080),  # 目标分辨率（width, height）- 默认1080P
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化视频缓存服务
        
        Args:
            stream_url: 视频流地址（RTSP/本地文件）
            cache_dir: 缓存目录
            cache_duration: 缓存时长（秒）
            target_fps: 目标帧率，None表示使用视频流原始帧率
            max_cache_size_gb: 最大缓存大小（GB）
            target_resolution: 目标分辨率（width, height）- 默认1080P
            logger: 日志记录器
        """
        self.stream_url = stream_url
        self.cache_duration = cache_duration
        self.target_fps = target_fps  # 可能为None，将在初始化视频捕获时设置为原始FPS
        self.max_cache_size_gb = max_cache_size_gb
        self.target_resolution = target_resolution  # 新增：目标分辨率
        
        # 视频流信息（将在初始化时设置）
        self.original_fps = None  # 原始视频流帧率
        
        # 【新增】动态帧率监测
        self.dynamic_fps_tracking = True  # 是否启用动态帧率跟踪
        self.fps_window_size = 30  # FPS计算窗口大小（帧数）
        self.recent_frame_times = deque(maxlen=self.fps_window_size)  # 最近帧的时间戳
        self.current_measured_fps = None  # 当前测量的实际FPS
        self.adaptive_frame_interval = None  # 自适应帧间隔
        
        # 【关键新增】帧丢失检测和补偿
        self.frame_gap_threshold = 2.0  # 帧间隔阈值（秒），超过此值认为有丢帧
        self.last_system_time = None  # 上一帧的系统时间
        self.system_time_baseline = None  # 系统时间基准
        self.enable_frame_compensation = True  # 是否启用帧补偿
        
        # 设置缓存目录
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 帧索引文件路径
        self.index_file = self.cache_dir / "frame_index.json"
        
        # 日志设置
        self.logger = logger or self._setup_logger()
        
        # 缓存状态
        self.is_running = False
        self.capture_thread = None
        self.cleanup_thread = None
        self.lock = threading.Lock()
        
        # 视频捕获
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=100)
        
        # 帧索引 - 内存中维护的元数据
        self.frame_index = deque()  # 存储 {"timestamp": float, "frame_id": int, "file_path": str}
        
        # 【新增】分析结果缓存 - 存储每帧的分析结果
        self.analysis_cache = deque()  # 存储 {"timestamp": float, "frame_id": int, "analysis_data": dict}
        self.analysis_index_file = self.cache_dir / "analysis_index.json"
        
        # 【新增】存储原始视频分辨率信息
        self.original_resolution = None
        
        # 统计信息
        self._stats = {
            "total_frames_captured": 0,
            "total_frames_cached": 0,
            "cache_size": 0,
            "cache_size_gb": 0.0,
            "start_time": 0,
            "last_frame_time": 0,
            "cleanup_count": 0,
            "fps": 0.0
        }
        
        # 加载现有索引
        self._load_frame_index()
        self._load_analysis_index()
        
        self.logger.info(f"📹 视频缓存服务初始化完成:")
        self.logger.info(f"   - 流地址: {stream_url}")
        self.logger.info(f"   - 缓存时长: {cache_duration}秒")
        self.logger.info(f"   - 目标帧率: {target_fps if target_fps else '自动检测'}fps")
        self.logger.info(f"   - 目标分辨率: {target_resolution[0]}x{target_resolution[1]}")
        self.logger.info(f"   - 缓存目录: {self.cache_dir}")
        self.logger.info(f"   - 最大缓存: {max_cache_size_gb}GB")
        self.logger.info(f"   - 现有帧数: {len(self.frame_index)}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"VideoCacheService")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 控制台输出
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(name)s: %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # 文件输出
            log_file = self.cache_dir / "video_cache.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def start(self):
        """启动视频缓存服务"""
        if self.is_running:
            self.logger.warning("⚠️ 视频缓存服务已经在运行")
            return False
        
        self.logger.info("🚀 启动视频缓存服务...")
        
        # 初始化视频捕获
        if not self._init_video_capture():
            return False
        
        # 启动状态
        self.is_running = True
        self._stats["start_time"] = time.time()
        
        # 启动捕获线程
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # 启动清理线程
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        self.logger.info("✅ 视频缓存服务启动成功")
        return True
    
    def stop(self):
        """停止视频缓存服务"""
        if not self.is_running:
            return
        
        self.logger.info("⏹️ 停止视频缓存服务...")
        
        self.is_running = False
        
        # 释放视频捕获
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 等待线程结束
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
        
        # 保存索引
        self._save_frame_index()
        
        self.logger.info("✅ 视频缓存服务已停止")
    
    def _init_video_capture(self) -> bool:
        """初始化视频捕获"""
        try:
            self.logger.info(f"📹 初始化视频捕获: {self.stream_url}")
            
            self.cap = cv2.VideoCapture(self.stream_url)
            
            if not self.cap.isOpened():
                self.logger.error(f"❌ 无法打开视频流: {self.stream_url}")
                return False
            
            # 设置缓冲区大小
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 获取原始视频信息
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # 存储原始分辨率和帧率
            self.original_resolution = (width, height)
            self.original_fps = fps if fps > 0 else 25.0  # 如果无法获取FPS，默认25fps
            
            # 如果没有指定目标帧率，使用原始帧率
            if self.target_fps is None:
                self.target_fps = self.original_fps
                self.logger.info(f"🎯 自动设置目标帧率为原始帧率: {self.target_fps:.1f}fps")
            
            self.logger.info(f"📹 视频流信息: {width}x{height}@{fps:.1f}fps (原始)")
            self.logger.info(f"🎯 缓存配置: {self.target_resolution[0]}x{self.target_resolution[1]}@{self.target_fps:.1f}fps")
            
            # 如果原始分辨率与目标分辨率不同，记录缩放信息
            if self.original_resolution != self.target_resolution:
                scale_x = self.target_resolution[0] / width
                scale_y = self.target_resolution[1] / height
                self.logger.info(f"🔧 分辨率缩放: {width}x{height} -> {self.target_resolution[0]}x{self.target_resolution[1]}, 缩放比例: {scale_x:.3f}x{scale_y:.3f}")
            else:
                self.logger.info(f"✅ 分辨率无需缩放: {width}x{height}")
            
            # 如果目标帧率与原始帧率不同，记录帧率调整信息
            if abs(self.target_fps - self.original_fps) > 0.1:
                fps_ratio = self.target_fps / self.original_fps
                self.logger.info(f"🔧 帧率调整: {self.original_fps:.1f}fps -> {self.target_fps:.1f}fps, 比例: {fps_ratio:.3f}")
            else:
                self.logger.info(f"✅ 帧率无需调整: {self.target_fps:.1f}fps")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 初始化视频捕获失败: {e}")
            return False
    
    def _capture_loop(self):
        """视频捕获循环 - 支持动态帧率适配"""
        self.logger.info("🔄 开始视频捕获循环（动态帧率适配模式）")
        
        frame_id = 0
        last_fps_time = time.time()
        fps_frame_count = 0
        last_cached_frame_time = 0  # 上一个缓存帧的时间
        
        # 初始帧间隔（如果有目标帧率）
        initial_interval = 1.0 / self.target_fps if self.target_fps and self.target_fps > 0 else 0
        self.adaptive_frame_interval = initial_interval
        
        self.logger.info(f"🎯 初始目标帧率: {self.target_fps}fps, 初始帧间隔: {initial_interval:.3f}s")
        self.logger.info(f"📊 动态帧率跟踪: {'启用' if self.dynamic_fps_tracking else '禁用'}")
        
        try:
            while self.is_running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.warning("⚠️ 无法读取视频帧，尝试重连...")
                    if not self._reconnect():
                        break
                    continue
                
                current_time = time.time()
                frame_id += 1
                
                # 【关键新增】动态帧率测量和适配
                self._update_dynamic_fps_measurement(current_time)
                
                # 智能帧率控制：是否应该缓存这一帧
                should_cache = self._should_cache_frame(current_time, last_cached_frame_time)
                
                if should_cache:
                    # 保存帧到磁盘
                    self._save_frame_to_disk(frame, frame_id, current_time)
                    last_cached_frame_time = current_time
                    
                    # 统计缓存的帧（用于FPS计算）
                    fps_frame_count += 1
                    
                    # 更新统计
                    self._stats["total_frames_captured"] += 1
                    self._stats["last_frame_time"] = current_time
                
                # 统计FPS（基于实际缓存的帧）
                if current_time - last_fps_time >= 1.0:
                    actual_fps = fps_frame_count / (current_time - last_fps_time)
                    self._stats["fps"] = actual_fps
                    fps_frame_count = 0
                    last_fps_time = current_time
                
                # 定期报告状态
                if frame_id % 100 == 0:
                    cache_count = len(self.frame_index)
                    actual_fps = self._stats.get("fps", 0)
                    measured_fps = self.current_measured_fps or 0
                    self.logger.info(f"📹 缓存状态: 读取帧={frame_id}, 缓存帧={cache_count}, "
                                   f"实际缓存FPS={actual_fps:.1f}, 视频流FPS={measured_fps:.1f}, "
                                   f"目标FPS={self.target_fps}, 自适应间隔={self.adaptive_frame_interval:.3f}s")
                
        except Exception as e:
            self.logger.error(f"❌ 视频捕获循环异常: {e}")
        finally:
            self.logger.info("🏁 视频捕获循环结束")
    
    def _update_dynamic_fps_measurement(self, current_time: float):
        """更新动态帧率测量"""
        if not self.dynamic_fps_tracking:
            return
        
        # 添加当前帧时间到窗口
        self.recent_frame_times.append(current_time)
        
        # 需要至少2帧才能计算FPS
        if len(self.recent_frame_times) < 2:
            return
        
        # 计算窗口内的平均FPS
        time_span = self.recent_frame_times[-1] - self.recent_frame_times[0]
        if time_span > 0:
            frame_count = len(self.recent_frame_times) - 1
            measured_fps = frame_count / time_span
            self.current_measured_fps = measured_fps
            
            # 【智能适配】根据测量的FPS调整缓存策略
            self._adapt_caching_strategy(measured_fps)
    
    def _adapt_caching_strategy(self, measured_fps: float):
        """根据测量的FPS智能调整缓存策略"""
        if not self.target_fps or self.target_fps <= 0:
            # 如果没有目标FPS，直接使用测量的FPS
            self.adaptive_frame_interval = 1.0 / measured_fps if measured_fps > 0 else 0.04  # 默认25fps
            return
        
        # 计算FPS偏差
        fps_ratio = measured_fps / self.target_fps if self.target_fps > 0 else 1.0
        
        if fps_ratio >= 1.2:
            # 视频流FPS比目标高20%以上，需要降采样
            self.adaptive_frame_interval = 1.0 / self.target_fps
        elif fps_ratio <= 0.8:
            # 视频流FPS比目标低20%以上，缓存所有帧
            self.adaptive_frame_interval = 1.0 / measured_fps if measured_fps > 0 else 0
        else:
            # FPS接近目标，使用测量的FPS
            self.adaptive_frame_interval = 1.0 / measured_fps if measured_fps > 0 else 1.0 / self.target_fps
    
    def _should_cache_frame(self, current_time: float, last_cached_time: float) -> bool:
        """智能判断是否应该缓存当前帧"""
        # 如果是第一帧，直接缓存
        if last_cached_time == 0:
            return True
        
        # 如果没有自适应间隔，缓存所有帧
        if not self.adaptive_frame_interval or self.adaptive_frame_interval <= 0:
            return True
        
        # 检查时间间隔
        elapsed = current_time - last_cached_time
        
        # 如果超过自适应间隔，缓存这一帧
        return elapsed >= self.adaptive_frame_interval
    
    def _save_frame_to_disk(self, frame: np.ndarray, frame_id: int, timestamp: float):
        """保存帧到磁盘（支持帧补偿）"""
        try:
            # 过滤异常时间戳
            if timestamp < 1000000000:  # 1970年后的合理时间戳
                self.logger.warning(f"⚠️ 跳过异常时间戳: {timestamp}")
                return
            
            # 【关键新增】帧丢失检测和补偿
            current_system_time = timestamp  # 使用传入的时间戳作为系统时间
            frame_gap_detected = False
            
            if self.enable_frame_compensation and self.last_system_time is not None:
                time_gap = current_system_time - self.last_system_time
                
                if time_gap > self.frame_gap_threshold:
                    frame_gap_detected = True
                    self.logger.warning(f"⚠️ 检测到帧丢失: 时间间隔={time_gap:.2f}s > 阈值={self.frame_gap_threshold}s")
                    
                    # 生成补偿帧填充时间空白
                    self._generate_compensation_frames(self.last_system_time, current_system_time, frame, frame_id)
            
            # 更新时间基准
            self.last_system_time = current_system_time
            if self.system_time_baseline is None:
                self.system_time_baseline = current_system_time
            
            # 【关键新增】调整帧分辨率到目标分辨率
            processed_frame = self._resize_frame_to_target_resolution(frame)
            
            # 生成文件路径
            timestamp_str = f"{timestamp:.3f}".replace('.', '_')
            filename = f"frame_{frame_id}_{timestamp_str}.jpg"
            file_path = self.cache_dir / filename
            
            # 保存图像（JPG格式，质量85%）
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            success = cv2.imwrite(str(file_path), processed_frame, encode_params)
            
            if success:
                # 添加到索引
                with self.lock:
                    frame_info = {
                        "timestamp": timestamp,
                        "frame_id": frame_id,
                        "file_path": str(file_path),
                        "cached_at": time.time(),
                        "original_resolution": self.original_resolution,  # 新增：存储原始分辨率
                        "cached_resolution": self.target_resolution      # 新增：存储缓存分辨率
                    }
                    
                    self.frame_index.append(frame_info)
                    self._stats["total_frames_cached"] += 1
                    self._stats["cache_size"] = len(self.frame_index)
            else:
                self.logger.error(f"❌ 保存帧失败: {file_path}")
                
        except Exception as e:
            self.logger.error(f"❌ 保存帧到磁盘失败: {e}")
    
    def _resize_frame_to_target_resolution(self, frame: np.ndarray) -> np.ndarray:
        """
        将帧调整到目标分辨率
        
        Args:
            frame: 原始帧
            
        Returns:
            np.ndarray: 调整后的帧
        """
        try:
            current_height, current_width = frame.shape[:2]
            target_width, target_height = self.target_resolution
            
            # 如果已经是目标分辨率，直接返回
            if current_width == target_width and current_height == target_height:
                return frame
            
            # 使用高质量插值进行缩放
            resized_frame = cv2.resize(
                frame, 
                (target_width, target_height), 
                interpolation=cv2.INTER_LANCZOS4  # 使用LANCZOS4高质量插值
            )
            
            self.logger.debug(f"🔧 帧分辨率调整: {current_width}x{current_height} -> {target_width}x{target_height}")
            
            return resized_frame
            
        except Exception as e:
            self.logger.error(f"❌ 调整帧分辨率失败: {e}")
            return frame  # 返回原始帧作为回退
    
    def get_current_resolution(self) -> Optional[tuple]:
        """
        获取当前使用的分辨率
        
        Returns:
            tuple: (width, height) 当前分辨率，如果无法获取则返回None
        """
        # 优先返回目标分辨率（即1080P）
        if self.target_resolution:
            return self.target_resolution
        
        # 如果没有目标分辨率，返回原始分辨率
        if self.original_resolution:
            return self.original_resolution
        
        return None
    
    def get_original_resolution(self) -> Optional[tuple]:
        """
        获取原始视频分辨率
        
        Returns:
            tuple: (width, height) 原始分辨率，如果无法获取则返回None
        """
        return self.original_resolution
    
    def get_target_resolution(self) -> Optional[tuple]:
        """
        获取目标分辨率
        
        Returns:
            tuple: (width, height) 目标分辨率，如果无法获取则返回None
        """
        return self.target_resolution
    
    def _generate_compensation_frames(self, start_time: float, end_time: float, reference_frame: np.ndarray, base_frame_id: int):
        """
        生成补偿帧填充时间空白
        
        Args:
            start_time: 开始时间（上一帧时间）
            end_time: 结束时间（当前帧时间）
            reference_frame: 参考帧（用于复制）
            base_frame_id: 基准帧ID
        """
        try:
            time_gap = end_time - start_time
            
            # 根据期望FPS计算需要补偿的帧数
            expected_fps = self.current_measured_fps or self.target_fps or 25.0
            expected_frames = int(time_gap * expected_fps)
            
            # 限制补偿帧数（避免过多补偿帧）
            max_compensation_frames = 50  # 最多补偿50帧
            compensation_frames = min(expected_frames, max_compensation_frames)
            
            if compensation_frames <= 1:
                return  # 不需要补偿
            
            self.logger.info(f"🔧 生成补偿帧: 时间间隔={time_gap:.2f}s, 期望帧数={expected_frames}, 实际补偿={compensation_frames}帧")
            
            # 生成均匀分布的时间点
            time_interval = time_gap / (compensation_frames + 1)
            
            for i in range(compensation_frames):
                compensation_time = start_time + (i + 1) * time_interval
                compensation_frame_id = base_frame_id * 1000 + i  # 使用特殊ID标识补偿帧
                
                # 【关键】直接保存补偿帧（复用参考帧）
                self._save_compensation_frame_to_disk(reference_frame, compensation_frame_id, compensation_time, is_compensation=True)
                
        except Exception as e:
            self.logger.error(f"❌ 生成补偿帧失败: {e}")
    
    def _save_compensation_frame_to_disk(self, frame: np.ndarray, frame_id: int, timestamp: float, is_compensation: bool = False):
        """保存补偿帧到磁盘"""
        try:
            # 调整帧分辨率到目标分辨率
            processed_frame = self._resize_frame_to_target_resolution(frame)
            
            # 生成文件路径（补偿帧使用特殊前缀）
            timestamp_str = f"{timestamp:.3f}".replace('.', '_')
            prefix = "comp_" if is_compensation else "frame_"
            filename = f"{prefix}{frame_id}_{timestamp_str}.jpg"
            file_path = self.cache_dir / filename
            
            # 保存图像
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            success = cv2.imwrite(str(file_path), processed_frame, encode_params)
            
            if success:
                # 添加到索引
                with self.lock:
                    frame_info = {
                        "timestamp": timestamp,
                        "frame_id": frame_id,
                        "file_path": str(file_path),
                        "cached_at": time.time(),
                        "original_resolution": self.original_resolution,
                        "cached_resolution": self.target_resolution,
                        "is_compensation": is_compensation  # 标记为补偿帧
                    }
                    
                    self.frame_index.append(frame_info)
                    self._stats["total_frames_cached"] += 1
                    self._stats["compensation_frames"] = self._stats.get("compensation_frames", 0) + (1 if is_compensation else 0)
                    self._stats["cache_size"] = len(self.frame_index)
                    
                if is_compensation:
                    self.logger.debug(f"📋 补偿帧已缓存: {filename}, 时间={timestamp:.3f}")
            else:
                self.logger.error(f"❌ 保存补偿帧失败: {file_path}")
                
        except Exception as e:
            self.logger.error(f"❌ 保存补偿帧到磁盘失败: {e}")
    
    def _cleanup_loop(self):
        """清理循环 - 定期清理过期文件"""
        self.logger.info("🧹 开始清理循环")
        
        try:
            while self.is_running:
                time.sleep(30)  # 每30秒清理一次
                
                if not self.is_running:
                    break
                
                with self.lock:
                    self._cleanup_expired_frames()
                    self._cleanup_by_size()
                
                # 定期保存索引
                if self._stats["cleanup_count"] % 10 == 0:  # 每10次清理保存一次索引
                    self._save_frame_index()
                
        except Exception as e:
            self.logger.error(f"❌ 清理循环异常: {e}")
        finally:
            self.logger.info("🏁 清理循环结束")
    
    def _cleanup_expired_frames(self):
        """清理过期的帧文件"""
        current_time = time.time()
        cutoff_time = current_time - self.cache_duration
        
        # 清理过期的帧
        while self.frame_index and self.frame_index[0]["timestamp"] < cutoff_time:
            expired_frame = self.frame_index.popleft()
            file_path = expired_frame["file_path"]
            
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self._stats["cleanup_count"] += 1
            except Exception as e:
                self.logger.warning(f"⚠️ 删除过期帧文件失败 {file_path}: {e}")
    
    def _cleanup_by_size(self):
        """按大小清理缓存"""
        cache_size_gb = self._calculate_cache_size_gb()
        if cache_size_gb > self.max_cache_size_gb:
            target_size_gb = self.max_cache_size_gb * 0.8  # 清理到80%
            
            while self.frame_index:
                cache_size_gb = self._calculate_cache_size_gb()
                if cache_size_gb <= target_size_gb:
                    break
                
                # 删除最老的帧
                oldest_frame = self.frame_index.popleft()
                file_path = oldest_frame["file_path"]
                
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        self._stats["cleanup_count"] += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ 删除帧文件失败 {file_path}: {e}")
    
    def _calculate_cache_size_gb(self) -> float:
        """计算缓存大小（GB）"""
        total_size = 0
        for frame_info in self.frame_index:
            file_path = frame_info["file_path"]
            try:
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
            except:
                continue
        
        return total_size / (1024 * 1024 * 1024)  # 转换为GB
    
    def add_analysis_result(self, frame_id: int, timestamp: float, analysis_data: Dict[str, Any]):
        """
        添加帧的分析结果到缓存
        
        Args:
            frame_id: 帧ID
            timestamp: 时间戳
            analysis_data: 分析结果数据
        """
        try:
            # 过滤异常时间戳
            if timestamp < 1000000000:  # 1970年后的合理时间戳
                self.logger.warning(f"⚠️ 跳过异常时间戳的分析结果: {timestamp}")
                return
            
            with self.lock:
                analysis_info = {
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "analysis_data": analysis_data,
                    "cached_at": time.time()
                }
                
                self.analysis_cache.append(analysis_info)
                
                # 清理过期的分析结果（超过缓存时长）
                current_time = time.time()
                cutoff_time = current_time - self.cache_duration
                
                while self.analysis_cache and self.analysis_cache[0]["timestamp"] < cutoff_time:
                    self.analysis_cache.popleft()
                
                self.logger.debug(f"📊 添加分析结果: frame_id={frame_id}, timestamp={timestamp:.3f}, 缓存数量={len(self.analysis_cache)}")
                
        except Exception as e:
            self.logger.error(f"❌ 添加分析结果失败: {e}")
    
    def get_analysis_results_for_time_range(self, center_timestamp: float, duration: float) -> List[Dict[str, Any]]:
        """
        获取指定时间范围的分析结果
        
        Args:
            center_timestamp: 中心时间戳
            duration: 总时长（秒）
            
        Returns:
            List[Dict]: 分析结果列表，按时间戳排序
        """
        half_duration = duration / 2.0
        start_time = center_timestamp - half_duration
        end_time = center_timestamp + half_duration
        
        with self.lock:
            matching_results = []
            
            for analysis_info in self.analysis_cache:
                analysis_timestamp = analysis_info["timestamp"]
                # 过滤异常时间戳并检查时间范围
                if analysis_timestamp > 1000000000 and start_time <= analysis_timestamp <= end_time:
                    matching_results.append(analysis_info)
            
            # 按时间戳排序
            matching_results.sort(key=lambda x: x["timestamp"])
            
            self.logger.info(f"📊 分析结果查询: 中心时间={center_timestamp:.3f}, "
                           f"查询范围=[{start_time:.3f}, {end_time:.3f}], "
                           f"时长={duration}s, 获取结果={len(matching_results)}个")
            
            return matching_results
    
    def _reconnect(self) -> bool:
        """重连视频流"""
        self.logger.info("🔄 尝试重连视频流...")
        
        if self.cap:
            self.cap.release()
        
        time.sleep(2)  # 等待2秒再重连
        
        return self._init_video_capture()
    
    def get_frames_for_time_range(self, center_timestamp: float, duration: float) -> List[np.ndarray]:
        """
        获取指定时间范围的帧数据
        
        Args:
            center_timestamp: 中心时间戳
            duration: 总时长（秒）
            
        Returns:
            List[np.ndarray]: 帧数据列表
        """
        half_duration = duration / 2.0
        start_time = center_timestamp - half_duration
        end_time = center_timestamp + half_duration
        
        with self.lock:
            matching_frames = []
            
            # 查找匹配的帧文件
            matching_files = []
            for frame_info in self.frame_index:
                frame_timestamp = frame_info["timestamp"]
                # 过滤异常时间戳并检查时间范围
                if frame_timestamp > 1000000000 and start_time <= frame_timestamp <= end_time:
                    matching_files.append(frame_info)
            
            # 按时间戳排序
            matching_files.sort(key=lambda x: x["timestamp"])
            
            # 加载图像数据
            for frame_info in matching_files:
                try:
                    file_path = frame_info["file_path"]
                    if os.path.exists(file_path):
                        frame = cv2.imread(file_path)
                        if frame is not None:
                            matching_frames.append(frame)
                except Exception as e:
                    self.logger.error(f"❌ 加载帧文件失败 {frame_info['file_path']}: {e}")
                    continue
            
            # 记录查询结果
            expected_frames = int(duration * self.target_fps)
            frame_coverage = len(matching_frames) / expected_frames * 100 if expected_frames > 0 else 0
            
            self.logger.info(f"📹 帧查询结果: 中心时间={center_timestamp:.3f}, "
                           f"查询范围=[{start_time:.3f}, {end_time:.3f}], "
                           f"时长={duration}s, 获取帧数={len(matching_frames)}/{expected_frames} ({frame_coverage:.1f}%)")
            
            # 如果帧覆盖率过低，记录警告
            if frame_coverage < 80.0:
                self.logger.warning(f"⚠️ 帧覆盖率过低: {frame_coverage:.1f}% < 80%, "
                                  f"这可能导致回放视频时长不足")
            
            return matching_frames
    
    def get_frames_with_metadata_for_time_range(self, center_timestamp: float, duration: float) -> List[Dict[str, Any]]:
        """
        获取指定时间范围内的帧数据和元数据
        
        Args:
            center_timestamp: 中心时间戳
            duration: 时间范围（总时长）
            
        Returns:
            List[Dict]: 包含帧数据和元数据的列表 [{"frame": np.ndarray, "timestamp": float, "frame_id": int}]
        """
        with self.lock:
            half_duration = duration / 2
            start_time = center_timestamp - half_duration
            end_time = center_timestamp + half_duration
            
            matching_frames = []
            
            for frame_info in self.frame_index:
                frame_timestamp = frame_info["timestamp"]
                
                # 过滤异常时间戳
                if frame_timestamp < 1000000000:
                    continue
                
                if start_time <= frame_timestamp <= end_time:
                    try:
                        # 加载帧数据
                        frame_path = frame_info["file_path"]
                        frame = cv2.imread(frame_path)
                        
                        if frame is not None:
                            matching_frames.append({
                                "frame": frame,
                                "timestamp": frame_timestamp,
                                "frame_id": frame_info["frame_id"],
                                "file_path": frame_path
                            })
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ 加载帧失败: {frame_path}, 错误: {e}")
                        continue
            
            # 按时间戳排序
            matching_frames.sort(key=lambda x: x["timestamp"])
            
            # 记录查询结果
            expected_frames = int(duration * self.target_fps)
            frame_coverage = len(matching_frames) / expected_frames * 100 if expected_frames > 0 else 0
            
            self.logger.info(f"📹 帧查询结果(带元数据): 中心时间={center_timestamp:.3f}, "
                           f"查询范围=[{start_time:.3f}, {end_time:.3f}], "
                           f"时长={duration}s, 获取帧数={len(matching_frames)}/{expected_frames} ({frame_coverage:.1f}%)")
            
            # 如果帧覆盖率过低，记录警告
            if frame_coverage < 80:
                self.logger.warning(f"⚠️ 帧覆盖率过低: {frame_coverage:.1f}% < 80%，可能影响视频质量")
            
            return matching_frames
    
    def _load_frame_index(self):
        """加载帧索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    index_data = json.load(f)
                    
                # 验证文件是否存在，清理无效条目
                valid_frames = []
                for frame_info in index_data:
                    if os.path.exists(frame_info["file_path"]):
                        valid_frames.append(frame_info)
                
                self.frame_index = deque(valid_frames)
                self.logger.info(f"📂 加载帧索引: {len(valid_frames)} 个有效帧")
                
            except Exception as e:
                self.logger.error(f"❌ 加载帧索引失败: {e}")
                self.frame_index = deque()
    
    def _load_analysis_index(self):
        """加载分析结果索引"""
        try:
            if self.analysis_index_file.exists():
                with open(self.analysis_index_file, 'r') as f:
                    analysis_data = json.load(f)
                
                # 过滤过期的分析结果
                current_time = time.time()
                cutoff_time = current_time - self.cache_duration
                
                valid_analysis = [
                    item for item in analysis_data 
                    if item.get("timestamp", 0) > cutoff_time
                ]
                
                self.analysis_cache = deque(valid_analysis)
                self.logger.info(f"📊 加载分析结果索引: {len(valid_analysis)} 个有效结果")
            else:
                self.analysis_cache = deque()
                self.logger.info("📊 分析结果索引文件不存在，创建新索引")
                
        except Exception as e:
            self.logger.error(f"❌ 加载分析结果索引失败: {e}")
            self.analysis_cache = deque()

    def _save_frame_index(self):
        """保存帧索引"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(list(self.frame_index), f, indent=2)
            self.logger.debug(f"💾 保存帧索引: {len(self.frame_index)} 个帧")
        except Exception as e:
            self.logger.error(f"❌ 保存帧索引失败: {e}")
    
    def _save_analysis_index(self):
        """保存分析结果索引"""
        try:
            with open(self.analysis_index_file, 'w') as f:
                json.dump(list(self.analysis_cache), f, indent=2)
            self.logger.debug(f"💾 保存分析结果索引: {len(self.analysis_cache)} 个结果")
        except Exception as e:
            self.logger.error(f"❌ 保存分析结果索引失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            current_time = time.time()
            start_time = self._stats.get("start_time", current_time)
            runtime = current_time - start_time
            
            cache_size_gb = self._calculate_cache_size_gb()
            
            stats = self._stats.copy()
            stats.update({
                "runtime_seconds": runtime,
                "current_cache_size": len(self.frame_index),
                "cache_size_gb": cache_size_gb,
                "cache_full": cache_size_gb >= self.max_cache_size_gb,
                "cache_time_range": self._get_cache_time_range(),
                "is_running": self.is_running,
                "stream_url": self.stream_url,
                "cache_dir": str(self.cache_dir),
                # 【新增】动态FPS统计
                "original_fps": self.original_fps,
                "target_fps": self.target_fps,
                "current_measured_fps": self.current_measured_fps,
                "adaptive_frame_interval": self.adaptive_frame_interval,
                "dynamic_fps_tracking": self.dynamic_fps_tracking,
                "fps_window_size": self.fps_window_size,
                # 【新增】帧补偿统计
                "frame_compensation_enabled": self.enable_frame_compensation,
                "frame_gap_threshold": self.frame_gap_threshold,
                "last_system_time": self.last_system_time
            })
            
            return stats
    
    def _get_cache_time_range(self) -> tuple:
        """获取缓存的时间范围"""
        if not self.frame_index:
            return (0.0, 0.0)
        
        timestamps = [item["timestamp"] for item in self.frame_index]
        if not timestamps:
            return (0.0, 0.0)
        
        return (min(timestamps), max(timestamps))


def run_video_cache_service(stream_url: str, cache_dir: str, cache_duration: int = 60):
    """运行视频缓存服务的独立进程函数"""
    
    # 设置信号处理器
    def signal_handler(signum, frame):
        print(f"收到信号 {signum}，正在停止视频缓存服务...")
        if 'service' in locals():
            service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建并启动服务
    service = VideoCacheService(
        stream_url=stream_url,
        cache_dir=cache_dir,
        cache_duration=cache_duration
    )
    
    if service.start():
        print(f"✅ 视频缓存服务已启动，缓存目录: {cache_dir}")
        
        try:
            # 主循环 - 保持进程运行并定期报告状态
            while service.is_running:
                time.sleep(60)  # 每分钟报告一次状态
                stats = service.get_stats()
                print(f"📊 缓存状态: 帧数={stats['current_cache_size']}, "
                      f"大小={stats['cache_size_gb']:.2f}GB, "
                      f"FPS={stats['fps']:.1f}")
        except KeyboardInterrupt:
            print("收到中断信号，正在停止...")
        finally:
            service.stop()
    else:
        print("❌ 视频缓存服务启动失败")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='视频缓存服务')
    parser.add_argument('--stream-url', required=True, help='视频流地址')
    parser.add_argument('--cache-dir', default='./storage/video_cache', help='缓存目录')
    parser.add_argument('--cache-duration', type=int, default=60, help='缓存时长（秒）')
    
    args = parser.parse_args()
    
    run_video_cache_service(
        stream_url=args.stream_url,
        cache_dir=args.cache_dir,
        cache_duration=args.cache_duration
    ) 