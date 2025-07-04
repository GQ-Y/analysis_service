#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: stream_capture.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 流捕获器

基于timelinetool架构实现的流捕获器，负责拉流并将帧存储到内存池。

本文件是分析服务项目的一部分。
"""

import cv2
import time
import threading
import logging
import os
from typing import Optional, Callable
from queue import Queue, Full

from .memory_pool import MemoryPool
from .time_axis import TimeAxis


class StreamCapture:
    """
    流捕获器
    
    负责从RTSP/HTTP流或本地文件捕获视频帧，
    使用独立线程进行I/O操作，避免阻塞主流程。
    """
    
    def __init__(
        self,
        stream_url: str,
        memory_pool: MemoryPool,
        time_axis: TimeAxis,
        stream_id: str = "",
        reconnect_delay: float = 5.0,
        display_queue: Optional[Queue] = None,
        video_player: Optional['VideoPlayer'] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化流捕获器
        
        Args:
            stream_url: 流URL或文件路径
            memory_pool: 内存池
            time_axis: 时间轴
            stream_id: 流ID
            reconnect_delay: 重连延迟（秒）
            display_queue: 显示队列（可选）
            logger: 日志记录器
        """
        self.stream_url = stream_url
        self.memory_pool = memory_pool
        self.time_axis = time_axis
        self.stream_id = stream_id or f"stream_{id(self)}"
        self.reconnect_delay = reconnect_delay
        self.display_queue = display_queue
        self.video_player = video_player
        self.logger = logger or logging.getLogger(__name__)
        
        # 捕获状态
        self.running = False
        self.connected = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        
        # 统计信息
        self._stats = {
            "total_frames": 0,
            "successful_frames": 0,
            "dropped_frames": 0,
            "reconnect_count": 0,
            "start_time": 0,
            "last_frame_time": 0,
            "fps": 0.0
        }
        
        # 配置OpenCV
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        
        self.logger.info(f"📹 流捕获器初始化: {stream_url} -> {self.stream_id}")
    
    def start(self) -> bool:
        """
        启动捕获线程
        
        Returns:
            bool: 是否成功启动
        """
        if self.running:
            self.logger.warning(f"⚠️ 流捕获器 {self.stream_id} 已在运行")
            return True
        
        self.running = True
        self._stats["start_time"] = time.time()
        
        # 创建并启动捕获线程
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        self.logger.info(f"🚀 流捕获器 {self.stream_id} 已启动")
        return True
    
    def stop(self):
        """停止捕获"""
        if not self.running:
            return
        
        self.running = False
        
        # 等待线程结束
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        
        # 释放资源
        self._disconnect()
        
        self.logger.info(f"⏹️ 流捕获器 {self.stream_id} 已停止")
    
    def _connect(self) -> bool:
        """
        连接到流
        
        Returns:
            bool: 是否连接成功
        """
        try:
            self.logger.info(f"🔗 连接流: {self.stream_url}")
            
            # 创建VideoCapture
            self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
            
            if not self.cap.isOpened():
                self.logger.error(f"❌ 无法打开流: {self.stream_url}")
                self.cap = None
                return False
            
            # 设置分辨率（如果支持）
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            # 获取实际分辨率
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.connected = True
            self.logger.info(f"✅ 流连接成功: {width}x{height}@{fps:.1f}fps")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 连接流失败: {e}")
            self.cap = None
            return False
    
    def _disconnect(self):
        """断开连接"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
    
    def _capture_loop(self):
        """捕获循环（在独立线程中运行）"""
        frame_id = 0
        last_fps_time = time.time()
        fps_frame_count = 0
        
        while self.running:
            # 检查连接状态
            if not self.connected or not self.cap or not self.cap.isOpened():
                self.logger.warning(f"⚠️ 流 {self.stream_id} 连接丢失，尝试重连...")
                self._reconnect()
                continue
            
            try:
                # 读取帧
                start_time = time.time()
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.warning(f"⚠️ 流 {self.stream_id} 读取帧失败")
                    self._reconnect()
                    continue
                
                # 更新统计
                frame_id += 1
                self._stats["total_frames"] += 1
                self._stats["last_frame_time"] = start_time
                
                # 计算FPS
                fps_frame_count += 1
                if start_time - last_fps_time >= 1.0:
                    self._stats["fps"] = fps_frame_count / (start_time - last_fps_time)
                    fps_frame_count = 0
                    last_fps_time = start_time
                
                # 存储到内存池
                buffer = self.memory_pool.put_frame(
                    frame, frame_id, start_time, self.stream_id
                )
                
                if buffer:
                    # 添加到时间轴
                    if self.time_axis.add_frame(buffer):
                        self._stats["successful_frames"] += 1
                    else:
                        self._stats["dropped_frames"] += 1
                        buffer.release()  # 时间轴添加失败，释放引用
                    
                    # 添加到显示队列（如果存在）
                    if self.display_queue:
                        try:
                            # 创建显示用的帧副本
                            display_frame = frame.copy()
                            self.display_queue.put((display_frame, frame_id), block=False)
                        except Full:
                            pass  # 显示队列满了，跳过

                    # 添加到视频播放器（如果存在）
                    if self.video_player:
                        try:
                            # 发送帧到播放器
                            self.video_player.add_frame(frame, frame_id)
                        except Exception as e:
                            self.logger.debug(f"⚠️ 发送帧到播放器失败: {e}")
                else:
                    self._stats["dropped_frames"] += 1
                
                # 记录处理时间
                process_time = (time.time() - start_time) * 1000
                self.logger.debug(f"📥 流 {self.stream_id} 帧 #{frame_id}: "
                                f"尺寸={frame.shape[1]}x{frame.shape[0]}, "
                                f"处理时间={process_time:.2f}ms")
                
            except Exception as e:
                self.logger.error(f"❌ 流 {self.stream_id} 捕获异常: {e}")
                self._reconnect()
        
        # 清理
        self._disconnect()
        self.logger.info(f"🏁 流 {self.stream_id} 捕获循环结束")
    
    def _reconnect(self):
        """重连流"""
        self._disconnect()
        self._stats["reconnect_count"] += 1
        
        self.logger.info(f"🔄 流 {self.stream_id} 等待 {self.reconnect_delay}s 后重连...")
        time.sleep(self.reconnect_delay)
        
        if self.running:
            self._connect()
    
    def get_stats(self) -> dict:
        """
        获取捕获统计信息
        
        Returns:
            dict: 统计信息
        """
        current_time = time.time()
        runtime = current_time - self._stats["start_time"] if self._stats["start_time"] > 0 else 0
        
        stats = self._stats.copy()
        stats.update({
            "stream_id": self.stream_id,
            "stream_url": self.stream_url,
            "connected": self.connected,
            "running": self.running,
            "runtime_seconds": runtime,
            "avg_fps": self._stats["total_frames"] / max(1, runtime),
            "success_rate": (self._stats["successful_frames"] / max(1, self._stats["total_frames"])) * 100
        })
        
        return stats
    
    def get_frame_dimensions(self) -> tuple:
        """
        获取帧尺寸
        
        Returns:
            tuple: (width, height) 或 (None, None)
        """
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return None, None
    
    def __str__(self) -> str:
        stats = self.get_stats()
        return (f"StreamCapture(id={self.stream_id}, "
                f"connected={self.connected}, "
                f"fps={stats['fps']:.1f}, "
                f"frames={stats['total_frames']})")
    
    def __repr__(self) -> str:
        return self.__str__()


class MultiStreamCapture:
    """多流捕获管理器"""
    
    def __init__(
        self,
        memory_pool: MemoryPool,
        time_axis: TimeAxis,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化多流捕获管理器
        
        Args:
            memory_pool: 内存池
            time_axis: 时间轴
            logger: 日志记录器
        """
        self.memory_pool = memory_pool
        self.time_axis = time_axis
        self.logger = logger or logging.getLogger(__name__)
        
        self.captures = {}
        self.lock = threading.Lock()
        
        self.logger.info("🌊 多流捕获管理器初始化")
    
    def add_stream(
        self,
        stream_id: str,
        stream_url: str,
        reconnect_delay: float = 5.0,
        display_queue: Optional[Queue] = None
    ) -> StreamCapture:
        """
        添加流捕获器
        
        Args:
            stream_id: 流ID
            stream_url: 流URL
            reconnect_delay: 重连延迟
            display_queue: 显示队列
            
        Returns:
            StreamCapture: 流捕获器实例
        """
        with self.lock:
            if stream_id in self.captures:
                self.logger.warning(f"⚠️ 流 {stream_id} 已存在，将被替换")
                self.remove_stream(stream_id)
            
            capture = StreamCapture(
                stream_url=stream_url,
                memory_pool=self.memory_pool,
                time_axis=self.time_axis,
                stream_id=stream_id,
                reconnect_delay=reconnect_delay,
                display_queue=display_queue,
                logger=self.logger
            )
            
            self.captures[stream_id] = capture
            self.logger.info(f"➕ 添加流捕获器: {stream_id}")
            
            return capture
    
    def remove_stream(self, stream_id: str) -> bool:
        """移除流捕获器"""
        with self.lock:
            if stream_id in self.captures:
                capture = self.captures.pop(stream_id)
                capture.stop()
                self.logger.info(f"➖ 移除流捕获器: {stream_id}")
                return True
            return False
    
    def start_all(self):
        """启动所有流捕获器"""
        with self.lock:
            for stream_id, capture in self.captures.items():
                capture.start()
                self.logger.info(f"🚀 启动流捕获器: {stream_id}")
    
    def stop_all(self):
        """停止所有流捕获器"""
        with self.lock:
            for stream_id, capture in self.captures.items():
                capture.stop()
                self.logger.info(f"⏹️ 停止流捕获器: {stream_id}")
    
    def get_all_stats(self) -> dict:
        """获取所有流的统计信息"""
        stats = {}
        with self.lock:
            for stream_id, capture in self.captures.items():
                stats[stream_id] = capture.get_stats()
        return stats
