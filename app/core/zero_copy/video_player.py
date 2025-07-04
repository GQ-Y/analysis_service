#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: video_player.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 实时视频播放器

基于timelinetool架构实现的实时视频播放器，支持显示原始帧和分析结果。

本文件是分析服务项目的一部分。
"""

import cv2
import time
import threading
import logging
import numpy as np
import os
from typing import Optional, Dict, Any, Callable
from queue import Queue, Empty
from collections import deque
from sortedcontainers import SortedDict

from .frame_buffer import FrameBuffer


class VideoPlayer:
    """
    实时视频播放器
    
    基于timelinetool的设计，支持：
    - 实时显示原始视频帧
    - 叠加AI分析结果
    - 多窗口显示
    - 性能监控
    """
    
    def __init__(
        self,
        window_name: str = "实时AI分析",
        display_queue_size: int = 5,
        results_cache_size: int = 100,
        show_fps: bool = True,
        show_info: bool = True,
        headless: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化视频播放器
        
        Args:
            window_name: 窗口名称
            display_queue_size: 显示队列大小
            results_cache_size: 结果缓存大小
            show_fps: 是否显示FPS
            show_info: 是否显示信息
            logger: 日志记录器
        """
        self.window_name = window_name
        self.show_fps = show_fps
        self.show_info = show_info
        self.logger = logger or logging.getLogger(__name__)

        # 检测是否为无头模式
        self.headless = headless or self._detect_headless_mode()
        
        # 显示队列（原始帧）
        self.display_queue = Queue(maxsize=display_queue_size)
        
        # 分析结果缓存（按帧ID排序）
        self.results_cache = SortedDict()
        self.results_cache_size = results_cache_size
        self.cache_lock = threading.Lock()
        
        # 播放状态
        self.running = False
        self.paused = False
        self.thread: Optional[threading.Thread] = None
        
        # 统计信息
        self._stats = {
            "frames_displayed": 0,
            "frames_dropped": 0,
            "start_time": 0,
            "last_fps_time": 0,
            "fps_frame_count": 0,
            "current_fps": 0.0
        }
        
        # 显示配置
        self.colors = {
            "person": (0, 255, 0),      # 绿色
            "vehicle": (255, 0, 0),     # 蓝色
            "car": (255, 0, 0),         # 蓝色
            "face": (0, 128, 255),      # 橙色
            "default": (128, 128, 128)  # 灰色
        }
        
        if self.headless:
            self.logger.info(f"🎬 视频播放器初始化: 无头模式 - {window_name}")
        else:
            self.logger.info(f"🎬 视频播放器初始化: 窗口={window_name}")

    def _detect_headless_mode(self) -> bool:
        """检测是否为无头模式"""
        # 检查DISPLAY环境变量
        if not os.environ.get('DISPLAY'):
            return True

        # 检查SSH连接
        if os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_CLIENT'):
            return True

        # 尝试创建测试窗口
        try:
            cv2.namedWindow("test_window", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("test_window")
            return False
        except:
            return True
    
    def start(self):
        """启动播放器"""
        if self.running:
            self.logger.warning("⚠️ 视频播放器已在运行")
            return
        
        self.running = True
        self.paused = False
        self._stats["start_time"] = time.time()
        self._stats["last_fps_time"] = time.time()
        
        # 创建OpenCV窗口（非无头模式）
        if not self.headless:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 1280, 720)
            except Exception as e:
                self.logger.warning(f"⚠️ 无法创建窗口，切换到无头模式: {e}")
                self.headless = True
        
        # 创建并启动播放线程
        self.thread = threading.Thread(target=self._play_loop, daemon=True)
        self.thread.start()
        
        self.logger.info(f"🚀 视频播放器已启动: {self.window_name}")
    
    def stop(self):
        """停止播放器"""
        if not self.running:
            return
        
        self.running = False
        
        # 等待线程结束
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        
        # 销毁窗口（非无头模式）
        if not self.headless:
            try:
                cv2.destroyWindow(self.window_name)
            except:
                pass
        
        self.logger.info(f"⏹️ 视频播放器已停止: {self.window_name}")
    
    def add_frame(self, frame: np.ndarray, frame_id: int):
        """
        添加帧到显示队列
        
        Args:
            frame: 视频帧
            frame_id: 帧ID
        """
        try:
            # 非阻塞添加，如果队列满了就丢弃旧帧
            while not self.display_queue.empty():
                try:
                    self.display_queue.get_nowait()
                    self._stats["frames_dropped"] += 1
                except Empty:
                    break
            
            self.display_queue.put((frame.copy(), frame_id), block=False)
            
        except Exception as e:
            self._stats["frames_dropped"] += 1
            self.logger.debug(f"⚠️ 添加显示帧失败: {e}")
    
    def add_analysis_result(self, frame_id: int, results: Dict[str, Any]):
        """
        添加分析结果到缓存
        
        Args:
            frame_id: 帧ID
            results: 分析结果
        """
        with self.cache_lock:
            self.results_cache[frame_id] = results
            
            # 限制缓存大小
            while len(self.results_cache) > self.results_cache_size:
                self.results_cache.popitem(0)  # 移除最旧的结果
    
    def toggle_pause(self):
        """切换暂停状态"""
        self.paused = not self.paused
        status = "暂停" if self.paused else "播放"
        self.logger.info(f"🎬 播放器状态: {status}")
    
    def _play_loop(self):
        """播放循环（在独立线程中运行）"""
        self.logger.info(f"🔄 播放循环开始: {self.window_name}")
        
        latest_frame_tuple = None
        latest_results = None
        
        while self.running:
            try:
                # 检查暂停状态
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # 获取最新帧（清空队列，只保留最新的）
                while not self.display_queue.empty():
                    try:
                        latest_frame_tuple = self.display_queue.get_nowait()
                    except Empty:
                        break
                
                if latest_frame_tuple is not None:
                    frame, frame_id = latest_frame_tuple
                    
                    # 获取对应的分析结果
                    with self.cache_lock:
                        # 查找最接近的分析结果
                        pos = self.results_cache.bisect_right(frame_id)
                        if pos > 0:
                            _, latest_results = self.results_cache.peekitem(pos - 1)
                    
                    # 绘制分析结果
                    if latest_results:
                        frame = self._draw_analysis_results(frame, latest_results)
                    
                    # 绘制信息叠加
                    if self.show_info or self.show_fps:
                        frame = self._draw_overlay_info(frame, frame_id)
                    
                    # 显示帧（非无头模式）
                    if not self.headless:
                        try:
                            cv2.imshow(self.window_name, frame)
                        except Exception as e:
                            self.logger.warning(f"⚠️ 显示帧失败，切换到无头模式: {e}")
                            self.headless = True

                    # 更新统计
                    self._update_stats()

                # 处理键盘事件（非无头模式）
                if not self.headless:
                    try:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.logger.info("🎬 按下 'q' 键，退出播放器")
                            self.running = False
                            break
                        elif key == ord(' '):  # 空格键暂停/继续
                            self.toggle_pause()
                        elif key == ord('s'):  # 's' 键截图
                            self._save_screenshot(frame if latest_frame_tuple else None)
                    except Exception as e:
                        self.logger.debug(f"⚠️ 键盘事件处理失败: {e}")
                else:
                    # 无头模式下的简单延迟
                    time.sleep(0.033)  # ~30fps
                
            except Exception as e:
                self.logger.error(f"❌ 播放循环异常: {e}")
                time.sleep(0.1)
        
        self.logger.info(f"🏁 播放循环结束: {self.window_name}")
    
    def _draw_analysis_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """在帧上绘制分析结果"""
        # 遍历所有分析器的结果
        for analyzer_name, result in results.items():
            detections = result.get("detections", [])
            
            for detection in detections:
                # 获取检测信息
                class_name = detection.get("class", "unknown")
                confidence = detection.get("confidence", 0.0)
                bbox = detection.get("bbox", [])
                
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    
                    # 选择颜色
                    color = self.colors.get(class_name, self.colors["default"])
                    
                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制标签
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # 标签背景
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # 标签文字
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _draw_overlay_info(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        """绘制叠加信息"""
        height, width = frame.shape[:2]
        
        # 准备信息文本
        info_lines = []
        
        if self.show_info:
            info_lines.append(f"Frame ID: {frame_id}")
            info_lines.append(f"Time: {time.strftime('%H:%M:%S')}")
            info_lines.append(f"Window: {self.window_name}")
        
        if self.show_fps:
            info_lines.append(f"FPS: {self._stats['current_fps']:.1f}")
            info_lines.append(f"Displayed: {self._stats['frames_displayed']}")
            info_lines.append(f"Dropped: {self._stats['frames_dropped']}")
        
        # 绘制信息背景
        if info_lines:
            line_height = 25
            bg_height = len(info_lines) * line_height + 10
            bg_width = 250
            
            # 半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (bg_width, bg_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # 绘制文字
            for i, line in enumerate(info_lines):
                y = 30 + i * line_height
                cv2.putText(frame, line, (20, y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 绘制状态指示
        if self.paused:
            cv2.putText(frame, "PAUSED", (width - 150, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return frame
    
    def _update_stats(self):
        """更新统计信息"""
        self._stats["frames_displayed"] += 1
        self._stats["fps_frame_count"] += 1
        
        current_time = time.time()
        if current_time - self._stats["last_fps_time"] >= 1.0:
            # 计算FPS
            elapsed = current_time - self._stats["last_fps_time"]
            self._stats["current_fps"] = self._stats["fps_frame_count"] / elapsed
            
            # 重置计数器
            self._stats["fps_frame_count"] = 0
            self._stats["last_fps_time"] = current_time
    
    def _save_screenshot(self, frame: Optional[np.ndarray]):
        """保存截图"""
        if frame is None:
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        
        try:
            cv2.imwrite(filename, frame)
            self.logger.info(f"📸 截图已保存: {filename}")
        except Exception as e:
            self.logger.error(f"❌ 截图保存失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取播放器统计信息"""
        current_time = time.time()
        runtime = current_time - self._stats["start_time"] if self._stats["start_time"] > 0 else 0
        
        stats = self._stats.copy()
        stats.update({
            "running": self.running,
            "paused": self.paused,
            "window_name": self.window_name,
            "queue_size": self.display_queue.qsize(),
            "cache_size": len(self.results_cache),
            "runtime_seconds": runtime,
            "avg_fps": self._stats["frames_displayed"] / max(1, runtime)
        })
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        self.stop()
        
        # 清空队列和缓存
        while not self.display_queue.empty():
            try:
                self.display_queue.get_nowait()
            except Empty:
                break
        
        with self.cache_lock:
            self.results_cache.clear()
        
        self.logger.info("🧹 视频播放器已清理")


class MultiWindowPlayer:
    """多窗口播放器管理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化多窗口播放器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.players: Dict[str, VideoPlayer] = {}
        self.lock = threading.Lock()
        
        self.logger.info("🎬 多窗口播放器管理器初始化")
    
    def create_player(
        self,
        name: str,
        window_name: Optional[str] = None,
        **kwargs
    ) -> VideoPlayer:
        """创建播放器"""
        with self.lock:
            if name in self.players:
                self.logger.warning(f"⚠️ 播放器 '{name}' 已存在，将被替换")
                self.remove_player(name)
            
            player = VideoPlayer(
                window_name=window_name or f"Player_{name}",
                logger=self.logger,
                **kwargs
            )
            
            self.players[name] = player
            self.logger.info(f"✅ 创建播放器: {name}")
            
            return player
    
    def get_player(self, name: str) -> Optional[VideoPlayer]:
        """获取播放器"""
        return self.players.get(name)
    
    def remove_player(self, name: str) -> bool:
        """移除播放器"""
        with self.lock:
            if name in self.players:
                player = self.players.pop(name)
                player.cleanup()
                self.logger.info(f"🗑️ 移除播放器: {name}")
                return True
            return False
    
    def start_all(self):
        """启动所有播放器"""
        with self.lock:
            for name, player in self.players.items():
                player.start()
                self.logger.info(f"🚀 启动播放器: {name}")
    
    def stop_all(self):
        """停止所有播放器"""
        with self.lock:
            for name, player in self.players.items():
                player.stop()
                self.logger.info(f"⏹️ 停止播放器: {name}")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有播放器的统计信息"""
        stats = {}
        with self.lock:
            for name, player in self.players.items():
                stats[name] = player.get_stats()
        return stats
    
    def cleanup_all(self):
        """清理所有播放器"""
        with self.lock:
            for name, player in self.players.items():
                player.cleanup()
                self.logger.info(f"🧹 清理播放器: {name}")
            self.players.clear()


# 全局多窗口播放器管理器
multi_window_player = MultiWindowPlayer()
