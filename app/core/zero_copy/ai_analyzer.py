#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: ai_analyzer.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: AI分析器

基于timelinetool架构实现的AI分析器，支持单帧和批量处理。

本文件是分析服务项目的一部分。
"""

import time
import threading
import logging
import asyncio
from typing import List, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod

from .frame_buffer import FrameBuffer
from .time_axis import TimeAxis


class FPSController:
    """
    FPS控制器，用于控制不同算法模型的分析帧率
    
    在分析工作器和时间轴之间插入，按照设定的FPS进行帧抽取
    """
    
    def __init__(
        self,
        time_axis: TimeAxis,
        target_fps: Optional[float] = None,
        model_code: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化FPS控制器
        
        Args:
            time_axis: 原始时间轴
            target_fps: 目标分析帧率，None表示不限制（全帧分析）
            model_code: 算法模型代码
            logger: 日志记录器
        """
        self.time_axis = time_axis
        self.target_fps = target_fps
        self.model_code = model_code or "unknown"
        self.logger = logger or logging.getLogger(__name__)
        
        # FPS控制状态
        self.last_frame_time = 0.0
        self.frame_interval = 1.0 / target_fps if target_fps and target_fps > 0 else 0.0
        self.dropped_frames = 0
        self.processed_frames = 0
        
        # 统计信息
        self._stats = {
            "target_fps": target_fps,
            "actual_fps": 0.0,
            "dropped_frames": 0,
            "processed_frames": 0,
            "drop_rate": 0.0
        }
        
        self.logger.info(f"🎯 FPS控制器初始化: 模型={model_code}, 目标FPS={target_fps}")
    
    def get_frame(self) -> Optional[FrameBuffer]:
        """
        获取帧（带FPS控制）
        
        Returns:
            Optional[FrameBuffer]: 符合FPS要求的帧，无可用帧返回None
        """
        if not self.target_fps or self.target_fps <= 0:
            # 无FPS限制，直接返回原始帧
            return self.time_axis.get_frame()
        
        current_time = time.time()
        
        # 检查是否到达下一帧时间
        if current_time - self.last_frame_time < self.frame_interval:
            # 还没到时间，丢弃当前可用的帧
            while True:
                frame = self.time_axis.get_frame()
                if not frame:
                    break
                
                # 释放丢弃的帧
                frame.release()
                self.dropped_frames += 1
                self._stats["dropped_frames"] = self.dropped_frames
                
                # 检查是否有更多帧
                if not self._has_more_frames():
                    break
            
            return None
        
        # 获取帧
        frame = self.time_axis.get_frame()
        if frame:
            self.last_frame_time = current_time
            self.processed_frames += 1
            self._stats["processed_frames"] = self.processed_frames
            
            # 计算实际FPS
            if self.processed_frames > 1:
                elapsed = current_time - (self.last_frame_time - self.frame_interval)
                self._stats["actual_fps"] = self.processed_frames / max(elapsed, 0.001)
            
            # 计算丢帧率
            total_frames = self.processed_frames + self.dropped_frames
            self._stats["drop_rate"] = self.dropped_frames / max(total_frames, 1) * 100
            
            self.logger.debug(f"🎯 {self.model_code}: 获取帧 {frame.frame_id}, "
                            f"实际FPS={self._stats['actual_fps']:.1f}, "
                            f"丢帧率={self._stats['drop_rate']:.1f}%")
        
        return frame
    
    def get_batch(self, max_size: int) -> List[FrameBuffer]:
        """
        获取批量帧（带FPS控制）
        
        Args:
            max_size: 最大批次大小
            
        Returns:
            List[FrameBuffer]: 符合FPS要求的帧列表
        """
        if not self.target_fps or self.target_fps <= 0:
            # 无FPS限制，直接返回原始批次
            return self.time_axis.get_batch(max_size)
        
        batch = []
        current_time = time.time()
        
        # 根据FPS计算需要的帧数
        time_since_last = current_time - self.last_frame_time
        frames_needed = int(time_since_last * self.target_fps)
        frames_needed = min(frames_needed, max_size)
        
        if frames_needed <= 0:
            return batch
        
        # 获取更多帧用于筛选
        available_frames = self.time_axis.get_batch(max_size * 2)
        
        if not available_frames:
            return batch
        
        # 按时间间隔筛选帧
        selected_frames = []
        dropped_count = 0
        
        for i, frame in enumerate(available_frames):
            if len(selected_frames) >= frames_needed:
                # 已获取足够帧，释放剩余帧
                frame.release()
                dropped_count += 1
                continue
            
            # 检查时间间隔
            frame_time = frame.timestamp
            if not selected_frames or (frame_time - selected_frames[-1].timestamp) >= self.frame_interval:
                selected_frames.append(frame)
            else:
                # 时间间隔不够，丢弃帧
                frame.release()
                dropped_count += 1
        
        # 更新统计
        self.dropped_frames += dropped_count
        self.processed_frames += len(selected_frames)
        self._stats["dropped_frames"] = self.dropped_frames
        self._stats["processed_frames"] = self.processed_frames
        
        if selected_frames:
            self.last_frame_time = current_time
            
            # 计算实际FPS
            if self.processed_frames > 1:
                elapsed = current_time - (self.last_frame_time - self.frame_interval)
                self._stats["actual_fps"] = self.processed_frames / max(elapsed, 0.001)
            
            # 计算丢帧率
            total_frames = self.processed_frames + self.dropped_frames
            self._stats["drop_rate"] = self.dropped_frames / max(total_frames, 1) * 100
            
            self.logger.debug(f"🎯 {self.model_code}: 获取批次 {len(selected_frames)} 帧, "
                            f"丢弃 {dropped_count} 帧, "
                            f"实际FPS={self._stats['actual_fps']:.1f}, "
                            f"丢帧率={self._stats['drop_rate']:.1f}%")
        
        return selected_frames
    
    def _has_more_frames(self) -> bool:
        """检查时间轴是否还有更多帧"""
        stats = self.time_axis.get_stats()
        return stats.get("current_size", 0) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取FPS控制器统计信息"""
        stats = self._stats.copy()
        stats.update({
            "model_code": self.model_code,
            "frame_interval": self.frame_interval,
            "last_frame_time": self.last_frame_time
        })
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.dropped_frames = 0
        self.processed_frames = 0
        self.last_frame_time = 0.0
        self._stats.update({
            "actual_fps": 0.0,
            "dropped_frames": 0,
            "processed_frames": 0,
            "drop_rate": 0.0
        })


class BaseAnalyzer(ABC):
    """AI分析器基类"""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        """
        初始化分析器
        
        Args:
            name: 分析器名称
            logger: 日志记录器
        """
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        
        # 统计信息
        self._stats = {
            "total_processed": 0,
            "total_batches": 0,
            "total_time": 0.0,
            "avg_time_per_frame": 0.0,
            "avg_time_per_batch": 0.0,
            "start_time": 0.0
        }
        
        self.logger.info(f"🤖 AI分析器初始化: {name}")
    
    @abstractmethod
    def analyze_frame(self, frame_buffer: FrameBuffer) -> Dict[str, Any]:
        """
        分析单帧
        
        Args:
            frame_buffer: 帧缓冲区
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        pass
    
    @abstractmethod
    def analyze_batch(self, frame_buffers: List[FrameBuffer]) -> List[Dict[str, Any]]:
        """
        批量分析
        
        Args:
            frame_buffers: 帧缓冲区列表
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        pass
    
    def process_frame(self, frame_buffer: FrameBuffer) -> Dict[str, Any]:
        """
        处理单帧（包含统计）
        
        Args:
            frame_buffer: 帧缓冲区
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        start_time = time.time()
        
        try:
            result = self.analyze_frame(frame_buffer)
            
            # 将结果存储到帧缓冲区
            frame_buffer.add_analysis_result(self.name, result)
            
            # 更新统计
            process_time = time.time() - start_time
            self._update_stats(1, 1, process_time)
            
            self.logger.debug(f"🔍 {self.name}: 分析帧 {frame_buffer.frame_id}, "
                            f"耗时 {process_time*1000:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {self.name}: 分析帧 {frame_buffer.frame_id} 失败: {e}")
            return {"error": str(e)}
    
    def process_batch(self, frame_buffers: List[FrameBuffer]) -> List[Dict[str, Any]]:
        """
        处理批量帧（包含统计）
        
        Args:
            frame_buffers: 帧缓冲区列表
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        if not frame_buffers:
            return []
        
        start_time = time.time()
        
        try:
            results = self.analyze_batch(frame_buffers)
            
            # 将结果存储到对应的帧缓冲区
            for frame_buffer, result in zip(frame_buffers, results):
                frame_buffer.add_analysis_result(self.name, result)
            
            # 更新统计
            process_time = time.time() - start_time
            self._update_stats(len(frame_buffers), 1, process_time)
            
            self.logger.debug(f"🔍 {self.name}: 批量分析 {len(frame_buffers)} 帧, "
                            f"耗时 {process_time*1000:.2f}ms")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ {self.name}: 批量分析失败: {e}")
            return [{"error": str(e)} for _ in frame_buffers]
    
    def _update_stats(self, frame_count: int, batch_count: int, process_time: float):
        """更新统计信息"""
        self._stats["total_processed"] += frame_count
        self._stats["total_batches"] += batch_count
        self._stats["total_time"] += process_time
        
        if self._stats["total_processed"] > 0:
            self._stats["avg_time_per_frame"] = self._stats["total_time"] / self._stats["total_processed"]
        
        if self._stats["total_batches"] > 0:
            self._stats["avg_time_per_batch"] = self._stats["total_time"] / self._stats["total_batches"]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取分析器统计信息"""
        stats = self._stats.copy()
        stats["name"] = self.name
        
        if self._stats["start_time"] > 0:
            runtime = time.time() - self._stats["start_time"]
            stats["runtime_seconds"] = runtime
            stats["fps"] = self._stats["total_processed"] / max(1, runtime)
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self._stats = {
            "total_processed": 0,
            "total_batches": 0,
            "total_time": 0.0,
            "avg_time_per_frame": 0.0,
            "avg_time_per_batch": 0.0,
            "start_time": time.time()
        }


class AnalysisWorker:
    """分析工作器，在独立线程中运行分析器"""
    
    def __init__(
        self,
        analyzer: BaseAnalyzer,
        time_axis: TimeAxis,
        batch_size: int = 1,
        timeout: float = 0.1,
        target_fps: Optional[float] = None,
        model_code: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化分析工作器
        
        Args:
            analyzer: AI分析器
            time_axis: 时间轴
            batch_size: 批处理大小（1为单帧处理）
            timeout: 获取帧的超时时间
            target_fps: 目标分析帧率，None表示不限制
            model_code: 算法模型代码
            logger: 日志记录器
        """
        self.analyzer = analyzer
        self.batch_size = batch_size
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        
        # 创建FPS控制器
        self.fps_controller = FPSController(
            time_axis=time_axis,
            target_fps=target_fps,
            model_code=model_code,
            logger=self.logger
        )
        
        # 工作状态
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # 结果回调
        self.result_callbacks: List[Callable] = []
        
        fps_info = f"FPS={target_fps}" if target_fps else "无限制"
        self.logger.info(f"👷 分析工作器初始化: {analyzer.name}, 批大小={batch_size}, {fps_info}")
    
    def add_result_callback(self, callback: Callable):
        """添加结果回调函数"""
        self.result_callbacks.append(callback)
    
    def start(self):
        """启动工作器"""
        if self.running:
            self.logger.warning(f"⚠️ 分析工作器 {self.analyzer.name} 已在运行")
            return
        
        self.running = True
        self.analyzer.reset_stats()
        self.fps_controller.reset_stats()
        
        # 创建并启动工作线程
        self.thread = threading.Thread(target=self._work_loop, daemon=True)
        self.thread.start()
        
        self.logger.info(f"🚀 分析工作器 {self.analyzer.name} 已启动")
    
    def stop(self):
        """停止工作器"""
        if not self.running:
            return
        
        self.running = False
        
        # 等待线程结束
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        
        self.logger.info(f"⏹️ 分析工作器 {self.analyzer.name} 已停止")
    
    def _work_loop(self):
        """工作循环（在独立线程中运行）"""
        self.logger.info(f"🔄 分析工作器 {self.analyzer.name} 开始工作循环")

        while self.running:
            try:
                if self.batch_size == 1:
                    # 单帧处理模式 - 使用FPS控制器
                    frame_buffer = self.fps_controller.get_frame()
                    if frame_buffer:
                        result = self.analyzer.process_frame(frame_buffer)
                        # 先通知结果回调，再释放引用
                        self._notify_result([frame_buffer], [result])
                        frame_buffer.release()  # 释放引用
                    else:
                        time.sleep(self.timeout)
                else:
                    # 批处理模式 - 使用FPS控制器
                    frame_buffers = self.fps_controller.get_batch(self.batch_size)
                    if frame_buffers:
                        results = self.analyzer.process_batch(frame_buffers)
                        # 先通知结果回调，再释放引用
                        self._notify_result(frame_buffers, results)

                        # 释放所有引用
                        for frame_buffer in frame_buffers:
                            frame_buffer.release()
                    else:
                        time.sleep(self.timeout)

            except Exception as e:
                self.logger.error(f"❌ 分析工作器 {self.analyzer.name} 异常: {e}")
                time.sleep(1.0)  # 异常后等待一下

        self.logger.info(f"🏁 分析工作器 {self.analyzer.name} 工作循环结束")
    
    def _notify_result(self, frame_buffers: List[FrameBuffer], results: List[Dict[str, Any]]):
        """通知结果回调"""
        for callback in self.result_callbacks:
            try:
                callback(frame_buffers, results)
            except Exception as e:
                self.logger.error(f"❌ 结果回调异常: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取工作器统计信息"""
        stats = self.analyzer.get_stats()
        fps_stats = self.fps_controller.get_stats()
        
        stats.update({
            "running": self.running,
            "batch_size": self.batch_size,
            "timeout": self.timeout,
            "fps_controller": fps_stats
        })
        return stats


class AnalysisEngine:
    """分析引擎，管理多个分析工作器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化分析引擎
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.workers: Dict[str, AnalysisWorker] = {}
        self.lock = threading.Lock()
        
        self.logger.info("🏭 分析引擎初始化")
    
    def add_worker(
        self,
        name: str,
        analyzer: BaseAnalyzer,
        time_axis: TimeAxis,
        batch_size: int = 1,
        timeout: float = 0.1,
        target_fps: Optional[float] = None,
        model_code: Optional[str] = None
    ) -> AnalysisWorker:
        """
        添加分析工作器
        
        Args:
            name: 工作器名称
            analyzer: AI分析器
            time_axis: 时间轴
            batch_size: 批处理大小
            timeout: 超时时间
            target_fps: 目标分析帧率
            model_code: 算法模型代码
        """
        with self.lock:
            if name in self.workers:
                self.logger.warning(f"⚠️ 分析工作器 {name} 已存在，将被替换")
                self.remove_worker(name)
            
            worker = AnalysisWorker(
                analyzer=analyzer,
                time_axis=time_axis,
                batch_size=batch_size,
                timeout=timeout,
                target_fps=target_fps,
                model_code=model_code,
                logger=self.logger
            )
            self.workers[name] = worker
            
            fps_info = f"FPS={target_fps}" if target_fps else "无限制"
            self.logger.info(f"➕ 添加分析工作器: {name} ({fps_info})")
            return worker
    
    def remove_worker(self, name: str) -> bool:
        """移除分析工作器"""
        with self.lock:
            if name in self.workers:
                worker = self.workers.pop(name)
                worker.stop()
                self.logger.info(f"➖ 移除分析工作器: {name}")
                return True
            return False
    
    def start_all(self):
        """启动所有工作器"""
        with self.lock:
            for name, worker in self.workers.items():
                worker.start()
                self.logger.info(f"🚀 启动分析工作器: {name}")
    
    def stop_all(self):
        """停止所有工作器"""
        with self.lock:
            for name, worker in self.workers.items():
                worker.stop()
                self.logger.info(f"⏹️ 停止分析工作器: {name}")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有工作器的统计信息"""
        stats = {}
        with self.lock:
            for name, worker in self.workers.items():
                stats[name] = worker.get_stats()
        return stats
