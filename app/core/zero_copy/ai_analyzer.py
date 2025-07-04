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


class MockAnalyzer(BaseAnalyzer):
    """模拟分析器（用于测试）"""
    
    def __init__(self, name: str = "mock", process_time: float = 0.1, logger: Optional[logging.Logger] = None):
        """
        初始化模拟分析器
        
        Args:
            name: 分析器名称
            process_time: 模拟处理时间（秒）
            logger: 日志记录器
        """
        super().__init__(name, logger)
        self.process_time = process_time
    
    def analyze_frame(self, frame_buffer: FrameBuffer) -> Dict[str, Any]:
        """模拟单帧分析"""
        # 模拟处理时间
        time.sleep(self.process_time)
        
        return {
            "detections": [
                {
                    "class": "person",
                    "confidence": 0.95,
                    "bbox": [100, 100, 200, 300]
                },
                {
                    "class": "car",
                    "confidence": 0.87,
                    "bbox": [300, 200, 500, 400]
                }
            ],
            "frame_id": frame_buffer.frame_id,
            "timestamp": frame_buffer.timestamp,
            "analyzer": self.name
        }
    
    def analyze_batch(self, frame_buffers: List[FrameBuffer]) -> List[Dict[str, Any]]:
        """模拟批量分析"""
        # 模拟批量处理时间（比单帧处理更高效）
        batch_time = self.process_time * len(frame_buffers) * 0.7
        time.sleep(batch_time)
        
        results = []
        for frame_buffer in frame_buffers:
            result = {
                "detections": [
                    {
                        "class": "person",
                        "confidence": 0.95,
                        "bbox": [100, 100, 200, 300]
                    }
                ],
                "frame_id": frame_buffer.frame_id,
                "timestamp": frame_buffer.timestamp,
                "analyzer": self.name,
                "batch_processed": True
            }
            results.append(result)
        
        return results


class AnalysisWorker:
    """分析工作器，在独立线程中运行分析器"""
    
    def __init__(
        self,
        analyzer: BaseAnalyzer,
        time_axis: TimeAxis,
        batch_size: int = 1,
        timeout: float = 0.1,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化分析工作器
        
        Args:
            analyzer: AI分析器
            time_axis: 时间轴
            batch_size: 批处理大小（1为单帧处理）
            timeout: 获取帧的超时时间
            logger: 日志记录器
        """
        self.analyzer = analyzer
        self.time_axis = time_axis
        self.batch_size = batch_size
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        
        # 工作状态
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # 结果回调
        self.result_callbacks: List[Callable] = []
        
        self.logger.info(f"👷 分析工作器初始化: {analyzer.name}, 批大小={batch_size}")
    
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
                    # 单帧处理模式
                    frame_buffer = self.time_axis.get_frame()
                    if frame_buffer:
                        result = self.analyzer.process_frame(frame_buffer)
                        # 先通知结果回调，再释放引用
                        self._notify_result([frame_buffer], [result])
                        frame_buffer.release()  # 释放引用
                    else:
                        time.sleep(self.timeout)
                else:
                    # 批处理模式
                    frame_buffers = self.time_axis.get_batch(self.batch_size)
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
        stats.update({
            "running": self.running,
            "batch_size": self.batch_size,
            "timeout": self.timeout
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
        timeout: float = 0.1
    ) -> AnalysisWorker:
        """添加分析工作器"""
        with self.lock:
            if name in self.workers:
                self.logger.warning(f"⚠️ 分析工作器 {name} 已存在，将被替换")
                self.remove_worker(name)
            
            worker = AnalysisWorker(analyzer, time_axis, batch_size, timeout, self.logger)
            self.workers[name] = worker
            
            self.logger.info(f"➕ 添加分析工作器: {name}")
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
