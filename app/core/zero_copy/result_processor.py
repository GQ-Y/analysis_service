#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: result_processor.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 分析结果处理器

负责保存分析结果图片、绘制检测框、存储元数据等。

本文件是分析服务项目的一部分。
"""

import os
import cv2
import json
import time
import threading
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from queue import Queue, Empty
from datetime import datetime

from .frame_buffer import FrameBuffer


class ResultProcessor:
    """
    分析结果处理器
    
    负责处理AI分析结果，包括：
    - 在图片上绘制检测框
    - 保存带标注的图片
    - 保存分析元数据
    - 统计分析结果
    """
    
    def __init__(
        self,
        output_dir: str = None,
        save_images: bool = True,
        save_metadata: bool = True,
        draw_boxes: bool = True,
        max_queue_size: int = 100,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化结果处理器
        
        Args:
            output_dir: 输出目录，None则使用默认的storage/results
            save_images: 是否保存图片
            save_metadata: 是否保存元数据
            draw_boxes: 是否绘制检测框
            max_queue_size: 最大队列大小
            logger: 日志记录器
        """
        # 设置默认输出目录
        if output_dir is None:
            from config.settings import get_settings
            settings = get_settings()
            output_dir = str(settings.BASE_DIR / "storage" / "results")
            
        self.output_dir = Path(output_dir)
        self.save_images = save_images
        self.save_metadata = save_metadata
        self.draw_boxes = draw_boxes
        self.logger = logger or logging.getLogger(__name__)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if save_images:
            (self.output_dir / "images").mkdir(exist_ok=True)
        if save_metadata:
            (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        # 处理队列
        self.result_queue = Queue(maxsize=max_queue_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # 统计信息
        self._stats = {
            "total_processed": 0,
            "images_saved": 0,
            "metadata_saved": 0,
            "queue_full_drops": 0,
            "processing_errors": 0,
            "start_time": 0
        }
        
        # 颜色配置（BGR格式）
        self.colors = {
            "person": (0, 255, 0),      # 绿色
            "vehicle": (255, 0, 0),     # 蓝色
            "car": (255, 0, 0),         # 蓝色
            "truck": (0, 0, 255),       # 红色
            "bus": (255, 255, 0),       # 青色
            "motorcycle": (255, 0, 255), # 紫色
            "bicycle": (0, 255, 255),   # 黄色
            "face": (0, 128, 255),      # 橙色
            "default": (128, 128, 128)  # 灰色
        }
        
        self.logger.info(f"📁 结果处理器初始化: 输出目录={output_dir}")
    
    def start(self):
        """启动结果处理器"""
        if self.running:
            self.logger.warning("⚠️ 结果处理器已在运行")
            return
        
        self.running = True
        self._stats["start_time"] = time.time()
        
        # 创建并启动处理线程
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
        self.logger.info("🚀 结果处理器已启动")
    
    def stop(self):
        """停止结果处理器"""
        if not self.running:
            return
        
        self.running = False
        
        # 等待线程结束
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        
        self.logger.info("⏹️ 结果处理器已停止")
    
    def process_result(self, frame_buffer: FrameBuffer, task_id: int = None):
        """
        处理分析结果

        Args:
            frame_buffer: 包含分析结果的帧缓冲区
            task_id: 任务ID（可选）
        """
        try:
            # 增加引用计数，防止在处理过程中被释放
            frame_buffer.add_ref()

            # 添加到处理队列
            result_data = {
                "frame_buffer": frame_buffer,
                "task_id": task_id,
                "timestamp": time.time()
            }

            self.result_queue.put(result_data, block=False)

        except Exception as e:
            self._stats["queue_full_drops"] += 1
            # 如果添加失败，释放引用
            frame_buffer.release()
            self.logger.warning(f"⚠️ 结果队列已满，丢弃帧 {frame_buffer.frame_id}: {e}")
    
    def _process_loop(self):
        """处理循环（在独立线程中运行）"""
        self.logger.info("🔄 结果处理循环开始")
        
        while self.running:
            try:
                # 从队列获取结果
                result_data = self.result_queue.get(timeout=1.0)
                
                # 处理结果
                self._process_single_result(result_data)
                
                # 更新统计
                self._stats["total_processed"] += 1
                
            except Empty:
                continue
            except Exception as e:
                self._stats["processing_errors"] += 1
                self.logger.error(f"❌ 处理结果异常: {e}")
        
        self.logger.info("🏁 结果处理循环结束")
    
    def _process_single_result(self, result_data: Dict[str, Any]):
        """处理单个结果"""
        frame_buffer = result_data["frame_buffer"]
        task_id = result_data.get("task_id")

        try:
            # 检查是否有分析结果
            if not frame_buffer.analysis_results:
                self.logger.debug(f"⏭️ 跳过无分析结果的帧: {frame_buffer.frame_id}")
                return

            # 检查帧数据是否有效
            if frame_buffer.frame_id <= 0 or frame_buffer.timestamp <= 0:
                self.logger.debug(f"⏭️ 跳过无效帧: ID={frame_buffer.frame_id}, 时间戳={frame_buffer.timestamp}")
                return

            # 获取帧数据
            frame = frame_buffer.get_frame_copy()

            # 绘制检测框
            if self.draw_boxes:
                frame = self._draw_detections(frame, frame_buffer)

            # 保存图片
            if self.save_images:
                self._save_image(frame, frame_buffer, task_id)

            # 保存元数据
            if self.save_metadata:
                self._save_metadata(frame_buffer, task_id)

            self.logger.debug(f"💾 处理完成: 帧 {frame_buffer.frame_id}")

        except Exception as e:
            self.logger.error(f"❌ 处理帧 {frame_buffer.frame_id} 失败: {e}")
            raise
        finally:
            # 处理完成后释放引用
            frame_buffer.release()
    
    def _draw_detections(self, frame, frame_buffer: FrameBuffer):
        """在帧上绘制检测结果"""
        # 获取所有分析结果
        for analyzer_name, result in frame_buffer.analysis_results.items():
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
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # 标签背景
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # 标签文字
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 添加时间戳和帧信息
        timestamp_str = datetime.fromtimestamp(frame_buffer.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        info_text = f"Frame: {frame_buffer.frame_id} | {timestamp_str} | Stream: {frame_buffer.stream_id}"
        
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _save_image(self, frame, frame_buffer: FrameBuffer, task_id: int = None):
        """保存图片"""
        # 生成文件名
        timestamp_str = datetime.fromtimestamp(frame_buffer.timestamp).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        if task_id:
            filename = f"task_{task_id}_frame_{frame_buffer.frame_id}_{timestamp_str}.jpg"
        else:
            filename = f"frame_{frame_buffer.frame_id}_{timestamp_str}.jpg"
        
        # 保存路径
        image_path = self.output_dir / "images" / filename
        
        # 保存图片
        success = cv2.imwrite(str(image_path), frame)
        
        if success:
            self._stats["images_saved"] += 1
            self.logger.debug(f"💾 图片已保存: {filename}")
        else:
            self.logger.error(f"❌ 图片保存失败: {filename}")
    
    def _save_metadata(self, frame_buffer: FrameBuffer, task_id: int = None):
        """保存元数据"""
        # 生成文件名
        timestamp_str = datetime.fromtimestamp(frame_buffer.timestamp).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        if task_id:
            filename = f"task_{task_id}_frame_{frame_buffer.frame_id}_{timestamp_str}.json"
        else:
            filename = f"frame_{frame_buffer.frame_id}_{timestamp_str}.json"
        
        # 元数据
        metadata = {
            "frame_id": frame_buffer.frame_id,
            "timestamp": frame_buffer.timestamp,
            "stream_id": frame_buffer.stream_id,
            "task_id": task_id,
            "analysis_results": frame_buffer.analysis_results,
            "saved_at": datetime.now().isoformat()
        }
        
        # 保存路径
        metadata_path = self.output_dir / "metadata" / filename
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self._stats["metadata_saved"] += 1
            self.logger.debug(f"💾 元数据已保存: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ 元数据保存失败: {filename}, 错误: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        current_time = time.time()
        runtime = current_time - self._stats["start_time"] if self._stats["start_time"] > 0 else 0
        
        stats = self._stats.copy()
        stats.update({
            "running": self.running,
            "queue_size": self.result_queue.qsize(),
            "runtime_seconds": runtime,
            "processing_rate": self._stats["total_processed"] / max(1, runtime)
        })
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        self.stop()
        
        # 清空队列
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except Empty:
                break
        
        self.logger.info("🧹 结果处理器已清理")


class BatchResultProcessor:
    """批量结果处理器"""
    
    def __init__(
        self,
        output_dir: str = None,
        batch_size: int = 10,
        save_interval: float = 5.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化批量结果处理器
        
        Args:
            output_dir: 输出目录，None则使用默认的storage/results
            batch_size: 批处理大小
            save_interval: 保存间隔（秒）
            logger: 日志记录器
        """
        # 设置默认输出目录
        if output_dir is None:
            from config.settings import get_settings
            settings = get_settings()
            output_dir = str(settings.BASE_DIR / "storage" / "results")
            
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.logger = logger or logging.getLogger(__name__)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 批处理缓存
        self.batch_buffer: List[Dict[str, Any]] = []
        self.last_save_time = time.time()
        self.lock = threading.Lock()
        
        # 统计信息
        self._stats = {
            "total_batches": 0,
            "total_items": 0,
            "last_batch_size": 0
        }
        
        self.logger.info(f"📦 批量结果处理器初始化: 批大小={batch_size}")
    
    def add_result(self, frame_buffer: FrameBuffer, task_id: int = None):
        """添加结果到批处理缓存"""
        with self.lock:
            # 添加到缓存
            result_data = {
                "frame_id": frame_buffer.frame_id,
                "timestamp": frame_buffer.timestamp,
                "stream_id": frame_buffer.stream_id,
                "task_id": task_id,
                "analysis_results": frame_buffer.analysis_results.copy(),
                "added_at": time.time()
            }
            
            self.batch_buffer.append(result_data)
            
            # 检查是否需要保存
            current_time = time.time()
            should_save = (
                len(self.batch_buffer) >= self.batch_size or
                current_time - self.last_save_time >= self.save_interval
            )
            
            if should_save:
                self._save_batch()
    
    def _save_batch(self):
        """保存批处理结果"""
        if not self.batch_buffer:
            return
        
        # 生成批次文件名
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        batch_filename = f"batch_{timestamp_str}.json"
        batch_path = self.output_dir / batch_filename
        
        # 保存批次数据
        batch_data = {
            "batch_id": timestamp_str,
            "batch_size": len(self.batch_buffer),
            "created_at": datetime.now().isoformat(),
            "results": self.batch_buffer.copy()
        }
        
        try:
            with open(batch_path, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, indent=2, ensure_ascii=False)
            
            # 更新统计
            self._stats["total_batches"] += 1
            self._stats["total_items"] += len(self.batch_buffer)
            self._stats["last_batch_size"] = len(self.batch_buffer)
            
            self.logger.info(f"💾 批次已保存: {batch_filename}, 大小: {len(self.batch_buffer)}")
            
            # 清空缓存
            self.batch_buffer.clear()
            self.last_save_time = time.time()
            
        except Exception as e:
            self.logger.error(f"❌ 批次保存失败: {batch_filename}, 错误: {e}")
    
    def flush(self):
        """强制保存当前批次"""
        with self.lock:
            if self.batch_buffer:
                self._save_batch()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                **self._stats,
                "current_buffer_size": len(self.batch_buffer),
                "batch_size": self.batch_size,
                "save_interval": self.save_interval
            }
