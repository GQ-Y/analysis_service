#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: image_processor.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 图片处理器

专门处理单张图片或图片集的处理器，支持批量分析和零拷贝架构。

本文件是分析服务项目的一部分。
"""

import cv2
import time
import threading
import logging
import os
import glob
from typing import List, Optional, Callable, Union
from pathlib import Path

from .memory_pool import MemoryPool
from .time_axis import TimeAxis


class ImageProcessor:
    """
    图片处理器
    
    专门处理图片文件（单张或批量），特点：
    - 支持单张图片分析
    - 支持图片集批量处理
    - 支持多种图片格式
    - 零拷贝架构集成
    - 自动完成检测
    """
    
    def __init__(
        self,
        image_path: Union[str, List[str]],
        memory_pool: MemoryPool,
        time_axis: TimeAxis,
        stream_id: str = "image_analysis",
        batch_size: int = 10,  # 批处理大小
        processing_delay: float = 0.1,  # 处理间隔（秒）
        supported_formats: List[str] = None,
        logger: Optional[logging.Logger] = None,
        on_batch_complete_callback: Optional[Callable] = None,
        on_all_complete_callback: Optional[Callable] = None
    ):
        """
        初始化图片处理器
        
        Args:
            image_path: 图片路径（单张图片、图片文件夹或图片路径列表）
            memory_pool: 内存池
            time_axis: 时间轴
            stream_id: 流ID
            batch_size: 批处理大小
            processing_delay: 处理间隔
            supported_formats: 支持的图片格式
            logger: 日志记录器
            on_batch_complete_callback: 批处理完成回调
            on_all_complete_callback: 全部完成回调
        """
        self.memory_pool = memory_pool
        self.time_axis = time_axis
        self.stream_id = stream_id
        self.batch_size = batch_size
        self.processing_delay = processing_delay
        self.logger = logger or logging.getLogger(__name__)
        self.on_batch_complete_callback = on_batch_complete_callback
        self.on_all_complete_callback = on_all_complete_callback
        
        # 支持的图片格式
        if supported_formats is None:
            self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        else:
            self.supported_formats = [fmt.lower() for fmt in supported_formats]
        
        # 处理状态
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # 图片列表
        self.image_paths: List[str] = []
        self._prepare_image_list(image_path)
        
        # 统计信息
        self._stats = {
            "total_images": len(self.image_paths),
            "processed_images": 0,
            "successful_images": 0,
            "failed_images": 0,
            "current_batch": 0,
            "total_batches": 0,
            "start_time": 0,
            "end_time": 0,
            "progress_percentage": 0.0,
            "current_image_index": 0
        }
        
        # 计算批次信息
        if self.batch_size > 0:
            self._stats["total_batches"] = (len(self.image_paths) + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"🖼️ 图片处理器初始化: {len(self.image_paths)}张图片 -> {self.stream_id}")
        self.logger.info(f"📊 处理配置: 批大小={batch_size}, 处理间隔={processing_delay}s")
    
    def _prepare_image_list(self, image_path: Union[str, List[str]]):
        """
        准备图片列表
        
        Args:
            image_path: 图片路径（单张、文件夹或列表）
        """
        if isinstance(image_path, list):
            # 图片路径列表
            for path in image_path:
                if os.path.isfile(path) and self._is_supported_format(path):
                    self.image_paths.append(path)
                    
        elif os.path.isfile(image_path):
            # 单张图片
            if self._is_supported_format(image_path):
                self.image_paths.append(image_path)
            else:
                self.logger.warning(f"⚠️ 不支持的图片格式: {image_path}")
                
        elif os.path.isdir(image_path):
            # 图片文件夹
            for ext in self.supported_formats:
                pattern = os.path.join(image_path, f"*{ext}")
                self.image_paths.extend(glob.glob(pattern))
                pattern = os.path.join(image_path, f"*{ext.upper()}")
                self.image_paths.extend(glob.glob(pattern))
            
            # 去重并排序
            self.image_paths = sorted(list(set(self.image_paths)))
            
        else:
            self.logger.error(f"❌ 无效的图片路径: {image_path}")
        
        self.logger.info(f"📂 找到 {len(self.image_paths)} 张支持的图片")
    
    def _is_supported_format(self, file_path: str) -> bool:
        """
        检查是否为支持的图片格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否支持
        """
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_formats
    
    def start(self) -> bool:
        """
        启动图片处理
        
        Returns:
            bool: 是否成功启动
        """
        if self.running:
            self.logger.warning(f"⚠️ 图片处理器 {self.stream_id} 已在运行")
            return True
        
        if not self.image_paths:
            self.logger.error(f"❌ 没有找到可处理的图片")
            return False
        
        self.running = True
        self._stats["start_time"] = time.time()
        
        # 创建并启动处理线程
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
        self.logger.info(f"🚀 图片处理器已启动: {self.stream_id}")
        return True
    
    def stop(self):
        """停止图片处理"""
        if not self.running:
            return
        
        self.running = False
        
        # 等待线程结束
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        
        self.logger.info(f"⏹️ 图片处理器已停止: {self.stream_id}")
    
    def _process_loop(self):
        """图片处理循环（在独立线程中运行）"""
        self.logger.info(f"🔄 图片处理循环开始: {self.stream_id}")
        
        frame_id = 0
        batch_count = 0
        
        try:
            for i, image_path in enumerate(self.image_paths):
                if not self.running:
                    break
                
                self._stats["current_image_index"] = i
                self._stats["progress_percentage"] = (i / len(self.image_paths)) * 100
                
                # 处理单张图片
                success = self._process_single_image(image_path, frame_id)
                
                # 更新统计
                self._stats["processed_images"] += 1
                if success:
                    self._stats["successful_images"] += 1
                else:
                    self._stats["failed_images"] += 1
                
                frame_id += 1
                
                # 检查是否完成一个批次
                if (i + 1) % self.batch_size == 0 or (i + 1) == len(self.image_paths):
                    batch_count += 1
                    self._stats["current_batch"] = batch_count
                    
                    self.logger.info(f"📦 批次 {batch_count}/{self._stats['total_batches']} 完成, "
                                   f"进度: {self._stats['progress_percentage']:.1f}%")
                    
                    # 调用批次完成回调
                    if self.on_batch_complete_callback:
                        try:
                            self.on_batch_complete_callback(batch_count, self._stats['total_batches'])
                        except Exception as e:
                            self.logger.error(f"❌ 批次完成回调执行失败: {e}")
                
                # 处理间隔
                if self.processing_delay > 0:
                    time.sleep(self.processing_delay)
            
            # 处理完成
            self._stats["end_time"] = time.time()
            self._stats["progress_percentage"] = 100.0
            
            self.logger.info(f"🏁 图片处理完成: {self.stream_id}, "
                           f"成功: {self._stats['successful_images']}, "
                           f"失败: {self._stats['failed_images']}")
            
            # 调用全部完成回调
            if self.on_all_complete_callback:
                try:
                    self.logger.info(f"📞 调用图片处理完成回调...")
                    self.on_all_complete_callback()
                except Exception as e:
                    self.logger.error(f"❌ 图片处理完成回调执行失败: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ 图片处理循环异常: {e}")
        finally:
            self.running = False
            self.logger.info(f"🏁 图片处理循环结束: {self.stream_id}")
    
    def _process_single_image(self, image_path: str, frame_id: int) -> bool:
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            frame_id: 帧ID
            
        Returns:
            bool: 是否处理成功
        """
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"❌ 无法读取图片: {image_path}")
                return False
            
            # 获取图片信息
            height, width = image.shape[:2]
            file_size = os.path.getsize(image_path)
            
            self.logger.debug(f"📷 处理图片: {os.path.basename(image_path)} "
                            f"({width}x{height}, {file_size/1024:.1f}KB)")
            
            # 存储到内存池
            timestamp = time.time()
            buffer = self.memory_pool.put_frame(
                image, frame_id, timestamp, self.stream_id
            )
            
            if buffer:
                # 添加图片路径信息到buffer（如果支持）
                if hasattr(buffer, 'metadata'):
                    buffer.metadata = {
                        'image_path': image_path,
                        'width': width,
                        'height': height,
                        'file_size': file_size,
                        'format': Path(image_path).suffix.lower()
                    }
                
                # 添加到时间轴
                if self.time_axis.add_frame(buffer):
                    return True
                else:
                    buffer.release()  # 时间轴添加失败，释放引用
                    return False
            else:
                self.logger.warning(f"⚠️ 内存池无可用缓冲区，跳过图片: {image_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 处理图片失败 {image_path}: {e}")
            return False
    
    def get_image_list(self) -> List[str]:
        """
        获取图片列表
        
        Returns:
            List[str]: 图片路径列表
        """
        return self.image_paths.copy()
    
    def get_stats(self) -> dict:
        """
        获取处理统计信息
        
        Returns:
            dict: 统计信息
        """
        current_time = time.time()
        runtime = current_time - self._stats["start_time"] if self._stats["start_time"] > 0 else 0
        
        stats = self._stats.copy()
        stats.update({
            "stream_id": self.stream_id,
            "running": self.running,
            "runtime_seconds": runtime,
            "batch_size": self.batch_size,
            "processing_delay": self.processing_delay,
            "avg_processing_time": runtime / max(1, self._stats["processed_images"]),
            "success_rate": (self._stats["successful_images"] / max(1, self._stats["processed_images"])) * 100,
            "images_per_second": self._stats["processed_images"] / max(1, runtime),
            "supported_formats": self.supported_formats
        })
        
        return stats
    
    def get_current_image(self) -> Optional[str]:
        """
        获取当前正在处理的图片路径
        
        Returns:
            Optional[str]: 当前图片路径
        """
        index = self._stats["current_image_index"]
        if 0 <= index < len(self.image_paths):
            return self.image_paths[index]
        return None
    
    def get_remaining_images(self) -> List[str]:
        """
        获取剩余未处理的图片列表
        
        Returns:
            List[str]: 剩余图片路径列表
        """
        index = self._stats["current_image_index"] + 1
        return self.image_paths[index:]
    
    def __str__(self) -> str:
        stats = self.get_stats()
        return (f"ImageProcessor(id={self.stream_id}, "
                f"progress={stats['progress_percentage']:.1f}%, "
                f"processed={stats['processed_images']}/{stats['total_images']}, "
                f"success_rate={stats['success_rate']:.1f}%)")
    
    def __repr__(self) -> str:
        return self.__str__() 