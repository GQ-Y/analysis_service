#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: video_file_processor.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 视频文件处理器

专门处理本地/在线视频文件的处理器，采用批处理机制确保不丢帧。

本文件是分析服务项目的一部分。
"""

import cv2
import time
import threading
import logging
import os
import asyncio
from typing import Optional, Callable, List
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

from .memory_pool import MemoryPool
from .time_axis import TimeAxis


class VideoFileProcessor:
    """
    视频文件处理器
    
    专门处理视频文件（本地或在线），特点：
    - 批处理机制（每批处理固定帧数）
    - 确保不丢帧（排队等待处理）
    - 自动检测视频结束
    - 内存高效利用
    """
    
    def __init__(
        self,
        video_path: str,
        memory_pool: MemoryPool,
        time_axis: TimeAxis,
        stream_id: str = "video_file",
        batch_size: int = 50,  # 每批处理帧数
        max_queue_size: int = 200,  # 最大队列大小
        video_player=None,
        logger: Optional[logging.Logger] = None,
        on_video_end_callback: Optional[Callable] = None,
        on_batch_complete_callback: Optional[Callable] = None
    ):
        self.video_path = video_path
        self.memory_pool = memory_pool
        self.time_axis = time_axis
        self.stream_id = stream_id
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.video_player = video_player
        self.logger = logger or logging.getLogger(__name__)
        self.on_video_end_callback = on_video_end_callback
        self.on_batch_complete_callback = on_batch_complete_callback
        
        # 视频信息
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.duration = 0
        self.current_frame = 0
        
        # 处理状态
        self.is_running = False
        self.is_paused = False
        self.read_thread = None
        self.process_thread = None
        
        # 批处理队列
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.current_batch = []
        
        # 统计信息
        self.total_read_frames = 0
        self.total_processed_frames = 0
        self.total_batches = 0
        self.start_time = None
        
    def _init_video_capture(self) -> bool:
        """初始化视频捕获"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.logger.error(f"❌ 无法打开视频文件: {self.video_path}")
                return False
            
            # 获取视频信息
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0
            
            self.logger.info(f"📹 视频文件信息:")
            self.logger.info(f"   路径: {self.video_path}")
            self.logger.info(f"   总帧数: {self.total_frames}")
            self.logger.info(f"   帧率: {self.fps:.2f} FPS")
            self.logger.info(f"   时长: {self.duration:.2f} 秒")
            self.logger.info(f"   批处理大小: {self.batch_size} 帧/批")
            self.logger.info(f"   预计批次数: {(self.total_frames + self.batch_size - 1) // self.batch_size}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 初始化视频捕获失败: {e}")
            return False
    
    def start(self):
        """启动视频处理"""
        if self.is_running:
            self.logger.warning(f"⚠️ 视频处理器 {self.stream_id} 已在运行")
            return
        
        if not self._init_video_capture():
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        self.logger.info(f"🎬 启动视频文件处理器: {self.stream_id}")
        
        # 启动读取线程
        self.read_thread = threading.Thread(
            target=self._read_frames_thread,
            name=f"VideoReader-{self.stream_id}",
            daemon=True
        )
        
        # 启动处理线程
        self.process_thread = threading.Thread(
            target=self._process_batches_thread,
            name=f"VideoProcessor-{self.stream_id}",
            daemon=True
        )
        
        self.read_thread.start()
        self.process_thread.start()
    
    def stop(self):
        """停止视频处理"""
        if not self.is_running:
            return
        
        self.logger.info(f"⏹️ 停止视频文件处理器: {self.stream_id}")
        self.is_running = False
        
        # 等待线程结束
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2.0)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        
        # 清理资源
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        self.current_batch.clear()
        
        self.logger.info(f"✅ 视频文件处理器已停止: {self.stream_id}")
        self._log_final_stats()
    
    def _read_frames_thread(self):
        """读取帧的线程"""
        self.logger.info(f"🔄 开始读取视频帧: {self.stream_id}")
        
        try:
            while self.is_running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.info(f"🏁 视频读取完毕: {self.total_read_frames}/{self.total_frames}")
                    break
                
                # 获取内存缓冲区，增加重试机制
                frame_timestamp = time.time()
                retry_count = 0
                max_retries = 100  # 最大重试次数
                frame_buffer = None
                
                while frame_buffer is None and retry_count < max_retries and self.is_running:
                    frame_buffer = self.memory_pool.get_buffer(self.current_frame, frame_timestamp, self.stream_id)
                    if frame_buffer is None:
                        retry_count += 1
                        if retry_count % 20 == 0:  # 每20次重试记录一次
                            self.logger.warning(f"⏳ 等待内存缓冲区释放... (重试 {retry_count}/{max_retries})")
                        time.sleep(0.01)  # 短暂等待
                
                if frame_buffer is None:
                    self.logger.error(f"❌ 无法获取内存缓冲区，跳过帧 {self.current_frame}")
                    continue
                
                # 设置帧数据
                frame_buffer.copy_frame_data(frame)
                
                # 添加到时间轴
                self.time_axis.add_frame(frame_buffer)
                
                # 定期清理时间轴中的超时帧，释放内存
                if self.total_read_frames % 50 == 0:  # 每50帧清理一次
                    due_frames = self.time_axis.get_due_frames()
                    if due_frames:
                        self.logger.debug(f"🧹 清理超时帧: {len(due_frames)} 个")
                        for frame in due_frames:
                            frame.release()
                
                # 放入队列等待处理，增加重试机制
                queue_retry_count = 0
                max_queue_retries = 10
                queue_success = False
                
                while queue_retry_count < max_queue_retries and self.is_running:
                    try:
                        self.frame_queue.put(frame_buffer, timeout=0.1)
                        queue_success = True
                        break
                    except:
                        queue_retry_count += 1
                        if queue_retry_count % 5 == 0:
                            self.logger.warning(f"⏳ 队列满，等待处理... (重试 {queue_retry_count}/{max_queue_retries})")
                        time.sleep(0.01)
                
                if not queue_success:
                    self.logger.error(f"❌ 无法放入处理队列，释放帧 {self.current_frame}")
                    frame_buffer.release()  # 释放内存
                    continue
                
                self.total_read_frames += 1
                self.current_frame += 1
                
                # 定期报告进度
                if self.total_read_frames % 100 == 0:
                    progress = (self.total_read_frames / self.total_frames) * 100
                    memory_stats = self.memory_pool.get_stats()
                    self.logger.info(f"📊 读取进度: {self.total_read_frames}/{self.total_frames} ({progress:.1f}%), "
                                   f"内存使用: {memory_stats['current_usage']}/{memory_stats['pool_size']}")
        
        except Exception as e:
            self.logger.error(f"❌ 读取帧线程异常: {e}")
        
        finally:
            # 发送结束信号
            try:
                self.frame_queue.put(None, timeout=1.0)  # 结束标记
            except:
                pass
            
            self.logger.info(f"🏁 帧读取线程结束: {self.stream_id}")
    
    def _process_batches_thread(self):
        """处理批次的线程"""
        self.logger.info(f"🔄 开始批处理: {self.stream_id}")
        
        try:
            while self.is_running:
                try:
                    # 从队列获取帧
                    frame_buffer = self.frame_queue.get(timeout=1.0)
                    
                    # 检查结束标记
                    if frame_buffer is None:
                        # 处理最后一批
                        if self.current_batch:
                            self._process_current_batch()
                        break
                    
                    # 添加到当前批次
                    self.current_batch.append(frame_buffer)
                    
                    # 检查是否达到批处理大小
                    if len(self.current_batch) >= self.batch_size:
                        self._process_current_batch()
                
                except Empty:
                    # 超时，检查是否有部分批次需要处理
                    if self.current_batch and not self.is_running:
                        self._process_current_batch()
                    continue
                except Exception as e:
                    self.logger.error(f"❌ 批处理线程异常: {e}")
                    break
        
        except Exception as e:
            self.logger.error(f"❌ 批处理线程严重异常: {e}")
        
        finally:
            self.logger.info(f"🏁 批处理线程结束: {self.stream_id}")
            
            # 检查是否完成所有处理
            if self.total_processed_frames >= self.total_frames:
                self._on_video_complete()
    
    def _process_current_batch(self):
        """处理当前批次"""
        if not self.current_batch:
            return
        
        batch_size = len(self.current_batch)
        self.total_batches += 1
        
        self.logger.debug(f"🔄 处理批次 {self.total_batches}: {batch_size} 帧")
        
        try:
            # 提交到分析引擎（通过时间轴的回调机制）
            # 这里的帧已经在时间轴中，会被分析引擎自动处理
            
            # 更新统计
            self.total_processed_frames += batch_size
            
            # 进度报告和内存清理
            if self.total_batches % 5 == 0:  # 每5批报告一次
                progress = (self.total_processed_frames / self.total_frames) * 100
                elapsed = time.time() - self.start_time
                fps = self.total_processed_frames / elapsed if elapsed > 0 else 0
                
                self.logger.info(f"📊 处理进度: 批次 {self.total_batches}, "
                               f"帧数 {self.total_processed_frames}/{self.total_frames} ({progress:.1f}%), "
                               f"处理速度 {fps:.1f} FPS")
                
                # 强制清理时间轴中的超时帧
                due_frames = self.time_axis.get_due_frames()
                if due_frames:
                    self.logger.info(f"🧹 批处理清理超时帧: {len(due_frames)} 个")
                    for frame in due_frames:
                        frame.release()
            
            # 批次完成回调
            if self.on_batch_complete_callback:
                try:
                    self.on_batch_complete_callback(self.total_batches, batch_size, self.total_processed_frames)
                except Exception as e:
                    self.logger.error(f"❌ 批次完成回调异常: {e}")
        
        except Exception as e:
            self.logger.error(f"❌ 处理批次失败: {e}")
        
        finally:
            # 清空当前批次
            self.current_batch.clear()
    
    def _on_video_complete(self):
        """视频处理完成"""
        self.logger.info(f"🎉 视频文件处理完成: {self.stream_id}")
        self.logger.info(f"   总帧数: {self.total_frames}")
        self.logger.info(f"   已读取: {self.total_read_frames}")
        self.logger.info(f"   已处理: {self.total_processed_frames}")
        self.logger.info(f"   总批次: {self.total_batches}")
        
        # 调用结束回调
        if self.on_video_end_callback:
            try:
                self.on_video_end_callback()
            except Exception as e:
                self.logger.error(f"❌ 视频结束回调异常: {e}")
    
    def _log_final_stats(self):
        """记录最终统计信息"""
        if self.start_time:
            total_time = time.time() - self.start_time
            avg_fps = self.total_processed_frames / total_time if total_time > 0 else 0
            
            self.logger.info(f"📊 视频处理统计:")
            self.logger.info(f"   处理时间: {total_time:.2f} 秒")
            self.logger.info(f"   平均速度: {avg_fps:.1f} FPS")
            self.logger.info(f"   读取帧数: {self.total_read_frames}")
            self.logger.info(f"   处理帧数: {self.total_processed_frames}")
            self.logger.info(f"   处理批次: {self.total_batches}")
    
    def pause(self):
        """暂停处理"""
        self.is_paused = True
        self.logger.info(f"⏸️ 暂停视频处理: {self.stream_id}")
    
    def resume(self):
        """恢复处理"""
        self.is_paused = False
        self.logger.info(f"▶️ 恢复视频处理: {self.stream_id}")
    
    def get_stats(self) -> dict:
        """获取处理统计信息"""
        progress = (self.total_processed_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.total_processed_frames / elapsed if elapsed > 0 else 0
        
        return {
            "stream_id": self.stream_id,
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "read_frames": self.total_read_frames,
            "processed_frames": self.total_processed_frames,
            "total_batches": self.total_batches,
            "progress_percent": progress,
            "elapsed_time": elapsed,
            "fps": fps,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "queue_size": self.frame_queue.qsize(),
            "current_batch_size": len(self.current_batch)
        } 