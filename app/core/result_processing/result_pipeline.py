#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
结果处理管道 - 统一管理多个结果处理器
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.core.zero_copy.frame_buffer import FrameBuffer
from app.models.analysis_result import AnalysisResult
from app.core.zero_copy.time_axis import TimeAxis
from .detection_filter import DetectionFilter, FilterPipeline
from .callback_processor import CallbackProcessor
from .storage_processor import StorageProcessor
# 旧的处理器已被新的服务架构替代
# from .video_processor import VideoProcessor
# from .video_cache_processor import VideoCacheProcessor


class ResultProcessingPipeline:
    """结果处理管道 - 协调多个处理器"""
    
    def __init__(
        self,
        task_id: int,
        task_config: Dict[str, Any],
        time_axis_manager: TimeAxis,  # 实际上是TimeAxis对象
        output_dir: str,
        video_playback_service: Optional[object] = None,  # 视频回放服务
        video_cache_service: Optional[object] = None,  # 视频缓存服务
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化结果处理管道
        
        Args:
            task_id: 任务ID
            task_config: 任务配置
            time_axis_manager: 时间轴对象
            output_dir: 输出目录
            video_playback_service: 视频回放服务（可选）
            logger: 日志记录器
        """
        self.task_id = task_id
        self.task_config = task_config
        self.time_axis_manager = time_axis_manager  # 实际存储的是TimeAxis对象
        self.output_dir = output_dir
        self.video_playback_service = video_playback_service
        self.video_cache_service = video_cache_service  # 视频缓存服务
        self.logger = logger or logging.getLogger(__name__)
        
        # 创建任务专用日志记录器
        self.file_logger = self._create_file_logger()
        
        # 处理器列表
        self.processors = []
        
        # 处理状态
        self.is_running = False
        self.start_time = None
        
        # 统计信息
        self.stats = {
            "total_results": 0,
            "successful_results": 0,
            "failed_results": 0,
            "start_time": 0,
            "processing_times": []
        }
        
        # 初始化处理器
        self._initialize_processors()
        
        self.file_logger.info(f"🏭 结果处理管道初始化完成")
        self.file_logger.info(f"   任务ID: {task_id}")
        self.file_logger.info(f"   处理器数量: {len(self.processors)}")
        
        # 如果有视频回放服务，记录在日志中
        if self.video_playback_service:
            self.file_logger.info(f"   视频回放服务: 已配置")
        else:
            self.file_logger.info(f"   视频回放服务: 未配置")
    
    def _create_file_logger(self):
        """设置文件日志记录器"""
        try:
            # 创建日志目录
            log_dir = Path("storage/logs/result_processing")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建专门的文件日志记录器
            file_logger = logging.getLogger(f"result_pipeline_task_{self.task_id}")
            file_logger.setLevel(logging.DEBUG)
            
            # 清除现有的处理器
            file_logger.handlers.clear()
            
            # 创建文件处理器
            log_file = log_dir / f"task_{self.task_id}_result_pipeline.log"
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # 设置详细的日志格式
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # 添加处理器
            file_logger.addHandler(file_handler)
            
            # 防止日志传播到根日志记录器（避免控制台输出）
            file_logger.propagate = False
            
            file_logger.info("="*80)
            file_logger.info(f"结果处理管道日志启动 - 任务ID: {self.task_id}")
            file_logger.info(f"任务配置: {self.task_config}")
            file_logger.info("="*80)
            
            return file_logger
            
        except Exception as e:
            # 确保有一个有效的logger对象
            if self.logger:
                self.logger.error(f"❌ 设置文件日志记录器失败: {e}")
                return self.logger
            else:
                # 如果self.logger也为None，创建一个默认的logger
                fallback_logger = logging.getLogger(__name__)
                fallback_logger.error(f"❌ 设置文件日志记录器失败: {e}")
                return fallback_logger

    def _initialize_processors(self):
        """初始化处理器"""
        self.file_logger.info("🔧 开始初始化处理器...")
        self.processors = []
        
        # 获取分析类型和回放配置
        analysis_type = self.task_config.get("analysis_type", 1)
        playback_duration = self.task_config.get("playback_duration", 0)
        
        self.file_logger.info(f"📋 任务配置检查: analysis_type={analysis_type}, playback_duration={playback_duration}")
        
        # 1. 回调处理器（如果配置了回调URL）
        callback_url = self.task_config.get("callback_url")
        if callback_url:
            self.file_logger.info("📞 创建回调处理器...")
            callback_processor = CallbackProcessor(
                callback_url=callback_url,
                task_id=self.task_id,
                logger=self.logger
            )
            self.processors.append(callback_processor)
            self.file_logger.info("✅ 回调处理器已创建")
        else:
            self.file_logger.info("⏭️ 跳过回调处理器：未配置回调URL")
        
        # 2. 存储处理器（总是创建）
        self.file_logger.info("💾 创建存储处理器...")
        storage_processor = StorageProcessor(
            save_images=self.task_config.get("save_images", True),
            save_metadata=self.task_config.get("save_result", True),
            draw_boxes=True,
            image_quality=95,
            logger=self.logger
        )
        self.processors.append(storage_processor)
        self.file_logger.info("✅ 存储处理器已创建")
        
        # 3. 视频回放处理 - 现在由VideoPlaybackService负责，不在这里创建处理器
        should_create_video_processor = (
            analysis_type in [2, 3] and  # 视频或流分析
            playback_duration > 0        # 配置了回放时长
        )
        
        if should_create_video_processor:
            self.file_logger.info("🎬 视频回放功能已启用:")
            self.file_logger.info(f"   - 分析类型: {analysis_type} ({'视频分析' if analysis_type == 2 else '流分析'})")
            self.file_logger.info(f"   - 回放时长: {playback_duration}秒")
            if self.video_playback_service:
                self.file_logger.info("   - 视频回放服务: 已配置，将自动处理分析结果")
            else:
                self.file_logger.info("   - 视频回放服务: 未配置，视频回放功能将不可用")
        else:
            self.file_logger.info("⏭️ 视频回放功能已禁用:")
            if analysis_type not in [2, 3]:
                self.file_logger.info(f"   - 分析类型不符合: {analysis_type} (需要2或3)")
            if playback_duration <= 0:
                self.file_logger.info(f"   - 回放时长不符合: {playback_duration}秒 (需要>0)")
        
        self.file_logger.info(f"🔧 处理器初始化完成，共创建 {len(self.processors)} 个处理器")
    
    async def start(self):
        """启动处理管道"""
        try:
            self.file_logger.info("🚀 启动结果处理管道...")
            
            # 启动需要异步初始化的处理器
            for processor in self.processors:
                processor_name = getattr(processor, 'name', processor.__class__.__name__)
                self.file_logger.info(f"🔄 启动处理器: {processor_name}")
                if hasattr(processor, 'start'):
                    await processor.start()
                    self.file_logger.info(f"✅ {processor_name}处理器已启动")
                else:
                    self.file_logger.info(f"ℹ️ {processor_name}处理器无需异步启动")
            
            self.file_logger.info("🚀 结果处理管道启动完成")
            self.logger.info("🚀 结果处理管道已启动")
            
        except Exception as e:
            error_msg = f"❌ 启动结果处理管道失败: {e}"
            self.file_logger.error(error_msg)
            self.logger.error(error_msg)
            raise
    
    async def stop(self):
        """停止处理管道"""
        try:
            self.file_logger.info("⏹️ 停止结果处理管道...")
            
            # 停止所有处理器
            for processor in self.processors:
                processor_name = getattr(processor, 'name', processor.__class__.__name__)
                try:
                    self.file_logger.info(f"⏹️ 停止处理器: {processor_name}")
                    if hasattr(processor, 'stop'):
                        await processor.stop()
                        self.file_logger.info(f"✅ {processor_name}处理器已停止")
                    elif hasattr(processor, 'cleanup'):
                        processor.cleanup()
                        self.file_logger.info(f"🧹 {processor_name}处理器已清理")
                    else:
                        self.file_logger.info(f"ℹ️ {processor_name}处理器无需特殊停止操作")
                except Exception as e:
                    error_msg = f"❌ 停止{processor_name}处理器失败: {e}"
                    self.file_logger.error(error_msg)
                    self.logger.error(error_msg)
            
            self.file_logger.info("⏹️ 结果处理管道停止完成")
            self.logger.info("⏹️ 结果处理管道已停止")
            
        except Exception as e:
            error_msg = f"❌ 停止结果处理管道失败: {e}"
            self.file_logger.error(error_msg)
            self.logger.error(error_msg)
            raise
    
    def process_result(self, frame_buffer: FrameBuffer, task_id: int = None):
        """
        处理分析结果 - 兼容原有接口
        
        Args:
            frame_buffer: 包含分析结果的帧缓冲区
            task_id: 任务ID
        """
        start_time = time.time()
        self.file_logger.info("-"*60)
        self.file_logger.info(f"📥 接收到分析结果")
        self.file_logger.info(f"   帧ID: {getattr(frame_buffer, 'frame_id', 'N/A')}")
        self.file_logger.info(f"   时间戳: {getattr(frame_buffer, 'timestamp', 'N/A')}")
        self.file_logger.info(f"   流ID: {getattr(frame_buffer, 'stream_id', 'N/A')}")
        self.file_logger.info(f"   任务ID: {task_id}")
        
        try:
            # 保存frame_buffer引用（处理器需要）
            self._current_frame_buffer = frame_buffer
            
            # 检查是否有检测结果
            analysis_result = getattr(frame_buffer, 'analysis_result', None)
            if analysis_result and hasattr(analysis_result, 'detections') and analysis_result.detections:
                self.file_logger.info(f"🎯 检测到 {len(analysis_result.detections)} 个目标")
                
                # 分发到所有处理器（存储、回调等）
                self._distribute_to_processors_sync(frame_buffer, self.processors, task_id)
                
                # 【新架构】如果有视频回放服务，直接将分析结果提交到队列
                if self.video_playback_service:
                    try:
                        # 【修复】将Detection对象转换为字典格式，并添加分辨率信息
                        detections_dict = []
                        for detection in analysis_result.detections:
                            detections_dict.append(detection.to_dict())
                        
                        # 【关键修复】获取图像分辨率信息
                        image_shape = getattr(analysis_result, 'image_shape', None) or getattr(frame_buffer, 'image_shape', None)
                        
                        # 从视频缓存服务获取分辨率信息作为补充
                        original_resolution = None
                        target_resolution = None
                        if self.video_cache_service:
                            original_resolution = getattr(self.video_cache_service, 'original_resolution', None)
                            target_resolution = getattr(self.video_cache_service, 'target_resolution', None)
                        
                        analysis_data = {
                            "frame_id": getattr(frame_buffer, 'frame_id', 0),
                            "timestamp": getattr(frame_buffer, 'timestamp', 0.0),
                            "detections": detections_dict,  # 使用转换后的字典列表
                            "model_name": getattr(analysis_result, 'model_name', 'unknown'),
                            "confidence": getattr(analysis_result, 'confidence', 0.0),
                            "stream_id": getattr(frame_buffer, 'stream_id', 'default'),
                            # 【关键修复】添加分辨率信息
                            "image_shape": image_shape,  # 原始图像尺寸
                            "analysis_resolution": original_resolution,  # 分析时的分辨率
                            "target_resolution": target_resolution  # 目标分辨率
                        }
                        
                        self.file_logger.info(f"📊 准备传递的分辨率信息:")
                        self.file_logger.info(f"   image_shape: {image_shape}")
                        self.file_logger.info(f"   analysis_resolution: {original_resolution}")
                        self.file_logger.info(f"   target_resolution: {target_resolution}")
                        
                        # 将分析结果添加到视频回放队列
                        self.video_playback_service.add_analysis_result(analysis_data)
                        self.file_logger.info(f"📋 分析结果已提交到视频回放队列")
                        
                    except Exception as e:
                        self.file_logger.error(f"❌ 提交分析结果到视频回放队列失败: {e}")
                        import traceback
                        self.file_logger.error(f"详细错误信息: {traceback.format_exc()}")
                else:
                    self.file_logger.debug("📋 无视频回放服务，跳过视频回放处理")
                
                # 【新增】将分析结果同步到视频缓存服务
                if self.video_cache_service:
                    try:
                        frame_id = getattr(frame_buffer, 'frame_id', 0)
                        timestamp = getattr(frame_buffer, 'timestamp', 0.0)
                        
                        # 准备分析数据（与视频回放队列格式一致）
                        cache_analysis_data = {
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "detections": detections_dict,  # 使用已转换的字典列表
                            "model_name": getattr(analysis_result, 'model_name', 'unknown'),
                            "confidence": getattr(analysis_result, 'confidence', 0.0),
                            "stream_id": getattr(frame_buffer, 'stream_id', 'default')
                        }
                        
                        # 将分析结果添加到视频缓存
                        self.video_cache_service.add_analysis_result(frame_id, timestamp, cache_analysis_data)
                        self.file_logger.info(f"📊 分析结果已同步到视频缓存服务")
                        
                    except Exception as e:
                        self.file_logger.error(f"❌ 同步分析结果到视频缓存服务失败: {e}")
                        import traceback
                        self.file_logger.error(f"详细错误信息: {traceback.format_exc()}")
                else:
                    self.file_logger.debug("📊 无视频缓存服务，跳过分析结果缓存")
            else:
                self.file_logger.info("ℹ️ 无检测结果，跳过结果处理器分发")
            
            # 更新统计信息
            self.stats["total_results"] += 1
            self.stats["successful_results"] += 1
            
            process_time = (time.time() - start_time) * 1000
            self.stats["processing_times"].append(process_time)
            
            self.file_logger.info(f"✅ 结果处理完成，耗时: {process_time:.2f}ms")
            
        except Exception as e:
            self.stats["failed_results"] += 1
            self.file_logger.error(f"❌ 结果处理失败: {e}")
            raise
        finally:
            # 清理引用
            self._current_frame_buffer = None
    
    def _distribute_to_processors_sync(self, frame_buffer: FrameBuffer, processors: List, task_id: int):
        """同步分发结果到所有处理器"""
        start_time = time.time()
        self.file_logger.info("📤 开始同步分发结果到处理器...")
        self.file_logger.info(f"   处理器数量: {len(processors)}")
        
        # 安全获取检测结果数量
        analysis_result = getattr(frame_buffer, 'analysis_result', None)
        if analysis_result:
            if hasattr(analysis_result, 'detections'):
                detections_count = len(analysis_result.detections) if analysis_result.detections else 0
            elif isinstance(analysis_result, dict):
                detections_count = len(analysis_result.get('detections', []))
            else:
                detections_count = 0
        else:
            detections_count = 0
        self.file_logger.info(f"   检测结果数量: {detections_count}")
        
        try:
            # 同步处理所有处理器
            for processor in processors:
                processor_name = getattr(processor, 'name', processor.__class__.__name__)
                self.file_logger.info(f"🔄 处理处理器: {processor_name}")
                
                try:
                    # 先调用基础的 process 方法
                    processor_result = self._safe_processor_call(
                        processor, processor_name, frame_buffer, self.task_config
                    )
                    self.file_logger.info(f"✅ 处理器 {processor_name} 基础调用完成: {processor_result}")
                    
                    # 【新架构】视频处理现在由VideoPlaybackService负责，这里不需要特殊处理
                    # 所有处理器都只需要调用基础的process方法
                    
                except Exception as e:
                    self.file_logger.error(f"❌ 处理器 {processor_name} 执行失败: {e}")
                    self.file_logger.exception("详细错误信息:")
            
            distribution_time = (time.time() - start_time) * 1000
            self.file_logger.info(f"✅ 所有处理器分发完成，总耗时: {distribution_time:.2f}ms")
            
        except Exception as e:
            distribution_time = (time.time() - start_time) * 1000
            error_msg = f"❌ 分发处理器失败: {e}"
            self.file_logger.error(f"{error_msg}，耗时: {distribution_time:.2f}ms")
            self.file_logger.exception("详细错误信息:")
    
    def _safe_processor_call(self, processor, name: str, frame_buffer: FrameBuffer, task_config: Dict[str, Any]):
        """安全调用处理器"""
        call_start = time.time()
        self.file_logger.info(f"🔧 开始调用处理器: {name}")
        
        try:
            result = processor.process(frame_buffer, task_config)
            call_time = (time.time() - call_start) * 1000
            self.file_logger.info(f"✅ 处理器 {name} 调用成功，耗时: {call_time:.2f}ms，结果: {result}")
            return result
            
        except Exception as e:
            call_time = (time.time() - call_start) * 1000
            error_msg = f"❌ 处理器 {name} 调用失败: {e}"
            self.file_logger.error(f"{error_msg}，耗时: {call_time:.2f}ms")
            self.file_logger.exception("详细错误信息:")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = {
            "task_id": self.task_id,
            "processor_count": len(self.processors),
            "processor_names": [getattr(processor, 'name', processor.__class__.__name__) for processor in self.processors],
            "config": {
                "save_result": self.task_config.get("save_result", False),
                "save_images": self.task_config.get("save_images", False),
                "callback_urls": self.task_config.get("callback_urls", []),
                "analysis_type": self.task_config.get("analysis_type", 1)
            }
        }
        
        # 获取各处理器的统计信息
        for name, processor in self.processors:
            if hasattr(processor, 'get_statistics'):
                stats[f"{name}_stats"] = processor.get_statistics()
        
        self.file_logger.info(f"📊 处理管道统计: {stats}")
        return stats
    
    def log_statistics(self):
        """记录统计信息"""
        stats = self.get_statistics()
        self.file_logger.info("📊 处理管道统计信息:")
        for key, value in stats.items():
            self.file_logger.info(f"   {key}: {value}")
        
        self.logger.info(f"📊 任务 {self.task_id} 结果处理统计: "
                        f"{stats['processor_count']} 个处理器, "
                        f"配置: save_result={stats['config']['save_result']}, "
                        f"save_images={stats['config']['save_images']}") 