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
from app.core.memory.time_axis_manager import TimeAxisManager
from .detection_filter import DetectionFilter, FilterPipeline
from .callback_processor import CallbackProcessor
from .storage_processor import StorageProcessor
from .video_processor import VideoProcessor


class ResultProcessingPipeline:
    """结果处理管道 - 协调多个处理器"""
    
    def __init__(
        self,
        task_id: int,
        task_config: Dict[str, Any],
        time_axis_manager,
        output_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化结果处理管道
        
        Args:
            task_id: 任务ID
            task_config: 任务配置
            time_axis_manager: 时间轴管理器
            output_dir: 输出目录
            logger: 日志记录器
        """
        self.task_id = task_id
        self.task_config = task_config
        self.time_axis_manager = time_axis_manager
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # 处理器列表
        self.processors = []
        
        # 运行时状态
        self._current_frame_buffer = None
        
        # 设置专门的文件日志记录器
        self._setup_file_logger()
        
        # 初始化过滤器
        self.filter_pipeline = self._create_filter_pipeline()
        
        # 初始化处理器
        self._initialize_processors()
    
    def _setup_file_logger(self):
        """设置文件日志记录器"""
        try:
            # 创建日志目录
            log_dir = Path("storage/logs/result_processing")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建专门的文件日志记录器
            self.file_logger = logging.getLogger(f"result_pipeline_task_{self.task_id}")
            self.file_logger.setLevel(logging.DEBUG)
            
            # 清除现有的处理器
            self.file_logger.handlers.clear()
            
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
            self.file_logger.addHandler(file_handler)
            
            # 防止日志传播到根日志记录器（避免控制台输出）
            self.file_logger.propagate = False
            
            self.file_logger.info("="*80)
            self.file_logger.info(f"结果处理管道日志启动 - 任务ID: {self.task_id}")
            self.file_logger.info(f"任务配置: {self.task_config}")
            self.file_logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"❌ 设置文件日志记录器失败: {e}")
            # 如果文件日志器设置失败，使用主日志记录器
            self.file_logger = self.logger

    def _create_filter_pipeline(self) -> FilterPipeline:
        """创建过滤器管道"""
        self.file_logger.info("🔧 创建过滤器管道...")
        
        filter_pipeline = FilterPipeline(logger=self.file_logger)
        
        # 添加类别过滤器
        target_classes = self.task_config.get("target_classes", [])
        if target_classes and any(target_classes):  # 如果有有效的目标类别
            self.file_logger.info(f"🎯 添加类别过滤器: {target_classes}")
            class_filter = DetectionFilter(target_classes=target_classes, logger=self.file_logger)
            filter_pipeline.add_filter(class_filter)
        else:
            self.file_logger.info("🎯 未配置类别过滤器，接受所有检测结果")
        
        # 添加处理器回调
        filter_pipeline.add_processor(self._process_filtered_result)
        
        self.file_logger.info("✅ 过滤器管道创建完成")
        return filter_pipeline
    
    def _initialize_processors(self):
        """初始化处理器"""
        self.file_logger.info("🔧 开始初始化处理器...")
        self.processors = []
        
        # 1. 回调处理器
        callback_urls = self.task_config.get("callback_urls")
        self.file_logger.info(f"📞 检查回调配置: {callback_urls}")
        
        if callback_urls and any(url.strip() for url in callback_urls if url):
            self.file_logger.info(f"📞 创建回调处理器: {len(callback_urls)} 个URL")
            callback_processor = CallbackProcessor(
                callback_urls=callback_urls,
                callback_interval=self.task_config.get("callback_interval", 0),
                logger=self.file_logger
            )
            self.processors.append(("callback", callback_processor))
            self.file_logger.info("✅ 回调处理器已添加")
        else:
            self.file_logger.info("⏭️ 跳过回调处理器（未配置有效URL）")
        
        # 2. 存储处理器
        save_result = self.task_config.get("save_result", True)
        save_images = self.task_config.get("save_images", False)
        
        self.file_logger.info(f"💾 检查存储配置: save_result={save_result}, save_images={save_images}")
        
        if save_result or save_images:
            self.file_logger.info("💾 创建存储处理器...")
            storage_processor = StorageProcessor(
                save_images=save_images,
                save_metadata=save_result,
                draw_boxes=True,
                logger=self.file_logger
            )
            self.processors.append(("storage", storage_processor))
            self.file_logger.info("✅ 存储处理器已添加")
        else:
            self.file_logger.info("⏭️ 跳过存储处理器（未启用保存功能）")
        
        # 3. 视频处理器
        analysis_type = self.task_config.get("analysis_type", 1)
        playback_duration = self.task_config.get("playback_duration", 0)
        
        self.file_logger.info(f"🎬 检查视频处理器配置: analysis_type={analysis_type}, playback_duration={playback_duration}")
        
        if analysis_type in [2, 3] and playback_duration >= 5:
            self.file_logger.info("🎬 创建视频处理器...")
            video_processor = VideoProcessor(
                output_dir=f"{self.output_dir}/videos",
                playback_duration=playback_duration,
                time_axis_manager=self.time_axis_manager,
                analysis_type=analysis_type,
                logger=self.file_logger
            )
            self.processors.append(("video", video_processor))
            self.file_logger.info("✅ 视频处理器已添加")
        else:
            self.file_logger.info("⏭️ 跳过视频处理器（条件不满足）")
        
        processor_names = [name for name, _ in self.processors]
        self.file_logger.info(f"🔧 处理器初始化完成: {processor_names} (共{len(self.processors)}个)")
        
        if not self.processors:
            self.file_logger.warning("⚠️ 警告：没有任何处理器被创建！")
        
        # 在控制台也输出关键信息
        self.logger.info(
            f"🔧 已初始化处理器: "
            f"{', '.join(processor_names)}"
        )
    
    async def start(self):
        """启动处理管道"""
        try:
            self.file_logger.info("🚀 启动结果处理管道...")
            
            # 启动需要异步初始化的处理器
            for name, processor in self.processors:
                self.file_logger.info(f"🔄 启动处理器: {name}")
                if hasattr(processor, 'start'):
                    await processor.start()
                    self.file_logger.info(f"✅ {name}处理器已启动")
                else:
                    self.file_logger.info(f"ℹ️ {name}处理器无需异步启动")
            
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
            for name, processor in self.processors:
                try:
                    self.file_logger.info(f"⏹️ 停止处理器: {name}")
                    if hasattr(processor, 'stop'):
                        await processor.stop()
                        self.file_logger.info(f"✅ {name}处理器已停止")
                    elif hasattr(processor, 'cleanup'):
                        processor.cleanup()
                        self.file_logger.info(f"🧹 {name}处理器已清理")
                    else:
                        self.file_logger.info(f"ℹ️ {name}处理器无需特殊停止操作")
                except Exception as e:
                    error_msg = f"❌ 停止{name}处理器失败: {e}"
                    self.file_logger.error(error_msg)
                    self.logger.error(error_msg)
            
            self.file_logger.info("⏹️ 结果处理管道停止完成")
            self.logger.info("⏹️ 结果处理管道已停止")
            
        except Exception as e:
            error_msg = f"❌ 停止结果处理管道失败: {e}"
            self.file_logger.error(error_msg)
            self.logger.error(error_msg)
    
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
            # 检查分析结果 - FrameBuffer.analysis_results是字典格式
            analysis_results_dict = getattr(frame_buffer, 'analysis_results', {})
            self.file_logger.info(f"🔍 分析结果字典键数量: {len(analysis_results_dict)}")
            
            if not analysis_results_dict:
                self.file_logger.warning("⚠️ FrameBuffer中没有分析结果，跳过处理")
                return
            
            # 记录分析结果详情
            for key, result in analysis_results_dict.items():
                self.file_logger.info(f"   结果键 {key}: {type(result)}")
            
            # 从frame_buffer提取AnalysisResult
            self.file_logger.info("🔄 开始提取AnalysisResult...")
            analysis_result = self._extract_analysis_result_from_frame_buffer(frame_buffer)
            
            if not analysis_result:
                self.file_logger.warning("⚠️ 无法提取AnalysisResult，跳过处理")
                return
            
            self.file_logger.info(f"✅ AnalysisResult提取成功: 检测数量={len(analysis_result.detections)}")
            
            # 保存frame_buffer引用（处理器需要）
            self._current_frame_buffer = frame_buffer
            
            # 通过过滤器管道处理
            self.file_logger.info("🎯 开始过滤器管道处理...")
            self.filter_pipeline.process(analysis_result)
            
            processing_time = (time.time() - start_time) * 1000
            self.file_logger.info(f"✅ 分析结果处理完成，耗时: {processing_time:.2f}ms")
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_msg = f"❌ 处理结果失败: {e}"
            self.file_logger.error(f"{error_msg}，耗时: {processing_time:.2f}ms")
            self.file_logger.exception("详细错误信息:")
            self.logger.error(error_msg)
    
    def _extract_analysis_result_from_frame_buffer(self, frame_buffer: FrameBuffer) -> Optional[AnalysisResult]:
        """从FrameBuffer提取AnalysisResult"""
        try:
            self.file_logger.info("🔍 开始从FrameBuffer提取AnalysisResult...")
            
            if not frame_buffer.analysis_results:
                self.file_logger.warning("⚠️ FrameBuffer.analysis_results为空")
                return None
            
            # 从分析结果字典获取主要结果（通常键为"analysis_result"）
            main_result = None
            if "analysis_result" in frame_buffer.analysis_results:
                main_result = frame_buffer.analysis_results["analysis_result"]
                self.file_logger.info("📋 使用'analysis_result'键的结果")
            else:
                # 如果没有"analysis_result"键，使用第一个结果
                first_key = list(frame_buffer.analysis_results.keys())[0]
                main_result = frame_buffer.analysis_results[first_key]
                self.file_logger.info(f"📋 使用'{first_key}'键的结果")
            
            self.file_logger.info(f"📋 主要分析结果: {type(main_result)}")
            
            # 构建检测列表
            detections = []
            detection_count = 0
            
            if "model_results" in main_result:
                self.file_logger.info("🤖 发现model_results结构")
                for model_code, model_result in main_result["model_results"].items():
                    model_detections = model_result.get("detections", [])
                    self.file_logger.info(f"   模型 {model_code}: {len(model_detections)} 个检测")
                    
                    for det_data in model_detections:
                        try:
                            from app.models.analysis_result import Detection
                            detection = Detection(
                                class_id=det_data.get("class_id", 0),
                                class_name=det_data.get("class_name", "unknown"),
                                confidence=det_data.get("confidence", 0.0),
                                bbox=[
                                    det_data["bbox"]["x1"],
                                    det_data["bbox"]["y1"], 
                                    det_data["bbox"]["x2"],
                                    det_data["bbox"]["y2"]
                                ],
                                area=det_data.get("area")
                            )
                            detections.append(detection)
                            detection_count += 1
                            self.file_logger.info(f"     检测 {detection_count}: {detection.class_name} ({detection.confidence:.2f})")
                        except Exception as e:
                            self.file_logger.error(f"❌ 创建Detection对象失败: {e}")
            else:
                self.file_logger.info("📋 使用直接检测结构")
                direct_detections = main_result.get("detections", [])
                self.file_logger.info(f"   直接检测: {len(direct_detections)} 个")
                
                for det_data in direct_detections:
                    try:
                        from app.models.analysis_result import Detection
                        detection = Detection(
                            class_id=det_data.get("class_id", 0),
                            class_name=det_data.get("class_name", "unknown"),
                            confidence=det_data.get("confidence", 0.0),
                            bbox=det_data.get("bbox", [0, 0, 0, 0]),
                            area=det_data.get("area")
                        )
                        detections.append(detection)
                        detection_count += 1
                        self.file_logger.info(f"     检测 {detection_count}: {detection.class_name} ({detection.confidence:.2f})")
                    except Exception as e:
                        self.file_logger.error(f"❌ 创建Detection对象失败: {e}")
            
            # 创建AnalysisResult对象
            analysis_result = AnalysisResult(
                frame_id=frame_buffer.frame_id,
                timestamp=frame_buffer.timestamp,
                model_name=main_result.get("model_code", "unknown"),
                image_shape=main_result.get("image_shape", (720, 1280, 3)),
                detections=detections,
                inference_time=main_result.get("inference_time", 0.0)
            )
            
            self.file_logger.info(f"✅ AnalysisResult创建成功: 模型={analysis_result.model_name}, 检测数={len(detections)}")
            return analysis_result
            
        except Exception as e:
            self.file_logger.error(f"❌ 从FrameBuffer提取AnalysisResult失败: {e}")
            self.file_logger.exception("详细错误信息:")
            return None
    
    def _process_filtered_result(self, filtered_result: AnalysisResult):
        """处理过滤后的结果"""
        self.file_logger.info("🎯 过滤器处理完成，开始分发到处理器...")
        
        if not filtered_result:
            self.file_logger.warning("⚠️ 过滤后的结果为空，跳过处理器分发")
            return
        
        self.file_logger.info(f"✅ 过滤后保留 {len(filtered_result.detections)} 个检测结果")
        
        # 【修复】同步处理所有处理器，避免异步事件循环问题
        self._distribute_to_processors_sync(filtered_result)
    
    def _distribute_to_processors_sync(self, result: AnalysisResult):
        """同步分发结果到所有处理器"""
        start_time = time.time()
        self.file_logger.info("📤 开始同步分发结果到处理器...")
        self.file_logger.info(f"   处理器数量: {len(self.processors)}")
        self.file_logger.info(f"   检测结果数量: {len(result.detections)}")
        
        try:
            # 同步处理所有处理器
            for name, processor in self.processors:
                self.file_logger.info(f"🔄 处理处理器: {name}")
                
                try:
                    # 先调用基础的 process 方法
                    processor_result = self._safe_processor_call(
                        processor, name, self._current_frame_buffer, self.task_config
                    )
                    self.file_logger.info(f"✅ 处理器 {name} 基础调用完成: {processor_result}")
                    
                    # 【修复】如果是视频处理器，还需要调用 process_result 方法
                    if name == "video" and hasattr(processor, 'process_result'):
                        self.file_logger.info(f"🎬 调用视频处理器的 process_result 方法...")
                        try:
                            # 使用 asyncio 运行异步方法
                            import asyncio
                            try:
                                # 尝试在现有事件循环中运行
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    # 如果事件循环正在运行，使用 create_task
                                    task = loop.create_task(processor.process_result(result))
                                    # 不等待完成，让它在后台运行
                                    self.file_logger.info(f"🎬 视频处理任务已提交到后台执行")
                                else:
                                    # 如果事件循环未运行，同步运行
                                    loop.run_until_complete(processor.process_result(result))
                                    self.file_logger.info(f"🎬 视频处理器 process_result 执行完成")
                            except RuntimeError:
                                # 如果没有事件循环，创建新的
                                asyncio.run(processor.process_result(result))
                                self.file_logger.info(f"🎬 视频处理器 process_result 执行完成（新事件循环）")
                        except Exception as video_e:
                            self.file_logger.error(f"❌ 视频处理器 process_result 执行失败: {video_e}")
                            self.file_logger.exception("视频处理详细错误:")
                    
                except Exception as e:
                    self.file_logger.error(f"❌ 处理器 {name} 执行失败: {e}")
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
            "processor_names": [name for name, _ in self.processors],
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