#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析处理器 - 从时间轴读取数据，使用分析器进行推理，将结果传递给结果处理器
"""

import time
import threading
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Callable

from app.core.zero_copy.time_axis import TimeAxis
from app.core.zero_copy.result_processor import ResultProcessor


class ImageAnalysisProcessor:
    """图像分析处理器 - 从时间轴读取图像数据，进行AI分析"""
    
    def __init__(self, 
                 analyzers: Dict[str, Any],
                 time_axis: TimeAxis,
                 result_processor: ResultProcessor,
                 task_id: int,
                 batch_size: int = 5,
                 processing_interval: float = 0.1,
                 model_fps_config: Optional[Dict[str, float]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化分析处理器
        
        Args:
            analyzers: 分析器字典 {model_code: analyzer}
            time_axis: 时间轴
            result_processor: 结果处理器
            task_id: 任务ID
            batch_size: 批处理大小
            processing_interval: 处理间隔
            model_fps_config: 模型FPS配置 {model_code: fps}
            logger: 日志记录器
        """
        self.analyzers = analyzers
        self.time_axis = time_axis
        self.result_processor = result_processor
        self.task_id = task_id
        self.batch_size = batch_size
        self.processing_interval = processing_interval
        self.model_fps_config = model_fps_config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # 处理状态
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # 为每个模型创建FPS控制器
        self.fps_controllers = {}
        self._create_fps_controllers()
        
        # 统计信息
        self.stats = {
            "total_processed": 0,
            "total_detections": 0,
            "successful_analysis": 0,
            "failed_analysis": 0,
            "start_time": 0,
            "processing_time": 0.0
        }
        
        self.logger.info(f"🔍 分析处理器初始化: {len(analyzers)} 个分析器, FPS控制: {len(self.fps_controllers)} 个")
    
    def _create_fps_controllers(self):
        """为每个模型创建FPS控制器"""
        from app.core.zero_copy.ai_analyzer import FPSController
        
        for model_code in self.analyzers.keys():
            target_fps = self.model_fps_config.get(model_code)
            
            fps_controller = FPSController(
                time_axis=self.time_axis,
                target_fps=target_fps,
                model_code=model_code,
                logger=self.logger
            )
            
            self.fps_controllers[model_code] = fps_controller
            
            fps_info = f"FPS={target_fps}" if target_fps else "无限制"
            self.logger.info(f"🎯 创建FPS控制器: {model_code} ({fps_info})")
    
    def start(self):
        """启动分析处理器"""
        if self.running:
            self.logger.warning("分析处理器已在运行")
            return
        
        self.running = True
        self.stats["start_time"] = time.time()
        self.thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.thread.start()
        self.logger.info("🚀 分析处理器已启动")
    
    def stop(self):
        """停止分析处理器"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        self.logger.info("⏹️ 分析处理器已停止")
    
    def _analysis_loop(self):
        """分析处理循环"""
        self.logger.info("🔄 开始分析处理循环")
        
        while self.running:
            try:
                # 从时间轴获取帧数据（按模型分组）
                model_frames = self._get_frames_from_time_axis()
                
                if not model_frames or all(not frames for frames in model_frames.values()):
                    time.sleep(self.processing_interval)
                    continue
                
                # 按模型分别处理帧
                self._process_model_frames(model_frames)
                
            except Exception as e:
                self.logger.error(f"❌ 分析处理循环错误: {e}")
                time.sleep(1.0)
        
        self.logger.info("🏁 分析处理循环结束")
    
    def _get_frames_from_time_axis(self) -> Dict[str, List[Any]]:
        """从时间轴获取待处理的帧，按模型分组"""
        model_frames = {}
        
        try:
            # 为每个模型使用对应的FPS控制器获取帧
            for model_code, fps_controller in self.fps_controllers.items():
                frames = fps_controller.get_batch(self.batch_size)
                
                if frames:
                    model_frames[model_code] = frames
                    self.logger.debug(f"📥 模型 {model_code} 获取到 {len(frames)} 帧")
                else:
                    model_frames[model_code] = []
                    
        except Exception as e:
            self.logger.error(f"❌ 从时间轴获取帧失败: {e}")
        
        return model_frames
    
    def _process_model_frames(self, model_frames: Dict[str, List[Any]]):
        """按模型处理帧"""
        for model_code, frames in model_frames.items():
            if not frames:
                continue
                
            try:
                # 使用对应的分析器处理帧
                analyzer = self.analyzers.get(model_code)
                if not analyzer:
                    self.logger.warning(f"⚠️ 模型 {model_code} 没有对应的分析器")
                    # 释放帧
                    for frame_buffer in frames:
                        self._release_frame_buffer(frame_buffer)
                    continue
                
                # 批量处理帧
                self._process_frames_batch_for_model(frames, model_code, analyzer)
                
            except Exception as e:
                self.logger.error(f"❌ 处理模型 {model_code} 帧失败: {e}")
                # 释放帧
                for frame_buffer in frames:
                    self._release_frame_buffer(frame_buffer)
    
    def _process_frames_batch_for_model(self, frames: List[Any], model_code: str, analyzer: Any):
        """为特定模型批量处理帧"""
        start_time = time.time()
        processed_frames = []
        
        for frame_buffer in frames:
            try:
                # 提取图像数据
                image = self._extract_image_from_frame(frame_buffer)
                if image is None:
                    # 即使处理失败也要释放帧缓冲区
                    self._release_frame_buffer(frame_buffer)
                    continue
                
                # 使用指定的分析器进行分析
                try:
                    result = analyzer.detect(image)
                    
                    # 统计检测数量
                    detections = result.get("detections", [])
                    self.stats["total_detections"] += len(detections)
                    
                    # 【新增】打印详细的检测结果日志（DEBUG级别）
                    self._log_detection_results(frame_buffer, model_code, result, detections)
                    
                    # 【新增】简洁的控制台提示（仅当有检测结果时）
                    if len(detections) > 0:
                        frame_id = getattr(frame_buffer, 'frame_id', 'unknown')
                        class_counts = {}
                        for detection in detections:
                            class_name = detection.get("class_name", "unknown")
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        class_summary = ", ".join([f"{cls}:{count}" for cls, count in class_counts.items()])
                        self.logger.info(f"🎯 [{model_code}] 帧{frame_id}: {class_summary}")
                    
                    # 构建分析结果（只包含当前模型的结果）
                    analysis_result = {
                        "task_id": self.task_id,
                        "frame_id": getattr(frame_buffer, 'frame_id', 0),
                        "timestamp": getattr(frame_buffer, 'timestamp', time.time()),
                        "stream_id": getattr(frame_buffer, 'stream_id', 'image_analysis'),
                        "image_shape": image.shape,
                        "model_results": {model_code: result},
                        "analysis_time": time.time() - start_time
                    }
                    
                    # 【关键修复】将分析结果添加到FrameBuffer
                    # FrameBuffer.analysis_results 是字典类型，使用add_analysis_result方法
                    frame_buffer.add_analysis_result(f"analysis_result_{model_code}", analysis_result)
                    
                    self.logger.debug(f"✅ 已将分析结果添加到FrameBuffer: 帧{getattr(frame_buffer, 'frame_id', 'unknown')}, "
                                    f"模型{model_code}, 检测数量: {len(detections)}")
                    
                    # 发送给结果处理器
                    if self.result_processor:
                        self.result_processor.process_result(frame_buffer, self.task_id)
                    
                    # 添加到已处理列表，延后释放
                    processed_frames.append(frame_buffer)
                    
                    self.stats["successful_analysis"] += 1
                    self.stats["total_processed"] += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ 分析器 {model_code} 处理失败: {e}")
                    self.stats["failed_analysis"] += 1
                    self._release_frame_buffer(frame_buffer)
                    continue
                
            except Exception as e:
                self.logger.error(f"❌ 处理帧失败: {e}")
                self.stats["failed_analysis"] += 1
                # 处理失败也要释放帧缓冲区
                self._release_frame_buffer(frame_buffer)
        
        # 批量释放已处理的帧缓冲区（关键修复）
        for frame_buffer in processed_frames:
            self._release_frame_buffer(frame_buffer)
        
        # 更新处理时间统计
        batch_time = time.time() - start_time
        self.stats["processing_time"] += batch_time
        
        # 【新增】批量处理汇总日志（调整显示频率）
        if len(processed_frames) > 0:
            avg_time_per_frame = batch_time / len(processed_frames) if len(processed_frames) > 0 else 0
            fps = len(processed_frames) / batch_time if batch_time > 0 else 0
            
            # 每处理50帧或检测到目标时才在控制台显示汇总，其他时候只记录到DEBUG
            current_detections = self.stats['total_detections']
            should_show_summary = (
                self.stats['total_processed'] % 50 == 0 or  # 每50帧显示一次
                current_detections > getattr(self, '_last_detections_count', 0) or  # 有新的检测结果
                batch_time > 1.0  # 处理时间超过1秒
            )
            
            # 记录当前检测数量，用于下次比较
            self._last_detections_count = current_detections
            
            log_message = (f"✅ 批量处理汇总: {len(processed_frames)}帧 "
                         f"耗时{batch_time:.3f}s "
                         f"平均{avg_time_per_frame*1000:.1f}ms/帧 "
                         f"FPS:{fps:.1f} "
                         f"累计检测{self.stats['total_detections']}个目标 "
                         f"累计处理{self.stats['total_processed']}帧")
            
            if should_show_summary:
                self.logger.info(log_message)
            else:
                self.logger.debug(log_message)
        else:
            self.logger.debug(f"📊 批次处理完成: 输入{len(frames)}帧, 成功处理0帧, 耗时: {batch_time:.3f}s")
    
    def _extract_image_from_frame(self, frame_buffer) -> Optional[np.ndarray]:
        """从帧缓冲区提取图像数据"""
        try:
            # 检查frame_buffer是否有效
            if not frame_buffer or not frame_buffer.is_valid:
                self.logger.warning("⚠️ 帧缓冲区无效或已失效")
                return None
            
            # 从FrameBuffer提取图像数据
            if hasattr(frame_buffer, 'get_frame_view'):
                # 使用零拷贝视图
                return frame_buffer.get_frame_view()
            elif hasattr(frame_buffer, 'frame_data'):
                # 直接访问帧数据
                return frame_buffer.frame_data
            elif hasattr(frame_buffer, 'get_frame_copy'):
                # 使用副本（如果需要）
                return frame_buffer.get_frame_copy()
            elif isinstance(frame_buffer, np.ndarray):
                return frame_buffer
            else:
                self.logger.error(f"❌ 不支持的帧缓冲区类型: {type(frame_buffer)}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 提取图像失败: {e}")
            return None
    
    def _log_detection_results(self, frame_buffer, model_code: str, result: Dict[str, Any], detections: List[Dict[str, Any]]):
        """打印详细的检测结果日志"""
        try:
            frame_id = getattr(frame_buffer, 'frame_id', 'unknown')
            inference_time = result.get("inference_time", 0.0)
            image_shape = result.get("image_shape", "unknown")
            
            # 基本信息（DEBUG级别，只记录到文件）
            self.logger.debug(f"🔍 [{model_code}] 帧{frame_id} 分析完成: "
                            f"检测{len(detections)}个目标, "
                            f"耗时{inference_time*1000:.1f}ms, "
                            f"图像尺寸{image_shape}")
            
            # 如果有检测结果，打印详细信息（DEBUG级别）
            if detections:
                # 统计各类别的检测数量
                class_counts = {}
                confidence_stats = {}
                
                for detection in detections:
                    class_name = detection.get("class_name", "unknown")
                    confidence = detection.get("confidence", 0.0)
                    
                    # 统计类别数量
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # 收集置信度统计
                    if class_name not in confidence_stats:
                        confidence_stats[class_name] = []
                    confidence_stats[class_name].append(confidence)
                
                # 打印类别统计（DEBUG级别）
                class_summary = ", ".join([f"{cls}:{count}个" for cls, count in class_counts.items()])
                self.logger.debug(f"📊 [{model_code}] 帧{frame_id} 检测统计: {class_summary}")
                
                # 打印置信度统计（DEBUG级别）
                for class_name, confidences in confidence_stats.items():
                    avg_conf = sum(confidences) / len(confidences)
                    max_conf = max(confidences)
                    min_conf = min(confidences)
                    self.logger.debug(f"📈 [{model_code}] {class_name}: "
                                    f"置信度 平均{avg_conf:.2f} 最高{max_conf:.2f} 最低{min_conf:.2f}")
                
                # 打印前3个最高置信度的检测结果详情（DEBUG级别）
                sorted_detections = sorted(detections, key=lambda x: x.get("confidence", 0.0), reverse=True)
                for i, detection in enumerate(sorted_detections[:3]):
                    bbox = detection.get("bbox", {})
                    self.logger.debug(f"🎯 [{model_code}] TOP{i+1}: "
                                    f"{detection.get('class_name', 'unknown')} "
                                    f"置信度{detection.get('confidence', 0.0):.2f} "
                                    f"位置({bbox.get('x1', 0):.0f},{bbox.get('y1', 0):.0f},"
                                    f"{bbox.get('x2', 0):.0f},{bbox.get('y2', 0):.0f}) "
                                    f"面积{detection.get('area', 0):.0f}px²")
            else:
                self.logger.debug(f"🚫 [{model_code}] 帧{frame_id}: 未检测到任何目标")
                
            # 检测是否为Mock模式
            if result.get("mock_mode", False):
                self.logger.warning(f"⚠️ [{model_code}] 帧{frame_id}: 使用Mock模式分析")
            
            # 检测是否有错误
            if "error" in result:
                self.logger.error(f"❌ [{model_code}] 帧{frame_id}: 分析出错 - {result['error']}")
                
        except Exception as e:
            self.logger.error(f"❌ 打印检测结果日志失败: {e}")

    def _release_frame_buffer(self, frame_buffer):
        """释放帧缓冲区（关键方法）"""
        try:
            if not frame_buffer:
                return
            
            # 调用帧缓冲区的释放方法
            if hasattr(frame_buffer, 'release'):
                frame_buffer.release()
            elif hasattr(frame_buffer, 'free'):
                frame_buffer.free()
            elif hasattr(frame_buffer, 'close'):
                frame_buffer.close()
            else:
                # 如果没有明确的释放方法，尝试标记为无效
                if hasattr(frame_buffer, 'is_valid'):
                    frame_buffer.is_valid = False
                    
        except Exception as e:
            self.logger.error(f"❌ 释放帧缓冲区失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        runtime = time.time() - self.stats["start_time"] if self.stats["start_time"] > 0 else 0
        
        return {
            **self.stats,
            "runtime": runtime,
            "avg_processing_time": (self.stats["processing_time"] / max(1, self.stats["total_processed"])),
            "detection_rate": (self.stats["total_detections"] / max(1, self.stats["total_processed"])),
            "success_rate": (self.stats["successful_analysis"] / max(1, self.stats["total_processed"])),
            "analyzers_count": len(self.analyzers)
        }
    
    def cleanup(self):
        """清理资源"""
        self.stop()
        
        # 清理分析器
        for model_code, analyzer in self.analyzers.items():
            try:
                if hasattr(analyzer, 'cleanup'):
                    analyzer.cleanup()
            except Exception as e:
                self.logger.error(f"❌ 清理分析器失败 {model_code}: {e}")
        
        self.logger.info("🧹 分析处理器资源已清理")


class VideoAnalysisProcessor(ImageAnalysisProcessor):
    """视频分析处理器 - 继承图像分析处理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("🎬 视频分析处理器初始化")


class StreamAnalysisProcessor(ImageAnalysisProcessor):
    """流分析处理器 - 继承图像分析处理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("📡 流分析处理器初始化") 