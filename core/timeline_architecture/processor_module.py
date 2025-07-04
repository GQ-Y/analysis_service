"""
动态分析处理器模块 - 流水线架构组件
实现智能的CPU多核并行分析，动态调整处理器数量
"""
import asyncio
import time
import threading
import uuid
import multiprocessing as mp
import psutil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from core.analyzer.analyzer_factory import analyzer_factory # Added import

try:
    from shared.utils.logger import get_normal_logger, get_exception_logger, get_analysis_logger
    normal_logger = get_normal_logger(__name__)
    exception_logger = get_exception_logger(__name__)
    analysis_logger = get_analysis_logger()
except ImportError:
    import logging
    normal_logger = logging.getLogger(__name__)
    exception_logger = logging.getLogger(__name__)
    analysis_logger = logging.getLogger(__name__)

@dataclass
class ProcessorConfig:
    """处理器配置 - 参考timelinetool的批量处理设计"""
    processor_id: str
    batch_size: int = 5
    max_processing_time: float = 5.0
    target_fps: float = 20.0
    device: str = "auto"  # auto, cpu, cuda
    enable_gpu_batch: bool = True
    max_gpu_batch_size: int = 16
    cpu_workers: int = 4
    enable_dynamic_batching: bool = True

@dataclass
class ProcessingResult:
    """处理结果"""
    frame_id: str
    stream_id: str
    timestamp: float
    processing_time: float
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None

class AnalysisProcessor:
    """单个分析处理器"""
    
    def __init__(self, config: ProcessorConfig, timeline_manager, memory_module, result_callback: Callable, analyzer: Any):
        self.config = config
        self.timeline_manager = timeline_manager
        self.memory_module = memory_module
        self.result_callback = result_callback
        self.analyzer = analyzer # Now directly receive the analyzer instance
        
        self.running = False
        self.processing_task: Optional[asyncio.Task] = None
        
        self.stats = {
            "total_processed": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "success_rate": 100.0,
            "active_frames_count": 0
        }
        
        normal_logger.info(f"[分析处理器] 初始化处理器: {config.processor_id}")
    
    async def start(self):
        """启动处理器"""
        if self.running:
            return
        
        self.running = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        
        normal_logger.info(f"[分析处理器] 启动完成: {self.config.processor_id}")
    
    async def stop(self):
        """停止处理器"""
        if not self.running:
            return
        
        self.running = False
        
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
        
        normal_logger.info(f"[分析处理器] 停止完成: {self.config.processor_id}")
    
    async def _processing_loop(self):
        """处理循环 - 参考timelinetool的动态批量处理"""
        consecutive_empty_count = 0

        while self.running:
            try:
                # 动态调整批量大小
                current_batch_size = self._adjust_batch_size()

                # 从时间轴获取待处理帧
                frames = await self.timeline_manager.get_next_frames_for_processing(
                    processor_id=self.config.processor_id,
                    batch_size=current_batch_size
                )

                if not frames:
                    consecutive_empty_count += 1
                    if consecutive_empty_count > 10:
                        await asyncio.sleep(0.1)
                    else:
                        await asyncio.sleep(0.01)
                    continue

                consecutive_empty_count = 0
                self.stats["active_frames_count"] = len(frames)

                # 批量处理帧
                await self._process_frames_batch(frames)

            except Exception as e:
                exception_logger.exception(f"[分析处理器] 处理循环异常: {self.config.processor_id}, {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _process_frames_batch(self, frames: List):
        """批量处理帧 - 参考timelinetool的批量处理设计"""
        start_time = time.time()

        try:
            # 获取帧数据
            frame_data_list = []
            valid_frames = []

            for frame in frames:
                frame_data = self._get_frame_data_from_memory_pool(frame.memory_block_id)
                if frame_data is not None:
                    frame_data_list.append(frame_data)
                    valid_frames.append(frame)

            if not valid_frames:
                return

            # 根据设备类型选择处理方式（参考timelinetool的设备适配）
            if self.config.device == 'cuda' and self.config.enable_gpu_batch:
                await self._process_gpu_batch(valid_frames, frame_data_list)
            else:
                await self._process_cpu_batch(valid_frames, frame_data_list)

            total_time = time.time() - start_time
            analysis_logger.info(f"[分析处理器] 批量处理完成: {self.config.processor_id}, "
                  f"帧数: {len(valid_frames)}, 耗时: {total_time:.3f}s")

        except Exception as e:
            analysis_logger.info(f"[分析处理器] 批量处理异常: {self.config.processor_id}, {str(e)}")
        finally:
            self.stats["active_frames_count"] = 0

    async def _process_gpu_batch(self, frames: List, frame_data_list: List[np.ndarray]):
        """GPU批量处理 - 参考timelinetool的GPU批量推理"""
        try:
            # 动态调整批量大小
            effective_batch_size = min(len(frames), self.config.max_gpu_batch_size)

            # 分批处理
            for i in range(0, len(frames), effective_batch_size):
                batch_frames = frames[i:i + effective_batch_size]
                batch_data = frame_data_list[i:i + effective_batch_size]

                # GPU批量推理
                batch_results = await self._analyze_batch_gpu(batch_frames, batch_data)

                # 处理结果
                for frame, result in zip(batch_frames, batch_results):
                    if result and self.result_callback:
                        self.result_callback(result)
                        self._update_stats(result.processing_time, result.success)

                        # 标记帧处理完成（参考timelinetool的完成标记）
                        await self.timeline_manager.mark_frame_completed(frame.frame_id, result.processing_time)

                        # 释放内存块引用（参考timelinetool的自动释放）
                        self._release_memory_block(frame.memory_block_id)
                    else:
                        self._update_stats(0.0, False)
                        # 即使处理失败也要释放内存
                        await self.timeline_manager.mark_frame_completed(frame.frame_id, 0.0)
                        self._release_memory_block(frame.memory_block_id)

        except Exception as e:
            analysis_logger.error(f"GPU批量处理异常: {str(e)}")
            # 降级到CPU处理
            await self._process_cpu_batch(frames, frame_data_list)

    async def _process_cpu_batch(self, frames: List, frame_data_list: List[np.ndarray]):
        """CPU并行处理 - 参考timelinetool的CPU并行设计"""
        try:
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=self.config.cpu_workers) as executor:
                future_to_frame = {}
                for frame, frame_data in zip(frames, frame_data_list):
                    future = executor.submit(self._analyze_single_frame, frame, frame_data)
                    future_to_frame[future] = frame

                # 收集结果
                for future in as_completed(future_to_frame, timeout=self.config.max_processing_time):
                    frame = future_to_frame[future]
                    try:
                        result = future.result()
                        if result and self.result_callback:
                            self.result_callback(result)
                            self._update_stats(result.processing_time, result.success)

                            # 标记帧处理完成（参考timelinetool的完成标记）
                            await self.timeline_manager.mark_frame_completed(frame.frame_id, result.processing_time)

                            # 释放内存块引用（参考timelinetool的自动释放）
                            self._release_memory_block(frame.memory_block_id)
                        else:
                            self._update_stats(0.0, False)
                            # 即使处理失败也要释放内存
                            await self.timeline_manager.mark_frame_completed(frame.frame_id, 0.0)
                            self._release_memory_block(frame.memory_block_id)
                    except Exception as e:
                        analysis_logger.info(f"处理帧异常: {frame.frame_id}, {str(e)}")
                        self._update_stats(0.0, False)
                        # 异常情况下也要释放内存
                        try:
                            await self.timeline_manager.mark_frame_completed(frame.frame_id, 0.0)
                            self._release_memory_block(frame.memory_block_id)
                        except Exception as cleanup_e:
                            analysis_logger.error(f"清理帧资源异常: {frame.frame_id}, {str(cleanup_e)}")

        except Exception as e:
            analysis_logger.error(f"CPU并行处理异常: {str(e)}")

    async def _analyze_batch_gpu(self, frames: List, frame_data_list: List[np.ndarray]) -> List[Optional[ProcessingResult]]:
        """GPU批量分析 - 参考timelinetool的批量推理"""
        start_time = time.time()
        results = []

        try:
            # 检查分析器是否支持批量处理
            if hasattr(self.analyzer, 'analyze_batch'):
                # 批量分析
                batch_results = self.analyzer.analyze_batch(frame_data_list)
                processing_time = (time.time() - start_time) / len(frames)  # 平均处理时间

                # 构建结果
                for frame, analysis_result in zip(frames, batch_results):
                    result = ProcessingResult(
                        frame_id=frame.frame_id,
                        stream_id=frame.stream_id,
                        timestamp=frame.timestamp,
                        processing_time=processing_time,
                        results=analysis_result,
                        metadata={
                            "processor_id": self.config.processor_id,
                            "frame_index": frame.frame_index,
                            "memory_block_id": frame.memory_block_id,
                            "batch_processing": True
                        },
                        success=True
                    )
                    results.append(result)
            else:
                # 降级到单帧处理
                for frame, frame_data in zip(frames, frame_data_list):
                    result = self._analyze_single_frame(frame, frame_data)
                    results.append(result)

        except Exception as e:
            analysis_logger.error(f"GPU批量分析异常: {str(e)}")
            # 返回失败结果
            for frame in frames:
                result = ProcessingResult(
                    frame_id=frame.frame_id,
                    stream_id=frame.stream_id,
                    timestamp=frame.timestamp,
                    processing_time=0.0,
                    results={},
                    metadata={"processor_id": self.config.processor_id},
                    success=False,
                    error_message=str(e)
                )
                results.append(result)

        return results

    def _adjust_batch_size(self) -> int:
        """
        动态调整批量大小 - 参考timelinetool的动态批量调整
        """
        if not self.config.enable_dynamic_batching:
            return self.config.batch_size

        # 基于处理性能调整批量大小
        avg_time = self.stats.get("avg_processing_time", 0.1)
        target_time = 1.0 / self.config.target_fps  # 目标处理时间

        if avg_time > target_time * 1.5:
            # 处理太慢，减少批量大小
            new_batch_size = max(1, self.config.batch_size - 1)
        elif avg_time < target_time * 0.5:
            # 处理很快，增加批量大小
            max_batch = self.config.max_gpu_batch_size if self.config.device == 'cuda' else 8
            new_batch_size = min(max_batch, self.config.batch_size + 1)
        else:
            new_batch_size = self.config.batch_size

        if new_batch_size != self.config.batch_size:
            analysis_logger.info(f"调整批量大小: {self.config.batch_size} -> {new_batch_size}")
            self.config.batch_size = new_batch_size

        return new_batch_size

    def _release_memory_block(self, memory_block_id: str):
        """
        释放内存块 - 参考timelinetool的内存释放机制

        Args:
            memory_block_id: 内存块ID
        """
        try:
            # 获取全局内存池实例（参考timelinetool的内存池访问方式）
            from shared.utils.app_state import app_state_manager
            memory_pool = app_state_manager.get_service("memory_pool")

            if memory_pool and hasattr(memory_pool, 'block_manager'):
                block_manager = memory_pool.block_manager

                # 通过block_id获取内存块对象
                try:
                    block_id = int(memory_block_id)
                    memory_block = None

                    # 直接从blocks字典中获取内存块
                    memory_block = block_manager.blocks.get(block_id)
                    if not memory_block:
                        analysis_logger.warning(f"未找到内存块: {block_id}")

                    if memory_block:
                        # 释放内存块（参考timelinetool的release机制）
                        success = memory_block.release()
                        if success:
                            analysis_logger.debug(f"成功释放内存块: {memory_block_id}")
                            return
                        else:
                            analysis_logger.warning(f"内存块释放失败: {memory_block_id}")
                    else:
                        analysis_logger.warning(f"未找到内存块: {memory_block_id}")

                except ValueError:
                    analysis_logger.error(f"无效的内存块ID: {memory_block_id}")
            else:
                analysis_logger.error("无法获取内存池实例")

        except Exception as e:
            analysis_logger.error(f"释放内存块异常: {memory_block_id}, {str(e)}")

    def _get_frame_data_from_memory_pool(self, memory_block_id: str):
        """
        从内存池获取帧数据 - 参考timelinetool的数据获取方式

        Args:
            memory_block_id: 内存块ID

        Returns:
            np.ndarray: 帧数据，失败返回None
        """
        try:
            # 获取全局内存池实例
            from shared.utils.app_state import app_state_manager
            memory_pool = app_state_manager.get_service("memory_pool")

            if memory_pool and hasattr(memory_pool, 'block_manager'):
                block_manager = memory_pool.block_manager

                # 通过block_id获取内存块对象
                try:
                    # 处理字符串或整数类型的memory_block_id
                    if isinstance(memory_block_id, str):
                        block_id = int(memory_block_id)
                    else:
                        block_id = memory_block_id

                    memory_block = None

                    # 直接从blocks字典中获取内存块（修复allocated_blocks错误）
                    memory_block = block_manager.blocks.get(block_id)
                    if not memory_block:
                        analysis_logger.warning(f"未找到内存块: {block_id}")

                    if memory_block:
                        # 获取numpy视图（参考timelinetool的数据访问）
                        numpy_view = memory_block.get_numpy_view()
                        if numpy_view is not None:
                            analysis_logger.debug(f"成功获取帧数据: {memory_block_id}")
                            return numpy_view.copy()  # 返回副本，避免数据竞争
                        else:
                            analysis_logger.warning(f"无法获取内存块numpy视图: {memory_block_id}")
                    else:
                        analysis_logger.warning(f"未找到内存块: {memory_block_id}")

                except ValueError:
                    analysis_logger.error(f"无效的内存块ID: {memory_block_id}")
            else:
                analysis_logger.error("无法获取内存池实例")

            return None

        except Exception as e:
            analysis_logger.error(f"获取帧数据异常: {memory_block_id}, {str(e)}")
            return None
    
    def _analyze_single_frame(self, frame, frame_data: np.ndarray) -> Optional[ProcessingResult]:
        """分析单帧"""
        start_time = time.time()
        
        try:
            analysis_results = self.analyzer.analyze_frame(frame_data)
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                frame_id=frame.frame_id,
                stream_id=frame.stream_id,
                timestamp=frame.timestamp,
                processing_time=processing_time,
                results=analysis_results,
                metadata={
                    "processor_id": self.config.processor_id,
                    "frame_index": frame.frame_index,
                    "memory_block_id": frame.memory_block_id
                },
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                frame_id=frame.frame_id,
                stream_id=frame.stream_id,
                timestamp=frame.timestamp,
                processing_time=processing_time,
                results={},
                metadata={"processor_id": self.config.processor_id},
                success=False,
                error_message=str(e)
            )
    
    def _update_stats(self, processing_time: float, success: bool):
        """更新统计信息 - 参考timelinetool的性能统计"""
        self.stats["total_processed"] += 1

        if success:
            self.stats["total_processing_time"] += processing_time
            self.stats["avg_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["total_processed"]
            )
            self.stats["successful_frames"] = self.stats.get("successful_frames", 0) + 1
        else:
            self.stats["failed_frames"] = self.stats.get("failed_frames", 0) + 1

        # 更新成功率
        self.stats["success_rate"] = (
            self.stats.get("successful_frames", 0) / self.stats["total_processed"]
        ) * 100

        # 更新吞吐量统计
        current_time = time.time()
        if not hasattr(self, '_last_throughput_update'):
            self._last_throughput_update = current_time
            self._throughput_counter = 0

        self._throughput_counter += 1
        time_diff = current_time - self._last_throughput_update

        if time_diff >= 1.0:  # 每秒更新一次吞吐量
            self.stats["current_fps"] = self._throughput_counter / time_diff
            self._last_throughput_update = current_time
            self._throughput_counter = 0

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            "processor_id": self.config.processor_id,
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "processing_stats": self.stats.copy(),
            "config": {
                "enable_gpu_batch": self.config.enable_gpu_batch,
                "max_gpu_batch_size": self.config.max_gpu_batch_size,
                "cpu_workers": self.config.cpu_workers,
                "enable_dynamic_batching": self.config.enable_dynamic_batching
            }
        }

class ProcessorModule:
    """动态分析处理器模块"""
    
    def __init__(self, timeline_manager, memory_module, target_cpu_utilization: float = 0.8):
        self.timeline_manager = timeline_manager
        self.memory_module = memory_module
        self.target_cpu_utilization = target_cpu_utilization
        self.min_processors = 1
        self.max_processors = mp.cpu_count()
        
        self.processors: Dict[str, AnalysisProcessor] = {}
        self.result_callbacks: List[Callable] = []
        self.running = False
        self.adjustment_task: Optional[asyncio.Task] = None
        self._analyzers: Dict[str, Any] = {} # Added _analyzers dictionary
        
        self.global_stats = {
            "total_processors": 0,
            "active_processors": 0,
            "total_frames_processed": 0,
            "avg_cpu_utilization": 0.0,
            "target_cpu_utilization": target_cpu_utilization
        }
        
        normal_logger.info(f"[处理器模块] 初始化完成 - 目标CPU利用率: {target_cpu_utilization*100}%")
    
    async def start(self):
        """启动处理器模块"""
        if self.running:
            return
        
        self.running = True
        
        # 启动初始处理器
        initial_count = min(self.max_processors, max(self.min_processors, mp.cpu_count() - 1))
        await self._adjust_processor_count(initial_count)
        
        # 启动动态调整任务
        self.adjustment_task = asyncio.create_task(self._dynamic_adjustment_loop())
        
        normal_logger.info(f"[处理器模块] 启动完成 - 初始处理器数: {initial_count}")
    
    async def stop(self):
        """停止处理器模块"""
        if not self.running:
            return
        
        self.running = False
        
        if self.adjustment_task and not self.adjustment_task.done():
            self.adjustment_task.cancel()
        
        # 停止所有处理器
        tasks = []
        for processor in list(self.processors.values()):
            tasks.append(processor.stop())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.processors.clear()
        
        # 清理分析器
        for analyzer in self._analyzers.values():
            if hasattr(analyzer, 'shutdown'):
                await analyzer.shutdown()
        self._analyzers.clear()

        normal_logger.info("[处理器模块] 停止完成")
    
    async def _dynamic_adjustment_loop(self):
        """动态调整循环"""
        while self.running:
            try:
                await asyncio.sleep(10.0)  # 每10秒调整一次
                
                # 获取当前CPU利用率
                cpu_percent = psutil.cpu_percent(interval=1)
                current_utilization = cpu_percent / 100.0
                
                self.global_stats["avg_cpu_utilization"] = current_utilization
                self.global_stats["active_processors"] = len(self.processors)
                
                # 计算目标处理器数量
                target_count = self._calculate_target_processor_count(current_utilization)
                current_count = len(self.processors)
                
                if target_count != current_count:
                    normal_logger.info(f"[处理器模块] CPU利用率: {cpu_percent:.1f}%, "
                          f"调整处理器: {current_count} -> {target_count}")
                    await self._adjust_processor_count(target_count)
                
            except Exception as e:
                exception_logger.exception(f"[处理器模块] 动态调整异常: {str(e)}")
                await asyncio.sleep(5.0)
    
    def _calculate_target_processor_count(self, current_utilization: float) -> int:
        """计算目标处理器数量"""
        current_count = len(self.processors)
        
        # CPU利用率过低，减少处理器
        if current_utilization < self.target_cpu_utilization * 0.6 and current_count > self.min_processors:
            return max(self.min_processors, current_count - 1)
        
        # CPU利用率过高，增加处理器
        elif current_utilization > self.target_cpu_utilization * 1.2 and current_count < self.max_processors:
            return min(self.max_processors, current_count + 1)
        
        return current_count
    
    async def _adjust_processor_count(self, target_count: int):
        """调整处理器数量"""
        current_count = len(self.processors)
        
        if target_count > current_count:
            for i in range(target_count - current_count):
                await self._add_processor()
        elif target_count < current_count:
            for i in range(current_count - target_count):
                await self._remove_processor()
        
        self.global_stats["total_processors"] = len(self.processors)
    
    async def _add_processor(self):
        """添加处理器"""
        try:
            processor_id = f"processor_{uuid.uuid4().hex[:8]}"
            
            # Request an analyzer instance for this processor
            # For now, we'll use a dummy model_code and analysis_type
            # In a real scenario, this would come from task configuration
            analyzer_instance = await self.request_analyzer(
                task_id=processor_id, # Use processor_id as task_id for now
                model_code="default_yolo", 
                analysis_type="detection", 
                device="auto", 
                config={}
            )

            if not analyzer_instance:
                normal_logger.warning(f"[处理器模块] 无法为处理器 {processor_id} 获取分析器，跳过添加")
                return

            config = ProcessorConfig(
                processor_id=processor_id,
                batch_size=5,
                max_processing_time=5.0
            )
            
            processor = AnalysisProcessor(
                config=config,
                timeline_manager=self.timeline_manager,
                memory_module=self.memory_module,
                result_callback=self._on_processing_result,
                analyzer=analyzer_instance # Pass the actual analyzer instance
            )
            
            await processor.start()
            self.processors[processor_id] = processor
            
            normal_logger.info(f"[处理器模块] 添加处理器: {processor_id}")
            
        except Exception as e:
            exception_logger.exception(f"[处理器模块] 添加处理器失败: {str(e)}")
    
    async def _remove_processor(self):
        """移除处理器"""
        try:
            if not self.processors:
                return
            
            processor_id = next(iter(self.processors))
            processor = self.processors[processor_id]
            await processor.stop()
            
            del self.processors[processor_id]
            normal_logger.info(f"[处理器模块] 移除处理器: {processor_id}")
            
        except Exception as e:
            exception_logger.exception(f"[处理器模块] 移除处理器失败: {str(e)}")
    
    def _on_processing_result(self, result: ProcessingResult):
        """处理结果回调"""
        try:
            self.global_stats["total_frames_processed"] += 1
            
            for callback in self.result_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    exception_logger.exception(f"[处理器模块] 结果回调异常: {str(e)}")
            
        except Exception as e:
            exception_logger.exception(f"[处理器模块] 处理结果异常: {str(e)}")
    
    def add_result_callback(self, callback: Callable[[ProcessingResult], None]):
        """添加结果回调"""
        self.result_callbacks.append(callback)
    
    def get_processor_statistics(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        processor_stats = {}
        for processor_id, processor in self.processors.items():
            processor_stats[processor_id] = processor.stats
        
        return {
            "global_stats": self.global_stats,
            "processor_stats": processor_stats,
            "processor_count": len(self.processors)
        }

    async def request_analyzer(self, task_id: str, model_code: str, analysis_type: str, device: str, config: Dict[str, Any]) -> Any:
        """
        请求一个分析器实例
        
        Args:
            task_id: 任务ID
            model_code: 模型代码
            analysis_type: 分析类型
            device: 设备
            config: 配置
            
        Returns:
            Any: 分析器实例
        """
        analyzer_key = f"{model_code}_{analysis_type}_{device}"
        
        if analyzer_key not in self._analyzers:
            normal_logger.info(f"[处理器模块] 创建新的分析器: {analyzer_key}")
            try:
                # 创建分析器
                analyzer = analyzer_factory.create_analyzer(
                    analysis_type, # type
                    "YOLODetectionAnalyzer",    # name (hardcoded for now)
                    {
                        "model_code": model_code,
                        "device": device,
                        **config
                    } # config
                )
                
                if not analyzer:
                    raise RuntimeError(f"无法创建分析器: {analyzer_key}")
                
                # 设置内存池引用
                if hasattr(analyzer, 'set_memory_pool') and self.memory_module:
                    analyzer.set_memory_pool(self.memory_module)
                
                

                self._analyzers[analyzer_key] = analyzer
            except Exception as e:
                exception_logger.exception(f"[处理器模块] 创建分析器失败: {analyzer_key}, {str(e)}")
                return None
        
        return self._analyzers[analyzer_key] 