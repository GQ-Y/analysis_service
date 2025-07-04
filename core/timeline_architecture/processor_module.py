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

@dataclass
class ProcessorConfig:
    """处理器配置"""
    processor_id: str
    batch_size: int = 5
    max_processing_time: float = 5.0
    target_fps: float = 20.0

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
    
    def __init__(self, config: ProcessorConfig, timeline_manager, memory_module, result_callback: Callable):
        self.config = config
        self.timeline_manager = timeline_manager
        self.memory_module = memory_module
        self.result_callback = result_callback
        
        self.running = False
        self.processing_task: Optional[asyncio.Task] = None
        self.analyzer = None
        
        self.stats = {
            "total_processed": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "success_rate": 100.0,
            "active_frames_count": 0
        }
        
        print(f"[分析处理器] 初始化处理器: {config.processor_id}")
    
    async def start(self):
        """启动处理器"""
        if self.running:
            return
        
        self.running = True
        await self._initialize_analyzer()
        self.processing_task = asyncio.create_task(self._processing_loop())
        
        print(f"[分析处理器] 启动完成: {self.config.processor_id}")
    
    async def stop(self):
        """停止处理器"""
        if not self.running:
            return
        
        self.running = False
        
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
        
        print(f"[分析处理器] 停止完成: {self.config.processor_id}")
    
    async def _initialize_analyzer(self):
        """初始化分析器"""
        self.analyzer = self._create_mock_analyzer()
        print(f"[分析处理器] 分析器初始化完成: {self.config.processor_id}")
    
    def _create_mock_analyzer(self):
        """创建模拟分析器"""
        class MockAnalyzer:
            def analyze_frame(self, frame_data: np.ndarray) -> Dict[str, Any]:
                # 模拟YOLO分析
                import random
                processing_time = random.uniform(0.05, 0.2)
                time.sleep(processing_time)
                
                return {
                    "detections": [
                        {"class": "person", "confidence": 0.85, "bbox": [100, 100, 200, 300]},
                        {"class": "car", "confidence": 0.92, "bbox": [300, 150, 500, 350]}
                    ],
                    "processing_time": processing_time,
                    "frame_size": frame_data.shape if frame_data is not None else (0, 0, 0)
                }
        
        return MockAnalyzer()
    
    async def _processing_loop(self):
        """处理循环"""
        consecutive_empty_count = 0
        
        while self.running:
            try:
                # 从时间轴获取待处理帧
                frames = await self.timeline_manager.get_next_frames_for_processing(
                    processor_id=self.config.processor_id,
                    batch_size=self.config.batch_size
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
                print(f"[分析处理器] 处理循环异常: {self.config.processor_id}, {e}")
                await asyncio.sleep(1.0)
    
    async def _process_frames_batch(self, frames: List):
        """批量处理帧"""
        start_time = time.time()
        
        try:
            # 获取帧数据
            frame_data_list = []
            valid_frames = []
            
            for frame in frames:
                frame_data = self.memory_module.get_frame_data(frame.frame_id)
                if frame_data is not None:
                    frame_data_list.append(frame_data)
                    valid_frames.append(frame)
            
            if not valid_frames:
                return
            
            # 并行分析
            with ThreadPoolExecutor(max_workers=min(len(valid_frames), 4)) as executor:
                future_to_frame = {}
                for frame, frame_data in zip(valid_frames, frame_data_list):
                    future = executor.submit(self._analyze_single_frame, frame, frame_data)
                    future_to_frame[future] = frame
                
                # 收集结果
                for future in as_completed(future_to_frame, timeout=self.config.max_processing_time):
                    frame = future_to_frame[future]
                    try:
                        result = future.result()
                        if result:
                            self.result_callback(result)
                            self._update_stats(result.processing_time, True)
                    except Exception as e:
                        print(f"[分析处理器] 处理帧异常: {frame.frame_id}, {e}")
                        self._update_stats(0.0, False)
            
            total_time = time.time() - start_time
            print(f"[分析处理器] 批量处理完成: {self.config.processor_id}, "
                  f"帧数: {len(valid_frames)}, 耗时: {total_time:.3f}s")
            
        except Exception as e:
            print(f"[分析处理器] 批量处理异常: {self.config.processor_id}, {e}")
        finally:
            self.stats["active_frames_count"] = 0
    
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
        """更新统计信息"""
        self.stats["total_processed"] += 1
        
        if success:
            self.stats["total_processing_time"] += processing_time
            self.stats["avg_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["total_processed"]
            )
        
        # 更新成功率
        success_count = self.stats["total_processed"] - (0 if success else 1)
        self.stats["success_rate"] = (success_count / self.stats["total_processed"]) * 100

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
        
        self.global_stats = {
            "total_processors": 0,
            "active_processors": 0,
            "total_frames_processed": 0,
            "avg_cpu_utilization": 0.0,
            "target_cpu_utilization": target_cpu_utilization
        }
        
        print(f"[处理器模块] 初始化完成 - 目标CPU利用率: {target_cpu_utilization*100}%")
    
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
        
        print(f"[处理器模块] 启动完成 - 初始处理器数: {initial_count}")
    
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
        print("[处理器模块] 停止完成")
    
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
                    print(f"[处理器模块] CPU利用率: {cpu_percent:.1f}%, "
                          f"调整处理器: {current_count} -> {target_count}")
                    await self._adjust_processor_count(target_count)
                
            except Exception as e:
                print(f"[处理器模块] 动态调整异常: {e}")
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
            
            config = ProcessorConfig(
                processor_id=processor_id,
                batch_size=5,
                max_processing_time=5.0
            )
            
            processor = AnalysisProcessor(
                config=config,
                timeline_manager=self.timeline_manager,
                memory_module=self.memory_module,
                result_callback=self._on_processing_result
            )
            
            await processor.start()
            self.processors[processor_id] = processor
            
            print(f"[处理器模块] 添加处理器: {processor_id}")
            
        except Exception as e:
            print(f"[处理器模块] 添加处理器失败: {e}")
    
    async def _remove_processor(self):
        """移除处理器"""
        try:
            if not self.processors:
                return
            
            processor_id = next(iter(self.processors))
            processor = self.processors[processor_id]
            await processor.stop()
            
            del self.processors[processor_id]
            print(f"[处理器模块] 移除处理器: {processor_id}")
            
        except Exception as e:
            print(f"[处理器模块] 移除处理器失败: {e}")
    
    def _on_processing_result(self, result: ProcessingResult):
        """处理结果回调"""
        try:
            self.global_stats["total_frames_processed"] += 1
            
            for callback in self.result_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    print(f"[处理器模块] 结果回调异常: {e}")
            
        except Exception as e:
            print(f"[处理器模块] 处理结果异常: {e}")
    
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