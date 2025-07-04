"""
时间轴流水线工厂 - 完整架构集成
负责整合拉流、内存管理、处理器和结果推送模块，实现高效的流水线架构
"""
import asyncio
import time
import threading
import psutil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

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

# 导入各个模块
from .timeline_manager import TimelineManager
from .stream_module import StreamModule, StreamConfig, StreamProtocol
from .memory_module import MemoryModule
from .processor_module import ProcessorModule
from .result_module import ResultModule

@dataclass
class PipelineConfig:
    """流水线配置"""
    # 内存配置
    max_memory_mb: int = 1024
    memory_cleanup_interval: float = 0.5
    
    # 处理器配置
    target_cpu_utilization: float = 0.8
    min_processors: int = 1
    max_processors: Optional[int] = None
    
    # 结果推送配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    result_queue_prefix: str = "analysis_results"
    
    # 时间轴配置
    frame_expire_time: float = 1.0  # 帧过期时间(秒)
    timeline_cleanup_interval: float = 0.5
    
    # 系统配置
    enable_monitoring: bool = True
    monitoring_interval: float = 5.0

class TimelinePipelineFactory:
    """
    时间轴流水线工厂
    实现基于时间轴的高度并行化流水线架构
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # 核心模块
        self.timeline_manager: Optional[TimelineManager] = None
        self.stream_module: Optional[StreamModule] = None
        self.memory_module: Optional[MemoryModule] = None
        self.processor_module: Optional[ProcessorModule] = None
        self.result_module: Optional[ResultModule] = None
        
        # 运行状态
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # 统计信息
        self.pipeline_stats = {
            "start_time": 0.0,
            "total_frames_processed": 0,
            "total_streams": 0,
            "system_cpu_usage": 0.0,
            "system_memory_usage": 0.0,
            "pipeline_efficiency": 0.0
        }
        
        # 外部回调
        self.status_callbacks: List[Callable] = []
        
        normal_logger.info(f"[流水线工厂] 初始化完成 - CPU目标利用率: {config.target_cpu_utilization*100}%")
    
    async def initialize(self):
        """初始化流水线"""
        try:
            normal_logger.info("[流水线工厂] 开始初始化...")
            
            # 1. 初始化时间轴管理器
            self.timeline_manager = TimelineManager(
                frame_expire_time=self.config.frame_expire_time,
                cleanup_interval=self.config.timeline_cleanup_interval
            )
            
            # 2. 初始化内存模块
            self.memory_module = MemoryModule(
                max_memory_mb=self.config.max_memory_mb,
                cleanup_interval=self.config.memory_cleanup_interval
            )
            
            # 3. 初始化处理器模块
            self.processor_module = ProcessorModule(
                timeline_manager=self.timeline_manager,
                memory_module=self.memory_module,
                target_cpu_utilization=self.config.target_cpu_utilization
            )
            
            # 4. 初始化结果推送模块
            self.result_module = ResultModule(
                redis_host=self.config.redis_host,
                redis_port=self.config.redis_port,
                redis_db=self.config.redis_db,
                result_queue_prefix=self.config.result_queue_prefix
            )
            
            # 5. 初始化拉流模块
            self.stream_module = StreamModule(
                timeline_manager=self.timeline_manager,
                memory_module=self.memory_module
            )
            
            # 6. 配置模块间回调
            self._setup_module_callbacks()
            
            normal_logger.info("[流水线工厂] 初始化完成")
            
        except Exception as e:
            exception_logger.exception(f"[流水线工厂] 初始化失败: {str(e)}")
            raise
    
    def _setup_module_callbacks(self):
        """设置模块间回调"""
        try:
            # 处理器模块结果回调到结果推送模块
            self.processor_module.add_result_callback(self._on_processing_result)
            
            # 结果模块状态回调
            self.result_module.add_result_callback(self._on_result_pushed)
            
            normal_logger.info("[流水线工厂] 模块回调配置完成")
            
        except Exception as e:
            exception_logger.exception(f"[流水线工厂] 模块回调配置失败: {str(e)}")
    
    def _on_processing_result(self, result):
        """处理结果回调"""
        try:
            # 转换结果格式并发送到结果模块
            result_data = {
                "frame_id": result.frame_id,
                "stream_id": result.stream_id,
                "timestamp": result.timestamp,
                "frame_index": result.metadata.get("frame_index", 0),
                "processing_time": result.processing_time,
                "results": result.results,
                "metadata": result.metadata,
                "success": result.success,
                "error_message": result.error_message
            }
            
            # 异步添加到结果模块
            asyncio.create_task(self.result_module.add_result(result_data))
            
            # 更新统计
            self.pipeline_stats["total_frames_processed"] += 1
            
        except Exception as e:
            exception_logger.exception(f"[流水线工厂] 处理结果回调异常: {str(e)}")
    
    def _on_result_pushed(self, result_entry):
        """结果推送回调"""
        try:
            # 可以在这里添加结果推送后的处理逻辑
            # 例如：通知外部系统、更新统计等
            pass
            
        except Exception as e:
            exception_logger.exception(f"[流水线工厂] 结果推送回调异常: {str(e)}")
    
    async def start(self):
        """启动流水线"""
        if self.running:
            return
        
        try:
            normal_logger.info("[流水线工厂] 启动流水线...")
            
            # 确保已初始化
            if not all([
                self.timeline_manager, self.memory_module, 
                self.processor_module, self.result_module, self.stream_module
            ]):
                await self.initialize()
            
            self.running = True
            self.pipeline_stats["start_time"] = time.time()
            
            # 按顺序启动各模块
            await self.timeline_manager.start()
            await self.memory_module.start()
            await self.result_module.start()
            await self.processor_module.start()
            await self.stream_module.start()
            
            # 启动监控任务
            if self.config.enable_monitoring:
                self.monitor_task = asyncio.create_task(self._monitoring_loop())
            
            normal_logger.info("[流水线工厂] 流水线启动完成")
            
            # 通知状态回调
            for callback in self.status_callbacks:
                try:
                    callback("started", self.get_pipeline_status())
                except Exception as e:
                    exception_logger.exception(f"[流水线工厂] 状态回调异常: {str(e)}")
            
        except Exception as e:
            exception_logger.exception(f"[流水线工厂] 启动失败: {str(e)}")
            self.running = False
            raise
    
    async def stop(self):
        """停止流水线"""
        if not self.running:
            return
        
        try:
            normal_logger.info("[流水线工厂] 停止流水线...")
            
            self.running = False
            
            # 停止监控任务
            if self.monitor_task and not self.monitor_task.done():
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            
            # 按相反顺序停止各模块
            if self.stream_module:
                await self.stream_module.stop()
            
            if self.processor_module:
                await self.processor_module.stop()
            
            if self.result_module:
                await self.result_module.stop()
            
            if self.memory_module:
                await self.memory_module.stop()
            
            if self.timeline_manager:
                await self.timeline_manager.stop()
            
            normal_logger.info("[流水线工厂] 流水线停止完成")
            
            # 通知状态回调
            for callback in self.status_callbacks:
                try:
                    callback("stopped", self.get_pipeline_status())
                except Exception as e:
                    exception_logger.exception(f"[流水线工厂] 状态回调异常: {str(e)}")
            
        except Exception as e:
            exception_logger.exception(f"[流水线工厂] 停止异常: {str(e)}")
    
    async def add_stream(self, stream_config: StreamConfig) -> bool:
        """添加视频流"""
        try:
            if not self.stream_module or not self.running:
                normal_logger.warning(f"[流水线工厂] 流水线未运行，无法添加流: {stream_config.stream_id}")
                return False
            
            success = await self.stream_module.add_stream(stream_config)
            
            if success:
                self.pipeline_stats["total_streams"] += 1
                normal_logger.info(f"[流水线工厂] 成功添加流: {stream_config.stream_id}")
            else:
                normal_logger.warning(f"[流水线工厂] 添加流失败: {stream_config.stream_id}")
            
            return success
            
        except Exception as e:
            exception_logger.exception(f"[流水线工厂] 添加流异常: {stream_config.stream_id}, {str(e)}")
            return False
    
    async def remove_stream(self, stream_id: str) -> bool:
        """移除视频流"""
        try:
            if not self.stream_module:
                return False
            
            success = await self.stream_module.remove_stream(stream_id)
            
            if success:
                self.pipeline_stats["total_streams"] -= 1
                normal_logger.info(f"[流水线工厂] 成功移除流: {stream_id}")
            else:
                normal_logger.warning(f"[流水线工厂] 移除流失败: {stream_id}")
            
            return success
            
        except Exception as e:
            exception_logger.exception(f"[流水线工厂] 移除流异常: {stream_id}, {str(e)}")
            return False
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                await asyncio.sleep(self.config.monitoring_interval)
                
                # 更新系统统计
                self.pipeline_stats["system_cpu_usage"] = psutil.cpu_percent()
                
                memory_info = psutil.virtual_memory()
                self.pipeline_stats["system_memory_usage"] = memory_info.percent
                
                # 计算流水线效率
                self._calculate_pipeline_efficiency()
                
                # 打印监控信息
                self._print_monitoring_info()
                
                # 通知状态回调
                for callback in self.status_callbacks:
                    try:
                        callback("monitoring", self.get_pipeline_status())
                    except Exception as e:
                        exception_logger.exception(f"[流水线工厂] 监控回调异常: {str(e)}")
                
            except Exception as e:
                exception_logger.exception(f"[流水线工厂] 监控异常: {str(e)}")
                await asyncio.sleep(5.0)
    
    def _calculate_pipeline_efficiency(self):
        """计算流水线效率"""
        try:
            # 基于处理器利用率、内存使用情况等计算效率
            processor_stats = self.processor_module.get_processor_statistics() if self.processor_module else {}
            memory_stats = self.memory_module.get_memory_statistics() if self.memory_module else {}
            result_stats = self.result_module.get_result_statistics() if self.result_module else {}
            
            # 简化的效率计算
            processor_efficiency = min(100, processor_stats.get("global_stats", {}).get("avg_cpu_utilization", 0) * 100)
            memory_efficiency = 100 - memory_stats.get("usage_percent", 0)
            result_efficiency = 100 if result_stats.get("global_stats", {}).get("total_results_pushed", 0) > 0 else 0
            
            # 综合效率
            self.pipeline_stats["pipeline_efficiency"] = (
                processor_efficiency * 0.5 + 
                memory_efficiency * 0.3 + 
                result_efficiency * 0.2
            )
            
        except Exception as e:
            exception_logger.exception(f"[流水线工厂] 效率计算异常: {str(e)}")
            self.pipeline_stats["pipeline_efficiency"] = 0.0
    
    def _print_monitoring_info(self):
        """打印监控信息"""
        try:
            uptime = time.time() - self.pipeline_stats["start_time"]
            
            normal_logger.info(f"\n[流水线监控] "
                  f"运行时间: {uptime:.1f}s, "
                  f"处理帧数: {self.pipeline_stats['total_frames_processed']}, "
                  f"活跃流数: {self.pipeline_stats['total_streams']}")
            
            normal_logger.info(f"[系统资源] "
                  f"CPU: {self.pipeline_stats['system_cpu_usage']:.1f}%, "
                  f"内存: {self.pipeline_stats['system_memory_usage']:.1f}%")
            
            normal_logger.info(f"[流水线效率] {self.pipeline_stats['pipeline_efficiency']:.1f}%")
            
            # 各模块详细统计
            if self.processor_module:
                proc_stats = self.processor_module.get_processor_statistics()
                normal_logger.info(f"[处理器] 数量: {proc_stats['processor_count']}, "
                      f"总处理: {proc_stats['global_stats'].get('total_frames_processed', 0)}")
            
            if self.memory_module:
                mem_stats = self.memory_module.get_memory_statistics()
                normal_logger.info(f"[内存] 使用: {mem_stats['memory_usage_mb']}MB "
                      f"({mem_stats['usage_percent']:.1f}%), "
                      f"活跃块: {mem_stats['active_blocks']}")
            
            if self.result_module:
                result_stats = self.result_module.get_result_statistics()
                normal_logger.info(f"[结果] 已推送: {result_stats['global_stats'].get('total_results_pushed', 0)}, "
                      f"活跃流: {result_stats['global_stats'].get('active_streams', 0)}")
            
            normal_logger.info("-" * 80)
            
        except Exception as e:
            exception_logger.exception(f"[流水线工厂] 监控信息打印异常: {str(e)}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """获取流水线状态"""
        status = {
            "running": self.running,
            "pipeline_stats": self.pipeline_stats.copy(),
            "config": {
                "max_memory_mb": self.config.max_memory_mb,
                "target_cpu_utilization": self.config.target_cpu_utilization,
                "frame_expire_time": self.config.frame_expire_time
            }
        }
        
        # 各模块状态
        if self.timeline_manager:
            status["timeline"] = self.timeline_manager.get_statistics()
        
        if self.memory_module:
            status["memory"] = self.memory_module.get_memory_statistics()
        
        if self.processor_module:
            status["processors"] = self.processor_module.get_processor_statistics()
        
        if self.result_module:
            status["results"] = self.result_module.get_result_statistics()
        
        if self.stream_module:
            status["streams"] = self.stream_module.get_stream_statistics()
        
        return status
    
    def add_status_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """添加状态回调"""
        self.status_callbacks.append(callback)
    
    async def get_stream_queue_status(self, stream_id: str) -> Dict[str, Any]:
        """获取流队列状态"""
        if not self.result_module:
            return {"error": "结果模块未初始化"}
        
        return await self.result_module.get_queue_status(stream_id)
    
    async def force_push_stream_results(self, stream_id: str) -> int:
        """强制推送流结果"""
        if not self.result_module:
            return 0
        
        return await self.result_module.force_push_stream_results(stream_id)

# 便捷函数
async def create_timeline_pipeline(config: Optional[PipelineConfig] = None) -> TimelinePipelineFactory:
    """创建时间轴流水线"""
    if config is None:
        config = PipelineConfig()
    
    pipeline = TimelinePipelineFactory(config)
    await pipeline.initialize()
    
    return pipeline 