"""
零拷贝任务服务
完全替换原有的TaskService，使用零拷贝架构进行视频流处理
"""
import asyncio
import uuid
import time
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from models.requests import StreamTask
from core.task_management.zero_copy_processor import ZeroCopyTaskProcessor
from core.task_management.stream.zero_copy_manager import ZeroCopyStreamManager
from core.memory.memory_pool import MemoryPool
from core.analyzer.analyzer_factory import analyzer_factory

# 使用项目现有的日志系统
try:
    from shared.utils.logger import get_normal_logger, get_exception_logger
    normal_logger = get_normal_logger(__name__)
    exception_logger = get_exception_logger(__name__)
except ImportError:
    import logging
    normal_logger = logging.getLogger(__name__)
    exception_logger = logging.getLogger(__name__)


class ZeroCopyTaskService:
    """
    零拷贝任务服务
    完全使用零拷贝架构处理视频流分析任务
    """
    
    def __init__(self, task_manager, memory_pool: MemoryPool = None):
        """
        初始化零拷贝任务服务
        
        Args:
            task_manager: 任务管理器（使用零拷贝处理器）
            memory_pool: 内存池实例
        """
        self.task_manager = task_manager
        self.memory_pool = memory_pool
        
        # 零拷贝组件
        self.zero_copy_processor: Optional[ZeroCopyTaskProcessor] = None
        self.zero_copy_stream_manager: Optional[ZeroCopyStreamManager] = None
        
        # 任务跟踪
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
        # 性能统计
        self.performance_stats = {
            "total_tasks_created": 0,
            "total_tasks_completed": 0,
            "total_frames_processed": 0,
            "zero_copy_operations": 0,
            "avg_processing_time": 0.0,
        }
        
        normal_logger.info("零拷贝任务服务初始化完成")
    
    def set_memory_pool(self, memory_pool: MemoryPool):
        """设置内存池"""
        self.memory_pool = memory_pool
        normal_logger.info("内存池已设置到零拷贝任务服务")
    
    def set_zero_copy_components(self, processor: ZeroCopyTaskProcessor, 
                                stream_manager: ZeroCopyStreamManager):
        """设置零拷贝组件"""
        self.zero_copy_processor = processor
        self.zero_copy_stream_manager = stream_manager
        normal_logger.info("零拷贝组件已设置")
    
    async def create_task(self, task: StreamTask, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        创建零拷贝任务
        
        Args:
            task: 流任务对象
            task_id: 任务ID，如果不提供则自动生成
            
        Returns:
            Dict[str, Any]: 创建结果
        """
        try:
            # 生成任务ID
            if not task_id:
                task_id = str(uuid.uuid4())
            
            normal_logger.info(f"开始创建零拷贝任务: {task_id}")
            
            # 构建零拷贝任务配置
            task_config = self._build_zero_copy_task_config(task, task_id)
            
            # 检查内存池状态
            if not self.memory_pool or not self.memory_pool.initialized:
                raise RuntimeError("内存池未初始化")
            
            # 验证分析器可用性
            analyzer = await self._prepare_zero_copy_analyzer(task.model_code, task_config)
            if not analyzer:
                raise RuntimeError(f"无法加载分析器: {task.model_code}")
            
            # 添加任务到管理器
            if not self.task_manager.add_task(task_id, task_config):
                raise RuntimeError("添加任务到管理器失败")
            
            # 启动零拷贝任务
            success = await self._start_zero_copy_task(task_id, task_config, analyzer)
            if not success:
                raise RuntimeError("启动零拷贝任务失败")
            
            # 记录任务信息
            self.active_tasks[task_id] = {
                "task_config": task_config,
                "analyzer": analyzer,
                "start_time": time.time(),
                "status": "running",
                "frames_processed": 0,
                "zero_copy_enabled": True,
            }
            
            # 更新统计信息
            self.performance_stats["total_tasks_created"] += 1
            
            normal_logger.info(f"零拷贝任务创建成功: {task_id}")
            
            return {
                "success": True,
                "message": "零拷贝任务创建成功",
                "task_id": task_id,
                "zero_copy_enabled": True,
                "memory_pool_status": self.memory_pool.get_stats() if self.memory_pool else None
            }
            
        except Exception as e:
            exception_logger.exception(f"创建零拷贝任务失败: {str(e)}")
            return {
                "success": False,
                "message": f"创建零拷贝任务失败: {str(e)}",
                "task_id": task_id,
                "zero_copy_enabled": False
            }
    
    async def start_task(self, model_code: str, stream_url: str, task_name: Optional[str] = None,
                        callback_urls: Optional[str] = None, output_url: Optional[str] = None,
                        analysis_type: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
                        enable_callback: bool = False, save_result: bool = False, save_images: bool = False,
                        frame_rate: Optional[int] = None, device: Optional[int] = None,
                        enable_alarm_recording: bool = False, alarm_recording_before: Optional[int] = None,
                        alarm_recording_after: Optional[int] = None, analysis_interval: Optional[int] = None,
                        callback_interval: Optional[int] = None, stream_engine: Optional[str] = None,
                        enable_hardware_decode: bool = False, low_latency: bool = False, **kwargs) -> Dict[str, Any]:
        """
        启动零拷贝任务
        
        Args:
            model_code: 模型代码
            stream_url: 流地址
            task_name: 任务名称
            callback_urls: 回调地址
            output_url: 输出地址
            analysis_type: 分析类型
            config: 配置参数
            enable_callback: 是否启用回调
            save_result: 是否保存结果
            save_images: 是否保存图像
            frame_rate: 帧率设置
            device: 设备类型
            enable_alarm_recording: 是否启用报警录像
            alarm_recording_before: 报警前录像时长
            alarm_recording_after: 报警后录像时长
            analysis_interval: 分析间隔(帧)
            callback_interval: 回调间隔(秒)
            stream_engine: 流处理引擎（强制使用零拷贝引擎）
            enable_hardware_decode: 是否启用硬件解码
            low_latency: 是否启用低延迟模式
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 启动结果
        """
        try:
            normal_logger.info(f"启动零拷贝任务: model={model_code}, stream={stream_url}")
            
            # 生成任务ID
            task_id = str(uuid.uuid4())
            
            # 强制使用零拷贝引擎
            stream_engine = "zero_copy"
            
            # 创建StreamTask对象
            task = StreamTask(
                model_code=model_code,
                stream_url=stream_url,
                task_name=task_name or f"零拷贝任务_{int(time.time())}",
                output_url=output_url,
                analysis_type=analysis_type or "detection",
                enable_callback=enable_callback,
                callback_url=callback_urls,
                save_result=save_result,
                save_images=save_images,
                frame_rate=frame_rate,
                device=device,
                enable_alarm_recording=enable_alarm_recording,
                alarm_recording_before=alarm_recording_before,
                alarm_recording_after=alarm_recording_after,
                analysis_interval=analysis_interval,
                callback_interval=callback_interval,
                stream_engine=stream_engine,
                enable_hardware_decode=enable_hardware_decode,
                low_latency=low_latency
            )
            
            # 设置零拷贝特定配置
            if config:
                from models.requests import DetectionConfig
                task.config = DetectionConfig(**config)
            
            # 添加零拷贝优化参数
            task.zero_copy_enabled = True
            task.memory_pool_enabled = True
            
            # 创建任务
            result = await self.create_task(task, task_id)
            
            if result["success"]:
                normal_logger.info(f"零拷贝任务启动成功: {task_id}")
            else:
                normal_logger.error(f"零拷贝任务启动失败: {result['message']}")
            
            return result
            
        except Exception as e:
            exception_logger.exception(f"启动零拷贝任务异常: {str(e)}")
            return {
                "success": False,
                "message": f"启动零拷贝任务异常: {str(e)}",
                "task_id": None,
                "zero_copy_enabled": False
            }

    def _build_zero_copy_task_config(self, task: StreamTask, task_id: str) -> Dict[str, Any]:
        """
        构建零拷贝任务配置

        Args:
            task: 流任务对象
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 零拷贝任务配置
        """
        # 基础配置
        config = {
            "task_id": task_id,
            "model_code": task.model_code,
            "stream_url": task.stream_url,
            "task_name": task.task_name,
            "analysis_type": task.analysis_type,
            "stream_engine": "zero_copy",  # 强制使用零拷贝引擎

            # 零拷贝特定配置
            "zero_copy_enabled": True,
            "memory_pool_enabled": True,
            "enable_frame_reference": True,
            "enable_batch_processing": True,
            "max_batch_size": 8,

            # 性能优化配置
            "enable_hardware_decode": task.enable_hardware_decode,
            "low_latency": task.low_latency,
            "analysis_interval": task.analysis_interval or 1,

            # 回调和存储配置
            "enable_callback": task.enable_callback,
            "callback_url": task.callback_url,
            "save_result": task.save_result,
            "save_images": task.save_images,

            # 设备配置
            "device": task.device or "auto",

            # 内存管理配置
            "memory_pool_config": {
                "enable_auto_cleanup": True,
                "cleanup_interval": 30,
                "memory_pressure_threshold": 0.85,
            }
        }

        # 添加分析器配置
        if hasattr(task, 'config') and task.config:
            config["analyzer_config"] = asdict(task.config)

        return config

    async def _prepare_zero_copy_analyzer(self, model_code: str, task_config: Dict[str, Any]):
        """
        准备零拷贝分析器

        Args:
            model_code: 模型代码
            task_config: 任务配置

        Returns:
            零拷贝分析器实例
        """
        try:
            # 获取分析器配置
            analyzer_config = task_config.get("analyzer_config", {})
            device = task_config.get("device", "auto")

            # 添加零拷贝特定配置
            analyzer_config.update({
                "enable_zero_copy": True,
                "enable_in_place_analysis": True,
                "enable_batch_processing": True,
                "max_batch_size": task_config.get("max_batch_size", 8),
            })

            # 创建零拷贝分析器
            analyzer = await analyzer_factory.create_analyzer(
                model_code=model_code,
                device=device,
                **analyzer_config
            )

            if not analyzer:
                raise RuntimeError(f"无法创建分析器: {model_code}")

            # 设置内存池引用
            if hasattr(analyzer, 'set_memory_pool') and self.memory_pool:
                analyzer.set_memory_pool(self.memory_pool)

            normal_logger.info(f"零拷贝分析器准备完成: {model_code}")
            return analyzer

        except Exception as e:
            exception_logger.exception(f"准备零拷贝分析器失败: {str(e)}")
            return None

    async def _start_zero_copy_task(self, task_id: str, task_config: Dict[str, Any], analyzer) -> bool:
        """
        启动零拷贝任务

        Args:
            task_id: 任务ID
            task_config: 任务配置
            analyzer: 分析器实例

        Returns:
            bool: 是否启动成功
        """
        try:
            # 检查零拷贝处理器
            if not self.zero_copy_processor:
                raise RuntimeError("零拷贝处理器未设置")

            # 启动零拷贝流处理
            success = await self.zero_copy_processor.start_zero_copy_stream_analysis(
                task_id=task_id,
                stream_url=task_config["stream_url"],
                analyzer=analyzer,
                config=task_config
            )

            if success:
                normal_logger.info(f"零拷贝任务启动成功: {task_id}")
            else:
                normal_logger.error(f"零拷贝任务启动失败: {task_id}")

            return success

        except Exception as e:
            exception_logger.exception(f"启动零拷贝任务异常: {str(e)}")
            return False

    async def stop_task(self, task_id: str) -> Dict[str, Any]:
        """
        停止零拷贝任务

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 停止结果
        """
        try:
            normal_logger.info(f"停止零拷贝任务: {task_id}")

            # 检查任务是否存在
            if task_id not in self.active_tasks:
                return {
                    "success": False,
                    "message": f"任务不存在: {task_id}",
                    "task_id": task_id
                }

            # 停止零拷贝处理
            if self.zero_copy_processor:
                await self.zero_copy_processor.stop_zero_copy_task(task_id)

            # 停止任务管理器中的任务
            if self.task_manager:
                await self.task_manager.stop_task(task_id)

            # 更新任务状态
            task_info = self.active_tasks[task_id]
            task_info["status"] = "stopped"
            task_info["end_time"] = time.time()

            # 计算运行时间
            run_time = task_info["end_time"] - task_info["start_time"]

            # 更新统计信息
            self.performance_stats["total_tasks_completed"] += 1

            normal_logger.info(f"零拷贝任务停止成功: {task_id}, 运行时间: {run_time:.2f}秒")

            return {
                "success": True,
                "message": "零拷贝任务停止成功",
                "task_id": task_id,
                "run_time": run_time,
                "frames_processed": task_info.get("frames_processed", 0)
            }

        except Exception as e:
            exception_logger.exception(f"停止零拷贝任务失败: {str(e)}")
            return {
                "success": False,
                "message": f"停止零拷贝任务失败: {str(e)}",
                "task_id": task_id
            }

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取零拷贝任务状态

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 任务状态
        """
        try:
            # 检查任务是否存在
            if task_id not in self.active_tasks:
                return {
                    "success": False,
                    "message": f"任务不存在: {task_id}",
                    "task_id": task_id
                }

            task_info = self.active_tasks[task_id]

            # 获取零拷贝统计信息
            zero_copy_stats = {}
            if self.zero_copy_processor:
                zero_copy_stats = self.zero_copy_processor.get_zero_copy_stats()

            # 获取内存池状态
            memory_pool_status = {}
            if self.memory_pool:
                memory_pool_status = self.memory_pool.get_stats()

            # 计算运行时间
            current_time = time.time()
            run_time = current_time - task_info["start_time"]

            return {
                "success": True,
                "task_id": task_id,
                "status": task_info["status"],
                "run_time": run_time,
                "frames_processed": task_info.get("frames_processed", 0),
                "zero_copy_enabled": task_info.get("zero_copy_enabled", True),
                "zero_copy_stats": zero_copy_stats,
                "memory_pool_status": memory_pool_status,
                "analyzer_info": task_info.get("analyzer", {}).model_info if task_info.get("analyzer") else {}
            }

        except Exception as e:
            exception_logger.exception(f"获取零拷贝任务状态失败: {str(e)}")
            return {
                "success": False,
                "message": f"获取零拷贝任务状态失败: {str(e)}",
                "task_id": task_id
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息

        Returns:
            Dict[str, Any]: 性能统计信息
        """
        # 获取零拷贝处理器统计
        zero_copy_stats = {}
        if self.zero_copy_processor:
            zero_copy_stats = self.zero_copy_processor.get_zero_copy_stats()

        # 获取内存池统计
        memory_pool_stats = {}
        if self.memory_pool:
            memory_pool_stats = self.memory_pool.get_stats()

        return {
            "task_service_stats": self.performance_stats.copy(),
            "zero_copy_stats": zero_copy_stats,
            "memory_pool_stats": memory_pool_stats,
            "active_tasks_count": len(self.active_tasks),
            "active_task_ids": list(self.active_tasks.keys())
        }
