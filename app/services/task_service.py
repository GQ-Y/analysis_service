#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: task_service.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 任务管理服务

实现任务管理的5个核心功能：
1. 创建任务
2. 停止任务  
3. 查看任务详情
4. 删除任务
5. 重启任务

本文件是分析服务项目的一部分。
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from app.services.base_service import BaseService
from config.settings import get_settings
from app.core.storage.model_manager import ModelManager
from app.core.analyzer.analyzer_factory import AnalyzerFactory
from app.core.analyzer.analysis_processor import ImageAnalysisProcessor, VideoAnalysisProcessor, StreamAnalysisProcessor


@dataclass
class TaskInfo:
    """任务信息数据类"""
    id: int
    name: str
    description: Optional[str]
    analysis_type: int  # 1-图片分析, 2-视频分析, 3-流分析
    status: int  # 0-未启动, 1-运行中, 2-已停止, 3-错误, 4-已完成
    progress: float
    created_at: str
    updated_at: str
    model_codes: List[str]
    stream_urls: List[str]
    result_count: int


class TaskService(BaseService):
    """任务管理服务"""
    
    def __init__(self):
        super().__init__()
        self.tasks: Dict[int, Dict[str, Any]] = {}  # 内存存储任务
        self.task_counter = 1
        self.running_tasks: Dict[int, asyncio.Task] = {}  # 运行中的异步任务
        self.task_components: Dict[int, Dict[str, Any]] = {}  # 任务组件引用（用于停止时清理）
        
        # 初始化模型管理器和分析器工厂
        self.model_manager = ModelManager()
        self.analyzer_factory = AnalyzerFactory(self.model_manager)
    
    async def create_task(
        self,
        name: str,
        description: Optional[str],
        analysis_type: int,
        model_codes: List[str],
        stream_urls: Optional[List[str]] = None,
        video_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        config: Dict[str, Any] = None,
        enable_video_player: bool = False,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        save_result: bool = True,
        save_images: bool = False,
        callback_urls: Optional[List[str]] = None,
        callback_interval: Optional[int] = None,
        playback_duration: Optional[int] = None,
        roi_config: Optional[Dict[str, Any]] = None,
        target_classes: Optional[List[str]] = None
    ) -> TaskInfo:
        """创建任务
        
        Args:
            name: 任务名称
            description: 任务描述
            analysis_type: 分析类型 (1-图片, 2-视频, 3-流)
            model_codes: 模型代码列表
            stream_urls: 流URL列表
            video_path: 视频文件路径
            image_paths: 图片路径列表
            config: 任务配置
            enable_video_player: 是否启用视频播放器
            confidence_threshold: 检测置信度阈值 (0.0-1.0)
            iou_threshold: 非极大值抑制IoU阈值 (0.0-1.0)
            save_result: 是否保存分析结果
            save_images: 是否保存检测图片
            callback_urls: 回调地址列表
            callback_interval: 回调间隔（秒），0表示实时回调
            playback_duration: 回放视频时长（秒），适用于视频/流分析
            roi_config: ROI区域配置
            target_classes: 目标检测类别列表

        Returns:
            TaskInfo: 创建的任务信息
        """
        # 参数验证
        if not name:
            raise ValueError("任务名称不能为空")
        
        if not model_codes:
            raise ValueError("必须指定至少一个模型")
        
        if analysis_type == 1 and not image_paths:
            raise ValueError("图片分析任务必须提供图片路径")
        elif analysis_type == 2 and not video_path:
            raise ValueError("视频分析任务必须提供视频路径")
        elif analysis_type == 3 and not stream_urls:
            raise ValueError("流分析任务必须提供流URL")
        
        # 创建任务
        task_id = self.task_counter
        self.task_counter += 1
        
        now = datetime.now()
        task_data = {
            "id": task_id,
            "name": name,
            "description": description,
            "analysis_type": analysis_type,
            "model_codes": model_codes,
            "stream_urls": stream_urls or [],
            "video_path": video_path,
            "image_paths": image_paths or [],
            "config": config or {},
            "enable_video_player": enable_video_player,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "save_result": save_result,
            "save_images": save_images,
            "callback_urls": callback_urls,
            "callback_interval": callback_interval or 0,
            "playback_duration": playback_duration or 0,
            "roi_config": roi_config,
            "target_classes": target_classes,
            "status": 0,  # 未启动
            "progress": 0.0,
            "created_at": now,
            "updated_at": now,
            "results": []
        }
        
        self.tasks[task_id] = task_data
        
        self.logger.info(f"✅ 任务创建成功: ID={task_id}, 名称={name}, 类型={analysis_type}")
        
        return self._to_task_info(task_data)
    
    async def start_task(self, task_id: int) -> Dict[str, Any]:
        """启动任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 启动结果
        """
        if task_id not in self.tasks:
            raise ValueError(f"任务 {task_id} 不存在")
        
        task_data = self.tasks[task_id]
        
        if task_data["status"] == 1:
            raise ValueError(f"任务 {task_id} 已经在运行中")
        
        # 更新任务状态
        task_data["status"] = 1  # 运行中
        task_data["updated_at"] = datetime.now()
        
        # 启动异步任务执行
        task_coroutine = self._execute_task(task_id)
        self.running_tasks[task_id] = asyncio.create_task(task_coroutine)
        
        self.logger.info(f"🚀 任务启动成功: ID={task_id}")
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": "任务已启动"
        }
    
    async def stop_task(self, task_id: int) -> Dict[str, Any]:
        """停止任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 停止结果
        """
        if task_id not in self.tasks:
            return {
                "task_id": task_id,
                "status": "not_found",
                "message": f"⚠️ 任务 {task_id} 不存在"
            }
        
        task_data = self.tasks[task_id]
        
        # 如果任务已经停止，直接返回
        if task_data["status"] != 1:
            return {
                "task_id": task_id,
                "status": "already_stopped",
                "message": "⏹️ 任务已经是停止状态"
            }
        
        self.logger.info(f"⏹️ 开始停止任务: ID={task_id}")
        
        # 记录当前任务状态
        self.logger.info(f"📊 任务 {task_id} 停止前状态:")
        self.logger.info(f"   - 是否有组件: {task_id in self.task_components}")
        self.logger.info(f"   - 是否有异步任务: {task_id in self.running_tasks}")
        if task_id in self.running_tasks:
            async_task = self.running_tasks[task_id]
            self.logger.info(f"   - 异步任务状态: done={async_task.done()}, cancelled={async_task.cancelled()}")
        
        # 记录停止过程中的错误，但不让这些错误阻止停止流程
        stop_errors = []
        
        try:
            # 1. 首先停止底层组件（关键修复）
            await self._stop_task_components(task_id)
        except Exception as e:
            import traceback
            error_msg = f"停止组件时出现错误: {type(e).__name__}: {e}"
            self.logger.error(f"❌ {error_msg}")
            self.logger.error(f"❌ 停止组件异常堆栈:\n{traceback.format_exc()}")
            stop_errors.append(error_msg)
        
        try:
            # 2. 然后取消运行中的异步任务
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                self.logger.info(f"🔍 找到异步任务 {task_id}: done={task.done()}, cancelled={task.cancelled()}")
                
                if not task.done():
                    self.logger.info(f"⏹️ 开始取消异步任务 {task_id}")
                    task.cancel()
                    try:
                        # 等待任务完全取消
                        await asyncio.wait_for(task, timeout=3.0)
                    except asyncio.CancelledError:
                        # 任务正常取消，这是预期行为
                        self.logger.info(f"⏹️ 任务 {task_id} 异步任务已正常取消")
                    except asyncio.TimeoutError:
                        # 超时但任务可能仍在取消中
                        self.logger.warning(f"⚠️ 任务 {task_id} 异步任务取消超时，将强制清理")
                    except asyncio.InvalidStateError:
                        # 任务状态异常，可能已经完成或被取消
                        self.logger.info(f"ℹ️ 任务 {task_id} 异步任务状态已改变")
                    except Exception as cancel_e:
                        import traceback
                        # 检查是否是事件循环相关的异常
                        if "attached to a different loop" in str(cancel_e):
                            self.logger.warning(f"⚠️ 任务 {task_id} 异步任务在不同事件循环中，将强制清理")
                        else:
                            self.logger.warning(f"⚠️ 任务 {task_id} 异步任务取消时出现异常: {type(cancel_e).__name__}: {cancel_e}")
                            self.logger.warning(f"⚠️ 详细异常堆栈:\n{traceback.format_exc()}")
                            
                            # 记录任务的详细状态
                            self.logger.warning(f"⚠️ 任务详细信息: done={task.done()}, cancelled={task.cancelled()}")
                            if hasattr(task, '_exception'):
                                self.logger.warning(f"⚠️ 任务内部异常: {task._exception}")
                            if hasattr(task, '_result'):
                                self.logger.warning(f"⚠️ 任务结果: {task._result}")
                else:
                    self.logger.info(f"ℹ️ 任务 {task_id} 异步任务已完成，无需取消")
                
                # 等待一小段时间，让异步任务的finally块执行完毕
                await asyncio.sleep(0.1)
                
                # 再次检查并清理任务引用（可能已经被finally块清理了）
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
                    self.logger.info(f"🧹 任务 {task_id} 异步任务引用已清理")
                else:
                    self.logger.info(f"ℹ️ 任务 {task_id} 异步任务引用已被finally块清理")
            else:
                self.logger.info(f"ℹ️ 任务 {task_id} 没有异步任务需要取消")
        except Exception as e:
            import traceback
            error_msg = f"取消异步任务时出现错误: {type(e).__name__}: {e}"
            self.logger.error(f"❌ {error_msg}")
            self.logger.error(f"❌ 异步任务取消异常堆栈:\n{traceback.format_exc()}")
            
            # 记录更详细的任务状态信息
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                self.logger.error(f"❌ 异步任务状态: done={task.done()}, cancelled={task.cancelled()}")
            else:
                self.logger.error(f"❌ 异步任务不在running_tasks中")
            
            stop_errors.append(error_msg)
        
        try:
            # 3. 清理组件引用
            if task_id in self.task_components:
                del self.task_components[task_id]
                self.logger.info(f"🧹 任务 {task_id}: 组件引用已清理")
        except Exception as e:
            error_msg = f"清理组件引用时出现错误: {e}"
            self.logger.error(f"❌ {error_msg}")
            stop_errors.append(error_msg)
        
        # 4. 无论如何都要更新任务状态
        try:
            task_data["status"] = 2  # 已停止
            task_data["updated_at"] = datetime.now()
        except Exception as e:
            error_msg = f"更新任务状态时出现错误: {e}"
            self.logger.error(f"❌ {error_msg}")
            stop_errors.append(error_msg)
        
        # 5. 根据停止过程记录日志
        if stop_errors:
            self.logger.warning(f"⚠️ 任务 {task_id} 停止过程中出现 {len(stop_errors)} 个错误，但任务已强制停止")
            for error in stop_errors:
                self.logger.warning(f"   - {error}")
            
            return {
                "task_id": task_id,
                "status": "stopped_with_errors",
                "message": f"任务已停止，但停止过程中出现 {len(stop_errors)} 个错误",
                "errors": stop_errors
            }
        else:
            self.logger.info(f"✅ 任务停止完成: ID={task_id}")
            
            return {
                "task_id": task_id,
                "status": "stopped",
                "message": "任务已停止"
            }
    
    async def _stop_task_components(self, task_id: int):
        """停止任务的所有组件
        
        Args:
            task_id: 任务ID
        """
        if task_id not in self.task_components:
            self.logger.warning(f"⚠️ 任务 {task_id} 没有组件引用，可能已经清理")
            return
        
        components = self.task_components[task_id]
        stop_errors = []  # 收集停止过程中的错误
        
        # 停止处理器
        try:
            processors = components.get('processors', [])
            for i, processor in enumerate(processors):
                try:
                    processor_name = getattr(processor, 'stream_id', f"processor_{i+1}")
                    self.logger.info(f"⏹️ 停止处理器 {i+1}/{len(processors)}: {processor_name}")
                    processor.stop()
                except Exception as e:
                    processor_name = getattr(processor, 'stream_id', f"processor_{i+1}")
                    error_msg = f"停止处理器 {processor_name} 失败: {e}"
                    self.logger.error(f"❌ {error_msg}")
                    stop_errors.append(error_msg)
        except Exception as e:
            error_msg = f"停止处理器组失败: {e}"
            self.logger.error(f"❌ {error_msg}")
            stop_errors.append(error_msg)
        
        # 停止分析引擎
        try:
            analysis_engine = components.get('analysis_engine')
            if analysis_engine:
                self.logger.info(f"⏹️ 停止分析引擎")
                analysis_engine.stop_all()
        except Exception as e:
            error_msg = f"停止分析引擎失败: {e}"
            self.logger.error(f"❌ {error_msg}")
            stop_errors.append(error_msg)
        
        # 停止结果处理管道
        try:
            result_pipeline = components.get('result_pipeline')
            if result_pipeline:
                self.logger.info(f"⏹️ 停止结果处理管道")
                # 在异步上下文中运行
                import asyncio
                try:
                    await result_pipeline.stop()
                except RuntimeError as e:
                    if "There is no current event loop" in str(e):
                        # 创建新的事件循环
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(result_pipeline.stop())
                        loop.close()
                    else:
                        raise
        except Exception as e:
            error_msg = f"停止结果处理管道失败: {e}"
            self.logger.error(f"❌ {error_msg}")
            stop_errors.append(error_msg)
        
        # 停止视频播放器
        try:
            video_player = components.get('video_player')
            if video_player:
                self.logger.info(f"⏹️ 停止视频播放器")
                video_player.stop()
        except Exception as e:
            error_msg = f"停止视频播放器失败: {e}"
            self.logger.error(f"❌ {error_msg}")
            stop_errors.append(error_msg)
        
        # 清理内存池
        try:
            memory_pool = components.get('memory_pool')
            if memory_pool:
                self.logger.info(f"⏹️ 清理内存池")
                memory_pool.cleanup()
        except Exception as e:
            error_msg = f"清理内存池失败: {e}"
            self.logger.error(f"❌ {error_msg}")
            stop_errors.append(error_msg)
        
        # 记录停止结果
        if stop_errors:
            self.logger.warning(f"⚠️ 任务 {task_id} 组件停止过程中出现 {len(stop_errors)} 个错误")
            for error in stop_errors:
                self.logger.warning(f"   - {error}")
            # 不抛出异常，允许任务继续停止流程
        else:
            self.logger.info(f"✅ 任务 {task_id} 所有组件已停止")
    
    async def get_task_detail(self, task_id: int) -> Optional[TaskInfo]:
        """查看任务详情
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[TaskInfo]: 任务详情，不存在返回None
        """
        if task_id not in self.tasks:
            return None
        
        task_data = self.tasks[task_id]
        return self._to_task_info(task_data)
    
    async def delete_task(self, task_id: int) -> Dict[str, Any]:
        """删除任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 删除结果
        """
        if task_id not in self.tasks:
            raise ValueError(f"任务 {task_id} 不存在")
        
        task_data = self.tasks[task_id]
        
        # 如果任务正在运行，先停止
        if task_data["status"] == 1:
            await self.stop_task(task_id)
        
        # 删除任务
        del self.tasks[task_id]
        
        self.logger.info(f"🗑️ 任务删除成功: ID={task_id}")
        
        return {
            "task_id": task_id,
            "message": "任务已删除"
        }
    
    async def restart_task(self, task_id: int) -> Dict[str, Any]:
        """重启任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 重启结果
        """
        if task_id not in self.tasks:
            raise ValueError(f"任务 {task_id} 不存在")
        
        task_data = self.tasks[task_id]
        
        # 如果任务正在运行，先停止
        if task_data["status"] == 1:
            await self.stop_task(task_id)
            # 等待一下确保任务完全停止
            await asyncio.sleep(1)
        
        # 重置任务状态
        task_data["progress"] = 0.0
        task_data["results"] = []
        task_data["updated_at"] = datetime.now()
        
        # 重新启动任务
        result = await self.start_task(task_id)
        
        self.logger.info(f"🔄 任务重启成功: ID={task_id}")
        
        return {
            "task_id": task_id,
            "status": "restarted",
            "message": "任务已重启"
        }
    
    async def list_tasks(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[int] = None
    ) -> Dict[str, Any]:
        """获取任务列表
        
        Args:
            page: 页码
            page_size: 每页大小
            status: 状态筛选
            
        Returns:
            Dict[str, Any]: 任务列表和分页信息
        """
        # 筛选任务
        filtered_tasks = []
        for task_data in self.tasks.values():
            if status is not None and task_data["status"] != status:
                continue
            filtered_tasks.append(task_data)
        
        # 排序（按创建时间倒序）
        filtered_tasks.sort(key=lambda x: x["created_at"], reverse=True)
        
        # 分页
        total = len(filtered_tasks)
        start = (page - 1) * page_size
        end = start + page_size
        page_tasks = filtered_tasks[start:end]
        
        # 转换为TaskInfo
        task_infos = [self._to_task_info(task_data) for task_data in page_tasks]
        
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "items": [asdict(task_info) for task_info in task_infos]
        }
    
    async def _execute_task(self, task_id: int):
        """执行任务（零拷贝AI分析）"""
        task_data = self.tasks[task_id]

        try:
            self.logger.info(f"🚀 开始执行零拷贝AI分析任务: ID={task_id}")

            # 导入零拷贝组件
            from app.core.zero_copy import (
                MemoryPool, TimeAxis, StreamCapture, MultiStreamCapture,
                VideoFileProcessor, ImageProcessor,
                MockAnalyzer, AnalysisWorker, AnalysisEngine, ResultProcessor,
                VideoPlayer
            )

            # 1. 内存预分配
            self.logger.info(f"🔧 任务 {task_id}: 预分配内存池...")
            
            # 根据分析类型调整内存池大小
            if task_data["analysis_type"] == 2:  # 视频分析
                pool_size = 400  # 视频分析需要更大的缓冲区，增加到200个
                self.logger.info(f"📹 视频分析任务，使用超大容量内存池: {pool_size} 个缓冲区")
            elif task_data["analysis_type"] == 3:  # 流分析
                pool_size = 80   # 流分析使用中等缓冲区，适当增加
                self.logger.info(f"📡 流分析任务，使用大容量内存池: {pool_size} 个缓冲区")
            else:
                pool_size = 50   # 其他类型使用默认缓冲区，适当增加
                self.logger.info(f"🔍 其他分析任务，使用标准内存池: {pool_size} 个缓冲区")
            
            memory_pool = MemoryPool(
                pool_size=pool_size,
                height=1080,
                width=1920,
                logger=self.logger
            )

            # 2. 构建时间轴
            self.logger.info(f"⏰ 任务 {task_id}: 构建时间轴...")
            
            # 根据分析类型调整时间轴配置
            if task_data["analysis_type"] == 2:  # 视频分析
                max_frames = 500  # 视频批处理需要更大的时间轴缓冲
                timeout = 5.0
            elif task_data["analysis_type"] == 3:  # 流分析
                max_frames = 200  # 流分析中等缓冲
                timeout = 0.2     # 关键修复：流分析需要更短的超时时间以实现实时处理
            else:  # 图片分析
                max_frames = 50   # 图片分析较小缓冲
                timeout = 1.0
            
            time_axis = TimeAxis(
                timeout_seconds=timeout,
                max_frames=max_frames,
                logger=self.logger
            )

            # 3. 创建视频播放器（如果启用）
            video_player = None
            if task_data.get("enable_video_player", False):
                self.logger.info(f"🎬 任务 {task_id}: 创建视频播放器...")
                video_player = VideoPlayer(
                    window_name=f"任务{task_id} - {task_data['name']}",
                    show_fps=True,
                    show_info=True,
                    logger=self.logger
                )
                video_player.start()

            # 4. 创建结果处理管道
            self.logger.info(f"🏭 任务 {task_id}: 创建结果处理管道...")
            
            # 导入新的结果处理管道
            from app.core.result_processing.result_pipeline import ResultProcessingPipeline
            
            # 创建结果处理管道，包含task_config参数传递
            task_config_with_id = task_data.copy()  # 复制所有任务配置
            task_config_with_id["task_id"] = task_id  # 确保task_id存在
            
            result_pipeline = ResultProcessingPipeline(
                task_id=task_id,
                task_config=task_config_with_id,  # 传递完整的任务配置
                time_axis_manager=None,  # 暂时不需要时间轴管理器
                output_dir="unused",  # 存储处理器会自动管理路径
                logger=self.logger
            )
            
            # 启动结果处理管道
            await result_pipeline.start()
            
            self.logger.info(f"🏭 任务 {task_id}: 结果处理管道已启动，处理器配置已就绪")

            # 4. 创建分析引擎
            self.logger.info(f"🖥 任务 {task_id}: 创建AI分析引擎...")
            analysis_engine = AnalysisEngine(self.logger)

            # 为每个模型创建检测分析器
            analyzers = {}
            confidence_threshold = task_data.get("confidence_threshold", 0.5)
            iou_threshold = task_data.get("iou_threshold", 0.45)
            
            for model_code in task_data["model_codes"]:
                # 创建基于模型类型的分析器，使用任务配置的参数
                analyzer = self.analyzer_factory.create_analyzer(
                    model_code=model_code,
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold,
                    device="auto"
                )
                
                if analyzer:
                    analyzers[model_code] = analyzer
                    self.logger.info(f"✅ 分析器创建成功: {model_code}")
                else:
                    self.logger.error(f"❌ 分析器创建失败: {model_code}")
                    # 可以选择跳过该模型或抛出异常
                    continue

            # 4. 根据分析类型创建对应的处理器
            processors = []
            
            if task_data["analysis_type"] == 1:  # 图片分析
                self.logger.info(f"🖼️ 任务 {task_id}: 创建图片处理器...")
                
                # 获取图片路径
                image_paths = task_data.get("image_paths", [])
                if not image_paths:
                    raise ValueError("图片分析任务需要提供image_paths")
                
                # 创建图片处理器
                image_processor = ImageProcessor(
                    image_path=image_paths,
                    memory_pool=memory_pool,
                    time_axis=time_axis,
                    stream_id="image_analysis",
                    batch_size=5,  # 每批处理5张图片
                    processing_delay=0.2,  # 每张图片间隔0.2秒
                    logger=self.logger,
                    on_all_complete_callback=lambda: self._on_analysis_complete(task_id, "image_analysis")
                )
                processors.append(image_processor)
                
                # 创建分析处理器 - 专门从时间轴读取数据进行分析
                analysis_processor = ImageAnalysisProcessor(
                    analyzers=analyzers,
                    time_axis=time_axis,
                    result_processor=result_pipeline,
                    task_id=task_id,
                    logger=self.logger
                )
                processors.append(analysis_processor)
                
            elif task_data["analysis_type"] == 3:  # 流分析
                self.logger.info(f"📡 任务 {task_id}: 创建流捕获器...")
                multi_capture = MultiStreamCapture(memory_pool, time_axis, self.logger)

                for i, stream_url in enumerate(task_data["stream_urls"]):
                    stream_id = f"stream_{i+1}"
                    capture = multi_capture.add_stream(stream_id, stream_url)
                    # 为第一个流设置视频播放器
                    if i == 0 and video_player:
                        capture.video_player = video_player
                    processors.append(capture)
                
                # 创建流分析处理器
                stream_analysis_processor = StreamAnalysisProcessor(
                    analyzers=analyzers,
                    time_axis=time_axis,
                    result_processor=result_pipeline,
                    task_id=task_id,
                    logger=self.logger
                )
                processors.append(stream_analysis_processor)

            elif task_data["analysis_type"] == 2:  # 视频分析
                self.logger.info(f" 任务 {task_id}: 创建视频文件处理器...")
                
                # 获取视频路径
                video_path = task_data.get("video_path")
                if not video_path:
                    raise ValueError("视频分析任务需要提供video_path")
                
                video_processor = VideoFileProcessor(
                    video_path=video_path,
                    memory_pool=memory_pool,
                    time_axis=time_axis,
                    stream_id="video_file",
                    batch_size=100,    # 每批处理100帧
                    max_queue_size=300,  # 队列容量300帧
                    video_player=video_player,
                    logger=self.logger,
                    on_video_end_callback=lambda: self._on_video_end(task_id, "video_file"),
                    on_batch_complete_callback=lambda batch_num, batch_size, total_processed: 
                        self.logger.debug(f"🔄 批次 {batch_num} 完成: {batch_size} 帧, 总计 {total_processed} 帧")
                )
                processors.append(video_processor)
                
                # 创建视频分析处理器
                video_analysis_processor = VideoAnalysisProcessor(
                    analyzers=analyzers,
                    time_axis=time_axis,
                    result_processor=result_pipeline,
                    task_id=task_id,
                    logger=self.logger
                )
                processors.append(video_analysis_processor)
            
            else:
                raise ValueError(f"不支持的分析类型: {task_data['analysis_type']}")
            
            self.logger.info(f"✅ 任务 {task_id}: 已创建 {len(processors)} 个处理器")

            # 5. 启动双管道处理
            self.logger.info(f"🔄 任务 {task_id}: 启动双管道处理...")

            # 启动分析引擎（分析管道）
            analysis_engine.start_all()

            # 启动处理器（数据管道）
            for processor in processors:
                processor.start()

            # 🔧 保存组件引用（关键修复）
            self.task_components[task_id] = {
                'processors': processors,
                'analysis_engine': analysis_engine,
                'result_pipeline': result_pipeline,
                'video_player': video_player,
                'memory_pool': memory_pool,
                'time_axis': time_axis
            }
            
            self.logger.info(f"✅ 任务 {task_id}: 组件引用已保存，可以正确停止")

            # 6. 监控任务执行
            start_time = time.time()
            last_progress_time = start_time

            while task_data["status"] == 1:  # 运行中
                current_time = time.time()
                runtime = current_time - start_time

                # 更新进度（基于运行时间）
                if current_time - last_progress_time >= 2.0:  # 每2秒更新一次
                    # 获取统计信息
                    memory_stats = memory_pool.get_stats()
                    time_axis_stats = time_axis.get_stats()
                    
                    # 【关键修复】从分析处理器获取统计信息，而不是分析引擎
                    total_processed = 0
                    for processor in processors:
                        if hasattr(processor, 'get_stats'):
                            processor_stats = processor.get_stats()
                            total_processed += processor_stats.get("total_processed", 0)
                        elif hasattr(processor, 'stats'):
                            total_processed += processor.stats.get("total_processed", 0)

                    # 计算进度（基于处理的帧数）
                    progress = min(95.0, (total_processed / 100.0) * 100)  # 最多95%，完成时设为100%

                    task_data["progress"] = progress
                    task_data["updated_at"] = datetime.now()

                    self.logger.info(f"📊 任务 {task_id} 进度: {progress:.1f}% | "
                                   f"内存使用: {memory_stats['current_usage']}/{memory_stats['pool_size']} | "
                                   f"时间轴: {time_axis_stats['current_size']} 帧 | "
                                   f"已处理: {total_processed} 帧")

                    last_progress_time = current_time

                await asyncio.sleep(1)  # 每秒检查一次

            # 7. 清理资源
            self.logger.info(f"🧹 任务 {task_id}: 清理资源...")

            # 停止处理器
            for processor in processors:
                processor.stop()

            # 停止分析引擎
            analysis_engine.stop_all()

            # 停止结果处理管道
            await result_pipeline.stop()

            # 停止视频播放器
            if video_player:
                video_player.stop()

            # 清理内存池
            memory_pool.cleanup()

            # 任务完成
            if task_data["status"] == 1:
                task_data["status"] = 4  # 已完成
                task_data["progress"] = 100.0
                task_data["updated_at"] = datetime.now()

                self.logger.info(f"✅ 零拷贝AI分析任务完成: ID={task_id}")

        except asyncio.CancelledError:
            self.logger.info(f"⏹️ 任务被取消: ID={task_id}")
        except Exception as e:
            self.logger.error(f"❌ 任务执行失败: ID={task_id}, 错误={str(e)}")
            task_data["status"] = 3  # 错误
            task_data["updated_at"] = datetime.now()
        finally:
            # 清理运行中的任务记录
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            # 清理组件引用
            if task_id in self.task_components:
                del self.task_components[task_id]
                self.logger.info(f"🧹 任务 {task_id}: 组件引用已清理")

    def _on_analysis_result(self, task_id: int, frame_buffers: list, results: list, result_processor=None, video_player=None):
        """分析结果回调"""
        if task_id not in self.tasks:
            return

        task_data = self.tasks[task_id]

        # 存储分析结果并保存图片
        for frame_buffer, result in zip(frame_buffers, results):
            # 存储到任务结果
            analysis_result = {
                "id": len(task_data["results"]) + 1,
                "frame_id": frame_buffer.frame_id,
                "timestamp": frame_buffer.timestamp,
                "stream_id": frame_buffer.stream_id,
                "detections": result.get("detections", []),
                "analyzer": result.get("analyzer", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "processing_time": result.get("processing_time", 0.0)
            }
            task_data["results"].append(analysis_result)

            # 发送给结果处理器保存图片
            if result_processor:
                result_processor.process_result(frame_buffer, task_id)

            # 发送分析结果给视频播放器
            if video_player:
                # 构建播放器需要的结果格式
                player_results = {
                    result.get("analyzer", "unknown"): result
                }
                video_player.add_analysis_result(frame_buffer.frame_id, player_results)

        # 限制结果数量（避免内存过多）
        if len(task_data["results"]) > 1000:
            task_data["results"] = task_data["results"][-1000:]  # 保留最新1000个结果

        # 计算进度（基于处理的帧数）
        total_frames = task_data.get("total_frames", 0)
        processed_frames = len(task_data["results"])
        if total_frames > 0:
            task_data["progress"] = min(100.0, (processed_frames / total_frames) * 100)
        else:
            # 没有总帧数时，基于时间估算进度
            runtime = time.time() - task_data.get("start_time", time.time())
            estimated_total_time = 60.0  # 假设60秒的任务
            task_data["progress"] = min(100.0, (runtime / estimated_total_time) * 100)
    
    def _on_video_end(self, task_id: int, stream_id: str):
        """视频文件播放结束回调"""
        self.logger.info(f"🏁 视频播放结束回调: 任务ID={task_id}, 流ID={stream_id}")
        
        if task_id not in self.tasks:
            self.logger.warning(f"⚠️ 任务 {task_id} 不存在，无法处理视频结束回调")
            return
        
        task_data = self.tasks[task_id]
        
        # 检查任务是否还在运行
        if task_data["status"] != 1:
            self.logger.info(f"📋 任务 {task_id} 已经停止，忽略视频结束回调")
            return
        
        self.logger.info(f"🎬 视频文件播放完毕，自动停止任务: {task_id}")
        
        # 使用同步方式处理停止操作，避免事件循环问题
        try:
            # 直接更新任务状态为停止
            task_data["status"] = 2  # 已停止
            task_data["progress"] = 100.0
            task_data["updated_at"] = datetime.now()
            self.logger.info(f"✅ 任务 {task_id} 已标记为完成状态")
            
            # 停止运行中的异步任务
            if task_id in self.running_tasks:
                running_task = self.running_tasks[task_id]
                if not running_task.done():
                    running_task.cancel()
                    self.logger.info(f"🛑 任务 {task_id} 异步任务已取消")
                
        except Exception as e:
            self.logger.error(f"❌ 自动停止任务 {task_id} 失败: {e}")

    def _on_analysis_complete(self, task_id: int, stream_id: str):
        """分析完成回调（用于图片分析）"""
        self.logger.info(f"🖼️ 分析完成回调: 任务ID={task_id}, 流ID={stream_id}")
        
        if task_id not in self.tasks:
            self.logger.warning(f"⚠️ 任务 {task_id} 不存在，无法处理分析完成回调")
            return
        
        task_data = self.tasks[task_id]
        
        # 检查任务是否还在运行
        if task_data["status"] != 1:
            self.logger.info(f"📋 任务 {task_id} 已经停止，忽略分析完成回调")
            return
        
        self.logger.info(f"🎯 分析任务处理完毕，自动停止任务: {task_id}")
        
        # 使用同步方式处理停止操作，避免事件循环问题
        try:
            # 直接更新任务状态为完成
            task_data["status"] = 4  # 已完成
            task_data["progress"] = 100.0
            task_data["updated_at"] = datetime.now()
            self.logger.info(f"✅ 任务 {task_id} 已标记为完成状态")
            
            # 停止运行中的异步任务
            if task_id in self.running_tasks:
                running_task = self.running_tasks[task_id]
                if not running_task.done():
                    running_task.cancel()
                    self.logger.info(f"🛑 任务 {task_id} 异步任务已取消")
                
        except Exception as e:
            self.logger.error(f"❌ 自动完成任务 {task_id} 失败: {e}")

    def _to_task_info(self, task_data: Dict[str, Any]) -> TaskInfo:
        """将任务数据转换为TaskInfo"""
        return TaskInfo(
            id=task_data["id"],
            name=task_data["name"],
            description=task_data["description"],
            analysis_type=task_data["analysis_type"],
            status=task_data["status"],
            progress=task_data["progress"],
            created_at=task_data["created_at"].isoformat(),
            updated_at=task_data["updated_at"].isoformat(),
            model_codes=task_data["model_codes"],
            stream_urls=task_data["stream_urls"],
            result_count=len(task_data.get("results", []))
        )


# 全局任务服务实例
_task_service = None


def get_task_service() -> TaskService:
    """获取任务服务实例"""
    global _task_service
    if _task_service is None:
        _task_service = TaskService()
    return _task_service
