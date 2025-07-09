#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回调处理器 - 负责处理分析结果的回调需求
"""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, Any, Optional, List
import aiohttp

from app.core.zero_copy.frame_buffer import FrameBuffer
from app.models.analysis_result import AnalysisResult
from .base_processor import BaseResultProcessor


class CallbackProcessor(BaseResultProcessor):
    """回调处理器 - 发送结果到指定的回调地址"""
    
    def __init__(self, 
                 callback_urls: List[str] = None,
                 callback_interval: int = 0,
                 max_retries: int = 3,
                 timeout: int = 10,
                 logger: Optional[logging.Logger] = None):
        """
        初始化回调处理器
        
        Args:
            callback_urls: 回调地址列表
            callback_interval: 回调间隔（秒），0表示实时回调
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            logger: 日志记录器
        """
        super().__init__(logger)
        
        self.callback_urls = callback_urls or []
        self.callback_interval = callback_interval
        self.max_retries = max_retries
        self.timeout = timeout
        
        # 批量回调缓存
        self.callback_buffer = []
        self.last_callback_time = 0
        
        # 异步任务队列（后台处理）
        self.pending_tasks = []
        self.executor_thread = None
        self.running = True
        
        # 统计信息
        self.stats = {
            "total_callbacks": 0,
            "successful_callbacks": 0,
            "failed_callbacks": 0,
            "retry_count": 0
        }
        
        # 启动后台任务处理线程
        self._start_background_executor()
        
        self.logger.info(f"📞 回调处理器初始化: {len(self.callback_urls)} 个回调地址")
    
    def _start_background_executor(self):
        """启动后台任务执行器"""
        def executor():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            while self.running:
                try:
                    if self.pending_tasks:
                        # 处理待执行的任务
                        tasks_to_run = self.pending_tasks.copy()
                        self.pending_tasks.clear()
                        
                        for task_func, args in tasks_to_run:
                            try:
                                loop.run_until_complete(task_func(*args))
                            except Exception as e:
                                self.logger.error(f"❌ 后台任务执行失败: {e}")
                    
                    time.sleep(0.1)  # 避免CPU占用过高
                    
                except Exception as e:
                    self.logger.error(f"❌ 后台执行器异常: {e}")
                    time.sleep(1)
            
            loop.close()
        
        self.executor_thread = threading.Thread(target=executor, daemon=True)
        self.executor_thread.start()
    
    def process(self, frame_buffer: FrameBuffer, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理回调需求
        
        Args:
            frame_buffer: 帧缓冲区
            task_config: 任务配置
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        result = {
            "processor": "callback",
            "success": False,
            "callbacks_sent": 0,
            "errors": []
        }
        
        try:
            # 检查是否启用回调
            callback_urls = task_config.get("callback_urls", self.callback_urls)
            if not callback_urls:
                result["success"] = True
                result["message"] = "未配置回调地址"
                return result
            
            # 检查是否有分析结果
            if not frame_buffer.analysis_results:
                result["success"] = True
                result["message"] = "无分析结果，跳过回调"
                return result
            
            # 构建回调数据
            callback_data = self._build_callback_data(frame_buffer, task_config)
            
            # 检查是否需要立即回调或缓存
            current_time = time.time()
            if self.callback_interval == 0:
                # 实时回调 - 添加到后台任务队列
                self.pending_tasks.append((self._send_callbacks, (callback_urls, callback_data)))
                result["callbacks_sent"] = len(callback_urls)
                result["success"] = True
            else:
                # 批量回调
                self.callback_buffer.append(callback_data)
                
                # 检查是否达到回调间隔
                if current_time - self.last_callback_time >= self.callback_interval:
                    self.pending_tasks.append((self._send_batch_callbacks, (callback_urls,)))
                    result["callbacks_sent"] = len(callback_urls)
                    result["success"] = True
                else:
                    result["success"] = True
                    result["message"] = f"缓存回调数据，等待间隔时间({self.callback_interval}s)"
            
            self.logger.debug(f"📞 回调处理完成: 任务{task_config.get('task_id')}, "
                            f"帧{frame_buffer.frame_id}")
            
        except Exception as e:
            error_msg = f"回调处理异常: {e}"
            result["errors"].append(error_msg)
            self.logger.error(f"❌ {error_msg}")
        
        return result
    
    def _build_callback_data(self, frame_buffer: FrameBuffer, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """构建回调数据"""
        return {
            "task_id": task_config.get("task_id"),
            "frame_id": frame_buffer.frame_id,
            "timestamp": frame_buffer.timestamp,
            "stream_id": frame_buffer.stream_id,
            "analysis_results": frame_buffer.analysis_results,
            "callback_time": time.time()
        }
    
    async def _send_callbacks(self, callback_urls: List[str], data: Dict[str, Any]):
        """发送实时回调"""
        tasks = []
        for url in callback_urls:
            task = asyncio.create_task(self._send_single_callback(url, data))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_batch_callbacks(self, callback_urls: List[str]):
        """发送批量回调"""
        if not self.callback_buffer:
            return
        
        batch_data = {
            "type": "batch_callback",
            "data": self.callback_buffer.copy(),
            "batch_size": len(self.callback_buffer),
            "timestamp": time.time()
        }
        
        # 清空缓存
        self.callback_buffer.clear()
        self.last_callback_time = time.time()
        
        # 发送批量数据
        await self._send_callbacks(callback_urls, batch_data)
    
    async def _send_single_callback(self, url: str, data: Dict[str, Any]):
        """发送单个回调"""
        retries = 0
        
        while retries <= self.max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status == 200:
                            self.stats["successful_callbacks"] += 1
                            self.logger.debug(f"✅ 回调成功: {url}")
                            return
                        else:
                            self.logger.warning(f"⚠️ 回调响应异常: {url}, 状态码: {response.status}")
                
            except Exception as e:
                retries += 1
                self.stats["retry_count"] += 1
                
                if retries > self.max_retries:
                    self.stats["failed_callbacks"] += 1
                    self.logger.error(f"❌ 回调失败: {url}, 错误: {e}")
                    break
                else:
                    self.logger.warning(f"⚠️ 回调重试 {retries}/{self.max_retries}: {url}")
                    await asyncio.sleep(1 * retries)  # 递增延迟
        
        self.stats["total_callbacks"] += 1
    
    def get_callback_stats(self) -> Dict[str, Any]:
        """获取回调统计信息"""
        return {
            **self.stats,
            "callback_urls": self.callback_urls,
            "callback_interval": self.callback_interval,
            "buffer_size": len(self.callback_buffer),
            "pending_tasks": len(self.pending_tasks),
            "success_rate": (self.stats["successful_callbacks"] / max(1, self.stats["total_callbacks"])) * 100
        }
    
    def cleanup(self):
        """清理资源"""
        self.running = False
        
        # 等待后台线程结束
        if self.executor_thread and self.executor_thread.is_alive():
            self.executor_thread.join(timeout=3.0)
        
        # 清空缓存
        self.callback_buffer.clear()
        self.pending_tasks.clear()
        
        self.logger.info("🧹 回调处理器已清理")
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass 