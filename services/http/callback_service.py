"""
回调服务
负责处理HTTP回调的发送
"""
from typing import Dict, Any, List, Optional, Union
import asyncio
import aiohttp
import json
import time
from datetime import datetime
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class CallbackService:
    """HTTP回调服务"""

    def __init__(self):
        """初始化回调服务"""
        # 存储每个任务的最后回调时间
        self.last_callback_times: Dict[str, Dict[str, float]] = {}
        # 存储每个任务的回调间隔设置
        self.callback_intervals: Dict[str, Dict[str, int]] = {}
        # 存储每个任务的回调URL
        self.callback_urls: Dict[str, str] = {}
        # 存储每个任务的回调状态
        self.callback_enabled: Dict[str, bool] = {}

    def register_task(self, task_id: str, callback_url: str, enable_callback: bool, 
                     callback_interval: Optional[int] = None):
        """
        注册任务回调信息

        Args:
            task_id: 任务ID
            callback_url: 回调URL
            enable_callback: 是否启用回调
            callback_interval: 回调间隔(秒)，如果为None则不限制回调频率
        """
        if not callback_url or not enable_callback:
            normal_logger.info(f"任务 {task_id} 未启用回调或未提供回调URL")
            return

        normal_logger.info(f"注册任务回调: {task_id}, URL: {callback_url}, 间隔: {callback_interval}秒")
        
        # 存储回调URL和状态
        self.callback_urls[task_id] = callback_url
        self.callback_enabled[task_id] = enable_callback
        
        # 初始化最后回调时间
        if task_id not in self.last_callback_times:
            self.last_callback_times[task_id] = {}
        
        # 设置回调间隔
        if callback_interval is not None and callback_interval > 0:
            if task_id not in self.callback_intervals:
                self.callback_intervals[task_id] = {}
            self.callback_intervals[task_id]["default"] = callback_interval

    def register_object_callback_interval(self, task_id: str, object_id: str, interval: int):
        """
        注册对象回调间隔

        Args:
            task_id: 任务ID
            object_id: 对象ID
            interval: 回调间隔(秒)
        """
        if task_id not in self.callback_intervals:
            self.callback_intervals[task_id] = {}
        
        self.callback_intervals[task_id][object_id] = interval
        normal_logger.info(f"注册对象回调间隔: 任务 {task_id}, 对象 {object_id}, 间隔: {interval}秒")

    def unregister_task(self, task_id: str):
        """
        取消注册任务回调信息

        Args:
            task_id: 任务ID
        """
        # 移除任务相关的所有回调信息
        if task_id in self.callback_urls:
            del self.callback_urls[task_id]
        
        if task_id in self.callback_enabled:
            del self.callback_enabled[task_id]
        
        if task_id in self.last_callback_times:
            del self.last_callback_times[task_id]
        
        if task_id in self.callback_intervals:
            del self.callback_intervals[task_id]
        
        normal_logger.info(f"取消注册任务回调: {task_id}")

    async def send_callback(self, task_id: str, data: Dict[str, Any], 
                          object_id: Optional[str] = None) -> bool:
        """
        发送回调

        Args:
            task_id: 任务ID
            data: 回调数据（已经格式化的数据，包含task_id、timestamp和data字段）
            object_id: 对象ID，用于对象级回调间隔控制

        Returns:
            bool: 是否发送成功
        """
        # 检查任务是否启用回调
        if task_id not in self.callback_enabled or not self.callback_enabled[task_id]:
            return False
        
        # 检查是否有回调URL
        if task_id not in self.callback_urls or not self.callback_urls[task_id]:
            return False
        
        # 获取回调URL
        callback_url = self.callback_urls[task_id]
        
        # 检查回调间隔
        current_time = time.time()
        
        # 如果指定了对象ID，检查对象级回调间隔
        if object_id is not None:
            # 获取对象的最后回调时间
            last_time = self.last_callback_times.get(task_id, {}).get(object_id, 0)
            
            # 获取对象的回调间隔
            interval = self.callback_intervals.get(task_id, {}).get(object_id)
            if interval is None:
                # 如果没有对象级间隔，使用默认间隔
                interval = self.callback_intervals.get(task_id, {}).get("default", 0)
            
            # 检查是否满足回调间隔
            if interval > 0 and current_time - last_time < interval:
                normal_logger.debug(f"对象回调间隔未满足: 任务 {task_id}, 对象 {object_id}, "
                           f"间隔: {interval}秒, 上次: {current_time - last_time:.2f}秒前")
                return False
            
            # 更新最后回调时间
            if task_id not in self.last_callback_times:
                self.last_callback_times[task_id] = {}
            self.last_callback_times[task_id][object_id] = current_time
        else:
            # 检查任务级回调间隔
            last_time = self.last_callback_times.get(task_id, {}).get("default", 0)
            interval = self.callback_intervals.get(task_id, {}).get("default", 0)
            
            # 检查是否满足回调间隔
            if interval > 0 and current_time - last_time < interval:
                normal_logger.debug(f"任务回调间隔未满足: 任务 {task_id}, "
                           f"间隔: {interval}秒, 上次: {current_time - last_time:.2f}秒前")
                return False
            
            # 更新最后回调时间
            if task_id not in self.last_callback_times:
                self.last_callback_times[task_id] = {}
            self.last_callback_times[task_id]["default"] = current_time
        
        # 直接使用传入的数据，因为它已经是格式化后的数据
        callback_data = data
        
        # 发送回调
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    callback_url, 
                    json=callback_data,
                    timeout=aiohttp.ClientTimeout(total=5)  # 5秒超时
                ) as response:
                    if response.status == 200:
                        normal_logger.info(f"回调发送成功: 任务 {task_id}, URL: {callback_url}")
                        return True
                    else:
                        normal_logger.warning(f"回调发送失败: 任务 {task_id}, URL: {callback_url}, "
                                     f"状态码: {response.status}")
                        return False
        except Exception as e:
            exception_logger.exception(f"回调发送异常: 任务 {task_id}, URL: {callback_url}, 错误: {str(e)}")
            return False
