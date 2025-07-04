#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: rtsp_plugin.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: RTSP流媒体插件

处理RTSP协议的流媒体数据。

本文件是分析服务项目的一部分。
"""

import asyncio
import threading
from typing import Dict, Any, Optional
from datetime import datetime

from ..base_plugin import StreamPlugin, PluginInfo, PluginStatus


class RTSPPlugin(StreamPlugin):
    """RTSP流媒体插件"""
    
    def __init__(self):
        """初始化RTSP插件"""
        super().__init__()
        self._active_streams: Dict[str, Dict[str, Any]] = {}
        self._stream_tasks: Dict[str, asyncio.Task] = {}
        self._stop_events: Dict[str, asyncio.Event] = {}
    
    def get_plugin_info(self) -> PluginInfo:
        """获取插件信息"""
        return PluginInfo(
            name="rtsp_plugin",
            version="1.0.0",
            description="RTSP流媒体处理插件",
            author="Yanli",
            dependencies=[],
            category="stream",
            priority=90,
            enabled=True
        )
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """初始化插件"""
        try:
            self.logger.info("初始化RTSP插件")
            
            # 设置默认配置
            default_config = {
                'max_concurrent_streams': 20,
                'connection_timeout': 15,
                'read_timeout': 5,
                'buffer_size': 2 * 1024 * 1024,  # 2MB
                'retry_attempts': 5,
                'retry_delay': 3,
                'transport_protocol': 'tcp',  # tcp or udp
                'auth_timeout': 10
            }
            
            if config:
                default_config.update(config)
            
            self.set_config(default_config)
            
            self.logger.info("RTSP插件初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"RTSP插件初始化失败: {e}")
            return False
    
    async def start(self) -> bool:
        """启动插件"""
        try:
            self.logger.info("启动RTSP插件")
            
            # 这里可以添加插件启动逻辑
            # 例如：初始化连接池、启动监控任务等
            
            self.logger.info("RTSP插件启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"RTSP插件启动失败: {e}")
            return False
    
    async def stop(self) -> bool:
        """停止插件"""
        try:
            self.logger.info("停止RTSP插件")
            
            # 停止所有活跃的流
            stream_urls = list(self._active_streams.keys())
            for stream_url in stream_urls:
                await self.stop_stream(stream_url)
            
            self.logger.info("RTSP插件停止成功")
            return True
            
        except Exception as e:
            self.logger.error(f"RTSP插件停止失败: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """清理插件资源"""
        try:
            self.logger.info("清理RTSP插件资源")
            
            # 确保所有流都已停止
            await self.stop()
            
            # 清理数据结构
            self._active_streams.clear()
            self._stream_tasks.clear()
            self._stop_events.clear()
            
            self.logger.info("RTSP插件资源清理完成")
            return True
            
        except Exception as e:
            self.logger.error(f"RTSP插件资源清理失败: {e}")
            return False
    
    async def process_stream(self, stream_url: str, config: Dict[str, Any] = None) -> bool:
        """处理RTSP流
        
        Args:
            stream_url: RTSP流URL
            config: 处理配置
            
        Returns:
            bool: 是否处理成功
        """
        if not stream_url.startswith('rtsp://'):
            self.logger.error(f"无效的RTSP URL: {stream_url}")
            return False
        
        if stream_url in self._active_streams:
            self.logger.warning(f"RTSP流已在处理中: {stream_url}")
            return True
        
        try:
            self.logger.info(f"开始处理RTSP流: {stream_url}")
            
            # 检查并发限制
            max_streams = self.get_config().get('max_concurrent_streams', 20)
            if len(self._active_streams) >= max_streams:
                self.logger.error(f"达到最大并发流限制: {max_streams}")
                return False
            
            # 创建停止事件
            stop_event = asyncio.Event()
            self._stop_events[stream_url] = stop_event
            
            # 创建流信息
            stream_info = {
                'url': stream_url,
                'config': config or {},
                'start_time': datetime.now(),
                'status': 'connecting',
                'frames_processed': 0,
                'bytes_received': 0,
                'last_frame_time': None,
                'error_count': 0,
                'connection_attempts': 0
            }
            
            self._active_streams[stream_url] = stream_info
            
            # 启动处理任务
            task = asyncio.create_task(self._process_stream_async(stream_url, stop_event))
            self._stream_tasks[stream_url] = task
            
            self.logger.info(f"RTSP流处理启动成功: {stream_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"处理RTSP流失败 {stream_url}: {e}")
            
            # 清理失败的流
            await self._cleanup_stream(stream_url)
            return False
    
    async def stop_stream(self, stream_url: str) -> bool:
        """停止RTSP流处理
        
        Args:
            stream_url: RTSP流URL
            
        Returns:
            bool: 是否停止成功
        """
        if stream_url not in self._active_streams:
            self.logger.warning(f"RTSP流未在处理中: {stream_url}")
            return True
        
        try:
            self.logger.info(f"停止RTSP流处理: {stream_url}")
            
            # 设置停止事件
            if stream_url in self._stop_events:
                self._stop_events[stream_url].set()
            
            # 取消任务
            if stream_url in self._stream_tasks:
                task = self._stream_tasks[stream_url]
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        self.logger.warning(f"RTSP流处理任务强制取消: {stream_url}")
            
            # 清理流资源
            await self._cleanup_stream(stream_url)
            
            self.logger.info(f"RTSP流处理停止成功: {stream_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"停止RTSP流处理失败 {stream_url}: {e}")
            return False
    
    def get_stream_status(self, stream_url: str) -> Dict[str, Any]:
        """获取RTSP流状态
        
        Args:
            stream_url: RTSP流URL
            
        Returns:
            Dict[str, Any]: 流状态信息
        """
        if stream_url not in self._active_streams:
            return {
                'url': stream_url,
                'status': 'not_found',
                'message': '流未在处理中'
            }
        
        stream_info = self._active_streams[stream_url].copy()
        
        # 计算运行时间
        if 'start_time' in stream_info:
            runtime = (datetime.now() - stream_info['start_time']).total_seconds()
            stream_info['runtime_seconds'] = runtime
        
        # 计算帧率
        if stream_info.get('frames_processed', 0) > 0 and 'runtime_seconds' in stream_info:
            stream_info['fps'] = stream_info['frames_processed'] / stream_info['runtime_seconds']
        else:
            stream_info['fps'] = 0.0
        
        # 添加任务状态
        if stream_url in self._stream_tasks:
            task = self._stream_tasks[stream_url]
            stream_info['task_done'] = task.done()
            stream_info['task_cancelled'] = task.cancelled()
        
        return stream_info
    
    async def _process_stream_async(self, stream_url: str, stop_event: asyncio.Event):
        """异步流处理
        
        Args:
            stream_url: 流URL
            stop_event: 停止事件
        """
        try:
            self.logger.info(f"RTSP流处理任务启动: {stream_url}")
            
            stream_info = self._active_streams[stream_url]
            config = self.get_config()
            
            # 模拟连接过程
            stream_info['status'] = 'connecting'
            stream_info['connection_attempts'] += 1
            
            # 模拟连接延迟
            await asyncio.sleep(1.0)
            
            if stop_event.is_set():
                return
            
            # 模拟连接成功
            stream_info['status'] = 'connected'
            self.logger.info(f"RTSP流连接成功: {stream_url}")
            
            frame_count = 0
            while not stop_event.is_set():
                try:
                    # 模拟读取帧数据
                    # 在实际实现中，这里会使用FFmpeg或其他库读取RTSP流
                    
                    # 模拟处理延迟（25fps）
                    await asyncio.sleep(0.04)
                    
                    if stop_event.is_set():
                        break
                    
                    # 更新统计信息
                    frame_count += 1
                    stream_info['frames_processed'] = frame_count
                    stream_info['last_frame_time'] = datetime.now()
                    stream_info['bytes_received'] += 2048  # 模拟数据
                    
                    # 模拟偶尔的网络问题
                    if frame_count % 500 == 0:
                        self.logger.debug(f"RTSP流处理进度 {stream_url}: {frame_count} 帧")
                        
                        # 模拟网络抖动
                        if frame_count % 2000 == 0:
                            await asyncio.sleep(0.1)
                    
                except asyncio.CancelledError:
                    self.logger.info(f"RTSP流处理任务被取消: {stream_url}")
                    break
                except Exception as e:
                    stream_info['error_count'] += 1
                    self.logger.error(f"RTSP流处理错误 {stream_url}: {e}")
                    
                    # 如果错误太多，停止处理
                    max_errors = config.get('max_errors', 20)
                    if stream_info['error_count'] > max_errors:
                        self.logger.error(f"RTSP流错误过多，停止处理: {stream_url}")
                        break
                    
                    # 短暂等待后重试
                    await asyncio.sleep(config.get('retry_delay', 3))
            
            stream_info['status'] = 'stopped'
            self.logger.info(f"RTSP流处理任务结束: {stream_url}")
            
        except asyncio.CancelledError:
            self.logger.info(f"RTSP流处理任务被取消: {stream_url}")
            if stream_url in self._active_streams:
                self._active_streams[stream_url]['status'] = 'cancelled'
        except Exception as e:
            self.logger.error(f"RTSP流处理任务异常 {stream_url}: {e}")
            if stream_url in self._active_streams:
                self._active_streams[stream_url]['status'] = 'error'
                self._active_streams[stream_url]['error_message'] = str(e)
    
    async def _cleanup_stream(self, stream_url: str):
        """清理流资源
        
        Args:
            stream_url: 流URL
        """
        # 移除流信息
        if stream_url in self._active_streams:
            del self._active_streams[stream_url]
        
        # 移除任务引用
        if stream_url in self._stream_tasks:
            del self._stream_tasks[stream_url]
        
        # 移除停止事件
        if stream_url in self._stop_events:
            del self._stop_events[stream_url]
    
    def get_all_streams_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有流的状态
        
        Returns:
            Dict[str, Dict[str, Any]]: 所有流的状态信息
        """
        status = {}
        for stream_url in self._active_streams:
            status[stream_url] = self.get_stream_status(stream_url)
        return status
    
    def get_plugin_statistics(self) -> Dict[str, Any]:
        """获取插件统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        total_streams = len(self._active_streams)
        active_streams = sum(1 for info in self._active_streams.values() if info.get('status') == 'connected')
        total_frames = sum(info.get('frames_processed', 0) for info in self._active_streams.values())
        total_bytes = sum(info.get('bytes_received', 0) for info in self._active_streams.values())
        total_errors = sum(info.get('error_count', 0) for info in self._active_streams.values())
        
        return {
            'plugin_name': self.get_info().name,
            'plugin_version': self.get_info().version,
            'total_streams': total_streams,
            'active_streams': active_streams,
            'total_frames_processed': total_frames,
            'total_bytes_received': total_bytes,
            'total_errors': total_errors,
            'config': self.get_config()
        }
    
    async def reconnect_stream(self, stream_url: str) -> bool:
        """重连RTSP流
        
        Args:
            stream_url: RTSP流URL
            
        Returns:
            bool: 是否重连成功
        """
        if stream_url not in self._active_streams:
            self.logger.error(f"流不存在，无法重连: {stream_url}")
            return False
        
        self.logger.info(f"重连RTSP流: {stream_url}")
        
        # 保存原始配置
        original_config = self._active_streams[stream_url].get('config', {})
        
        # 停止当前流
        await self.stop_stream(stream_url)
        
        # 等待一小段时间
        await asyncio.sleep(1.0)
        
        # 重新启动流
        return await self.process_stream(stream_url, original_config)
