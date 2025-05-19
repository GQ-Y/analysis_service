"""
HTTP协议处理模块
实现HTTP/HTTPS流的拉取和处理
"""

import asyncio
import time
import cv2
import numpy as np
import os
import aiohttp
from typing import Dict, Any, Optional, Tuple, List, Union
from urllib.parse import urlparse, parse_qs
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

from ...base.base_stream import BaseStream
from ...base.stream_interface import StreamStatus, StreamHealthStatus
from .config import HttpConfig, http_config

class HttpStream(BaseStream):
    """HTTP流类，实现HTTP/HTTPS协议的流处理"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化HTTP流
        
        Args:
            config: 流配置
        """
        # 调用基类初始化
        super().__init__(config.get("stream_id", ""), config)
        
        # HTTP特有配置
        self._http_config = HttpConfig()
        
        # 如果配置中有HTTP特有配置，更新
        http_extra = config.get("http", {})
        if http_extra:
            self._http_config = HttpConfig.from_dict(http_extra)
        
        # OpenCV视频捕获对象
        self._cap = None
        
        # HTTP会话
        self._session = None
        
        # 是否为图像流（单张图片或MJPEG）或视频文件
        self._is_image_stream = False
        self._is_video_file = False
        self._is_hls_stream = False
        
        # 内容类型
        self._content_type = ""
        
        # 缓存相关
        self._cache_path = None
        self._cache_file = None
        self._cache_timestamp = 0
        
        # 重连配置
        self._retry_count = 0
        self._max_retries = self._http_config.retry_count
        self._retry_interval = self._http_config.retry_interval / 1000.0  # 转换为秒
        
        # 超时配置
        self._timeout = self._http_config.timeout / 1000.0  # 转换为秒
        
        # 解析URL
        self._parse_url()
        
        normal_logger.info(f"创建HTTP流: {self._stream_id}, URL: {self._url}")
    
    def _parse_url(self) -> None:
        """解析URL，判断是图像流还是视频文件"""
        # 解析URL
        parsed_url = urlparse(self._url)
        path = parsed_url.path.lower()
        
        # 判断是否为视频文件
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.webm', '.wmv', '.ts']
        self._is_video_file = any(path.endswith(ext) for ext in video_extensions)
        
        # 如果路径中包含图片扩展名，可能是单张图片或MJPEG流
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        if any(path.endswith(ext) for ext in image_extensions):
            self._is_image_stream = True
            
        # 检查是否为HLS流
        if path.endswith('.m3u8'):
            self._is_video_file = True
            self._is_hls_stream = True
    
    async def _start_pulling(self) -> bool:
        """开始拉流
        
        Returns:
            bool: 是否成功启动拉流
        """
        try:
            # 创建HTTP会话
            self._session = aiohttp.ClientSession()
            
            # 准备请求头
            headers = {
                'User-Agent': self._http_config.user_agent
            }
            
            # 添加自定义头
            if self._http_config.headers:
                headers.update(self._http_config.headers)
            
            # 准备认证信息
            auth = None
            if self._http_config.auth_enable and self._http_config.auth_user and self._http_config.auth_password:
                auth = aiohttp.BasicAuth(self._http_config.auth_user, self._http_config.auth_password)
            
            # 首先发送HEAD请求获取内容类型
            try:
                async with self._session.head(
                    self._url, 
                    headers=headers,
                    auth=auth,
                    timeout=self._timeout
                ) as response:
                    if response.status == 200:
                        self._content_type = response.headers.get('Content-Type', '').lower()
                        normal_logger.info(f"HTTP流内容类型: {self._content_type}")
                        
                        # 根据内容类型判断流类型
                        if 'video/' in self._content_type:
                            self._is_video_file = True
                            self._is_image_stream = False
                        elif 'application/vnd.apple.mpegurl' in self._content_type or 'application/x-mpegURL' in self._content_type:
                            # HLS流
                            self._is_video_file = True
                            self._is_hls_stream = True
                            self._is_image_stream = False
                        elif 'image/jpeg' in self._content_type or 'image/jpg' in self._content_type:
                            if 'multipart/x-mixed-replace' in self._content_type:
                                # MJPEG流
                                self._is_image_stream = True
                                self._is_video_file = False
                            else:
                                # 单张JPEG图片
                                self._is_image_stream = True
                                self._is_video_file = False
                        elif 'image/' in self._content_type:
                            # 其他图片格式
                            self._is_image_stream = True
                            self._is_video_file = False
            except Exception as e:
                normal_logger.warning(f"获取HTTP内容类型失败: {str(e)}，将根据URL推断流类型")
            
            # 如果是视频文件，并且需要缓存
            if self._is_video_file and self._http_config.use_cache:
                # 创建缓存目录
                cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'temp', 'http_cache')
                os.makedirs(cache_dir, exist_ok=True)
                
                # 生成缓存文件名
                url_hash = str(hash(self._url))
                ext = os.path.splitext(urlparse(self._url).path)[1] or '.mp4'
                self._cache_path = os.path.join(cache_dir, f"{url_hash}{ext}")
                
                # 检查缓存是否有效
                if os.path.exists(self._cache_path):
                    # 获取文件修改时间
                    mtime = os.path.getmtime(self._cache_path)
                    self._cache_timestamp = mtime
                    
                    # 检查是否过期
                    if time.time() - mtime < self._http_config.cache_timeout:
                        normal_logger.info(f"使用缓存文件: {self._cache_path}")
                        self._cap = cv2.VideoCapture(self._cache_path)
                        
                        if not self._cap.isOpened():
                            normal_logger.warning(f"缓存文件损坏，将重新下载: {self._cache_path}")
                            os.remove(self._cache_path)
                        else:
                            return True
                
                # 缓存不存在或已过期，下载文件
                normal_logger.info(f"下载HTTP视频到缓存: {self._cache_path}")
                
                try:
                    # 下载文件
                    async with self._session.get(
                        self._url,
                        headers=headers,
                        auth=auth,
                        timeout=self._timeout
                    ) as response:
                        if response.status != 200:
                            exception_logger.exception(f"HTTP请求失败: {response.status}")
                            self._last_error = f"HTTP请求失败: {response.status}"
                            return False
                        
                        # 写入文件
                        with open(self._cache_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        
                        normal_logger.info(f"HTTP视频下载完成: {self._cache_path}")
                        
                        # 打开视频文件
                        self._cap = cv2.VideoCapture(self._cache_path)
                        if not self._cap.isOpened():
                            exception_logger.exception(f"无法打开下载的HTTP视频: {self._cache_path}")
                            self._last_error = "无法打开下载的HTTP视频"
                            os.remove(self._cache_path)
                            return False
                        
                        return True
                except Exception as e:
                    exception_logger.exception(f"下载HTTP视频异常: {str(e)}")
                    self._last_error = f"下载HTTP视频异常: {str(e)}"
                    
                    # 删除可能损坏的缓存文件
                    if os.path.exists(self._cache_path):
                        os.remove(self._cache_path)
                    
                    return False
            
            # 对于非缓存视频文件或直接流，直接使用OpenCV打开
            if self._is_video_file and not self._http_config.use_cache:
                self._cap = cv2.VideoCapture(self._url)
                
                if not self._cap.isOpened():
                    exception_logger.exception(f"无法打开HTTP视频流: {self._url}")
                    self._last_error = "无法打开HTTP视频流"
                    return False
                
                return True
            
            # 对于图像流，将在_pull_stream_task中处理
            if self._is_image_stream:
                return True
            
            # 默认使用OpenCV尝试打开URL
            self._cap = cv2.VideoCapture(self._url)
            
            if not self._cap.isOpened():
                exception_logger.exception(f"无法打开HTTP流: {self._url}")
                self._last_error = "无法打开HTTP流"
                return False
            
            normal_logger.info(f"成功创建HTTP流捕获对象: {self._url}")
            return True
        except Exception as e:
            exception_logger.exception(f"创建HTTP流异常: {str(e)}")
            self._last_error = f"创建HTTP流异常: {str(e)}"
            return False
    
    async def _stop_pulling(self) -> bool:
        """停止拉流
        
        Returns:
            bool: 是否成功停止拉流
        """
        try:
            # 释放OpenCV捕获对象
            if self._cap:
                self._cap.release()
                self._cap = None
            
            # 关闭HTTP会话
            if self._session:
                await self._session.close()
                self._session = None
            
            normal_logger.info(f"成功停止HTTP流: {self._url}")
            return True
        except Exception as e:
            exception_logger.exception(f"停止HTTP流异常: {str(e)}")
            return False
    
    async def _fetch_image(self) -> Optional[np.ndarray]:
        """获取HTTP图像流的一帧
        
        Returns:
            Optional[np.ndarray]: 图像帧，失败返回None
        """
        if not self._session:
            return None
        
        try:
            # 准备请求头
            headers = {
                'User-Agent': self._http_config.user_agent
            }
            
            # 添加自定义头
            if self._http_config.headers:
                headers.update(self._http_config.headers)
            
            # 准备认证信息
            auth = None
            if self._http_config.auth_enable and self._http_config.auth_user and self._http_config.auth_password:
                auth = aiohttp.BasicAuth(self._http_config.auth_user, self._http_config.auth_password)
            
            # 获取图像
            async with self._session.get(
                self._url,
                headers=headers,
                auth=auth,
                timeout=self._timeout
            ) as response:
                if response.status != 200:
                    normal_logger.warning(f"HTTP图像请求失败: {response.status}")
                    return None
                
                # 读取图像数据
                image_data = await response.read()
                
                # 转换为OpenCV格式
                image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                
                if image is None:
                    normal_logger.warning("无法解码HTTP图像数据")
                    return None
                
                return image
        except Exception as e:
            normal_logger.warning(f"获取HTTP图像异常: {str(e)}")
            return None
    
    async def _pull_stream_task(self) -> None:
        """拉流任务，从HTTP服务器拉取视频帧"""
        normal_logger.info(f"启动HTTP流拉流任务: {self._url}")
        
        # 重置重试计数
        self._retry_count = 0
        
        try:
            # 图像流和视频流处理逻辑不同
            if self._is_image_stream:
                # 图像流处理
                await self._pull_image_stream()
            else:
                # 视频流处理
                await self._pull_video_stream()
        except asyncio.CancelledError:
            normal_logger.info(f"HTTP流 {self._stream_id} 拉流任务被取消")
        except Exception as e:
            exception_logger.exception(f"HTTP流 {self._stream_id} 拉流任务异常: {str(e)}")
            self._last_error = f"拉流任务异常: {str(e)}"
        finally:
            # 释放资源
            if self._cap:
                self._cap.release()
                self._cap = None
            
            if self._session:
                await self._session.close()
                self._session = None
            
            normal_logger.info(f"HTTP流 {self._stream_id} 拉流任务已停止")
            
            # 设置状态
            self.set_status(StreamStatus.STOPPED)
            self.set_health_status(StreamHealthStatus.OFFLINE)
    
    async def _pull_video_stream(self) -> None:
        """视频流处理任务"""
        while not self._stop_event.is_set():
            try:
                # 如果捕获对象未初始化或已关闭，重新打开
                if self._cap is None or not self._cap.isOpened():
                    if not await self._reconnect():
                        # 重连失败，停止任务
                        break
                
                # 读取一帧
                ret, frame = self._cap.read()
                
                if not ret or frame is None:
                    # 检查是否到达视频结尾
                    # 如果是缓存的视频文件，可以重新开始播放
                    if self._is_video_file and self._cache_path and os.path.exists(self._cache_path):
                        normal_logger.info(f"HTTP视频文件播放完毕，重新开始: {self._cache_path}")
                        self._cap.release()
                        self._cap = cv2.VideoCapture(self._cache_path)
                        continue
                    
                    # 读取失败，尝试重连
                    normal_logger.warning(f"读取HTTP流帧失败: {self._url}")
                    self._last_error = "读取HTTP流帧失败"
                    
                    if not await self._reconnect():
                        # 重连失败，停止任务
                        break
                        
                    continue
                
                # 读取成功，重置重试计数
                self._retry_count = 0
                
                # 添加帧到缓冲区
                self._add_frame_to_buffer(frame)
                
                # 帧处理完成，设置事件
                self._frame_processed_event.set()
                
                # 控制读取速度，避免占用太多CPU
                await asyncio.sleep(0.001)
                
            except asyncio.CancelledError:
                normal_logger.info(f"HTTP流 {self._stream_id} 拉流任务被取消")
                break
            except Exception as e:
                exception_logger.exception(f"HTTP流 {self._stream_id} 拉流异常: {str(e)}")
                self._last_error = f"拉流异常: {str(e)}"
                self._stats["errors"] += 1
                
                # 尝试重连
                if not await self._reconnect():
                    # 重连失败，停止任务
                    break
    
    async def _pull_image_stream(self) -> None:
        """图像流处理任务"""
        while not self._stop_event.is_set():
            try:
                # 获取图像
                frame = await self._fetch_image()
                
                if frame is None:
                    # 读取失败，尝试重连
                    normal_logger.warning(f"读取HTTP图像流失败: {self._url}")
                    self._last_error = "读取HTTP图像流失败"
                    
                    if not await self._reconnect():
                        # 重连失败，停止任务
                        break
                        
                    continue
                
                # 读取成功，重置重试计数
                self._retry_count = 0
                
                # 添加帧到缓冲区
                self._add_frame_to_buffer(frame)
                
                # 帧处理完成，设置事件
                self._frame_processed_event.set()
                
                # 对于静态图片，设置适当的刷新间隔
                # 对于MJPEG流，尽快获取下一帧
                if 'multipart/x-mixed-replace' in self._content_type:
                    # MJPEG流，最小延迟
                    await asyncio.sleep(0.001)
                else:
                    # 静态图片，间隔刷新
                    await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                normal_logger.info(f"HTTP流 {self._stream_id} 拉流任务被取消")
                break
            except Exception as e:
                exception_logger.exception(f"HTTP流 {self._stream_id} 拉流异常: {str(e)}")
                self._last_error = f"拉流异常: {str(e)}"
                self._stats["errors"] += 1
                
                # 尝试重连
                if not await self._reconnect():
                    # 重连失败，停止任务
                    break
    
    async def _reconnect(self) -> bool:
        """重新连接HTTP流
        
        Returns:
            bool: 是否成功重连
        """
        # 检查重试次数
        if self._retry_count >= self._max_retries:
            normal_logger.error(f"HTTP流 {self._stream_id} 重试次数已达上限 ({self._max_retries}次)")
            self._last_error = f"重试次数已达上限 ({self._max_retries}次)"
            self.set_status(StreamStatus.ERROR)
            self.set_health_status(StreamHealthStatus.UNHEALTHY)
            return False
        
        # 增加重试次数和统计
        self._retry_count += 1
        self._stats["reconnects"] += 1
        
        normal_logger.info(f"HTTP流 {self._stream_id} 尝试重连 (第 {self._retry_count}/{self._max_retries} 次)")
        
        # 释放旧的资源
        if self._cap:
            self._cap.release()
            self._cap = None
        
        if self._session:
            await self._session.close()
            self._session = None
        
        # 等待重试间隔
        await asyncio.sleep(self._retry_interval)
        
        # 设置状态
        self.set_status(StreamStatus.CONNECTING)
        
        # 重新创建捕获对象
        try:
            return await self._start_pulling()
        except Exception as e:
            exception_logger.exception(f"HTTP流 {self._stream_id} 重连异常: {str(e)}")
            self._last_error = f"重连异常: {str(e)}"
            return False
