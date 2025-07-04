"""
拉流模块 - 流水线架构组件
负责从各种流协议(RTSP/RTMP/HTTP等)拉取视频帧并投递到时间轴
"""
import asyncio
import time
import threading
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import cv2
import numpy as np

class StreamProtocol(Enum):
    """流协议类型"""
    RTSP = "rtsp"
    RTMP = "rtmp"
    HTTP = "http"
    FILE = "file"
    USB_CAMERA = "usb"

@dataclass
class StreamConfig:
    """流配置"""
    stream_id: str
    url: str
    protocol: StreamProtocol
    fps_limit: Optional[int] = None  # 限制FPS，None表示不限制
    resolution: Optional[tuple] = None  # (width, height)，None表示不调整
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0
    buffer_size: int = 1  # 缓冲区大小
    timeout: float = 30.0

@dataclass
class FrameData:
    """帧数据"""
    frame_id: str
    stream_id: str
    frame_array: np.ndarray
    timestamp: float
    frame_index: int
    metadata: Dict[str, Any]

class StreamPuller:
    """
    单个流的拉流器
    负责从指定URL拉取视频帧
    """
    
    def __init__(self, config: StreamConfig, frame_callback: Callable[[FrameData], None]):
        self.config = config
        self.frame_callback = frame_callback
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.pull_task: Optional[asyncio.Task] = None
        self.last_frame_time = 0.0
        self.frame_count = 0
        self.error_count = 0
        
        # 统计信息
        self.stats = {
            "total_frames": 0,
            "dropped_frames": 0,
            "error_count": 0,
            "avg_fps": 0.0,
            "last_error": None
        }
    
    async def start(self):
        """启动拉流"""
        if self.running:
            return
        
        self.running = True
        self.pull_task = asyncio.create_task(self._pull_frames())
        print(f"[拉流模块] 启动流: {self.config.stream_id} -> {self.config.url}")
    
    async def stop(self):
        """停止拉流"""
        if not self.running:
            return
        
        self.running = False
        
        if self.pull_task and not self.pull_task.done():
            self.pull_task.cancel()
            try:
                await self.pull_task
            except asyncio.CancelledError:
                pass
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print(f"[拉流模块] 停止流: {self.config.stream_id}")
    
    def _initialize_capture(self) -> bool:
        """初始化视频捕获"""
        try:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.config.url)
            
            # 设置缓冲区大小
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            
            # 设置分辨率（如果指定）
            if self.config.resolution:
                width, height = self.config.resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            if not self.cap.isOpened():
                print(f"[拉流模块] 无法打开流: {self.config.url}")
                return False
            
            print(f"[拉流模块] 成功连接流: {self.config.stream_id}")
            self.error_count = 0
            return True
            
        except Exception as e:
            print(f"[拉流模块] 初始化捕获失败: {self.config.stream_id}, {e}")
            self.stats["last_error"] = str(e)
            return False
    
    async def _pull_frames(self):
        """拉流主循环"""
        reconnect_attempts = 0
        
        while self.running:
            try:
                # 初始化连接
                if not self._initialize_capture():
                    reconnect_attempts += 1
                    if reconnect_attempts >= self.config.reconnect_attempts:
                        print(f"[拉流模块] 重连次数超限，停止拉流: {self.config.stream_id}")
                        break
                    
                    await asyncio.sleep(self.config.reconnect_delay)
                    continue
                
                reconnect_attempts = 0
                
                # 拉流循环
                while self.running and self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        print(f"[拉流模块] 读取帧失败: {self.config.stream_id}")
                        break
                    
                    current_time = time.time()
                    
                    # FPS限制检查
                    if self.config.fps_limit:
                        min_interval = 1.0 / self.config.fps_limit
                        if current_time - self.last_frame_time < min_interval:
                            continue
                    
                    # 创建帧数据
                    frame_data = FrameData(
                        frame_id=f"{self.config.stream_id}_{self.frame_count}_{uuid.uuid4().hex[:8]}",
                        stream_id=self.config.stream_id,
                        frame_array=frame,
                        timestamp=current_time,
                        frame_index=self.frame_count,
                        metadata={
                            "width": frame.shape[1],
                            "height": frame.shape[0],
                            "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
                            "protocol": self.config.protocol.value
                        }
                    )
                    
                    # 投递帧到回调
                    try:
                        self.frame_callback(frame_data)
                        self.stats["total_frames"] += 1
                        self.frame_count += 1
                        self.last_frame_time = current_time
                        
                        # 更新FPS统计
                        if self.frame_count > 1:
                            elapsed = current_time - self.last_frame_time if self.last_frame_time > 0 else 1.0
                            self.stats["avg_fps"] = 1.0 / elapsed if elapsed > 0 else 0.0
                        
                    except Exception as e:
                        print(f"[拉流模块] 帧回调异常: {self.config.stream_id}, {e}")
                        self.stats["dropped_frames"] += 1
                    
                    # 避免过度消耗CPU
                    await asyncio.sleep(0.001)
                
            except Exception as e:
                print(f"[拉流模块] 拉流异常: {self.config.stream_id}, {e}")
                self.error_count += 1
                self.stats["error_count"] = self.error_count
                self.stats["last_error"] = str(e)
                
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
                await asyncio.sleep(self.config.reconnect_delay)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "stream_id": self.config.stream_id,
            "url": self.config.url,
            "protocol": self.config.protocol.value,
            "running": self.running,
            **self.stats
        }

class StreamModule:
    """
    拉流模块 - 管理多个流的拉取
    负责协调所有流的拉取任务并投递到时间轴管理器
    """
    
    def __init__(self, timeline_manager):
        self.timeline_manager = timeline_manager
        self.stream_pullers: Dict[str, StreamPuller] = {}
        self.memory_module = None  # 将由流水线管理器设置
        
        # 全局统计
        self.global_stats = {
            "total_streams": 0,
            "active_streams": 0,
            "total_frames_delivered": 0,
            "total_frames_failed": 0
        }
        
        self.running = False
    
    async def start(self):
        """启动拉流模块"""
        if self.running:
            return
        
        self.running = True
        print("[拉流模块] 启动完成")
    
    async def stop(self):
        """停止拉流模块"""
        if not self.running:
            return
        
        self.running = False
        
        # 停止所有流拉取器
        tasks = []
        for puller in self.stream_pullers.values():
            tasks.append(puller.stop())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.stream_pullers.clear()
        print("[拉流模块] 停止完成")
    
    async def add_stream(self, config: StreamConfig) -> bool:
        """
        添加流
        
        Args:
            config: 流配置
            
        Returns:
            bool: 是否成功添加
        """
        try:
            if config.stream_id in self.stream_pullers:
                print(f"[拉流模块] 流已存在: {config.stream_id}")
                return False
            
            # 创建拉流器
            puller = StreamPuller(config, self._on_frame_received)
            
            # 启动拉流器
            await puller.start()
            
            # 注册到管理器
            self.stream_pullers[config.stream_id] = puller
            self.global_stats["total_streams"] += 1
            self.global_stats["active_streams"] = len(self.stream_pullers)
            
            print(f"[拉流模块] 成功添加流: {config.stream_id}")
            return True
            
        except Exception as e:
            print(f"[拉流模块] 添加流失败: {config.stream_id}, {e}")
            return False
    
    async def remove_stream(self, stream_id: str) -> bool:
        """
        移除流
        
        Args:
            stream_id: 流ID
            
        Returns:
            bool: 是否成功移除
        """
        try:
            if stream_id not in self.stream_pullers:
                return False
            
            puller = self.stream_pullers[stream_id]
            await puller.stop()
            
            del self.stream_pullers[stream_id]
            self.global_stats["active_streams"] = len(self.stream_pullers)
            
            print(f"[拉流模块] 成功移除流: {stream_id}")
            return True
            
        except Exception as e:
            print(f"[拉流模块] 移除流失败: {stream_id}, {e}")
            return False
    
    def _on_frame_received(self, frame_data: FrameData):
        """
        帧接收回调
        处理从拉流器接收到的帧数据
        """
        try:
            # 将帧存储到内存管理模块
            if self.memory_module:
                memory_block_id = self.memory_module.store_frame(
                    frame_data.frame_id,
                    frame_data.frame_array,
                    frame_data.metadata
                )
                
                if memory_block_id:
                    # 投递到时间轴管理器
                    asyncio.create_task(self.timeline_manager.add_frame(
                        frame_id=frame_data.frame_id,
                        stream_id=frame_data.stream_id,
                        timestamp=frame_data.timestamp,
                        memory_block_id=memory_block_id,
                        frame_index=frame_data.frame_index
                    ))
                    
                    self.global_stats["total_frames_delivered"] += 1
                else:
                    print(f"[拉流模块] 存储帧失败: {frame_data.frame_id}")
                    self.global_stats["total_frames_failed"] += 1
            else:
                print(f"[拉流模块] 内存模块未初始化")
                self.global_stats["total_frames_failed"] += 1
                
        except Exception as e:
            print(f"[拉流模块] 处理帧异常: {frame_data.frame_id}, {e}")
            self.global_stats["total_frames_failed"] += 1
    
    def set_memory_module(self, memory_module):
        """设置内存管理模块"""
        self.memory_module = memory_module
        print("[拉流模块] 内存管理模块已连接")
    
    def get_stream_list(self) -> List[str]:
        """获取流列表"""
        return list(self.stream_pullers.keys())
    
    def get_stream_statistics(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取指定流的统计信息"""
        if stream_id not in self.stream_pullers:
            return None
        
        return self.stream_pullers[stream_id].get_statistics()
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """获取全局统计信息"""
        return {
            **self.global_stats,
            "stream_details": {
                stream_id: puller.get_statistics()
                for stream_id, puller in self.stream_pullers.items()
            }
        }
    
    async def update_stream_config(self, stream_id: str, new_config: StreamConfig) -> bool:
        """
        更新流配置
        
        Args:
            stream_id: 流ID
            new_config: 新配置
            
        Returns:
            bool: 是否成功更新
        """
        try:
            # 先移除旧流
            await self.remove_stream(stream_id)
            
            # 添加新流
            return await self.add_stream(new_config)
            
        except Exception as e:
            print(f"[拉流模块] 更新流配置失败: {stream_id}, {e}")
            return False 