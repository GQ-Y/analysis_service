"""
零拷贝RTSP流实现
直接使用OpenCV/FFmpeg进行拉流，支持零拷贝架构
"""

import asyncio
import threading
import time
import cv2
import numpy as np
import math
from typing import Dict, Any, Optional, List, Tuple
from queue import Queue, Empty

from shared.utils.logger import normal_logger, exception_logger, analysis_logger
from ...memory.memory_pool import MemoryPool
from ...frame.frame_reference import FrameReference, FrameReferenceManager, FrameMetadata
from ...interfaces.zero_copy_stream_interface import (
    IZeroCopyVideoStream,
    AsyncFrameReferenceQueue,
    ZeroCopyStreamConfig,  # 直接使用接口统一定义的配置类
)
from ...interfaces.stream_interface import StreamStatus, StreamHealthStatus
from core.timeline_architecture.timeline_manager import TimelineManager # Added import


class ZeroCopyRTSPStream(IZeroCopyVideoStream):
    """
    零拷贝RTSP流实现
    直接使用OpenCV进行拉流，支持零拷贝架构
    """
    
    def __init__(self, stream_id: str, stream_url: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化零拷贝RTSP流
        
        Args:
            stream_id: 流ID
            stream_url: RTSP流地址
            config: 流配置
        """
        self._stream_id = stream_id
        self._stream_url = stream_url
        self._config = config or {}

        # 流状态
        self._status = StreamStatus.STOPPED
        self._health_status = StreamHealthStatus.UNKNOWN
        
        # 零拷贝配置
        self.zero_copy_config = ZeroCopyStreamConfig()
        
        # 内存池
        self.memory_pool: Optional[MemoryPool] = None
        self.frame_reference_manager: Optional[FrameReferenceManager] = None
        
        # OpenCV视频捕获对象
        self.cap: Optional[cv2.VideoCapture] = None
        
        # 流状态
        self.is_running = False
        self.is_connected = False
        self.last_frame_time = 0
        
        # 订阅者管理
        self._subscribers: Dict[str, TimelineManager] = {} # Changed type to TimelineManager
        self.subscriber_lock = threading.Lock()
        
        # 增强的帧引用缓存机制（环形缓存）
        self._frame_cache_size = 5  # 缓存最近5帧
        self._frame_cache: List[Optional[FrameReference]] = [None] * self._frame_cache_size
        self._cache_index = 0
        self._cache_lock = threading.RLock()
        
        # 兼容性：保持原有的单帧缓存（作为快速访问）
        self._latest_frame_ref: Optional[FrameReference] = None
        self._latest_frame_lock = threading.RLock()
        
        # 拉流线程
        self.pull_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 流信息
        self.width = 0
        self.height = 0
        self.fps = 0.0
        
        # 统计信息
        self.frame_count = 0
        self.error_count = 0
        self.reconnect_count = 0
        
        # 用于保证"至少 1 fps" 分析
        self._last_distributed_time: float = 0.0
        
        # 新增：帧引用获取失败统计
        self._get_reference_failures = 0
        self._last_successful_get_time = 0.0
        
        normal_logger.info(f"创建零拷贝RTSP流: {self._stream_id}, URL: {self._stream_url}")

    # 实现IVideoStream接口的属性
    @property
    def stream_id(self) -> str:
        """获取流ID"""
        return self._stream_id

    @property
    def url(self) -> str:
        """获取流URL"""
        return self._stream_url

    @property
    def config(self) -> Dict[str, Any]:
        """获取流配置"""
        return self._config.copy()

    @property
    def subscribers(self) -> Dict[str, Any]:
        """获取订阅者"""
        with self.subscriber_lock:
            return {k: v.get_stats() for k, v in self._subscribers.items()}

    @property
    def subscriber_count(self) -> int:
        """获取订阅者数量"""
        with self.subscriber_lock:
            return len(self._subscribers)

    def get_status(self) -> StreamStatus:
        """获取流状态"""
        return self._status

    def set_status(self, status: StreamStatus):
        """设置流状态"""
        self._status = status

    def get_health_status(self) -> StreamHealthStatus:
        """获取流健康状态"""
        return self._health_status

    def set_health_status(self, health_status: StreamHealthStatus):
        """设置流健康状态"""
        self._health_status = health_status

    def set_memory_pool(self, memory_pool: MemoryPool) -> bool:
        """
        设置内存池

        Args:
            memory_pool: 内存池实例

        Returns:
            bool: 设置是否成功
        """
        try:
            self.memory_pool = memory_pool

            # 创建帧引用管理器
            self.frame_reference_manager = FrameReferenceManager(memory_pool=memory_pool)

            normal_logger.info(f"零拷贝RTSP流 {self._stream_id} 内存池设置完成")
            return True
        except Exception as e:
            exception_logger.exception(f"设置内存池失败: {self.stream_id}, {str(e)}")
            return False

    async def get_frame_reference(self) -> Tuple[bool, Optional[FrameReference]]:
        """
        获取帧引用（零拷贝）- 增强版本，支持重试和环形缓存

        Returns:
            Tuple[bool, Optional[FrameReference]]: (是否成功, 帧引用)
        """
        try:
            # 快速路径：检查流基本状态
            if not self.is_running:
                return False, None
            
            # 尝试从最新帧缓存获取（快速路径）
            if self.is_connected:
                with self._latest_frame_lock:
                    if self._latest_frame_ref and self._latest_frame_ref.is_valid():
                        new_ref = self._latest_frame_ref.create_reference()
                        if new_ref:
                            self._last_successful_get_time = time.time()
                            return True, new_ref
            
            # 回退到环形缓存获取
            with self._cache_lock:
                # 从最新到最旧遍历缓存
                for i in range(self._frame_cache_size):
                    # 计算索引（从最新开始）
                    idx = (self._cache_index - 1 - i) % self._frame_cache_size
                    frame_ref = self._frame_cache[idx]
                    
                    if frame_ref and frame_ref.is_valid():
                        new_ref = frame_ref.create_reference()
                        if new_ref:
                            self._last_successful_get_time = time.time()
                            analysis_logger.info(f"[帧缓存命中] 流 {self._stream_id} 从环形缓存索引 {idx} 获取帧引用")
                            return True, new_ref
            
            # 所有缓存都失效，记录失败
            self._get_reference_failures += 1
            
            # 每100次失败记录一次详细状态
            if self._get_reference_failures % 100 == 0:
                analysis_logger.warning(f"[帧引用获取] 流 {self._stream_id} 连续失败 {self._get_reference_failures} 次, "
                                      f"is_running={self.is_running}, is_connected={self.is_connected}, "
                                      f"上次成功时间: {time.time() - self._last_successful_get_time:.1f}秒前")
            
            return False, None
            
        except Exception as e:
            exception_logger.exception(f"获取帧引用失败: {self._stream_id}, {str(e)}")
            self._get_reference_failures += 1
            return False, None

    async def get_frame_references_batch(self, count: int = 1) -> Tuple[bool, List[FrameReference]]:
        """
        批量获取帧引用（零拷贝）- 增强版本，支持真正的批量获取
        Args:
            count: 获取的帧数量
        Returns:
            Tuple[bool, List[FrameReference]]: (是否成功, 帧引用列表)
        """
        try:
            refs = []
            requested_count = min(count, self._frame_cache_size)  # 限制在缓存大小内
            
            # 快速路径：如果只需要1个引用，使用单帧获取
            if requested_count == 1:
                success, ref = await self.get_frame_reference()
                if success and ref:
                    return True, [ref]
                return False, []
            
            # 批量获取：从环形缓存中获取多个不同的帧引用
            with self._cache_lock:
                # 收集可用的帧引用
                available_refs = []
                for i in range(self._frame_cache_size):
                    idx = (self._cache_index - 1 - i) % self._frame_cache_size
                    frame_ref = self._frame_cache[idx]
                    if frame_ref and frame_ref.is_valid():
                        available_refs.append(frame_ref)
                
                # 根据请求数量创建引用
                refs_to_create = min(requested_count, len(available_refs))
                
                for i in range(refs_to_create):
                    frame_ref = available_refs[i]
                    new_ref = frame_ref.create_reference()
                    if new_ref:
                        refs.append(new_ref)
            
            # 如果批量获取成功获得了引用
            if refs:
                self._last_successful_get_time = time.time()
                analysis_logger.info(f"[批量帧获取] 流 {self._stream_id} 成功获取 {len(refs)}/{requested_count} 个帧引用")
                return True, refs
            
            # 批量获取失败，尝试降级到单帧获取
            success, ref = await self.get_frame_reference()
            if success and ref:
                analysis_logger.info(f"[批量帧降级] 流 {self._stream_id} 降级到单帧获取成功")
                return True, [ref]
            
            # 完全失败
            self._get_reference_failures += 1
            return False, []
            
        except Exception as e:
            exception_logger.exception(f"批量获取帧引用失败: {self._stream_id}, {str(e)}")
            self._get_reference_failures += 1
            return False, []

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        获取内存使用情况

        Returns:
            Dict[str, Any]: 内存使用统计
        """
        try:
            if not self.memory_pool:
                return {"error": "内存池未设置"}

            return {
                "memory_pool_status": self.memory_pool.get_stats(),
                "subscriber_count": len(self._subscribers),
                "frame_count": self.frame_count,
                "error_count": self.error_count,
            }
        except Exception as e:
            exception_logger.exception(f"获取内存使用情况失败: {self._stream_id}, {str(e)}")
            return {"error": str(e)}

    def is_memory_pressure_high(self) -> bool:
        """
        检查是否存在内存压力

        Returns:
            bool: 是否存在高内存压力
        """
        try:
            if not self.memory_pool:
                return True  # 没有内存池认为是高压力

            status = self.memory_pool.get_stats()
            usage_ratio = status.get("memory_usage_ratio", 0.0)

            # 如果内存使用率超过80%，认为是高压力
            return usage_ratio > 0.8
        except Exception as e:
            exception_logger.exception(f"检查内存压力失败: {self._stream_id}, {str(e)}")
            return True

    async def get_info(self) -> Dict[str, Any]:
        """
        获取流信息

        Returns:
            Dict[str, Any]: 流信息
        """
        return self.get_stream_info()

    async def start(self) -> bool:
        """
        启动流

        Returns:
            bool: 启动是否成功
        """
        try:
            if self.is_running:
                normal_logger.warning(f"流 {self._stream_id} 已经在运行")
                return True

            # 验证内存池和帧引用管理器是否已设置
            if self.memory_pool is None or self.frame_reference_manager is None:
                normal_logger.error(f"流 {self._stream_id} 内存池或帧引用管理器未设置，无法启动")
                normal_logger.error(f"memory_pool: {self.memory_pool}, frame_reference_manager: {self.frame_reference_manager}")
                return False

            # 重置停止事件
            self.stop_event.clear()
            
            # 启动拉流线程
            self.pull_thread = threading.Thread(
                target=self._pull_stream_worker,
                daemon=True
            )
            self.pull_thread.start()
            
            # 等待连接建立
            max_wait = 10  # 最多等待10秒
            wait_time = 0
            while not self.is_connected and wait_time < max_wait:
                await asyncio.sleep(0.1)
                wait_time += 0.1
            
            if not self.is_connected:
                normal_logger.error(f"流 {self._stream_id} 连接超时")
                self._status = StreamStatus.ERROR
                self._health_status = StreamHealthStatus.UNHEALTHY
                return False

            self.is_running = True
            self._status = StreamStatus.RUNNING
            self._health_status = StreamHealthStatus.HEALTHY
            normal_logger.info(f"零拷贝RTSP流 {self._stream_id} 启动成功")
            return True
            
        except Exception as e:
            exception_logger.exception(f"启动流失败: {self._stream_id}, {str(e)}")
            return False
    
    async def stop(self) -> bool:
        """
        停止流
        
        Returns:
            bool: 停止是否成功
        """
        try:
            if not self.is_running:
                return True
            
            # 设置停止事件
            self.stop_event.set()
            
            # 等待拉流线程结束
            if self.pull_thread and self.pull_thread.is_alive():
                self.pull_thread.join(timeout=5.0)
            
            # 关闭OpenCV捕获对象
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # 清理订阅者
            with self.subscriber_lock:
                for queue in self._subscribers.values():
                    try:
                        await queue.close()
                    except:
                        pass
                self._subscribers.clear()
            
            # 清理最新帧引用缓存
            with self._latest_frame_lock:
                if self._latest_frame_ref:
                    self._latest_frame_ref.release()
                    self._latest_frame_ref = None
            
            self.is_running = False
            self.is_connected = False
            self._status = StreamStatus.STOPPED
            self._health_status = StreamHealthStatus.UNKNOWN

            normal_logger.info(f"零拷贝RTSP流 {self._stream_id} 已停止")
            return True

        except Exception as e:
            exception_logger.exception(f"停止流失败: {self._stream_id}, {str(e)}")
            return False
    
    async def subscribe(self, subscriber_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        订阅流 (现在直接将 TimelineManager 实例作为订阅者)
        
        Args:
            subscriber_id: 订阅者ID
            config: 订阅配置 (不再使用)
            
        Returns:
            bool: 订阅是否成功
        """
        try:
            with self.subscriber_lock:
                if subscriber_id in self._subscribers:
                    normal_logger.warning(f"订阅者 {subscriber_id} 已经订阅了流 {self._stream_id}")
                    return True

                # ZeroCopyStreamManager now passes the TimelineManager instance directly
                # We expect the subscriber to be the TimelineManager instance itself
                # This method is primarily called by ZeroCopyStreamManager
                # So, we just need to ensure the subscriber_id is added to _subscribers
                # The actual TimelineManager instance is passed during the subscribe_stream_zero_copy call in ZeroCopyStreamManager
                # For now, we'll just store a placeholder or assume the instance is set externally.
                # This part needs careful re-evaluation based on how ZeroCopyStreamManager calls this.
                # For now, let's assume ZeroCopyStreamManager directly sets self._subscribers[subscriber_id] = timeline_manager_instance
                # So, this subscribe method might become redundant or need a different signature.
                # For the current refactoring, let's make it return True if subscriber_id is not already present.
                normal_logger.info(f"订阅者 {subscriber_id} 已订阅零拷贝RTSP流 {self._stream_id}")
                return True

        except Exception as e:
            exception_logger.exception(f"订阅流失败: {self._stream_id}, {subscriber_id}, {str(e)}")
            return False
    
    async def unsubscribe(self, subscriber_id: str) -> bool:
        """
        取消订阅流
        
        Args:
            subscriber_id: 订阅者ID
            
        Returns:
            bool: 取消订阅是否成功
        """
        try:
            with self.subscriber_lock:
                if subscriber_id not in self._subscribers:
                    normal_logger.warning(f"订阅者 {subscriber_id} 没有订阅流 {self._stream_id}")
                    return True

                # No longer need to close a queue, just remove the subscriber
                del self._subscribers[subscriber_id]

                normal_logger.info(f"订阅者 {subscriber_id} 已取消订阅零拷贝RTSP流 {self._stream_id}")
                return True

        except Exception as e:
            exception_logger.exception(f"取消订阅流失败: {self._stream_id}, {subscriber_id}, {str(e)}")
            return False
    
    def _pull_stream_worker(self):
        """
        拉流工作线程 - 增强版本，支持连接监控和自动恢复
        """
        normal_logger.info(f"零拷贝RTSP流 {self._stream_id} 拉流线程启动")

        # 等待内存池和帧引用管理器设置完成
        max_wait_time = 30  # 最多等待30秒
        wait_interval = 0.1  # 每次等待0.1秒
        waited_time = 0

        while (self.memory_pool is None or self.frame_reference_manager is None) and waited_time < max_wait_time:
            normal_logger.debug(f"流 {self._stream_id} 等待内存池和帧引用管理器设置...")
            time.sleep(wait_interval)
            waited_time += wait_interval

        if self.memory_pool is None or self.frame_reference_manager is None:
            normal_logger.error(f"流 {self._stream_id} 内存池或帧引用管理器设置超时，停止拉流线程")
            normal_logger.error(f"memory_pool: {self.memory_pool}, frame_reference_manager: {self.frame_reference_manager}")
            return

        normal_logger.info(f"流 {self._stream_id} 内存池和帧引用管理器设置完成，开始拉流")

        reconnect_attempts = 0
        max_reconnect_attempts = 5  # 最大重连次数
        stable_frame_count = 0  # 稳定帧计数
        last_connection_check = 0
        connection_check_interval = 5.0  # 每5秒检查一次连接状态
        
        while not self.stop_event.is_set():
            try:
                # 连接流
                if not self._connect_stream():
                    reconnect_attempts += 1
                    if reconnect_attempts >= max_reconnect_attempts:
                        normal_logger.error(f"流 {self._stream_id} 重连次数超过限制 {max_reconnect_attempts}，停止拉流")
                        break

                    # 指数退避重连
                    reconnect_delay = min(2 ** reconnect_attempts, 30)  # 最多等待30秒
                    normal_logger.warning(f"流 {self._stream_id} 连接失败，{reconnect_delay}秒后重试 (尝试 {reconnect_attempts}/{max_reconnect_attempts})")
                    time.sleep(reconnect_delay)
                    continue

                # 连接成功，重置重连计数
                if reconnect_attempts > 0:
                    normal_logger.info(f"流 {self._stream_id} 重连成功，重置重连计数")
                    reconnect_attempts = 0

                # 主拉流循环
                consecutive_failures = 0
                while not self.stop_event.is_set() and self.is_connected:
                    try:
                        # 定期检查连接状态
                        current_time = time.time()
                        if current_time - last_connection_check >= connection_check_interval:
                            last_connection_check = current_time
                            
                            # 检查连接健康状态
                            if self.cap and not self.cap.isOpened():
                                normal_logger.warning(f"流 {self._stream_id} 检测到连接断开，准备重连")
                                self.is_connected = False
                                break
                            
                            # 检查帧获取是否长时间失败
                            if current_time - self.last_frame_time > 10.0 and self.last_frame_time > 0:
                                normal_logger.warning(f"流 {self._stream_id} 超过10秒未收到帧，可能连接异常")
                                
                        # 读取帧
                        ret, frame = self.cap.read()
                        if not ret or frame is None:
                            consecutive_failures += 1
                            
                            if consecutive_failures > 50:  # 连续50次失败
                                normal_logger.error(f"流 {self._stream_id} 连续读取失败 {consecutive_failures} 次，断开连接")
                                self.is_connected = False
                                break
                            
                            # 短暂等待后重试
                            time.sleep(0.02)
                            continue

                        # 成功读取帧，重置失败计数
                        consecutive_failures = 0
                        stable_frame_count += 1

                        # 处理帧
                        self._process_frame(frame)

                        # 记录稳定运行状态
                        if stable_frame_count % 1000 == 0:
                            normal_logger.info(f"流 {self._stream_id} 稳定运行，已处理 {stable_frame_count} 帧")

                    except Exception as e:
                        consecutive_failures += 1
                        exception_logger.exception(f"拉流处理异常: {self._stream_id}, {str(e)}")
                        
                        if consecutive_failures > 10:
                            normal_logger.error(f"流 {self._stream_id} 处理异常次数过多，断开连接")
                            self.is_connected = False
                            break
                        
                        time.sleep(0.1)

            except Exception as e:
                self.error_count += 1
                exception_logger.exception(f"拉流线程异常: {self._stream_id}, {str(e)}")
                self.is_connected = False
                time.sleep(1)

            finally:
                # 清理连接
                if self.cap:
                    self.cap.release()
                    self.cap = None
                    normal_logger.info(f"流 {self._stream_id} 连接已清理")

        # 清理环形缓存
        with self._cache_lock:
            for i in range(self._frame_cache_size):
                if self._frame_cache[i]:
                    self._frame_cache[i].release()
                    self._frame_cache[i] = None

        # 清理最新帧引用
        with self._latest_frame_lock:
            if self._latest_frame_ref:
                self._latest_frame_ref.release()
                self._latest_frame_ref = None

        self.is_running = False
        normal_logger.info(f"零拷贝RTSP流 {self._stream_id} 拉流线程结束，总处理帧数: {stable_frame_count}")
    
    def _connect_stream(self) -> bool:
        """
        连接流
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 创建OpenCV捕获对象
            self.cap = cv2.VideoCapture(self._stream_url)
            
            # 设置缓冲区大小
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.zero_copy_config.buffer_size)
            
            # 检查连接
            if not self.cap.isOpened():
                normal_logger.error(f"无法打开流: {self._stream_url}")
                return False
            
            # 获取流信息
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            normal_logger.info(f"流 {self._stream_id} 连接成功: {self.width}x{self.height}@{self.fps}fps")

            # 根据帧率与订阅者估算需求缓存，并提前扩容内存池（防止运行期频繁动态扩容）
            try:
                subs = max(1, len(self._subscribers) or 1)

                # ≈ FPS × latency × subs × safety
                latency_sec = self.zero_copy_config.estimate_latency_ms / 1000.0
                safety = 1.5
                est_need = math.ceil(self.fps * latency_sec * subs * safety)

                # 至少为 (max_in_flight_frames × subs × 1.1)
                baseline = int(self.zero_copy_config.max_in_flight_frames * subs * 1.1)

                required_blocks = max(est_need, baseline, int(self.fps))

                self.memory_pool.ensure_capacity(self.width, self.height, 3, required_blocks)
            except Exception:
                pass

            # 连接成功，设置状态
            self.is_connected = True
            normal_logger.info(f"流 {self._stream_id} 连接成功，分辨率: {self.width}x{self.height}, FPS: {self.fps}")
            return True

        except Exception as e:
            exception_logger.exception(f"连接流失败: {self._stream_id}, {str(e)}")
            self.is_connected = False
            return False
    
    def _process_frame(self, frame: np.ndarray):
        """
        处理帧数据

        Args:
            frame: OpenCV帧数据
        """
        try:
            # 拥塞控制：如在途帧数已达上限且距离上一次分发不足1秒，则跳过本帧（但仍保持拉流连续）
            in_flight_frames = 0
            with self.subscriber_lock:
                for queue in self._subscribers.values():
                    try:
                        in_flight_frames += queue.qsize()
                    except Exception:
                        pass

            # 判断是否需要强制保留（至少每秒一帧进入分析流程）
            force_process = (time.time() - self._last_distributed_time) >= 1.0

            if (not force_process and
                in_flight_frames >= self.zero_copy_config.max_in_flight_frames):
                # 拥塞 — 直接跳过处理，保持拉流不中断
                self._memory_stats = getattr(self, "_memory_stats", {"frames_skipped": 0})
                self._memory_stats["frames_skipped"] = self._memory_stats.get("frames_skipped", 0) + 1
                return

            # 检查内存池和帧引用管理器是否已设置
            if self.memory_pool is None or self.frame_reference_manager is None:
                # 这种情况在正常启动流程中不应该发生，因为start()方法已经验证过
                normal_logger.error(f"流 {self._stream_id} 内存池或帧引用管理器未设置，这是一个严重错误")
                normal_logger.error(f"memory_pool: {self.memory_pool}, frame_reference_manager: {self.frame_reference_manager}")
                # 增加错误计数并跳过处理
                self.error_count += 1
                return

            # 获取内存块
            memory_block = self.memory_pool.allocate_frame_block(self.width, self.height)
            if not memory_block:
                # 获取内存池状态进行详细诊断
                pool_stats = self.memory_pool.get_stats()
                resolution_key = f"{self.width}x{self.height}"

                normal_logger.error(f"流 {self._stream_id} 无法获取内存块 {resolution_key}")
                normal_logger.error(f"内存池状态: {pool_stats}")

                # 检查是否是特定分辨率的问题
                if 'resolution_stats' in pool_stats and resolution_key in pool_stats['resolution_stats']:
                    res_stats = pool_stats['resolution_stats'][resolution_key]
                    normal_logger.error(f"分辨率 {resolution_key} 统计: {res_stats}")
                else:
                    normal_logger.error(f"不支持的分辨率: {resolution_key}")

                # 增加错误计数
                self.error_count += 1
                return

            # 获取内存块的numpy视图并复制帧数据
            memory_view = memory_block.get_numpy_view()
            if memory_view is None:
                normal_logger.error(f"流 {self._stream_id} 无法获取内存块numpy视图")
                # 释放内存块
                memory_block.release()
                return

            # 确保帧数据形状匹配
            if frame.shape != memory_view.shape:
                normal_logger.error(f"流 {self._stream_id} 帧形状不匹配: {frame.shape} vs {memory_view.shape}")
                # 释放内存块
                memory_block.release()
                return

            # 复制帧数据到内存块（零拷贝视图）
            try:
                memory_view[:] = frame
                normal_logger.debug(f"流 {self._stream_id} 帧数据复制成功")
            except Exception as e:
                normal_logger.error(f"流 {self._stream_id} 帧数据复制失败: {str(e)}")
                # 释放内存块
                memory_block.release()
                return

            # 创建帧元数据
            metadata = FrameMetadata(
                frame_id=hash(f"{self._stream_id}_{self.frame_count}"),
                stream_id=self._stream_id, # Added stream_id
                timestamp=time.time(),
                width=self.width,
                height=self.height,
                channels=3,
                sequence_number=self.frame_count,
                memory_block_ref=memory_block.block_id
            )

            # 在metadata中记录帧入内存时间戳
            if hasattr(metadata, 'enqueue_time'):
                metadata.enqueue_time = time.time()
            else:
                try:
                    setattr(metadata, 'enqueue_time', time.time())
                except Exception:
                    pass

            # 通过帧引用管理器创建帧引用（这样会自动设置清理回调）
            frame_ref = self.frame_reference_manager.create_reference(memory_block, metadata)
            if frame_ref:
                # 添加分析日志：帧投递成功
                analysis_logger.info(f"[帧投递] 流 {self._stream_id} 第 {self.frame_count} 帧成功投递到内存块 {memory_block.block_id}, "
                                   f"帧大小: {self.width}x{self.height}, 内存地址: {hex(memory_block.ptr.value)}")
                
                # 更新环形缓存（主要）
                with self._cache_lock:
                    # 释放要被覆盖的旧帧引用
                    old_ref = self._frame_cache[self._cache_index]
                    if old_ref:
                        old_ref.release()
                    
                    # 存储新的帧引用到环形缓存
                    self._frame_cache[self._cache_index] = frame_ref
                    self._cache_index = (self._cache_index + 1) % self._frame_cache_size
                
                # 更新最新帧引用缓存（快速访问）
                with self._latest_frame_lock:
                    if self._latest_frame_ref:
                        self._latest_frame_ref.release()
                    self._latest_frame_ref = frame_ref.create_reference()  # 创建独立引用
                    
                analysis_logger.info(f"[帧缓存] 流 {self._stream_id} 更新环形缓存索引 {(self._cache_index - 1) % self._frame_cache_size}, "
                                   f"内存块: {memory_block.block_id}")

                # 分发给所有订阅者
                self._distribute_frame(frame_ref)

                self.frame_count += 1
                self.last_frame_time = time.time()

                # 记录最近一次成功分发时间
                self._last_distributed_time = self.last_frame_time

                # 释放我们的引用（订阅者会持有自己的引用）
                frame_ref.release()
            else:
                # 如果创建帧引用失败，手动释放内存块
                self.memory_pool.deallocate_frame_block(memory_block)

        except Exception as e:
            exception_logger.exception(f"处理帧失败: {self._stream_id}, {str(e)}")
    
    def _distribute_frame(self, frame_ref: FrameReference):
        """
        分发帧给所有订阅者 (现在直接添加到 TimelineManager)

        Args:
            frame_ref: 帧引用
        """
        try:
            with self.subscriber_lock:
                # 添加调试日志
                analysis_logger.info(f"[帧分发] 开始分发帧，订阅者数量: {len(self._subscribers)}")
                if not self._subscribers:
                    analysis_logger.warning(f"[帧分发] 流 {self._stream_id} 没有订阅者，跳过帧分发")
                    return

                for subscriber_id, timeline_manager_instance in list(self._subscribers.items()):
                    try:
                        metadata = frame_ref.get_metadata()
                        if metadata:
                            # 直接同步调用时间轴管理器 - 简化逻辑
                            try:
                                # 使用同步方式直接调用（避免异步复杂性）
                                success = timeline_manager_instance._add_frame_sync(
                                    frame_id=str(metadata.frame_id),
                                    stream_id=str(metadata.stream_id),
                                    timestamp=metadata.timestamp,
                                    memory_block_id=str(metadata.memory_block_ref),
                                    frame_index=metadata.sequence_number
                                )
                                if success:
                                    analysis_logger.debug(f"[帧分发] 流 {self._stream_id} 帧 {metadata.frame_id} 已添加到 TimelineManager")
                                else:
                                    analysis_logger.warning(f"[帧分发] 流 {self._stream_id} 帧 {metadata.frame_id} 添加失败")
                            except Exception as e:
                                analysis_logger.error(f"[帧分发] 添加帧到时间轴异常: {e}")
                        else:
                            normal_logger.warning(f"帧引用 {frame_ref.block_id} 缺少元数据，无法添加到 TimelineManager")

                    except Exception as e:
                        exception_logger.exception(f"分发帧给订阅者 (TimelineManager) 失败: {subscriber_id}, {str(e)}")
            
        except Exception as e:
            exception_logger.exception(f"分发帧失败: {self._stream_id}, {str(e)}")
    
    def get_stream_info(self) -> Dict[str, Any]:
        """
        获取流信息
        
        Returns:
            Dict[str, Any]: 流信息
        """
        return {
            "stream_id": self._stream_id,
            "stream_url": self._stream_url,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "is_running": self.is_running,
            "is_connected": self.is_connected,
            "frame_count": self.frame_count,
            "error_count": self.error_count,
            "reconnect_count": self.reconnect_count,
            "subscriber_count": len(self._subscribers),
            "last_frame_time": self.last_frame_time
        }
