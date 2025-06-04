# core/stream_manager.py
import asyncio
import time
import threading
from typing import Dict, Optional, Tuple, Any, List
import traceback
import hashlib

# 导入共享工具
from shared.utils.logger import get_normal_logger, get_exception_logger, get_test_logger

# 导入状态定义
from .status import StreamStatus, StreamHealthStatus

# 导入流接口
from .interface import IVideoStream

# 导入ZLMediaKit流实现
from core.media_kit.zlm_stream import ZLMVideoStream

# 导入新的流工厂
from core.media_kit.factory.stream_factory import StreamFactory, StreamEngine

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()

class StreamManager:
    """视频流管理器，负责管理所有视频流的创建、共享和生命周期"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StreamManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化流管理器"""
        if self._initialized:
            return

        self._initialized = True
        self._streams: Dict[str, IVideoStream] = {}  # stream_key (URL derived) -> IVideoStream
        self._stream_id_to_key_map: Dict[str, str] = {} # original_stream_id -> stream_key
        
        normal_logger.info("流管理器初始化完成，支持多种流媒体引擎(ZLMediaKit/GStreamer/OpenCV)")

    async def initialize(self):
        """初始化流管理器，包括ZLMediaKit管理器"""
        try:
            # 导入ZLMediaKit管理器
            from core.media_kit.zlm_manager import zlm_manager

            # 初始化ZLMediaKit管理器
            await zlm_manager.initialize()

            normal_logger.info("ZLMediaKit管理器初始化成功")
        except Exception as e:
            exception_logger.exception(f"初始化ZLMediaKit管理器失败: {str(e)}")

        normal_logger.info("StreamManager.initialize() 完成")

    def _generate_stream_key(self, url: str) -> str:
        """根据URL生成规范化的流键。"""
        if not url:
            return ""
        # 简单规范化：小写，去除尾部斜杠
        normalized_url = url.lower().rstrip('/')
        # 使用MD5哈希确保键的格式和长度一致性，避免特殊字符问题
        return hashlib.md5(normalized_url.encode('utf-8')).hexdigest()

    async def get_or_create_stream(self, original_stream_id: str, config: Dict[str, Any]) -> Optional[IVideoStream]:
        """获取或创建视频流.
        现在基于config中的URL来共享流实例，并支持多种流引擎。
        original_stream_id 主要用于日志和映射。
        """
        stream_url = config.get("url")
        if not stream_url:
            normal_logger.error(f"配置中缺少 'url'，无法获取或创建流。original_stream_id: {original_stream_id}")
            return None

        stream_key = self._generate_stream_key(stream_url)
        if not stream_key: # Should not happen if URL is present
            normal_logger.error(f"从URL '{stream_url}' 生成 stream_key 失败。original_stream_id: {original_stream_id}")
            return None

        # 创建当前事件循环的锁
        async with asyncio.Lock():
            # 记录 original_stream_id 到 stream_key 的映射
            # 这允许我们之后通过 original_stream_id 找到 stream_key
            if original_stream_id: # 只有当original_stream_id有效时才记录映射
                 self._stream_id_to_key_map[original_stream_id] = stream_key
                 normal_logger.info(f"映射: original_stream_id '{original_stream_id}' -> stream_key '{stream_key}' (URL: {stream_url})")

            # 检查流是否已基于 stream_key 存在
            if stream_key in self._streams:
                normal_logger.info(f"返回已存在的共享视频流 (key: {stream_key}) for original_id '{original_stream_id}' (URL: {stream_url})")
                return self._streams[stream_key]

            # 检查是否有流引擎配置
            stream_config = config.get("stream_config", {})
            engine_type = stream_config.get("engine", "auto")
            
            # 创建新的视频流，根据配置选择引擎
            normal_logger.info(f"创建新的共享视频流 (key: {stream_key}) for original_id '{original_stream_id}' (URL: {stream_url}), 引擎: {engine_type}")
            
            stream = None
            
            if engine_type in ["gstreamer", "opencv"] or (engine_type == "auto" and self._should_use_media_kit_factory(stream_url, stream_config)):
                # 使用新的流工厂创建流
                stream = await self._create_media_kit_stream(stream_key, stream_url, stream_config)
            else:
                # 使用ZLMediaKit创建流（保持向后兼容）
                stream = await self._create_zlm_stream(stream_key, config)

            if not stream:
                # 清理映射
                if original_stream_id and original_stream_id in self._stream_id_to_key_map:
                    del self._stream_id_to_key_map[original_stream_id]
                return None

            # 添加到管理器，使用 stream_key 作为键
            self._streams[stream_key] = stream

            # 启动流
            try:
                await stream.start()
                # 添加测试标记
                protocol = self._detect_protocol(stream_url)
                
                test_logger.info(f"TEST_LOG_MARKER: {protocol}_STREAM_CONNECTED (key: {stream_key})")
                test_logger.info(f"TEST_LOG_MARKER: STREAM_CONNECTED (key: {stream_key})")
                
            except Exception as e:
                exception_logger.exception(f"启动流失败 (key: {stream_key}, URL: {stream_url}): {str(e)}")
                # 从管理器中移除失败的流
                if stream_key in self._streams:
                    del self._streams[stream_key]
                # 也从映射中移除（如果已添加）
                if original_stream_id and original_stream_id in self._stream_id_to_key_map:
                    del self._stream_id_to_key_map[original_stream_id]
                raise # 重新抛出异常，让上层处理

            return stream

    def _should_use_media_kit_factory(self, stream_url: str, stream_config: Dict[str, Any]) -> bool:
        """判断是否应该使用媒体工具包工厂而非ZLMediaKit"""
        # 如果明确指定了GStreamer或OpenCV，使用工厂
        engine = stream_config.get("engine", "auto")
        if engine in ["gstreamer", "opencv"]:
            return True
        
        # 如果启用了硬件解码或低延迟，推荐使用GStreamer
        if stream_config.get("enable_hardware_decode") or stream_config.get("low_latency"):
            return True
        
        # ONVIF协议使用专用处理器
        if stream_url.lower().startswith("onvif://"):
            return True
        
        # 其他情况使用ZLMediaKit（保持现有行为）
        return False

    async def _create_media_kit_stream(self, stream_key: str, stream_url: str, stream_config: Dict[str, Any]) -> Optional[IVideoStream]:
        """使用媒体工具包工厂创建流"""
        try:
            # 准备流工厂配置
            factory_config = {
                "stream_id": stream_key,
                "url": stream_url,
                "preferred_engine": stream_config.get("engine", "auto"),
                "enable_hardware_decode": stream_config.get("enable_hardware_decode", False),
                "low_latency": stream_config.get("low_latency", False),
                "gstreamer": {
                    "hardware_decoder": "auto",
                    "buffer_size": 100,
                    "max_buffer_ms": 1000,
                    "min_buffer_ms": 100,
                    "rtsp_latency": 200,
                    "drop_on_latency": True,
                    "network_timeout": 20,
                    "debug_pipeline": False,
                    "log_level": "WARNING"
                }
            }
            
            # 选择引擎
            engine_map = {
                "auto": StreamEngine.AUTO,
                "gstreamer": StreamEngine.GSTREAMER,
                "opencv": StreamEngine.OPENCV
            }
            engine = engine_map.get(stream_config.get("engine", "auto"), StreamEngine.AUTO)
            
            # 创建流
            base_stream = StreamFactory.create_stream(factory_config, engine)
            if not base_stream:
                normal_logger.error(f"流工厂创建流失败: {stream_url}")
                return None
            
            # 包装为IVideoStream接口（需要适配器）
            from .adapters import MediaKitStreamAdapter
            stream = MediaKitStreamAdapter(base_stream)
            
            normal_logger.info(f"使用媒体工具包工厂创建流成功: {stream_url}, 引擎: {engine.value}")
            return stream
            
        except Exception as e:
            exception_logger.exception(f"使用媒体工具包工厂创建流失败: {str(e)}")
            return None

    async def _create_zlm_stream(self, stream_key: str, config: Dict[str, Any]) -> Optional[IVideoStream]:
        """使用ZLMediaKit创建流（保持向后兼容）"""
        try:
            # 导入ZLMediaKit管理器
            from core.media_kit.zlm_manager import zlm_manager

            # 创建新的ZLMediaKit视频流
            stream = ZLMVideoStream(stream_key, config, zlm_manager)
            
            normal_logger.info(f"使用ZLMediaKit创建流成功: {config.get('url', '')}")
            return stream
            
        except Exception as e:
            exception_logger.exception(f"使用ZLMediaKit创建流失败: {str(e)}")
            return None

    def _detect_protocol(self, stream_url: str) -> str:
        """检测流协议类型"""
        if not stream_url:
            return "UNKNOWN"
        
        stream_url_lower = stream_url.lower()
        if stream_url_lower.startswith("rtsp://"):
            return "RTSP"
        elif stream_url_lower.startswith("rtmp://"):
            return "RTMP"
        elif stream_url_lower.startswith("http://") and stream_url_lower.endswith(".m3u8"):
            return "HLS"
        elif stream_url_lower.startswith("http://") and stream_url_lower.endswith(".flv"):
            return "HTTP-FLV"
        elif "webrtc" in stream_url_lower:
            return "WEBRTC"
        elif "gb28181" in stream_url_lower:
            return "GB28181"
        elif stream_url_lower.startswith("onvif://"):
            return "ONVIF"
        else:
            return "UNKNOWN"

    async def release_stream(self, original_stream_id: str) -> bool: # 参数名改为 original_stream_id
        """释放流资源. 当流不再被任何订阅者使用时，实际停止并移除.
        Args:
            original_stream_id: 任务最初用于订阅的ID.
        """
        stream_key = self._stream_id_to_key_map.get(original_stream_id)
        if not stream_key:
            normal_logger.warning(f"尝试释放流时，未找到 original_stream_id '{original_stream_id}' 到 stream_key 的映射。")
            return False
            
        # 创建当前事件循环的锁
        async with asyncio.Lock():
            if stream_key not in self._streams:
                normal_logger.warning(f"尝试释放不存在的流 (key: {stream_key}, original_id: {original_stream_id})")
                # 清理可能存在的无效映射
                if original_stream_id in self._stream_id_to_key_map:
                    del self._stream_id_to_key_map[original_stream_id]
                return False

            stream = self._streams[stream_key]

            # 如果没有订阅者，停止并移除流
            # 注意：这里的 stream.subscriber_count 是由 stream.subscribe 和 stream.unsubscribe 维护的
            if stream.subscriber_count == 0:
                normal_logger.info(f"共享流 (key: {stream_key}, URL: {stream.url if hasattr(stream, 'url') else 'N/A'}) 没有订阅者，停止并移除 (由 original_id '{original_stream_id}' 触发检查)")

                # 停止流
                await stream.stop()

                # 从管理器移除
                del self._streams[stream_key]
                
                # 清理所有指向此 stream_key 的 original_stream_id 映射
                # 这是一个简化的清理，假设一个 original_stream_id 只会映射到一个 stream_key
                # 如果多个 original_stream_id 可能映射到同一个 stream_key (虽然get_or_create中是覆盖),
                # 那么这里需要更复杂的逻辑来只移除与当前操作相关的映射。
                # 但基于当前 get_or_create_stream 的逻辑，一个 original_stream_id 只会映射一次。
                # 当流被删除时，所有相关的 original_stream_id 映射都应该被清理。
                # 然而，一个 stream_key 可能被多个 original_stream_id 映射过（虽然是覆盖式的）。
                # 更安全的做法是迭代查找并删除所有值为 stream_key 的条目。
                keys_to_remove_from_map = [k for k, v in self._stream_id_to_key_map.items() if v == stream_key]
                for k_map in keys_to_remove_from_map:
                    del self._stream_id_to_key_map[k_map]
                    normal_logger.info(f"移除映射: original_stream_id '{k_map}' -> stream_key '{stream_key}' (因共享流停止)")

                return True

            normal_logger.info(f"共享流 (key: {stream_key}) 仍有 {stream.subscriber_count} 个订阅者，保持运行 (检查由 original_id '{original_stream_id}' 触发)")
            return False

    async def get_stream(self, original_stream_id: str) -> Optional[IVideoStream]: # 参数名改为 original_stream_id
        """获取视频流
        Args:
            original_stream_id: 任务最初用于订阅的ID.
        """
        stream_key = self._stream_id_to_key_map.get(original_stream_id)
        if not stream_key:
            normal_logger.debug(f"获取流时，未找到 original_stream_id '{original_stream_id}' 的映射。")
            return None
            
        # 创建当前事件循环的锁
        async with asyncio.Lock(): # get 操作通常不需要锁整个字典，但为了与之前逻辑保持一致
            return self._streams.get(stream_key)

    async def subscribe_stream(self, stream_id_from_task: str, subscriber_id: str, config: Dict[str, Any]) -> Tuple[bool, Optional[asyncio.Queue]]:
        """订阅视频流
        Args:
            stream_id_from_task: 任务提供的流标识符 (例如 video_id 或 stream_<task_id>).
            subscriber_id: 订阅者ID (通常是任务ID).
            config: 流配置, 必须包含 'url'.
        Returns:
            Tuple[bool, Optional[asyncio.Queue]]: (是否成功, 帧队列)
        """
        # 确保 stream_id_from_task 有效，这是映射到 stream_key 的键之一
        # TaskProcessor.process_stream_worker 中会生成 stream_id_to_use，它会作为此处的 stream_id_from_task
        if not stream_id_from_task:
            # 如果 TaskProcessor 没有提供有效的 stream_id_from_task (不太可能发生，但作为防御)
            # 我们需要一个。之前这里会用 subscriber_id 生成。
            # 我们应该依赖 TaskProcessor 提供一个有意义的 stream_id_from_task。
            # 例如，TaskProcessor中的 stream_id_from_task_config (即 video_id 或 stream_id 或 stream_task_id)
            normal_logger.warning(f"subscribe_stream 调用时 stream_id_from_task 为空，将使用 subscriber_id '{subscriber_id}' 代替生成。这可能不符合预期。")
            stream_id_from_task = f"stream_sub_{subscriber_id}" # 创建一个临时的，用于映射

        # 检查URL格式 (这部分可以保留，用于日志或早期检查，但核心共享依赖 get_or_create_stream)
        stream_url = config.get("url", "")
        if not stream_url:
            normal_logger.error(f"流URL为空，无法订阅流。original_id: {stream_id_from_task}")
            return False, None
            
        # 检查URL协议 (这部分日志和配置类型设置可以保留)
        if stream_url.startswith("rtsp://"):
            normal_logger.info(f"检测到RTSP流: {stream_url}")
            config["type"] = "rtsp"
        elif stream_url.startswith("rtmp://"):
            normal_logger.info(f"检测到RTMP流: {stream_url}")
            config["type"] = "rtmp"
        elif stream_url.startswith("http://") or stream_url.startswith("https://"):
            if ".m3u8" in stream_url:
                normal_logger.info(f"检测到HLS流: {stream_url}")
                config["type"] = "hls"
            else: # 假设是 HTTP-FLV 或其他 HTTP 流
                normal_logger.info(f"检测到HTTP流: {stream_url}")
                config["type"] = "http" # 可以更具体，但ZLM通常能自动处理
        else:
            normal_logger.warning(f"未知流类型: {stream_url}，默认使用RTSP处理")
            config["type"] = "rtsp" # 默认

        # 获取或创建流. 传入 stream_id_from_task 作为 original_stream_id
        try:
            normal_logger.info(f"尝试获取或创建流 for original_id: {stream_id_from_task}, URL: {stream_url}")
            stream = await self.get_or_create_stream(stream_id_from_task, config)
            
            if not stream: # get_or_create_stream 失败 (例如无URL)
                 normal_logger.error(f"获取或创建流失败 for original_id: {stream_id_from_task}, URL: {stream_url}")
                 return False, None

            # 订阅流
            stream_key_for_logging = self._generate_stream_key(stream_url) # 用于日志
            normal_logger.info(f"开始订阅共享流 (key: {stream_key_for_logging}), original_id: {stream_id_from_task}, 订阅者: {subscriber_id}")
            success, queue = await stream.subscribe(subscriber_id)
            
            if success:
                normal_logger.info(f"订阅共享流成功 (key: {stream_key_for_logging}), original_id: {stream_id_from_task}, 订阅者: {subscriber_id}")
            else:
                normal_logger.error(f"订阅共享流失败 (key: {stream_key_for_logging}), original_id: {stream_id_from_task}, 订阅者: {subscriber_id}")

            return success, queue
        except Exception as e:
            exception_logger.exception(f"获取或订阅流 (original_id: {stream_id_from_task}, URL: {stream_url}) 时出错: {str(e)}")
            return False, None

    async def unsubscribe_stream(self, stream_id_from_task: str, subscriber_id: str) -> bool:
        """取消订阅视频流
        Args:
            stream_id_from_task: 任务最初用于订阅的ID.
            subscriber_id: 订阅者ID (通常是任务ID).
        """
        stream_key = self._stream_id_to_key_map.get(stream_id_from_task)
        if not stream_key:
            normal_logger.warning(f"取消订阅时，未找到 original_stream_id '{stream_id_from_task}' 的 stream_key 映射。")
            return False

        # 创建当前事件循环的锁
        async with asyncio.Lock():
            stream = self._streams.get(stream_key)
            if not stream:
                normal_logger.warning(f"取消订阅时，未找到 stream_key '{stream_key}' 对应的流实例 (original_id: {stream_id_from_task})。可能已被释放。")
                # 清理无效的映射
                if stream_id_from_task in self._stream_id_to_key_map:
                    del self._stream_id_to_key_map[stream_id_from_task]
                return False

            normal_logger.info(f"开始取消订阅共享流 (key: {stream_key}), original_id: {stream_id_from_task}, 订阅者: {subscriber_id}")
            unsubscribed_successfully = await stream.unsubscribe(subscriber_id)

            if unsubscribed_successfully:
                normal_logger.info(f"取消订阅共享流成功 (key: {stream_key}), original_id: {stream_id_from_task}, 订阅者: {subscriber_id}。剩余订阅者: {stream.subscriber_count}")
            else:
                # unsubscribe 内部应该已经打印了日志
                normal_logger.warning(f"取消订阅共享流似乎未成功或订阅者不存在 (key: {stream_key}), original_id: {stream_id_from_task}, 订阅者: {subscriber_id}")


            # 如果没有其他订阅者，则释放（停止并移除）共享流
            if stream.subscriber_count == 0:
                normal_logger.info(f"共享流 (key: {stream_key}) 已无订阅者，准备停止并移除。")
                await stream.stop()
                del self._streams[stream_key]
                normal_logger.info(f"共享流 (key: {stream_key}) 已停止并从管理器移除。")
                
                # 清理所有指向此已停止 stream_key 的 original_stream_id 映射
                keys_to_remove_from_map = [k for k, v in self._stream_id_to_key_map.items() if v == stream_key]
                for k_map in keys_to_remove_from_map:
                    del self._stream_id_to_key_map[k_map]
                    normal_logger.info(f"移除映射: original_stream_id '{k_map}' -> stream_key '{stream_key}' (因共享流停止)")
            else:
                 normal_logger.info(f"共享流 (key: {stream_key}) 仍有 {stream.subscriber_count} 个订阅者，保持运行。")

            # 不再需要单独调用 self.release_stream(stream_id_from_task)
            # 因为释放逻辑已在此处处理
            return unsubscribed_successfully

    async def update_stream_status(self, original_stream_id: str, status: StreamStatus, health_status: StreamHealthStatus, error_msg: str = "") -> None: # 参数名改为 original_stream_id
        """更新流的状态（如果存在）"""
        stream_key = self._stream_id_to_key_map.get(original_stream_id)
        if not stream_key:
            normal_logger.debug(f"更新流状态时，未找到 original_stream_id '{original_stream_id}' 的映射。")
            return

        async with asyncio.Lock(): # 保护对 _streams 的访问
            stream = self._streams.get(stream_key)
            if stream:
                # 假设 IVideoStream 有 update_status 方法，或者这里直接更新 StreamManager 维护的状态
                # 目前 IVideoStream 内部管理自己的状态，这里可能更多是日志或外部通知
                normal_logger.info(f"流状态更新 (key: {stream_key}, original_id: {original_stream_id}): Status={status.name}, Health={health_status.name}, Msg='{error_msg}'")
                # 如果需要 StreamManager 维护一个全局状态视图，可以在这里更新
                # stream.current_status = status
                # stream.current_health_status = health_status
                # stream.last_error = error_msg
            else:
                normal_logger.warning(f"尝试更新状态的流 (key: {stream_key}, original_id: {original_stream_id}) 不存在。")

    async def reconnect_stream(self, original_stream_id: str) -> bool: # 参数名改为 original_stream_id
        """尝试重新连接指定的流（如果存在且支持）"""
        stream_key = self._stream_id_to_key_map.get(original_stream_id)
        if not stream_key:
            normal_logger.warning(f"尝试重连流时，未找到 original_stream_id '{original_stream_id}' 的映射。")
            return False

        async with asyncio.Lock(): # 保护对 _streams 的访问
            stream = self._streams.get(stream_key)
            if stream:
                if hasattr(stream, 'reconnect'):
                    normal_logger.info(f"请求重新连接流 (key: {stream_key}, original_id: {original_stream_id})")
                    try:
                        return await stream.reconnect()
                    except Exception as e:
                        exception_logger.error(f"重新连接流 (key: {stream_key}, original_id: {original_stream_id}) 时发生错误: {e}")
                        return False
                else:
                    normal_logger.warning(f"流 (key: {stream_key}, original_id: {original_stream_id}) 不支持 reconnect 方法。")
                    return False
            else:
                normal_logger.warning(f"尝试重新连接的流 (key: {stream_key}, original_id: {original_stream_id}) 不存在。")
                return False

    async def get_all_streams(self) -> List[Dict[str, Any]]:
        """获取所有活动流的信息（基于共享流的视角）"""
        infos = []
        async with asyncio.Lock(): # 保护对 _streams 的迭代
            for stream_key, stream_instance in self._streams.items():
                stream_info = {
                    "stream_key": stream_key, # 这是URL派生的键
                    "url": getattr(stream_instance, 'url', 'N/A'), # 尝试获取流的URL
                    "type": getattr(stream_instance, 'stream_type', 'N/A'), # 尝试获取流类型
                    "status": getattr(stream_instance, 'status', StreamStatus.UNKNOWN).name,
                    "health_status": getattr(stream_instance, 'health_status', StreamHealthStatus.UNKNOWN).name,
                    "subscriber_count": getattr(stream_instance, 'subscriber_count', 0),
                    "created_at": getattr(stream_instance, 'created_at', 'N/A'),
                    # 获取映射到此 stream_key 的所有 original_stream_ids
                    "associated_original_ids": [k for k, v in self._stream_id_to_key_map.items() if v == stream_key]
                }
                infos.append(stream_info)
        return infos

    async def get_stream_info(self, original_stream_id: str) -> Optional[Dict[str, Any]]: # 参数名改为 original_stream_id
        """获取特定流的信息（通过其 original_stream_id）"""
        stream_key = self._stream_id_to_key_map.get(original_stream_id)
        if not stream_key:
            normal_logger.debug(f"获取流信息时，未找到 original_stream_id '{original_stream_id}' 的映射。")
            return None

        async with asyncio.Lock(): # 保护对 _streams 的访问
            stream_instance = self._streams.get(stream_key)
            if stream_instance:
                stream_info = {
                    "stream_key": stream_key,
                    "original_stream_id": original_stream_id, # 传入的ID
                    "url": getattr(stream_instance, 'url', 'N/A'),
                    "type": getattr(stream_instance, 'stream_type', 'N/A'),
                    "status": getattr(stream_instance, 'status', StreamStatus.UNKNOWN).name,
                    "health_status": getattr(stream_instance, 'health_status', StreamHealthStatus.UNKNOWN).name,
                    "subscriber_count": getattr(stream_instance, 'subscriber_count', 0),
                    "created_at": getattr(stream_instance, 'created_at', 'N/A'),
                    "subscribers": list(getattr(stream_instance, '_subscribers', {}).keys()) # 假设_subscribers存在且键是订阅者ID
                }
                return stream_info
            return None

    async def get_stream_status(self, original_stream_id: str) -> Optional[StreamStatus]: # 参数名改为 original_stream_id
        """获取流的当前状态（通过其 original_stream_id）"""
        stream_key = self._stream_id_to_key_map.get(original_stream_id)
        if not stream_key:
            # normal_logger.debug(f"获取流状态时，未找到 original_stream_id '{original_stream_id}' 的映射。") # 可能过于频繁
            return None
            
        async with asyncio.Lock(): # 保护对 _streams 的访问
            stream = self._streams.get(stream_key)
            if stream:
                return getattr(stream, 'status', StreamStatus.UNKNOWN)
            return None

    async def stop_all_streams(self):
        """停止所有活动流"""
        normal_logger.info("正在停止所有活动流...")
        async with asyncio.Lock(): # 保护对 _streams 的迭代和修改
            stream_keys_to_stop = list(self._streams.keys())
            for stream_key in stream_keys_to_stop:
                stream = self._streams.get(stream_key)
                if stream:
                    try:
                        normal_logger.info(f"正在停止流 (key: {stream_key})")
                        await stream.stop()
                    except Exception as e:
                        exception_logger.error(f"停止流 (key: {stream_key}) 时出错: {e}")
                if stream_key in self._streams: # 再次检查，因为 stop 可能异步且耗时
                    del self._streams[stream_key]
            
            # 清空映射
            self._stream_id_to_key_map.clear()
        normal_logger.info("所有活动流已停止并已从管理器移除。")

    async def shutdown(self):
        """关闭流管理器"""
        # 停止所有流
        await self.stop_all_streams()

        # 关闭ZLMediaKit管理器
        try:
            from core.media_kit.zlm_manager import zlm_manager
            await zlm_manager.shutdown()
            normal_logger.info("ZLMediaKit管理器已关闭")
        except Exception as e:
            exception_logger.exception(f"关闭ZLMediaKit管理器时出错: {str(e)}")

        normal_logger.info("流管理器已关闭")

    async def get_stream_proxy_url(self, original_stream_id: str) -> Optional[Dict[str, Any]]: # 参数名改为 original_stream_id
        """获取流的代理播放地址（通过其 original_stream_id）"""
        stream_key = self._stream_id_to_key_map.get(original_stream_id)
        if not stream_key:
            normal_logger.warning(f"获取代理URL时，未找到 original_stream_id '{original_stream_id}' 的映射。")
            return None

        async with asyncio.Lock(): # 保护对 _streams 的访问
            stream = self._streams.get(stream_key)
            if stream and isinstance(stream, ZLMVideoStream): # 确保是ZLMVideoStream类型
                try:
                    proxy_info = await stream.get_zlm_player_proxy_info()
                    if proxy_info:
                         normal_logger.info(f"获取到流 (key: {stream_key}, original_id: {original_stream_id}) 的代理播放地址: {proxy_info}")
                         return proxy_info
                    else:
                         normal_logger.warning(f"未能获取流 (key: {stream_key}, original_id: {original_stream_id}) 的代理播放地址。")
                         return None
                except Exception as e:
                    exception_logger.error(f"获取流 (key: {stream_key}, original_id: {original_stream_id}) 代理播放地址时出错: {e}")
                    return None
            elif not stream:
                normal_logger.warning(f"尝试获取代理URL的流 (key: {stream_key}, original_id: {original_stream_id}) 不存在。")
                return None
            else: # stream 存在但不是 ZLMVideoStream
                normal_logger.warning(f"流 (key: {stream_key}, original_id: {original_stream_id}) 不是 ZLMVideoStream 类型，无法获取代理播放地址。类型: {type(stream)}")
                return None

