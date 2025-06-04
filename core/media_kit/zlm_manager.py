"""
ZLMediaKit管理器模块
负责初始化和管理ZLMediaKit实例
"""
import os
import sys
import json
import threading
import ctypes
import traceback
import subprocess
from typing import Dict, Any, List, Optional, Callable
import asyncio
import platform

from shared.utils.logger import get_normal_logger, get_exception_logger
from core.config import settings
from .zlm_config import ZLMConfig, zlm_config

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class ZLMediaKitManager:
    """ZLMediaKit管理器类，负责初始化和管理ZLMediaKit实例"""

    _instance = None
    _lock = threading.Lock()
    _shutdown_lock = threading.Lock()

    def __new__(cls, config: Optional[ZLMConfig] = None):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ZLMediaKitManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[ZLMConfig] = None):
        """初始化ZLMediaKit HTTP API管理器

        Args:
            config: ZLMediaKit配置，如果为None则使用默认配置
        """
        if self._initialized:
            return

        self._config = config or zlm_config
        # 确保API URL格式正确，移除末尾斜杠
        server_address = self._config.server_address.rstrip('/')
        http_port = self._config.http_port
        self._api_url = f"http://{server_address}:{http_port}"
        self._secret = self._config.api_secret
        self._streams = {}  # 流ID到流对象的映射
        self._event_callbacks = {}  # 事件回调函数
        self._zlm_process = None  # ZLM进程对象

        # 锁，用于保护流相关操作
        self._stream_lock = threading.Lock()

        # 初始化标记
        self._is_running = False
        self._is_shutting_down = False
        self._api_ready = False  # 新增：标记API是否就绪
        self._initialized = True

        normal_logger.info(f"ZLMediaKit管理器初始化完成，配置已加载: {self._api_url}")

    @property
    def is_api_ready(self) -> bool:
        """获取API是否就绪"""
        return self._api_ready

    @property
    def zlm_internal_host(self) -> str:
        """获取ZLM内部主机地址，通常是server_address"""
        return self._config.server_address

    @property
    def rtmp_port(self) -> int:
        """获取ZLM配置的RTMP端口"""
        return self._config.rtmp_port

    @property
    def rtsp_port(self) -> int:
        """获取ZLM配置的RTSP端口"""
        return self._config.rtsp_port

    async def initialize(self) -> None:
        """初始化ZLMediaKit环境

        Returns:
            None
        """
        try:
            normal_logger.info("正在初始化ZLMediaKit环境...")

            # 如果已经运行则跳过
            if self._is_running:
                normal_logger.info("ZLMediaKit已经在运行中")
                return

            # 获取系统类型
            system = platform.system().lower()
            if system == "darwin":
                system_type = "darwin"  # macOS
            elif system == "linux":
                system_type = "linux"   # Linux x64
            elif system == "windows":
                system_type = "windows" # Windows x64
            else:
                raise RuntimeError(f"不支持的操作系统类型: {system}")

            normal_logger.info(f"当前系统类型: {system_type}")

            # 获取ZLMediaKit可执行文件路径
            zlm_binary = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                    "zlmos", system_type, "MediaServer")
            
            # Windows系统需要添加.exe后缀
            if system_type == "windows":
                zlm_binary += ".exe"
            
            if not os.path.exists(zlm_binary):
                raise FileNotFoundError(f"ZLMediaKit可执行文件不存在: {zlm_binary}")

            # 使用config/zlm/config.ini作为主配置文件
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                     "config", "zlm", "config.ini")
            
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"ZLMediaKit配置文件不存在: {config_file}")

            # 启动ZLM进程，使用指定的配置文件
            try:
                normal_logger.info(f"正在启动ZLMediaKit服务器: {zlm_binary}")
                normal_logger.info(f"使用配置文件: {config_file}")
                self._zlm_process = subprocess.Popen(
                    [zlm_binary, "-c", config_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except Exception as e:
                exception_logger.exception(f"启动ZLMediaKit服务器失败: {str(e)}")
                raise

            # 设置运行状态
            self._is_running = True

            # 等待一段时间让服务器完全启动
            await asyncio.sleep(2)

        except Exception as e:
            exception_logger.exception(f"ZLMediaKit环境初始化失败: {str(e)}")
            self._is_running = False
            raise

    async def test_api_connection(self) -> bool:
        """测试ZLMediaKit HTTP API连接

        Returns:
            bool: API连接是否成功
        """
        if self._api_ready:
            return True

        try:
            normal_logger.info("正在测试ZLMediaKit HTTP API连接...")
            
            # 等待ZLMediaKit服务启动
            max_retries = 5
            retry_interval = 2  # 秒
            
            for retry in range(max_retries):
                try:
                    # 测试HTTP API连接
                    test_result = self.call_api("getApiList")
                    if test_result.get("code") == 0:
                        self._api_ready = True
                        return True
                    else:
                        error_msg = test_result.get("msg", "未知错误")
                        normal_logger.warning(f"ZLMediaKit HTTP API连接失败: {error_msg}")
                except Exception as api_err:
                    normal_logger.warning(f"尝试连接ZLMediaKit HTTP API (第{retry + 1}次): {str(api_err)}")
                    if retry < max_retries - 1:
                        normal_logger.info(f"等待 {retry_interval} 秒后重试...")
                        await asyncio.sleep(retry_interval)
                    continue

            normal_logger.error("无法连接到ZLMediaKit HTTP API，请检查服务是否正常运行")
            return False

        except Exception as e:
            exception_logger.exception(f"测试ZLMediaKit HTTP API连接时出错: {str(e)}")
            return False

    async def shutdown(self) -> None:
        """关闭ZLMediaKit HTTP API连接并清理资源

        Returns:
            None
        """
        try:
            with self._shutdown_lock:
                if not self._is_running:
                    normal_logger.info("ZLMediaKit未运行，无需关闭")
                    return

                if self._is_shutting_down:
                    normal_logger.info("ZLMediaKit正在关闭中，跳过重复关闭")
                    return

                self._is_shutting_down = True
                normal_logger.info("正在关闭ZLMediaKit HTTP API连接...")

                # 1. 首先停止所有流
                with self._stream_lock:
                    stream_ids = list(self._streams.keys())

                for stream_id in stream_ids:
                    try:
                        await self.stop_stream(stream_id)
                    except Exception as e:
                        normal_logger.error(f"停止流 {stream_id} 时出错: {str(e)}")

                # 2. 等待所有流停止
                await asyncio.sleep(1)
                
                # 3. 清理所有回调
                self._event_callbacks.clear()
                
                # 4. 清理内部状态
                self._streams.clear()
                
                # 5. 停止ZLM进程
                if self._zlm_process:
                    try:
                        normal_logger.info("正在停止ZLMediaKit服务器...")
                        self._zlm_process.terminate()
                        try:
                            self._zlm_process.wait(timeout=5)  # 等待进程结束，最多5秒
                        except subprocess.TimeoutExpired:
                            normal_logger.warning("ZLMediaKit服务器未能在5秒内停止，强制结束进程")
                            self._zlm_process.kill()  # 强制结束进程
                        normal_logger.info("ZLMediaKit服务器已停止")
                    except Exception as e:
                        exception_logger.error(f"停止ZLMediaKit服务器时出错: {str(e)}")
                    finally:
                        self._zlm_process = None
                
                # 6. 强制等待更长时间确保所有清理完成
                await asyncio.sleep(1.0)
                
                # 7. 尝试强制垃圾回收
                import gc
                gc.collect()
                
                # 8. 设置状态
                self._is_running = False
                self._is_shutting_down = False
                self._api_ready = False

                normal_logger.info("ZLMediaKit HTTP API连接已关闭")

        except Exception as e:
            exception_logger.exception(f"关闭ZLMediaKit HTTP API连接时出错: {str(e)}")
            self._is_shutting_down = False

    async def create_stream(self, stream_id: str, stream_config: Dict[str, Any]) -> bool:
        """创建一个媒体流

        Args:
            stream_id: 流ID
            stream_config: 流配置

        Returns:
            bool: 是否成功创建
        """
        try:
            # 如果流已存在则返回
            with self._stream_lock:
                if stream_id in self._streams:
                    normal_logger.info(f"流 {stream_id} 已存在")
                    return True

            # 获取流类型和地址
            stream_type = stream_config.get("type", "rtsp")
            stream_url = stream_config.get("url", "")

            if not stream_url:
                normal_logger.error(f"创建流 {stream_id} 失败: 未提供URL")
                return False

            # 使用HTTP API创建流代理
            # 准备参数
            vhost = stream_config.get("vhost", "__defaultVhost__")
            app = stream_config.get("app", "live")
            stream_name = stream_config.get("stream_name", stream_id)
            
            # 根据流类型和其他配置，设置API请求参数
            api_params = {
                "vhost": vhost,
                "app": app,
                "stream": stream_name,
                "url": stream_url,
                "enable_rtsp": stream_config.get("enable_rtsp", 1),
                "enable_rtmp": stream_config.get("enable_rtmp", 1),
                "enable_hls": stream_config.get("enable_hls", 1),
                "enable_mp4": stream_config.get("enable_mp4", 0),
                "rtp_type": stream_config.get("rtp_type", 0),  # 0是TCP，1是UDP
                "timeout_sec": stream_config.get("timeout_sec", 10),
                "retry_count": stream_config.get("retry_count", -1),  # -1表示无限重试
            }
            
            # 调用addStreamProxy API
            normal_logger.info(f"使用HTTP API创建流代理: {stream_url} -> {app}/{stream_name}")
            result = self.call_api("addStreamProxy", api_params)
            
            if result.get("code") == 0:
                normal_logger.info(f"HTTP API创建流代理成功: {stream_id}")
                stream_key = result.get("data", {}).get("key", "")
                
                # 创建ZLM流对象
                from .zlm_stream import ZLMVideoStream
                stream = ZLMVideoStream(stream_id, stream_config, self)
                
                # 保存stream_key到配置中，方便后续操作
                stream_config["stream_key"] = stream_key
                
                # 启动流
                success = await stream.start()
                if not success:
                    normal_logger.error(f"启动流 {stream_id} 失败")
                    # 清理资源，删除流代理
                    self.call_api("delStreamProxy", {"key": stream_key})
                    return False

                # 保存流对象
                with self._stream_lock:
                    self._streams[stream_id] = stream

                normal_logger.info(f"成功创建并启动流 {stream_id} (使用HTTP API)")
                return True
            else:
                error_msg = result.get("msg", "未知错误")
                normal_logger.error(f"HTTP API创建流代理失败: {error_msg}")
                return False
            
        except Exception as e:
            exception_logger.exception(f"创建流 {stream_id} 时出错: {str(e)}")
            return False

    async def stop_stream(self, stream_id: str) -> bool:
        """停止一个媒体流

        Args:
            stream_id: 流ID

        Returns:
            bool: 是否成功停止
        """
        try:
            # 获取流对象
            stream = None
            stream_key = None

            with self._stream_lock:
                if stream_id in self._streams:
                    stream = self._streams[stream_id]
                    # 获取流关联的stream_key
                    if hasattr(stream, 'config') and isinstance(stream.config, dict):
                        stream_key = stream.config.get("stream_key", "")
                    del self._streams[stream_id]

            if not stream:
                normal_logger.warning(f"流 {stream_id} 不存在，无需停止")
                return True

            # 停止流
            await stream.stop()

            # 使用HTTP API关闭流代理
            if stream_key:
                # 使用delStreamProxy关闭流代理
                normal_logger.info(f"使用HTTP API关闭流代理: {stream_id}, key: {stream_key}")
                result = self.call_api("delStreamProxy", {"key": stream_key})
                if result.get("code") == 0:
                    normal_logger.info(f"HTTP API关闭流代理成功: {stream_id}")
                else:
                    error_msg = result.get("msg", "未知错误")
                    normal_logger.warning(f"HTTP API关闭流代理失败: {error_msg}")
                    
                    # 尝试使用close_streams API
                    app = stream.config.get("app", "live")
                    stream_name = stream.config.get("stream_name", stream_id)
                    vhost = stream.config.get("vhost", "__defaultVhost__")
                    close_params = {
                        "vhost": vhost,
                        "app": app,
                        "stream": stream_name,
                        "force": 1
                    }
                    normal_logger.info(f"尝试使用close_streams API关闭流: {stream_id}")
                    result = self.call_api("close_streams", close_params)
                    if result.get("code") == 0:
                        normal_logger.info(f"close_streams API关闭流成功: {stream_id}")
                    else:
                        normal_logger.warning(f"close_streams API关闭流失败: {result}")
            else:
                # 如果没有stream_key，尝试使用close_streams API
                app = stream.config.get("app", "live")
                stream_name = stream.config.get("stream_name", stream_id)
                vhost = stream.config.get("vhost", "__defaultVhost__")
                close_params = {
                    "vhost": vhost,
                    "app": app,
                    "stream": stream_name,
                    "force": 1
                }
                normal_logger.info(f"使用close_streams API关闭流: {stream_id}")
                result = self.call_api("close_streams", close_params)
                if result.get("code") == 0:
                    normal_logger.info(f"close_streams API关闭流成功: {stream_id}")
                else:
                    normal_logger.warning(f"close_streams API关闭流失败: {result}")

            normal_logger.info(f"成功停止流 {stream_id}")
            return True
        except Exception as e:
            exception_logger.exception(f"停止流 {stream_id} 时出错: {str(e)}")
            return False

    async def get_stream_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取流信息

        Args:
            stream_id: 流ID

        Returns:
            Optional[Dict[str, Any]]: 流信息，如果流不存在则返回None
        """
        try:
            # 首先检查本地缓存的流
            stream = None
            with self._stream_lock:
                if stream_id in self._streams:
                    stream = self._streams[stream_id]
            
            # 如果本地存在流对象，先获取基本信息
            if stream:
                # 获取流的配置信息
                app = stream.config.get("app", "live")
                stream_name = stream.config.get("stream_name", stream_id)
                vhost = stream.config.get("vhost", "__defaultVhost__")
                
                # 使用HTTP API获取媒体信息
                params = {
                    "schema": "rtsp",  # 这里可以根据实际情况选择协议
                    "vhost": vhost,
                    "app": app,
                    "stream": stream_name
                }
                
                # 调用getMediaInfo API
                normal_logger.info(f"使用HTTP API获取流信息: {stream_id}")
                result = self.call_api("getMediaInfo", params)
                
                if result.get("code") == 0:
                    # 合并HTTP API返回的信息与本地流信息
                    media_info = result
                    # 补充流ID信息
                    media_info["stream_id"] = stream_id
                    normal_logger.info(f"成功获取流信息: {stream_id}")
                    return media_info
                else:
                    # 如果API调用失败，返回本地流信息
                    normal_logger.warning(f"HTTP API获取流信息失败，使用本地信息: {stream_id}")
                    return await stream.get_info()
            else:
                normal_logger.warning(f"流 {stream_id} 不存在于本地缓存")
                return None
                
        except Exception as e:
            exception_logger.exception(f"获取流 {stream_id} 信息时出错: {str(e)}")
            return None

    async def get_all_streams(self) -> List[Dict[str, Any]]:
        """获取所有流信息

        Returns:
            List[Dict[str, Any]]: 所有流的信息列表
        """
        try:
            # 使用HTTP API获取所有媒体列表
            normal_logger.info("使用HTTP API获取所有流列表")
            result = self.call_api("getMediaList")
            
            if result.get("code") == 0:
                # 获取API返回的流列表
                media_list = result.get("data", [])
                normal_logger.info(f"HTTP API获取到 {len(media_list)} 个流")
                
                # 为每个流补充本地信息
                streams_info = []
                for media in media_list:
                    # 从API返回的媒体信息中提取流ID
                    stream_name = media.get("stream")
                    app = media.get("app")
                    
                    # 查找对应的本地流ID
                    local_stream_id = None
                    with self._stream_lock:
                        for sid, stream in self._streams.items():
                            if (stream.config.get("stream_name") == stream_name and 
                                stream.config.get("app") == app):
                                local_stream_id = sid
                                break
                    
                    # 补充流ID信息
                    if local_stream_id:
                        media["stream_id"] = local_stream_id
                    else:
                        # 如果找不到对应的本地流ID，使用stream作为ID
                        media["stream_id"] = stream_name
                    
                    streams_info.append(media)
                
                return streams_info
            else:
                # 如果API调用失败，返回本地流信息
                normal_logger.warning("HTTP API获取所有流列表失败，使用本地信息")
                
                # 获取所有流ID
                with self._stream_lock:
                    stream_ids = list(self._streams.keys())

                # 获取每个流的信息
                streams_info = []
                for stream_id in stream_ids:
                    info = await self.get_stream_info(stream_id)
                    if info:
                        streams_info.append(info)

                return streams_info
        except Exception as e:
            exception_logger.exception(f"获取所有流信息时出错: {str(e)}")
            return []

    def call_api(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """调用ZLMediaKit HTTP API

        Args:
            method: API方法名，例如 addStreamProxy
            params: API参数

        Returns:
            Dict[str, Any]: API响应
        """
        try:
            import requests
            import json

            # 确保参数是字典类型
            if params is None:
                params = {}
            
            # 添加鉴权密钥
            params['secret'] = self._secret
            
            # 构建完整URL
            if method.startswith('/'):
                method = method[1:]  # 移除开头的斜杠
            if not method.startswith('index/api/'):
                method = f"index/api/{method}"
            url = f"{self._api_url}/{method}"
            
            # 发送HTTP请求
            normal_logger.info(f"调用ZLM HTTP API: {url}, 参数: {params}")
            
            try:
                # 为所有请求添加超时，防止阻塞
                headers = {'Content-Type': 'application/x-www-form-urlencoded'}
                response = requests.post(url, data=params, headers=headers, timeout=5)
                
                # 解析响应
                result = response.json()
                return result
            except requests.exceptions.RequestException as e:
                return {"code": -1, "msg": f"请求失败: {str(e)}"}
            except json.JSONDecodeError as e:
                return {"code": -1, "msg": f"响应解析失败: {str(e)}"}
            
        except Exception as e:
            exception_logger.exception(f"调用API {method} 时出错: {str(e)}")
            return {"code": -1, "msg": str(e)}

    def register_event_callback(self, event_name: str, callback: Callable) -> None:
        """注册事件回调函数

        Args:
            event_name: 事件名称
            callback: 回调函数
        """
        if event_name not in self._event_callbacks:
            self._event_callbacks[event_name] = set()

        self._event_callbacks[event_name].add(callback)
        normal_logger.debug(f"注册事件回调: {event_name}")

    def unregister_event_callback(self, event_name: str, callback: Callable) -> None:
        """取消注册事件回调函数

        Args:
            event_name: 事件名称
            callback: 回调函数
        """
        if event_name in self._event_callbacks:
            if callback in self._event_callbacks[event_name]:
                self._event_callbacks[event_name].remove(callback)
                normal_logger.debug(f"取消注册事件回调: {event_name}")

    def trigger_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """触发事件

        Args:
            event_name: 事件名称
            data: 事件数据
        """
        if event_name in self._event_callbacks:
            for callback in self._event_callbacks[event_name]:
                try:
                    callback(data)
                except Exception as e:
                    exception_logger.exception(f"执行事件回调 {event_name} 时出错: {str(e)}")

# 单例实例
zlm_manager = ZLMediaKitManager()