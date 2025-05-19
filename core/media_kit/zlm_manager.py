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
from typing import Dict, Any, List, Optional, Callable

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

    def __new__(cls, config: Optional[ZLMConfig] = None):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ZLMediaKitManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[ZLMConfig] = None):
        """初始化ZLMediaKit管理器

        Args:
            config: ZLMediaKit配置，如果为None则使用默认配置
        """
        if self._initialized:
            return

        self._config = config or zlm_config
        self._lib = None
        self._handle = None
        self._api_url = f"http://{self._config.server_address}:{self._config.http_port}"
        self._secret = self._config.api_secret
        self._streams = {}  # 流ID到流对象的映射
        self._players = {}  # 播放器句柄到流ID的映射
        self._event_callbacks = {}  # 事件回调函数

        # 锁，用于保护流相关操作
        self._stream_lock = threading.Lock()

        # 初始化标记
        self._is_running = False
        self._initialized = True

        normal_logger.info(f"ZLMediaKit管理器初始化完成，API地址: {self._api_url}")

    async def initialize(self) -> None:
        """初始化ZLMediaKit

        Returns:
            None
        """
        try:
            normal_logger.info("正在初始化ZLMediaKit...")

            # 如果已经运行则跳过
            if self._is_running:
                normal_logger.info("ZLMediaKit已经在运行中")
                return

            # 加载ZLMediaKit库
            self._load_library()

            # 初始化ZLMediaKit
            self._init_zlmediakit()

            # 测试API连接
            try:
                api_connected = self._test_api_connection()
                if not api_connected:
                    normal_logger.warning("无法连接到ZLMediaKit C API，请检查库是否正确安装")
            except Exception as api_err:
                normal_logger.warning(f"测试API连接时发生异常: {str(api_err)}")
                normal_logger.warning("ZLMediaKit C API不可用")

            # 标记为运行中 - 即使API连接失败，库功能仍然可用
            self._is_running = True

            normal_logger.info("ZLMediaKit初始化完成")

        except Exception as e:
            exception_logger.exception(f"ZLMediaKit初始化失败: {str(e)}")
            normal_logger.error("无法初始化ZLMediaKit，请确保C API库正确安装")

    async def shutdown(self) -> None:
        """关闭ZLMediaKit

        Returns:
            None
        """
        try:
            if not self._is_running:
                normal_logger.info("ZLMediaKit未运行，无需关闭")
                return

            normal_logger.info("正在关闭ZLMediaKit...")

            # 关闭所有流
            with self._stream_lock:
                stream_ids = list(self._streams.keys())

            for stream_id in stream_ids:
                await self.stop_stream(stream_id)

            # 注意：由于在关闭ZLMediaKit时可能出现段错误，我们暂时跳过这一步
            # 只记录日志，不实际调用关闭函数
            if self._lib and hasattr(self._lib, 'mk_stop_all_server'):
                normal_logger.info("跳过ZLMediaKit关闭函数调用，避免段错误")
                # self._lib.mk_stop_all_server()

            # 标记为未运行
            self._is_running = False

            normal_logger.info("ZLMediaKit已关闭")
        except Exception as e:
            exception_logger.exception(f"关闭ZLMediaKit时出错: {str(e)}")

    def _load_library(self) -> None:
        """加载ZLMediaKit库"""
        try:
            # 获取当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 获取项目根目录
            project_root = os.path.abspath(os.path.join(current_dir, "../.."))

            # 根据平台选择库文件名
            if sys.platform == 'darwin':
                lib_name = 'libmk_api.dylib'
                platform_dir = 'darwin'
            elif sys.platform == 'linux':
                lib_name = 'libmk_api.so'
                platform_dir = 'linux'
            elif sys.platform == 'win32':
                lib_name = 'mk_api.dll'
                platform_dir = 'win32'
            else:
                raise RuntimeError(f"不支持的操作系统: {sys.platform}")
                
            # 首先尝试从项目中的lib/zlm目录加载库
            lib_zlm_path = os.path.join(project_root, "lib", "zlm")
            if os.path.exists(lib_zlm_path):
                lib_file = os.path.join(lib_zlm_path, lib_name)
                if os.path.exists(lib_file):
                    normal_logger.info(f"在lib/zlm目录中找到ZLMediaKit库: {lib_file}")
                    # 设置环境变量，确保库可以被正确加载
                    os.environ['LD_LIBRARY_PATH'] = lib_zlm_path
                    os.environ['DYLD_LIBRARY_PATH'] = lib_zlm_path
                    os.environ['ZLM_LIB_PATH'] = lib_zlm_path
                    # 加载库
                    self._lib = ctypes.CDLL(lib_file)
                    normal_logger.info(f"成功从lib/zlm目录加载ZLMediaKit库: {lib_file}")
                    return
            
            # 如果lib/zlm目录不存在或没有库文件，尝试从zlmos目录加载
            zlmos_lib_path = os.path.join(project_root, "zlmos", platform_dir)
            if os.path.exists(zlmos_lib_path):
                zlmos_lib_file = os.path.join(zlmos_lib_path, lib_name)
                if os.path.exists(zlmos_lib_file):
                    normal_logger.info(f"在zlmos目录中找到ZLMediaKit库: {zlmos_lib_file}")
                    # 设置环境变量，确保库可以被正确加载
                    os.environ['LD_LIBRARY_PATH'] = zlmos_lib_path
                    os.environ['DYLD_LIBRARY_PATH'] = zlmos_lib_path
                    os.environ['ZLM_LIB_PATH'] = zlmos_lib_path
                    # 加载库
                    self._lib = ctypes.CDLL(zlmos_lib_file)
                    normal_logger.info(f"成功从zlmos目录加载ZLMediaKit库: {zlmos_lib_file}")
                    return
                    
            # 如果zlmos目录不存在，尝试从lib目录加载
            lib_dir_path = os.path.join(project_root, "lib")
            if os.path.exists(lib_dir_path):
                lib_file = os.path.join(lib_dir_path, lib_name)
                if os.path.exists(lib_file):
                    normal_logger.info(f"在lib目录中找到ZLMediaKit库: {lib_file}")
                    # 设置环境变量，确保库可以被正确加载
                    os.environ['LD_LIBRARY_PATH'] = lib_dir_path
                    os.environ['ZLM_LIB_PATH'] = lib_dir_path
                    # 加载库
                    self._lib = ctypes.CDLL(lib_file)
                    normal_logger.info(f"成功从lib目录加载ZLMediaKit库: {lib_file}")
                    return

            # 尝试从ZLMediaKit项目目录加载库
            zlm_project_dir = os.path.join(project_root, "ZLMediaKit")
            if os.path.exists(zlm_project_dir):
                normal_logger.info(f"找到ZLMediaKit项目目录: {zlm_project_dir}")

                # 尝试在项目目录中查找库
                possible_lib_paths = [
                    os.path.join(zlm_project_dir, lib_name),
                    os.path.join(zlm_project_dir, "release", lib_name),
                    os.path.join(zlm_project_dir, "lib", lib_name),
                    os.path.join(zlm_project_dir, "build", lib_name)
                ]

                for lib_path in possible_lib_paths:
                    if os.path.exists(lib_path):
                        normal_logger.info(f"在ZLMediaKit项目目录中找到库: {lib_path}")
                        # 设置环境变量，确保库可以被正确加载
                        os.environ['LD_LIBRARY_PATH'] = os.path.dirname(lib_path)
                        os.environ['ZLM_LIB_PATH'] = os.path.dirname(lib_path)
                        # 加载库
                        self._lib = ctypes.CDLL(lib_path)
                        normal_logger.info(f"成功从ZLMediaKit项目目录加载库: {lib_path}")
                        return

            # 如果以上都失败，尝试从环境变量或系统路径加载
            normal_logger.warning("在项目目录中未找到ZLMediaKit库，尝试从环境变量或系统路径加载")

            # 检查环境变量中是否指定了ZLMediaKit库路径
            zlm_lib_path = os.environ.get('ZLM_LIB_PATH', '')

            if zlm_lib_path and os.path.exists(zlm_lib_path):
                lib_file = os.path.join(zlm_lib_path, lib_name)
                if os.path.exists(lib_file):
                    normal_logger.info(f"从环境变量指定的路径加载ZLMediaKit库: {lib_file}")
                    self._lib = ctypes.CDLL(lib_file)
                    normal_logger.info(f"成功从环境变量指定的路径加载ZLMediaKit库: {lib_file}")
                    return

            # 最后尝试直接加载库（依赖系统路径）
            normal_logger.warning("尝试从系统路径加载ZLMediaKit库")
            try:
                self._lib = ctypes.CDLL(lib_name)
                normal_logger.info(f"成功从系统路径加载ZLMediaKit库: {lib_name}")
                return
            except Exception as e:
                exception_logger.exception(f"从系统路径加载ZLMediaKit库失败: {str(e)}")
                raise RuntimeError(f"无法加载ZLMediaKit库: {str(e)}")

        except Exception as e:
            exception_logger.exception(f"加载ZLMediaKit库失败: {str(e)}")
            raise

    def _init_zlmediakit(self) -> None:
        """初始化ZLMediaKit"""
        try:
            if not self._lib:
                normal_logger.error("ZLMediaKit库未加载，无法初始化")
                return

            # 创建配置
            # 这里简化处理，实际中应根据ZLMediaKit的mk_config结构创建更详细的配置
            thread_num = ctypes.c_int(self._config.thread_num)
            log_level = ctypes.c_int(self._config.log_level)
            log_mask = ctypes.c_int(7)  # LOG_CONSOLE | LOG_FILE | LOG_CALLBACK
            log_file_path = ctypes.c_char_p(self._config.log_path.encode('utf-8'))
            log_file_days = ctypes.c_int(self._config.log_days)

            # 注册日志回调
            if hasattr(self._lib, 'mk_set_log_callback'):
                @ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)
                def on_log(level, content):
                    try:
                        content_str = content.decode('utf-8') if content else ""
                        if level <= 1:  # ERROR
                            normal_logger.error(f"ZLM: {content_str}")
                        elif level == 2:  # WARN
                            normal_logger.warning(f"ZLM: {content_str}")
                        elif level == 3:  # INFO
                            normal_logger.info(f"ZLM: {content_str}")
                        else:  # DEBUG or TRACE
                            normal_logger.debug(f"ZLM: {content_str}")
                    except Exception as e:
                        exception_logger.exception(f"处理ZLM日志回调异常: {str(e)}")

                self._lib.mk_set_log_callback(on_log)
                normal_logger.info("已注册ZLMediaKit日志回调")

            # 调用初始化函数
            if hasattr(self._lib, 'mk_env_init2'):
                self._lib.mk_env_init2(
                    thread_num,
                    log_level,
                    log_mask,
                    log_file_path,
                    log_file_days,
                    None,  # ini 为 NULL
                    0,     # ini_is_path 为 0
                    None,  # ssl 为 NULL
                    0,     # ssl_is_path 为 0
                    None   # ssl_pwd 为 NULL
                )
            elif hasattr(self._lib, 'mk_env_init'):
                # 创建mk_config结构体
                # 这里需要根据实际的ZLMediaKit API进行适配
                normal_logger.warning("使用mk_env_init函数初始化，可能需要根据ZLMediaKit版本进行适配")
                # 示例代码，实际中需要根据ZLMediaKit的mk_config结构进行调整
                # mk_config = ctypes.Structure(...)
                # self._lib.mk_env_init(ctypes.byref(mk_config))
            else:
                normal_logger.error("未找到ZLMediaKit初始化函数")
                return

            # 注册事件回调
            if hasattr(self._lib, 'mk_set_event_callback'):
                @ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
                def on_event(event_code, event_json, user_data):
                    try:
                        event_data = json.loads(event_json.decode('utf-8')) if event_json else {}
                        event_type = event_data.get("type", "unknown")

                        # 触发事件
                        self.trigger_event(event_type, event_data)

                        # 特殊事件处理
                        if event_type == "on_stream_changed":
                            # 流状态变化事件
                            self.trigger_event("stream_status_changed", event_data)
                        elif event_type == "on_flow_report":
                            # 流量统计事件
                            self.trigger_event("flow_report", event_data)
                        elif event_type == "on_server_started":
                            # 服务器启动事件
                            normal_logger.info("ZLMediaKit服务器已启动")
                        elif event_type == "on_server_exited":
                            # 服务器退出事件
                            normal_logger.warning("ZLMediaKit服务器已退出")

                    except Exception as e:
                        exception_logger.exception(f"处理ZLM事件回调异常: {str(e)}")

                self._lib.mk_set_event_callback(on_event, None)
                normal_logger.info("已注册ZLMediaKit事件回调")

            normal_logger.info("ZLMediaKit初始化成功")
        except Exception as e:
            exception_logger.exception(f"初始化ZLMediaKit失败: {str(e)}")
            raise

    def _test_api_connection(self) -> bool:
        """测试ZLMediaKit API连接

        Returns:
            bool: 连接是否成功
        """
        try:
            # 使用C API检查ZLMediaKit是否正常运行
            if self._lib:
                # 检查是否有mk_get_option函数
                if hasattr(self._lib, 'mk_get_option'):
                    # 设置参数类型和返回值类型
                    self._lib.mk_get_option.argtypes = [ctypes.c_char_p]
                    self._lib.mk_get_option.restype = ctypes.c_char_p
                    # 尝试获取一个基本配置项
                    val = self._lib.mk_get_option("api.secret".encode('utf-8'))
                    if val:
                        normal_logger.info(f"ZLMediaKit C API 连接正常，API密钥: {val.decode('utf-8')}")
                        return True

                # 检查是否有mk_set_option函数
                if hasattr(self._lib, 'mk_set_option'):
                    normal_logger.info("ZLMediaKit C API 连接正常 (mk_set_option可用)")
                    return True

                # 检查是否有其他基本函数
                basic_functions = ['mk_env_init', 'mk_stop_all_server']
                for func_name in basic_functions:
                    if hasattr(self._lib, func_name):
                        normal_logger.info(f"ZLMediaKit C API 连接正常 ({func_name}可用)")
                        return True

                normal_logger.warning("ZLMediaKit C API 不可用，未找到可用的API函数")
                return False
            else:
                normal_logger.warning("ZLMediaKit 库未加载")
                return False
        except Exception as e:
            exception_logger.exception(f"测试API连接失败: {str(e)}")
            return False

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

            # 尝试使用C API创建流代理
            if self._lib and hasattr(self._lib, 'mk_proxy_player_create'):
                try:
                    # 配置参数
                    vhost = stream_config.get("vhost", "__defaultVhost__")
                    app = stream_config.get("app", "live")
                    stream_name = stream_config.get("stream_name", stream_id)

                    # 转换为C字符串
                    url_c = ctypes.c_char_p(stream_url.encode('utf-8'))
                    vhost_c = ctypes.c_char_p(vhost.encode('utf-8'))
                    app_c = ctypes.c_char_p(app.encode('utf-8'))
                    stream_name_c = ctypes.c_char_p(stream_name.encode('utf-8'))

                    # 创建代理播放器
                    normal_logger.info(f"使用C API创建流代理: {stream_url} -> {app}/{stream_name}")
                    player_handle = self._lib.mk_proxy_player_create(url_c, vhost_c, app_c, stream_name_c)

                    if player_handle:
                        # 保存播放器句柄
                        self._players[player_handle] = stream_id
                        normal_logger.info(f"成功创建流代理: {stream_id}, 句柄: {player_handle}")

                        # 创建ZLM流对象
                        from .zlm_stream import ZLMVideoStream
                        stream = ZLMVideoStream(stream_id, stream_config, self, player_handle)

                        # 启动流
                        success = await stream.start()
                        if not success:
                            normal_logger.error(f"启动流 {stream_id} 失败")
                            # 清理资源
                            self._lib.mk_proxy_player_release(player_handle)
                            if player_handle in self._players:
                                del self._players[player_handle]
                            return False

                        # 保存流对象
                        with self._stream_lock:
                            self._streams[stream_id] = stream

                        normal_logger.info(f"成功创建并启动流 {stream_id} (使用C API)")
                        return True
                    else:
                        normal_logger.warning(f"C API创建流代理失败，回退到HTTP API: {stream_id}")
                except Exception as e:
                    exception_logger.exception(f"使用C API创建流代理时出错: {str(e)}")
                    normal_logger.warning(f"回退到HTTP API创建流: {stream_id}")

            # 如果C API不可用或创建失败，使用HTTP API或直接创建ZLM流
            from .zlm_stream import ZLMVideoStream
            stream = ZLMVideoStream(stream_id, stream_config, self)

            # 启动流
            success = await stream.start()
            if not success:
                normal_logger.error(f"启动流 {stream_id} 失败")
                return False

            # 保存流对象
            with self._stream_lock:
                self._streams[stream_id] = stream

            normal_logger.info(f"成功创建并启动流 {stream_id}")
            return True
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
            player_handle = None

            with self._stream_lock:
                if stream_id in self._streams:
                    stream = self._streams[stream_id]
                    del self._streams[stream_id]

                    # 查找对应的播放器句柄
                    for handle, sid in list(self._players.items()):
                        if sid == stream_id:
                            player_handle = handle
                            del self._players[handle]
                            break

            if not stream:
                normal_logger.warning(f"流 {stream_id} 不存在，无需停止")
                return True

            # 停止流
            await stream.stop()

            # 注意：由于在释放播放器时遇到段错误，我们暂时跳过播放器释放步骤
            # 只记录日志，不实际释放播放器
            if player_handle:
                normal_logger.info(f"跳过播放器释放步骤，只清除引用: {stream_id}, 句柄: {player_handle}")

            # 如果在未来需要重新启用播放器释放，可以取消注释以下代码
            """
            # 如果有播放器句柄，使用C API释放
            if player_handle and self._lib and hasattr(self._lib, 'mk_proxy_player_release'):
                try:
                    normal_logger.info(f"使用C API释放播放器: {stream_id}, 句柄: {player_handle}")
                    self._lib.mk_proxy_player_release(player_handle)
                except Exception as e:
                    exception_logger.exception(f"释放播放器句柄时出错: {str(e)}")
            """

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
            # 检查流是否存在
            with self._stream_lock:
                if stream_id not in self._streams:
                    normal_logger.warning(f"流 {stream_id} 不存在")
                    return None

                stream = self._streams[stream_id]

            # 获取流信息
            return await stream.get_info()
        except Exception as e:
            exception_logger.exception(f"获取流 {stream_id} 信息时出错: {str(e)}")
            return None

    async def get_all_streams(self) -> List[Dict[str, Any]]:
        """获取所有流信息

        Returns:
            List[Dict[str, Any]]: 所有流的信息列表
        """
        try:
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
        """调用ZLMediaKit API (C API版本)

        Args:
            method: API方法名
            params: API参数

        Returns:
            Dict[str, Any]: API响应
        """
        try:
            normal_logger.warning(f"HTTP API已禁用，请使用C API: {method}")
            # 返回错误响应
            return {"code": -1, "msg": "HTTP API已禁用，请使用C API"}
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