"""
流工厂模块
根据不同的协议类型创建对应的流实例
"""

import importlib
from typing import Dict, Any, Type, Optional

from ..base.stream_interface import IStream, IStreamFactory

class StreamFactory(IStreamFactory):
    """流工厂类，负责创建不同协议的流实例"""
    
    _instance = None
    _lock = None
    
    def __new__(cls):
        """单例模式"""
        if cls._lock is None:
            import threading
            cls._lock = threading.Lock()
            
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StreamFactory, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化流工厂"""
        if getattr(self, '_initialized', False):
            return
            
        # 协议处理器映射，格式: 协议名 -> 处理器类
        self._protocol_handlers: Dict[str, Type[IStream]] = {}
        
        # 注册内置协议处理器
        self._register_built_in_handlers()
        
        self._initialized = True
    
    def _register_built_in_handlers(self) -> None:
        """注册内置协议处理器"""
        # 这里不直接导入各协议处理器类，以避免循环导入
        # 将在首次使用时动态导入
        
        # 注册协议名到模块路径的映射
        self._protocol_modules = {
            'rtsp': 'core.media_kit.protocols.rtsp.handler',
            'rtmp': 'core.media_kit.protocols.rtmp.handler',
            'hls': 'core.media_kit.protocols.hls.handler',
            'http': 'core.media_kit.protocols.http.handler',
            'webrtc': 'core.media_kit.protocols.webrtc.handler',
            'gb28181': 'core.media_kit.protocols.gb28181.handler',
            'onvif': 'core.media_kit.protocols.onvif.handler'
        }
        
        self._protocol_classes = {
            'rtsp': 'RtspStream',
            'rtmp': 'RtmpStream',
            'hls': 'HlsStream',
            'http': 'HttpStream',
            'webrtc': 'WebRTCStream',
            'gb28181': 'Gb28181Stream',
            'onvif': 'OnvifStream'
        }
    
    def register_protocol_handler(self, protocol: str, handler_class: Type[IStream]) -> None:
        """注册协议处理器
        
        Args:
            protocol: 协议名，如'rtsp'、'rtmp'等
            handler_class: 处理器类，必须实现IStream接口
        """
        self._protocol_handlers[protocol.lower()] = handler_class
    
    def create_stream(self, stream_config: Dict[str, Any]) -> IStream:
        """创建流实例
        
        Args:
            stream_config: 流配置，必须包含protocol和url字段
            
        Returns:
            IStream: 流实例
            
        Raises:
            ValueError: 如果协议不支持或配置无效
        """
        # 获取协议类型
        protocol = stream_config.get("protocol", "").lower()
        if not protocol:
            # 尝试从URL推断协议类型
            url = stream_config.get("url", "")
            protocol = self._infer_protocol_from_url(url)
            
            if not protocol:
                raise ValueError("无法确定协议类型，请在配置中指定protocol字段")
        
        # 检查协议处理器是否已注册
        if protocol in self._protocol_handlers:
            # 使用已注册的处理器创建流实例
            handler_class = self._protocol_handlers[protocol]
            return handler_class(stream_config)
        
        # 动态导入协议处理器
        if protocol in self._protocol_modules:
            try:
                module_path = self._protocol_modules[protocol]
                class_name = self._protocol_classes.get(protocol)
                
                # 导入模块
                module = importlib.import_module(module_path)
                
                # 获取处理器类
                if class_name and hasattr(module, class_name):
                    handler_class = getattr(module, class_name)
                    
                    # 注册处理器类
                    self.register_protocol_handler(protocol, handler_class)
                    
                    # 创建流实例
                    return handler_class(stream_config)
                else:
                    raise ValueError(f"协议 {protocol} 的处理器类 {class_name} 不存在")
            except (ImportError, AttributeError) as e:
                raise ValueError(f"导入协议 {protocol} 的处理器失败: {str(e)}")
        
        # 不支持的协议
        raise ValueError(f"不支持的协议类型: {protocol}")
    
    def _infer_protocol_from_url(self, url: str) -> Optional[str]:
        """从URL推断协议类型
        
        Args:
            url: 流URL
            
        Returns:
            Optional[str]: 推断的协议类型，如果无法推断则返回None
        """
        if not url:
            return None
        
        url = url.lower()
        
        # 检查URL前缀
        if url.startswith('rtsp://'):
            return 'rtsp'
        elif url.startswith('rtmp://'):
            return 'rtmp'
        elif url.startswith(('http://', 'https://')) and url.endswith('.m3u8'):
            return 'hls'
        elif url.startswith(('http://', 'https://')) and url.endswith(('.flv', '.ts', '.mp4', '.jpg', '.jpeg', '.png', '.gif')):
            return 'http'
        elif url.startswith('webrtc://'):
            return 'webrtc'
        elif url.startswith(('http://', 'https://')) and '/whep' in url:
            return 'webrtc'
        elif url.startswith(('http://', 'https://')) and '/whip' in url:
            return 'webrtc'
        elif url.startswith('srt://'):
            return 'srt'
        elif url.startswith('gb28181://'):
            return 'gb28181'
        elif url.startswith('onvif://'):
            return 'onvif'
        
        # 无法确定
        return None

# 单例实例
stream_factory = StreamFactory()
