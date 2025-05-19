"""
分析服务基类
提供图片、视频、流的分析能力的基础接口和共享功能
"""
import os
import sys
import time
import asyncio
import uuid
import json
import numpy as np
import cv2
import base64
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from abc import ABC, abstractmethod

# 添加父级目录到sys.path以允许导入core模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.config import settings
from core.analyzer.detection.yolo_detector import YOLODetector
from core.analyzer.model_loader import ModelLoader
from core.analyzer.analyzer_factory import analyzer_factory
from core.task_management.utils.status import TaskStatus
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class BaseAnalyzerService(ABC):
    """
    基础分析器服务接口
    所有分析器服务必须继承此类并实现相应方法
    """

    def __init__(self):
        """初始化基础分析器服务"""
        normal_logger.info("初始化基础分析服务")

        # 加载的检测器
        self.detectors = {}

        # 确保输出目录存在
        os.makedirs(settings.OUTPUT.save_dir, exist_ok=True)

        # 初始化任务处理器
        self.task_handlers = {}

        self.task_manager = None
        self.model_loader = ModelLoader()
        self.task_results = {}  # 存储任务结果

        normal_logger.info("基础分析服务初始化完成")

    def get_detector(self, model_code: str) -> YOLODetector:
        """
        获取检测器实例，如果不存在则创建

        Args:
            model_code: 模型代码

        Returns:
            YOLODetector: 检测器实例
        """
        if model_code not in self.detectors:
            try:
                normal_logger.info(f"创建检测器实例: {model_code}")
                self.detectors[model_code] = YOLODetector(model_code)
                normal_logger.info(f"检测器实例创建成功: {model_code}")
            except Exception as e:
                exception_logger.error(f"创建检测器实例失败: {model_code}, 错误: {str(e)}")
                raise ValueError(f"创建检测器实例失败: {model_code}, 错误: {str(e)}")

        return self.detectors[model_code]

    def _get_available_models(self) -> List[str]:
        """
        获取可用的模型列表

        Returns:
            List[str]: 可用模型代码列表
        """
        # 检查模型目录中的模型文件
        model_dir = settings.STORAGE.model_dir
        if not os.path.exists(model_dir):
            normal_logger.warning(f"模型目录不存在: {model_dir}")
            return ["yolov8n"]  # 返回默认模型

        # 获取模型目录中的所有.pt或.onnx文件
        model_files = []
        for file in os.listdir(model_dir):
            if file.endswith(".pt") or file.endswith(".onnx"):
                model_name = os.path.splitext(file)[0]
                model_files.append(model_name)

        if not model_files:
            normal_logger.warning(f"模型目录中未找到模型文件: {model_dir}")
            return ["yolov8n"]  # 返回默认模型

        normal_logger.debug(f"找到可用模型: {model_files}")
        return model_files

    def analyze_image(self, *args, **kwargs) -> Dict[str, Any]:
        """
        分析图像的基础方法

        在子类中实现具体逻辑
        """
        raise NotImplementedError("子类必须实现此方法")

    def start_video_analysis(self, *args, **kwargs) -> Dict[str, Any]:
        """
        启动视频分析的基础方法

        在子类中实现具体逻辑
        """
        raise NotImplementedError("子类必须实现此方法")

    def start_stream_analysis(self, *args, **kwargs) -> Dict[str, Any]:
        """
        启动流分析的基础方法

        在子类中实现具体逻辑
        """
        raise NotImplementedError("子类必须实现此方法")

    def stop_task(self, task_id: str) -> Dict[str, Any]:
        """
        停止任务的基础方法

        在子类中实现具体逻辑
        """
        raise NotImplementedError("子类必须实现此方法")

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态的基础方法

        在子类中实现具体逻辑
        """
        raise NotImplementedError("子类必须实现此方法")

    def get_all_tasks(self) -> Dict[str, Any]:
        """
        获取所有任务的基础方法

        在子类中实现具体逻辑
        """
        raise NotImplementedError("子类必须实现此方法")

    def _get_local_ip(self) -> str:
        """获取本地IP地址"""
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            normal_logger.warning(f"获取本地IP失败: {str(e)}")
            return "127.0.0.1"

    def _get_mac_address(self) -> str:
        """
        获取MAC地址

        Returns:
            str: MAC地址
        """
        try:
            mac = uuid.getnode()
            mac_str = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])
            return mac_str
        except Exception as e:
            exception_logger.error(f"获取MAC地址失败: {str(e)}")
            # 使用随机生成的MAC地址
            import random
            mac = [random.randint(0x00, 0xff) for _ in range(6)]
            mac_str = ':'.join(['{:02x}'.format(x) for x in mac])
            return mac_str

    def _log_all_network_interfaces(self):
        """记录所有网络接口信息"""
        try:
            import psutil
            import socket
            addresses = psutil.net_if_addrs()

            for interface_name, interface_addresses in addresses.items():
                for address in interface_addresses:
                    if address.family == socket.AF_INET:  # IPv4
                        normal_logger.info(f"接口: {interface_name}, IPv4: {address.address}")
                    elif address.family == psutil.AF_LINK:  # MAC地址
                        normal_logger.info(f"接口: {interface_name}, MAC: {address.address}")
        except Exception as e:
            exception_logger.error(f"记录网络接口信息失败: {str(e)}")

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """绘制检测结果"""
        result = image.copy()

        for det in detections:
            bbox = det["bbox"]
            conf = det["confidence"]
            label = f"{det['class_name']} {conf:.2f}"

            # 绘制边界框
            cv2.rectangle(
                result,
                (bbox["x1"], bbox["y1"]),
                (bbox["x2"], bbox["y2"]),
                (0, 255, 0),
                2
            )

            # 绘制标签
            cv2.putText(
                result,
                label,
                (bbox["x1"], bbox["y1"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return result

    def stop(self):
        """停止分析服务"""
        normal_logger.info("停止基础分析服务")
        normal_logger.info("基础分析服务已停止")

    # 添加一个辅助方法来同步处理检测器的模型加载
    def sync_load_model(self, detector, model_code: str):
        """
        同步方式加载模型的辅助方法，解决在非异步环境中调用异步方法的问题

        Args:
            detector: 检测器实例
            model_code: 模型代码

        Returns:
            bool: 加载是否成功
        """
        import asyncio

        try:
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 同步执行异步方法
            result = loop.run_until_complete(detector.load_model(model_code))

            # 关闭事件循环
            loop.close()

            return True
        except Exception as e:
            exception_logger.error(f"同步加载模型失败: {str(e)}")
            import traceback
            exception_logger.error(traceback.format_exc())
            return False

    # 同步处理检测
    def sync_detect(self, detector, image, config: Optional[Dict] = None):
        """
        同步方式执行检测的辅助方法

        Args:
            detector: 检测器实例
            image: 输入图像
            config: 检测配置

        Returns:
            List[Dict]: 检测结果
        """
        import asyncio

        try:
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 同步执行异步方法
            detections = loop.run_until_complete(detector.detect(image, config=config))

            # 关闭事件循环
            loop.close()

            return detections
        except Exception as e:
            exception_logger.error(f"同步执行检测失败: {str(e)}")
            import traceback
            exception_logger.error(traceback.format_exc())
            return []

    def register_task_handler(self, task_type: str, handler: callable):
        """
        注册任务处理器

        Args:
            task_type: 任务类型
            handler: 处理函数
        """
        self.task_handlers[task_type] = handler
        normal_logger.info(f"注册任务处理器: {task_type}")

    async def process_task(self, task_type: str, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理任务

        Args:
            task_type: 任务类型
            task_id: 任务ID
            task_data: 任务数据

        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 获取任务处理器
            handler = self.task_handlers.get(task_type)
            if not handler:
                raise ValueError(f"未找到任务处理器: {task_type}")

            # 执行任务处理
            normal_logger.info(f"开始处理任务: {task_id}, 类型: {task_type}")
            result = await handler(task_id, task_data)
            normal_logger.info(f"任务处理完成: {task_id}")

            return result

        except Exception as e:
            exception_logger.error(f"处理任务失败: {task_id}, 错误: {str(e)}")
            raise

    async def connect(self) -> bool:
        """
        连接到服务

        Returns:
            bool: 是否连接成功
        """
        return True

    async def disconnect(self) -> bool:
        """
        断开服务连接

        Returns:
            bool: 是否断开成功
        """
        return True

    def get_service_info(self) -> Dict[str, Any]:
        """
        获取服务信息

        Returns:
            Dict[str, Any]: 服务信息
        """
        return {
            "name": "base_analyzer",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT
        }

    @abstractmethod
    async def start(self):
        """启动服务"""
        pass

    @abstractmethod
    async def process_image(self, image_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理图像

        Args:
            image_path: 图像路径
            params: 处理参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        pass

    @abstractmethod
    async def process_video(self, video_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理视频

        Args:
            video_path: 视频路径
            params: 处理参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        pass

    @abstractmethod
    async def process_stream(self, stream_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理视频流

        Args:
            stream_url: 视频流URL
            params: 处理参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """
        获取服务状态

        Returns:
            Dict[str, Any]: 服务状态
        """
        pass

    async def initialize(self) -> bool:
        """初始化服务
        
        Returns:
            bool: 是否初始化成功
        """
        try:
            normal_logger.info("初始化分析服务")
            
            # 加载模型
            await self.model_loader.load_models()
            
            normal_logger.info("分析服务初始化成功")
            return True
        except Exception as e:
            exception_logger.exception(f"初始化分析服务失败: {str(e)}")
            return False

    async def analyze_video(self, video_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """分析视频
        
        这是一个需要被子类实现的方法
        
        Args:
            video_path: 视频文件路径
            params: 分析参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        raise NotImplementedError("子类必须实现analyze_video方法")

    async def start_stream_analysis(self, stream_url: str, params: Dict[str, Any]) -> str:
        """开始流分析
        
        这是一个需要被子类实现的方法
        
        Args:
            stream_url: 流URL
            params: 分析参数
            
        Returns:
            str: 任务ID
        """
        raise NotImplementedError("子类必须实现start_stream_analysis方法")

    async def stop_stream_analysis(self, task_id: str) -> bool:
        """停止流分析
        
        这是一个需要被子类实现的方法
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否停止成功
        """
        raise NotImplementedError("子类必须实现stop_stream_analysis方法")

    async def get_stream_analysis_result(self, task_id: str) -> Dict[str, Any]:
        """获取流分析结果
        
        这是一个需要被子类实现的方法
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        raise NotImplementedError("子类必须实现get_stream_analysis_result方法")

    async def shutdown(self) -> bool:
        """关闭服务
        
        Returns:
            bool: 是否关闭成功
        """
        try:
            normal_logger.info("关闭分析服务")
            return True
        except Exception as e:
            exception_logger.exception(f"关闭分析服务失败: {str(e)}")
            return False

    def _create_analyzer(self, analyzer_type: str, params: Dict[str, Any]) -> Any:
        """创建分析器
        
        Args:
            analyzer_type: 分析器类型
            params: 分析器参数
            
        Returns:
            Any: 分析器实例
        """
        try:
            analyzer = analyzer_factory.create_analyzer(analyzer_type, **params)
            if analyzer:
                normal_logger.info(f"创建分析器成功: {analyzer_type}")
                return analyzer
            else:
                exception_logger.error(f"创建分析器失败: {analyzer_type}")
                return None
        except Exception as e:
            exception_logger.exception(f"创建分析器异常: {str(e)}")
            return None

    async def _load_image(self, image_path: str) -> Tuple[bool, Optional[np.ndarray]]:
        """加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (是否成功, 图像数据)
        """
        try:
            if not os.path.exists(image_path):
                exception_logger.error(f"图像文件不存在: {image_path}")
                return False, None
            
            image = cv2.imread(image_path)
            if image is None:
                exception_logger.error(f"无法读取图像: {image_path}")
                return False, None
            
            return True, image
        except Exception as e:
            exception_logger.exception(f"加载图像失败: {str(e)}")
            return False, None

    def _encode_image(self, image: np.ndarray, format: str = ".jpg", quality: int = 95) -> Optional[str]:
        """编码图像为Base64字符串
        
        Args:
            image: 图像数据
            format: 图像格式
            quality: 图像质量
            
        Returns:
            Optional[str]: Base64编码的图像
        """
        try:
            if image is None:
                return None
            
            # 编码参数
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            
            # 编码图像
            success, buffer = cv2.imencode(format, image, encode_params)
            if not success:
                exception_logger.error("编码图像失败")
                return None
            
            # 转换为Base64
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            return encoded_image
        except Exception as e:
            exception_logger.exception(f"编码图像失败: {str(e)}")
            return None