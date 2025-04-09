"""
分析服务基类
提供图片、视频、流的分析能力的基础接口和共享功能
"""
import os
import sys
import logging
import asyncio
import uuid
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Union

# 添加父级目录到sys.path以允许导入core模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.config import settings
from core.detector import YOLODetector
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseAnalyzerService:
    """分析服务基类，提供基础接口和共享功能"""
    
    def __init__(self):
        """初始化基础分析服务"""
        logger.info("初始化基础分析服务")
        
        # 加载的检测器
        self.detectors = {}
        
        # 确保输出目录存在
        self._ensure_output_dirs()
        
        logger.info("基础分析服务初始化完成")
    
    def _ensure_output_dirs(self):
        """确保输出目录存在"""
        os.makedirs(settings.OUTPUT.save_dir, exist_ok=True)
        os.makedirs(f"{settings.OUTPUT.save_dir}/images", exist_ok=True)
        os.makedirs(f"{settings.OUTPUT.save_dir}/videos", exist_ok=True)
        os.makedirs(f"{settings.OUTPUT.save_dir}/streams", exist_ok=True)
        logger.debug("输出目录已确保存在")
    
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
                logger.info(f"创建检测器实例: {model_code}")
                self.detectors[model_code] = YOLODetector(model_code)
                logger.info(f"检测器实例创建成功: {model_code}")
            except Exception as e:
                logger.error(f"创建检测器实例失败: {model_code}, 错误: {str(e)}")
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
            logger.warning(f"模型目录不存在: {model_dir}")
            return ["yolov8n"]  # 返回默认模型
        
        # 获取模型目录中的所有.pt或.onnx文件
        model_files = []
        for file in os.listdir(model_dir):
            if file.endswith(".pt") or file.endswith(".onnx"):
                model_name = os.path.splitext(file)[0]
                model_files.append(model_name)
        
        if not model_files:
            logger.warning(f"模型目录中未找到模型文件: {model_dir}")
            return ["yolov8n"]  # 返回默认模型
        
        logger.debug(f"找到可用模型: {model_files}")
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
            logger.warning(f"获取本地IP失败: {str(e)}")
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
            logger.error(f"获取MAC地址失败: {str(e)}")
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
                        logger.info(f"接口: {interface_name}, IPv4: {address.address}")
                    elif address.family == psutil.AF_LINK:  # MAC地址
                        logger.info(f"接口: {interface_name}, MAC: {address.address}")
        except Exception as e:
            logger.error(f"记录网络接口信息失败: {str(e)}")
            
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
        logger.info("停止基础分析服务")
        logger.info("基础分析服务已停止")
        
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
            logger.error(f"同步加载模型失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
            logger.error(f"同步执行检测失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return [] 