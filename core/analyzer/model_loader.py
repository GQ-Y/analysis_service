"""
模型加载器模块
负责加载不同版本和类型的模型
"""
import os
import aiohttp
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import torch

from shared.utils.logger import setup_logger
from core.config import settings

logger = setup_logger(__name__)

class ModelLoader:
    """模型加载器基类"""
    
    # 引擎类型名称映射
    ENGINE_TYPE_NAME_MAP = {
        0: "PyTorch",
        1: "ONNX",
        2: "TensorRT",
        3: "OpenVINO",
        4: "Pytron"
    }
    
    # YOLO版本名称映射
    YOLO_VERSION_NAME_MAP = {
        0: "YOLOv8n",
        1: "YOLOv8s",
        2: "YOLOv8l",
        3: "YOLOv8x",
        4: "YOLO11s",
        5: "YOLO11m",
        6: "YOLO11l"
    }
    
    # YOLO版本文件名映射
    YOLO_VERSION_FILENAME_MAP = {
        0: "yolov8n",
        1: "yolov8s",
        2: "yolov8l",
        3: "yolov8x",
        4: "yolo11s",
        5: "yolo11m",
        6: "yolo11l"
    }
    
    @classmethod
    async def load_model(cls, model_code: str, engine_type: int = 0, 
                        yolo_version: int = 0, device: str = "auto", **kwargs) -> Any:
        """
        加载模型
        
        Args:
            model_code: 模型代码
            engine_type: 推理引擎类型 (0=PyTorch, 1=ONNX, 2=TensorRT, 3=OpenVINO, 4=Pytron)
            yolo_version: YOLO版本 (0=v8n, 1=v8s, 2=v8l, 3=v8x, 4=11s, 5=11m, 6=11l)
            device: 推理设备 ("cpu", "cuda", "auto")
            **kwargs: 其他参数
            
        Returns:
            Any: 加载的模型
            
        Raises:
            ValueError: 当引擎类型不支持时
        """
        # 获取模型路径
        model_path = await cls.get_model_path(model_code, yolo_version)
        
        # 根据引擎类型选择加载方法
        if engine_type == 0:  # PyTorch
            return await cls._load_pytorch_model(model_path, device, **kwargs)
        elif engine_type == 1:  # ONNX
            return await cls._load_onnx_model(model_path, device, **kwargs)
        elif engine_type == 2:  # TensorRT
            return await cls._load_tensorrt_model(model_path, device, **kwargs)
        elif engine_type == 3:  # OpenVINO
            return await cls._load_openvino_model(model_path, device, **kwargs)
        elif engine_type == 4:  # Pytron
            return await cls._load_pytron_model(model_path, device, **kwargs)
        else:
            raise ValueError(f"不支持的引擎类型: {engine_type}")
    
    @classmethod
    async def get_model_path(cls, model_code: str, yolo_version: int = 0) -> str:
        """
        获取模型路径
        
        Args:
            model_code: 模型代码
            yolo_version: YOLO版本
            
        Returns:
            str: 模型路径
        """
        try:
            # 检查本地缓存
            cache_dir = os.path.join("data", "models", model_code)
            model_path = os.path.join(cache_dir, "best.pt")
            
            if os.path.exists(model_path):
                logger.info(f"找到本地缓存模型: {model_path}")
                return model_path
            
            # 本地不存在，从模型服务下载
            logger.info(f"本地未找到模型 {model_code}，准备从模型服务下载...")
            
            # 构建API URL
            model_service_url = settings.MODEL_SERVICE.url
            api_prefix = settings.MODEL_SERVICE.api_prefix
            api_url = f"{model_service_url}{api_prefix}/models/download?code={model_code}"
            
            logger.info(f"开始从模型服务下载: {api_url}")
            
            # 创建缓存目录
            os.makedirs(cache_dir, exist_ok=True)
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(api_url) as response:
                        if response.status == 200:
                            # 保存模型文件
                            with open(model_path, "wb") as f:
                                f.write(await response.read())
                            logger.info(f"模型下载成功并保存到: {model_path}")
                            return model_path
                        else:
                            error_msg = await response.text()
                            raise Exception(f"模型下载失败: HTTP {response.status} - {error_msg}")
            
            except aiohttp.ClientError as e:
                raise Exception(f"请求模型服务失败: {str(e)}")
        
        except Exception as e:
            logger.error(f"获取模型路径时出错: {str(e)}")
            raise Exception(f"获取模型失败: {str(e)}")
    
    @classmethod
    async def _load_pytorch_model(cls, model_path: str, device: str = "auto", **kwargs) -> Any:
        """
        加载PyTorch模型
        
        Args:
            model_path: 模型路径
            device: 推理设备
            **kwargs: 其他参数
            
        Returns:
            Any: 加载的PyTorch模型
        """
        try:
            from ultralytics import YOLO
            
            logger.info(f"正在加载PyTorch模型: {model_path}")
            
            # 加载模型
            model = YOLO(model_path)
            
            # 设置设备
            if device != "auto":
                model_device = device
            else:
                model_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model.to(model_device)
            
            # 设置模型参数
            if "confidence" in kwargs:
                model.conf = kwargs["confidence"]
            if "iou_threshold" in kwargs:
                model.iou = kwargs["iou_threshold"]
            if "max_detections" in kwargs:
                model.max_det = kwargs["max_detections"]
            
            logger.info(f"PyTorch模型加载成功: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"加载PyTorch模型失败: {str(e)}")
            raise Exception(f"加载PyTorch模型失败: {str(e)}")
    
    @classmethod
    async def _load_onnx_model(cls, model_path: str, device: str = "auto", **kwargs) -> Any:
        """
        加载ONNX模型
        
        Args:
            model_path: 模型路径
            device: 推理设备
            **kwargs: 其他参数
            
        Returns:
            Any: 加载的ONNX模型
        """
        try:
            import onnxruntime as ort
            
            logger.info(f"正在加载ONNX模型: {model_path}")
            
            # 设置设备
            if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            # 创建推理会话
            session = ort.InferenceSession(model_path, providers=providers)
            
            logger.info(f"ONNX模型加载成功: {model_path}")
            return session
            
        except Exception as e:
            logger.error(f"加载ONNX模型失败: {str(e)}")
            raise Exception(f"加载ONNX模型失败: {str(e)}")
    
    @classmethod
    async def _load_tensorrt_model(cls, model_path: str, device: str = "auto", **kwargs) -> Any:
        """
        加载TensorRT模型
        
        Args:
            model_path: 模型路径
            device: 推理设备
            **kwargs: 其他参数
            
        Returns:
            Any: 加载的TensorRT模型
        """
        # TODO: 实现TensorRT模型加载
        logger.warning("TensorRT模型加载尚未实现，使用PyTorch模型代替")
        return await cls._load_pytorch_model(model_path, device, **kwargs)
    
    @classmethod
    async def _load_openvino_model(cls, model_path: str, device: str = "auto", **kwargs) -> Any:
        """
        加载OpenVINO模型
        
        Args:
            model_path: 模型路径
            device: 推理设备
            **kwargs: 其他参数
            
        Returns:
            Any: 加载的OpenVINO模型
        """
        # TODO: 实现OpenVINO模型加载
        logger.warning("OpenVINO模型加载尚未实现，使用PyTorch模型代替")
        return await cls._load_pytorch_model(model_path, device, **kwargs)
    
    @classmethod
    async def _load_pytron_model(cls, model_path: str, device: str = "auto", **kwargs) -> Any:
        """
        加载Pytron模型
        
        Args:
            model_path: 模型路径
            device: 推理设备
            **kwargs: 其他参数
            
        Returns:
            Any: 加载的Pytron模型
        """
        # TODO: 实现Pytron模型加载
        logger.warning("Pytron模型加载尚未实现，使用PyTorch模型代替")
        return await cls._load_pytorch_model(model_path, device, **kwargs)
