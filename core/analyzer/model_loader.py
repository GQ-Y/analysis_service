"""
模型加载器模块
负责加载YOLO模型
"""
import os
import aiohttp
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import torch

# 使用新的日志记录器
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class ModelLoader:
    """模型加载器基类"""

    @classmethod
    async def load_model(cls, model_code: str, device: str = "auto", **kwargs) -> Any:
        """
        加载模型

        Args:
            model_code: 模型代码
            device: 推理设备 ("cpu", "cuda", "auto")
            **kwargs: 其他参数

        Returns:
            Any: 加载的模型

        Raises:
            ValueError: 当引擎类型不支持时
        """
        # 获取模型路径
        model_path = await cls.get_model_path(model_code)
        
        # 使用PyTorch加载模型
        return await cls._load_pytorch_model(model_path, device, **kwargs)

    @classmethod
    async def get_model_path(cls, model_code: str) -> str:
        """
        获取模型路径

        Args:
            model_code: 模型代码

        Returns:
            str: 模型路径
        """
        try:
            # 检查本地缓存
            cache_dir = os.path.join("data", "models", model_code)

            # 检查多个可能的模型文件名
            possible_filenames = ["best.pt", "base.pt", "model.pt", "weights.pt", f"{model_code}.pt"]

            for filename in possible_filenames:
                model_path = os.path.join(cache_dir, filename)
                if os.path.exists(model_path):
                    normal_logger.info(f"找到本地缓存模型: {model_path}")
                    return model_path

            # 如果目录存在但没有找到特定文件，尝试查找任何.pt文件
            if os.path.exists(cache_dir):
                pt_files = [f for f in os.listdir(cache_dir) if f.endswith('.pt')]
                if pt_files:
                    model_path = os.path.join(cache_dir, pt_files[0])
                    normal_logger.info(f"找到本地缓存模型: {model_path}")
                    return model_path

            # 本地不存在，尝试使用默认模型
            return await cls._find_default_model(model_code)

        except Exception as e:
            # 使用异常日志记录器
            exception_logger.error(f"获取模型路径时出错: {str(e)}")
            # 尝试使用默认模型
            return await cls._find_default_model(model_code)

    @classmethod
    async def _find_default_model(cls, model_code: str) -> str:
        """
        查找默认模型

        Args:
            model_code: 模型代码

        Returns:
            str: 默认模型路径

        Raises:
            Exception: 当找不到默认模型时
        """
        # 检查通用模型目录
        models_dir = os.path.join("data", "models")

        # 尝试在yolo目录中查找任何模型
        yolo_dir = os.path.join(models_dir, "yolo")
        if os.path.exists(yolo_dir):
            pt_files = [f for f in os.listdir(yolo_dir) if f.endswith('.pt')]
            if pt_files:
                model_path = os.path.join(yolo_dir, pt_files[0])
                normal_logger.warning(f"未找到{model_code}模型，使用默认YOLO模型: {model_path}")
                return model_path

        # 搜索data/models目录下的任何.pt文件
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith('.pt'):
                    model_path = os.path.join(root, file)
                    normal_logger.warning(f"未找到特定模型，使用找到的模型: {model_path}")
                    return model_path

        # 如果找不到任何模型，抛出异常
        exception_logger.error(f"无法找到任何可用的模型，请确保至少有一个模型文件在data/models目录下")
        raise Exception(f"无法找到任何可用的模型，请确保至少有一个模型文件在data/models目录下")

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

            normal_logger.info(f"正在加载PyTorch模型: {model_path}")

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

            normal_logger.info(f"PyTorch模型加载成功: {model_path}")
            return model

        except Exception as e:
            exception_logger.error(f"加载PyTorch模型失败: {str(e)}")
            raise Exception(f"加载PyTorch模型失败: {str(e)}")
