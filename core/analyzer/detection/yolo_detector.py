"""
YOLO检测器模块 - 重构版
基于Ultralytics YOLO的底层检测实现
"""
import os
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import base64
import cv2
from io import BytesIO

from shared.utils.logger import get_normal_logger, get_exception_logger
from core.analyzer.model_loader import ModelLoader

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class YOLODetector:
    """YOLO检测器 - 重构版，使用更灵活的参数传递"""

    def __init__(self, model_code: Optional[str] = None, device: str = "auto", **kwargs):
        """
        初始化YOLO检测器
        
        Args:
            model_code: 模型代码，如果提供则立即加载模型
            device: 推理设备 ("cpu", "cuda", "auto")
            **kwargs: 其他参数，包括：
                - custom_weights_path: 自定义权重路径
                - half_precision: 是否使用半精度
                - confidence: 置信度阈值
                - iou_threshold: IoU阈值
                - max_detections: 最大检测目标数
                - classes: 类别列表
        """
        self.model = None
        self.current_model_code = None
        self.device = device
        self.custom_weights_path = kwargs.get("custom_weights_path")
        self.half_precision = kwargs.get("half_precision", False)
        self.kwargs = kwargs
        
        # 加载模型（如果提供了模型代码）
        if model_code:
            import asyncio
            try:
                # 尝试获取现有事件循环
                loop = asyncio.get_running_loop()
            except RuntimeError:  # 如果没有正在运行的事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            try:
                if loop.is_running():
                    asyncio.create_task(self.load_model(model_code))
                    normal_logger.info(f"已为模型 {model_code} 创建异步加载任务。")
                else:
                    normal_logger.info(f"同步加载模型 {model_code}...")
                    loop.run_until_complete(self.load_model(model_code))
                    normal_logger.info(f"模型 {model_code} 同步加载完成。")
            except Exception as e:
                exception_logger.exception(f"在初始化过程中加载模型 {model_code} 失败: {e}")

    async def load_model(self, model_code: str) -> bool:
        """
        加载模型
        
        Args:
            model_code: 模型代码
            
        Returns:
            bool: 是否成功加载模型
        """
        try:
            # 获取模型路径
            model_path = await self._get_model_path(model_code)
            
            # 判断模型文件是否存在
            if not os.path.exists(model_path):
                exception_logger.error(f"模型文件不存在: {model_path}")
                return False
            
            # 记录要加载的模型信息
            normal_logger.info(f"开始加载YOLO模型: {model_path}, 设备={self.device}")
            
            # 加载Ultralytics YOLO模型
            from ultralytics import YOLO
            
            # 加载模型
            self.model = YOLO(model_path)
            
            # 设置设备
            if self.device != "auto":
                self.model.to(self.device)
                
            # 设置半精度
            if self.half_precision:
                normal_logger.info("使用半精度(FP16)推理")
                self.model.half()
                
            # 更新当前模型代码
            self.current_model_code = model_code
            
            normal_logger.info(f"YOLO模型加载成功: {model_code}")
            return True
            
        except Exception as e:
            exception_logger.exception(f"加载YOLO模型失败: {e}")
            return False

    async def _get_model_path(self, model_code: str) -> str:
        """
        获取模型路径
        
        Args:
            model_code: 模型代码
            
        Returns:
            str: 模型路径
        """
        # 如果提供了自定义权重路径，优先使用
        if self.custom_weights_path:
            # 判断是否是URL
            if self.custom_weights_path.startswith(("http://", "https://", "s3://", "oss://")):
                # TODO: 实现URL下载
                normal_logger.warning(f"暂不支持从URL加载模型: {self.custom_weights_path}")
                
            # 判断文件是否存在
            if os.path.exists(self.custom_weights_path):
                normal_logger.info(f"使用自定义权重路径: {self.custom_weights_path}")
                return self.custom_weights_path
                
            normal_logger.warning(f"自定义权重路径不存在，将使用默认路径: {self.custom_weights_path}")
            
        # 使用相对路径获取模型目录
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        model_base_path = os.path.join(project_root, "data", "models")
        normal_logger.info(f"使用模型基础路径: {model_base_path}")
        
        # 模型目录
        model_dir = os.path.join(model_base_path, model_code)
        
        # 检查模型文件是否存在 - 先尝试根目录
        model_path = os.path.join(model_dir, "best.pt")
        if os.path.exists(model_path):
            normal_logger.info(f"找到模型文件: {model_path}")
            return model_path
            
        # 检查是否有weights子目录
        weights_path = os.path.join(model_dir, "weights", "best.pt")
        if os.path.exists(weights_path):
            normal_logger.info(f"找到模型文件: {weights_path}")
            return weights_path
            
        # 尝试使用last.pt
        model_path = os.path.join(model_dir, "last.pt")
        if os.path.exists(model_path):
            normal_logger.info(f"找到模型文件: {model_path}")
            return model_path
            
        weights_path = os.path.join(model_dir, "weights", "last.pt")
        if os.path.exists(weights_path):
            normal_logger.info(f"找到模型文件: {weights_path}")
            return weights_path
            
        # 尝试使用yolov8n.pt
        model_path = os.path.join(model_dir, "yolov8n.pt")
        if os.path.exists(model_path):
            normal_logger.info(f"找到模型文件: {model_path}")
            return model_path
            
        # 尝试使用任何.pt文件
        for file in os.listdir(model_dir):
            if file.endswith(".pt"):
                model_path = os.path.join(model_dir, file)
                normal_logger.info(f"找到模型文件: {model_path}")
                return model_path
                
        # 如果找不到任何模型文件，报错
        normal_logger.error(f"模型目录中找不到任何.pt模型文件: {model_dir}")
        return os.path.join(model_dir, "best.pt")  # 返回一个默认路径，虽然它不存在

    async def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        检测图像中的目标
        
        Args:
            image: 输入图像
            **kwargs: 其他参数，包括：
                - confidence: 置信度阈值
                - iou_threshold: IoU阈值
                - max_detections: 最大检测目标数
                - classes: 类别列表
                - return_image: 是否返回标注后的图像
                
        Returns:
            Dict[str, Any]: 检测结果
        """
        if self.model is None:
            normal_logger.warning("YOLO检测器模型未加载，无法进行检测")
            return {"success": False, "error": "模型未加载", "detections": []}
            
        # 获取参数
        confidence = kwargs.get("confidence", 0.25)
        iou_threshold = kwargs.get("iou_threshold", 0.45)
        max_detections = kwargs.get("max_detections", 100)
        class_names_to_filter = kwargs.get("classes") # 这是字符串列表，如 ['person', 'car'] 或 []
        return_image = kwargs.get("return_image", False)
        
        class_indices_to_filter = None # 默认为 None，表示检测所有类别

        # 检查 class_names_to_filter 是否明确地是一个空列表，如果是，则意图是检测所有
        if class_names_to_filter == []: # 用户明确传入空列表
            normal_logger.info("请求中的 'classes' 为空列表，将检测所有类别。")
            class_indices_to_filter = None # 显式设置为None
        elif class_names_to_filter and self.model and hasattr(self.model, 'names') and self.model.names:
            # 处理非空的 class_names_to_filter
            model_class_names_list = []
            # self.model.names 可以是 list [name1, name2, ...] 或 dict {id1: name1, id2: name2, ...}
            if isinstance(self.model.names, list):
                model_class_names_list = self.model.names
            elif isinstance(self.model.names, dict):
                try:
                    max_idx = -1
                    valid_keys = [k for k in self.model.names.keys() if isinstance(k, int)]
                    if valid_keys:
                        max_idx = max(valid_keys)
                    
                    if max_idx != -1:
                        model_class_names_list = [""] * (max_idx + 1) 
                        for idx, name in self.model.names.items():
                            if isinstance(idx, int) and 0 <= idx <= max_idx:
                                model_class_names_list[idx] = name
                    else: 
                        normal_logger.warning(f"模型类别名称字典 (self.model.names) 键不符合预期: {self.model.names}，尝试直接从值构建查找。")
                        # 如果字典的键不是预期的整数索引，尝试直接使用值的列表进行查找，但顺序不保证
                        # model_class_names_list = list(self.model.names.values()) # 这可能不是最佳选择
                except Exception as e_names:
                    normal_logger.warning(f"处理 self.model.names (字典类型) 时出错: {e_names}。类别过滤可能不准确。")
            
            temp_indices_list = []
            processed_indices_set = set()

            if model_class_names_list: # 如果成功从列表转换 (或字典成功转为有序列表)
                for name_to_filter in class_names_to_filter:
                    try:
                        idx = model_class_names_list.index(name_to_filter)
                        if idx not in processed_indices_set:
                           temp_indices_list.append(idx)
                           processed_indices_set.add(idx)
                    except ValueError:
                        normal_logger.warning(f"类别 '{name_to_filter}' 在模型类别列表 {model_class_names_list} 中未找到，将被忽略。")
            elif isinstance(self.model.names, dict): # 如果是字典且列表转换不理想/未发生，尝试直接从字典查找
                name_to_index_map = {name: idx for idx, name in self.model.names.items() if isinstance(idx, int)} # 确保键是整数
                if not name_to_index_map and self.model.names: # 如果上面的构建失败（例如键不是整数），尝试更通用的反向查找
                     name_to_index_map = {v: k for k, v in self.model.names.items()}

                for name_to_filter in class_names_to_filter:
                    if name_to_filter in name_to_index_map:
                        idx = name_to_index_map[name_to_filter]
                        if isinstance(idx, int) and idx not in processed_indices_set: # 确保索引是整数
                           temp_indices_list.append(idx)
                           processed_indices_set.add(idx)
                        elif not isinstance(idx, int):
                            normal_logger.warning(f"类别 '{name_to_filter}' 对应的索引 '{idx}' 不是整数，将被忽略。")
                    else:
                        normal_logger.warning(f"类别 '{name_to_filter}' 在模型类别字典 {self.model.names} 中未找到，将被忽略。")
            
            if temp_indices_list: # 如果找到任何有效的类别索引
                class_indices_to_filter = temp_indices_list
                normal_logger.info(f"将过滤以下类别索引: {class_indices_to_filter} (来自名称: {class_names_to_filter})")
            else: # 如果没有找到任何有效类别索引 (class_names_to_filter 非空但所有名称都无效)
                normal_logger.warning(f"提供的所有类别名称 {class_names_to_filter} 均无效或未在模型中找到。将检测所有类别。")
                class_indices_to_filter = None # 设置为None以检测所有

        elif class_names_to_filter: # class_names_to_filter 非空，但模型信息不足 (self.model is None or self.model.names is empty/None)
             normal_logger.warning(f"提供了类别名称 {class_names_to_filter} 用于过滤，但无法访问模型类别信息。将检测所有类别。")
             class_indices_to_filter = None # 设置为None以检测所有
        # 如果 class_names_to_filter 原本就是 None (kwargs.get返回None), class_indices_to_filter 保持为 None

        # 记录时间
        start_time = time.time()
        
        # Log model names and the actual class indices being used for filtering
        if self.model and hasattr(self.model, 'names'):
            normal_logger.info(f"模型类别名称 (self.model.names): {self.model.names}")
        else:
            normal_logger.warning("无法访问 self.model.names")
        normal_logger.info(f"传递给 model.predict 的 confidence: {confidence}, iou: {iou_threshold}, max_det: {max_detections}")
        # 修改：始终传递 classes=None 给 predict，后续手动过滤
        normal_logger.info(f"内部调用 model.predict 时将传递 classes=None，agnostic_nms=True。原始请求的类别过滤将在之后应用。原始 class_indices_to_filter: {class_indices_to_filter}")

        normal_logger.info(f"开始执行YOLO检测，图像大小：{image.shape}，参数：conf={confidence}, iou={iou_threshold}, max_det={max_detections}")
        
        try:
            # 进行推理
            results = self.model.predict(
                image, 
                conf=confidence, 
                iou=iou_threshold,
                max_det=max_detections,
                classes=None,  # 修改：始终传递 None，避免内部 buggy 过滤
                agnostic_nms=True, # 保持类别无关的NMS
                verbose=False
            )
            
            # 处理结果
            result = results[0]  # 只处理第一个结果
            
            # 构建检测结果
            detections = []
            for i in range(len(result.boxes)):
                box = result.boxes[i]
                
                # 获取类别
                class_id = int(box.cls.item())
                
                # 手动应用类别过滤
                if class_indices_to_filter is not None and class_id not in class_indices_to_filter:
                    continue # 跳过不符合指定类别的检测

                # 获取置信度
                conf = float(box.conf.item())
                
                # 获取边界框 (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                
                # 获取类别名称
                class_name = result.names[class_id]
                
                # 添加检测结果
                detections.append({
                    "id": i,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "center_x": (x1 + x2) / 2,
                    "center_y": (y1 + y2) / 2,
                    "area": (x2 - x1) * (y2 - y1)
                })
                
            # 计算总时间
            total_time = time.time() - start_time
            
            normal_logger.info(f"YOLO检测完成，耗时：{total_time:.4f}秒，检测到{len(detections)}个目标")
            
            # 构建返回结果
            result_dict = {
                "success": True,
                "detections": detections,
                "stats": {
                    "total_time": total_time,
                    "objects_count": len(detections)
                }
            }
            
            # 如果需要返回标注图像
            if return_image:
                # 绘制结果
                plotted_img = results[0].plot()
                
                # 转换为base64
                _, buffer = cv2.imencode('.jpg', plotted_img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 添加到结果
                result_dict["annotated_image"] = img_base64
                
            return result_dict
            
        except Exception as e:
            exception_logger.exception(f"YOLO检测失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "detections": []
            }

    @property
    def model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        info = {
            "detector_type": "YOLODetector",
            "loaded": self.model is not None,
            "model_code": self.current_model_code,
            "device": self.device,
            "half_precision": self.half_precision,
            "custom_weights_path": self.custom_weights_path
        }
        
        # 添加模型信息
        if self.model:
            try:
                info.update({
                    "model_type": self.model.type,
                    "model_task": self.model.task,
                    "model_stride": int(self.model.stride),
                    "model_pt": bool(self.model.pt),
                    "model_names": self.model.names
                })
            except Exception as e:
                exception_logger.warning(f"获取模型详细信息失败: {e}")
                
        return info

    def release(self) -> None:
        """释放资源"""
        if self.model:
            try:
                # 释放GPU内存
                import torch
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # 移除模型引用
                self.model = None
                normal_logger.info("YOLO检测器资源已释放")
                
            except Exception as e:
                exception_logger.warning(f"释放YOLO检测器资源失败: {e}")