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
        self.device = self._auto_select_device(device)  # 自动选择设备
        self.custom_weights_path = kwargs.get("custom_weights_path")
        self.half_precision = kwargs.get("half_precision", False)
        self.kwargs = kwargs
        
        normal_logger.info(f"YOLO检测器设备自动选择结果: {self.device}")
        
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

    def _auto_select_device(self, device: str) -> str:
        """
        自动选择推理设备
        
        Args:
            device: 用户指定的设备
            
        Returns:
            str: 选择的设备 ("cpu" 或 "cuda")
        """
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    normal_logger.info("检测到CUDA可用，自动选择GPU设备")
                    return "cuda"
                else:
                    normal_logger.info("CUDA不可用，自动选择CPU设备")
                    return "cpu"
            except Exception as e:
                normal_logger.warning(f"设备检测失败，默认使用CPU: {e}")
                return "cpu"
        elif device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    normal_logger.info("用户指定CUDA设备，且CUDA可用")
                    return "cuda"
                else:
                    normal_logger.warning("用户指定CUDA设备，但CUDA不可用，强制使用CPU")
                    return "cpu"
            except Exception as e:
                normal_logger.warning(f"CUDA检测失败，强制使用CPU: {e}")
                return "cpu"
        else:
            # 用户指定了cpu或其他设备，直接返回
            return device

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
            
            # 自动选择设备
            actual_device = self._auto_select_device(self.device)
            
            # 记录要加载的模型信息
            normal_logger.info(f"开始加载YOLO模型: {model_path}, 设备={actual_device}")
            
            # 加载Ultralytics YOLO模型
            from ultralytics import YOLO
            
            # 加载模型
            self.model = YOLO(model_path)
            
            # 设置设备 - 使用自动选择的设备
            self.model.to(actual_device)
            
            # 设置半精度
            if self.half_precision and actual_device != "cpu":
                normal_logger.info("使用半精度(FP16)推理")
                self.model.half()
            elif self.half_precision and actual_device == "cpu":
                normal_logger.warning("CPU设备不支持半精度推理，跳过半精度设置")
                
            # 更新当前模型代码和实际设备
            self.current_model_code = model_code
            self.device = actual_device  # 更新为实际使用的设备
            
            normal_logger.info(f"YOLO模型加载成功: {model_code}, 实际使用设备: {actual_device}")
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
        检测图像中的目标, 返回更详细的结果结构。
        
        Args:
            image: 输入图像 (H, W, C)
            **kwargs: 其他参数，包括：
                - confidence: 置信度阈值
                - iou_threshold: IoU阈值
                - max_detections: 最大检测目标数
                - classes: 类别名称列表用于过滤
                - return_image: 是否返回标注后的图像 (base64)
                - imgsz: 推理图像尺寸 (例如 640 或 [640, 480])
                
        Returns:
            Dict[str, Any]: 检测结果, 结构如下:
            {
                "success": bool,
                "error": Optional[str],
                "detections": Optional[List[Dict]],
                "applied_config": {
                    "confidence_threshold": float,
                    "iou_threshold": float,
                    "max_detections": int,
                    "imgsz": Union[int, List[int]],
                    "filtered_classes": Optional[List[str]], // 实际用于过滤的类别名称
                    "half_precision": bool
                },
                "image_info": {
                    "original_height": int,
                    "original_width": int,
                    "processed_height": Optional[int], // 模型实际推理的图像高度
                    "processed_width": Optional[int]   // 模型实际推理的图像宽度
                },
                "timing_stats": { // 所有时间单位为毫秒
                    "analyzer_total_time_ms": float, 
                    "model_predict_time_ms": Optional[float], // ultralytics predict()耗时
                    "pre_processing_time_ms": Optional[float], 
                    "inference_time_ms": Optional[float], 
                    "post_processing_time_ms": Optional[float],
                    "nms_time_ms": Optional[float]
                },
                "annotated_image_base64": Optional[str]
            }
        """
        analyzer_start_time = time.perf_counter()

        if self.model is None:
            normal_logger.warning("YOLO检测器模型未加载，无法进行检测")
            return {
                "success": False, 
                "error": "模型未加载", 
                "detections": [],
                "applied_config": {},
                "image_info": {"original_height": image.shape[0], "original_width": image.shape[1]},
                "timing_stats": {"analyzer_total_time_ms": (time.perf_counter() - analyzer_start_time) * 1000},
                "annotated_image_base64": None
            }

        original_height, original_width = image.shape[:2]
            
        # 获取和记录应用的参数
        confidence = kwargs.get("confidence", 0.25)
        iou_threshold = kwargs.get("iou_threshold", 0.45)
        max_detections = kwargs.get("max_detections", 100)
        class_names_to_filter_input = kwargs.get("classes") 
        return_image_flag = kwargs.get("return_image", False)
        imgsz_arg = kwargs.get("imgsz", self.kwargs.get("image_size", {}).get("width", 640)) # 尝试从 kwargs 或初始化参数获取
        if isinstance(imgsz_arg, dict) and 'width' in imgsz_arg : # Handle cases where it might come as {'width': W, 'height': H}
            imgsz_arg = imgsz_arg['width'] # Default to width for single int value, model handles aspect ratio or specific H,W


        applied_config_log = {
            "confidence_threshold": confidence,
            "iou_threshold": iou_threshold,
            "max_detections": max_detections,
            "imgsz_arg": imgsz_arg, # Log the argument passed to predict
            "input_classes_filter": class_names_to_filter_input,
            "half_precision_enabled": self.half_precision
        }
        normal_logger.info(f"YOLO detect called with applied_config_log: {applied_config_log}")

        class_indices_to_filter = None
        actual_filtered_class_names = None

        if class_names_to_filter_input == []:
            normal_logger.info("请求中的 'classes' 为空列表，将检测所有类别。")
            class_indices_to_filter = None
        elif class_names_to_filter_input and self.model and hasattr(self.model, 'names') and self.model.names:
            model_class_names_map = self.model.names # 这是 {index: name} 格式
            actual_filtered_class_names = []
            temp_indices_list = []
            for name_to_filter in class_names_to_filter_input:
                found = False
                for index, model_class_name in model_class_names_map.items():
                    if model_class_name.lower() == name_to_filter.lower():
                        temp_indices_list.append(index)
                        actual_filtered_class_names.append(model_class_name) # Store the actual name from model
                        found = True
                        break
                if not found:
                    normal_logger.warning(f"请求过滤的类别 '{name_to_filter}' 在模型类别中未找到，将被忽略。")
            
            if temp_indices_list:
                class_indices_to_filter = sorted(list(set(temp_indices_list)))
                normal_logger.info(f"将过滤以下类别索引: {class_indices_to_filter} (来自名称: {actual_filtered_class_names})")
            else:
                # 如果提供的所有类别名称都无效，则不进行过滤 (或者根据需求可以报错或返回空)
                normal_logger.warning(f"提供的所有类别名称 {class_names_to_filter_input} 在模型中均未找到，实际不过滤任何类别。")
                class_indices_to_filter = None 
                actual_filtered_class_names = None # Reset if no valid classes found
        
        # 记录模型类别名称和最终传递给 predict 的过滤器
        if self.model and hasattr(self.model, 'names'):
            normal_logger.info(f"模型类别名称 (self.model.names): {self.model.names}")
        normal_logger.info(f"传递给 model.predict 的 class_indices_to_filter: {class_indices_to_filter} (如果为None则不过滤)")
        normal_logger.info(f"传递给 model.predict 的 confidence: {confidence}, iou: {iou_threshold}, max_det: {max_detections}, imgsz: {imgsz_arg}")
        
        detections_output = []
        annotated_image_b64 = None
        processed_img_h, processed_img_w = None, None # 模型实际推理尺寸

        timing_stats = {
            "analyzer_total_time_ms": 0.0, # Will be set at the end
            "model_predict_time_ms": None,
            "pre_processing_time_ms": None,
            "inference_time_ms": None,
            "post_processing_time_ms": None,
            "nms_time_ms": None
        }

        try:
            # --- 开始推理 ---
            predict_start_time = time.perf_counter()
            # Ultralytics YOLOv8 predict()
            # classes=None 表示不过滤，由后续手动完成
            # verbose=False 减少不必要的日志输出
            results = self.model.predict(
                source=image,
                conf=confidence,
                iou=iou_threshold,
                max_det=max_detections,
                classes=None, # 我们总是先获取所有结果，然后手动过滤
                agnostic_nms=True, # 使用类别无关的NMS，因为类别过滤是后置的
                imgsz=imgsz_arg,
                half=self.half_precision,
                verbose=False 
            )
            predict_end_time = time.perf_counter()
            timing_stats["model_predict_time_ms"] = (predict_end_time - predict_start_time) * 1000
            # --- 推理结束 ---

            # results 是一个列表，通常只包含一个结果对象 (对应单张输入图像)
            if results and len(results) > 0:
                result = results[0] # 获取第一个结果对象
                
                # 获取模型实际推理的图像尺寸 (如果有)
                if hasattr(result, 'orig_shape') and result.orig_shape is not None: # orig_shape 是输入给模型的尺寸
                     # Note: result.orig_shape might be the original image shape *before* letterboxing/resizing by predict
                     # We need the shape the model *actually* saw.
                     # result.speed gives timings, and result.im.shape could be the processed image shape
                    if hasattr(result, 'im') and result.im is not None:
                        processed_img_h, processed_img_w = result.im.shape[:2]
                    elif hasattr(result, 'ims') and result.ims and len(result.ims) > 0 and result.ims[0] is not None: # For some cases like segmentation
                        processed_img_h, processed_img_w = result.ims[0].shape[:2]


                # 尝试从 result.speed 获取更详细的时间 (如果可用)
                # result.speed 的格式 {'preprocess': ms, 'inference': ms, 'postprocess': ms} 或 {'preprocess': ms, 'inference': ms, 'NMS': ms}
                if hasattr(result, 'speed') and isinstance(result.speed, dict):
                    timing_stats["pre_processing_time_ms"] = result.speed.get('preprocess')
                    timing_stats["inference_time_ms"] = result.speed.get('inference')
                    timing_stats["post_processing_time_ms"] = result.speed.get('postprocess') # for older ultralytics versions
                    timing_stats["nms_time_ms"] = result.speed.get('NMS') # for newer ultralytics versions
                    if timing_stats["post_processing_time_ms"] is None and timing_stats["nms_time_ms"] is not None:
                        # Newer versions might only have NMS time as part of postprocessing
                        timing_stats["post_processing_time_ms"] = timing_stats["nms_time_ms"] 
                    elif timing_stats["post_processing_time_ms"] is not None and timing_stats["nms_time_ms"] is None:
                         # if postprocess includes NMS, we can consider them combined or try to estimate if needed.
                         # For now, just report what's available.
                         pass


                boxes = result.boxes  # Boxes object for bbox outputs
                if boxes is not None and len(boxes) > 0:
                    det_id_counter = 0
                    for i in range(len(boxes)):
                        box_data = boxes[i]
                        class_id = int(box_data.cls.item())
                        conf = float(box_data.conf.item())
                        
                        # 手动应用类别过滤
                        if class_indices_to_filter is not None and class_id not in class_indices_to_filter:
                            continue

                        xyxy_pixels = box_data.xyxy.cpu().numpy().squeeze().tolist() # [xmin, ymin, xmax, ymax]
                        
                        # 确保坐标是浮点数
                        x1p, y1p, x2p, y2p = map(float, xyxy_pixels)

                        width_p = x2p - x1p
                        height_p = y2p - y1p
                        center_x_p = x1p + width_p / 2
                        center_y_p = y1p + height_p / 2
                        area_p = width_p * height_p

                        # 计算归一化坐标
                        x1n = x1p / original_width
                        y1n = y1p / original_height
                        x2n = x2p / original_width
                        y2n = y2p / original_height
                        
                        detection = {
                            "id": det_id_counter,
                            "class_id": class_id,
                            "class_name": self.model.names[class_id] if self.model.names and class_id in self.model.names else "unknown",
                            "confidence": conf,
                            "bbox_pixels": [x1p, y1p, x2p, y2p],
                            "bbox_normalized": [x1n, y1n, x2n, y2n],
                            "center_x_pixels": center_x_p,
                            "center_y_pixels": center_y_p,
                            "width_pixels": width_p,
                            "height_pixels": height_p,
                            "area_pixels": area_p
                        }
                        detections_output.append(detection)
                        det_id_counter += 1
                
                normal_logger.info(f"YOLO检测完成，原始检测到 {len(boxes) if boxes else 0} 个目标, 过滤后输出 {len(detections_output)} 个目标。")

                # 如果需要返回标注图像
                if return_image_flag:
                    annotated_frame = result.plot()  # BGR numpy array with annotations
                    try:
                        # 将BGR图像编码为JPG并转换为Base64
                        is_success, buffer = cv2.imencode(".jpg", annotated_frame)
                        if is_success:
                            annotated_image_b64 = base64.b64encode(buffer).decode("utf-8")
                        else:
                            normal_logger.warning("无法将标注帧编码为JPG。")
                    except Exception as e_img:
                        exception_logger.error(f"转换标注图像到Base64时出错: {e_img}")
            else:
                normal_logger.info("YOLO模型 predict() 未返回任何结果。")

        except Exception as e:
            exception_logger.exception(f"YOLO检测过程中发生严重错误: {e}")
            analyzer_end_time = time.perf_counter()
            timing_stats["analyzer_total_time_ms"] = (analyzer_end_time - analyzer_start_time) * 1000
            return {
                "success": False,
                "error": f"检测失败: {str(e)}",
                "detections": [],
                "applied_config": {
                    "confidence_threshold": confidence, "iou_threshold": iou_threshold, 
                    "max_detections": max_detections, "imgsz": imgsz_arg,
                    "filtered_classes": actual_filtered_class_names, "half_precision": self.half_precision
                },
                "image_info": {
                    "original_height": original_height, "original_width": original_width,
                    "processed_height": processed_img_h, "processed_width": processed_img_w
                },
                "timing_stats": timing_stats,
                "annotated_image_base64": None
            }

        analyzer_end_time = time.perf_counter()
        timing_stats["analyzer_total_time_ms"] = (analyzer_end_time - analyzer_start_time) * 1000
        
        # 最终返回结构
        final_result = {
            "success": True,
            "error": None,
            "detections": detections_output,
            "applied_config": {
                "confidence_threshold": confidence,
                "iou_threshold": iou_threshold,
                "max_detections": max_detections,
                "imgsz": imgsz_arg, 
                "filtered_classes": actual_filtered_class_names, # 使用实际应用过滤的类别名称
                "half_precision": self.half_precision
            },
            "image_info": {
                "original_height": original_height,
                "original_width": original_width,
                "processed_height": processed_img_h, 
                "processed_width": processed_img_w
            },
            "timing_stats": timing_stats,
            "annotated_image_base64": annotated_image_b64
        }
        return final_result

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