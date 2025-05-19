"""
YOLOE分析器实现
提供基于YOLOE的目标检测、分割和跟踪功能
支持文本提示、图像提示和无提示推理
"""
import os
import time
import json
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
import base64
import io

# 使用新的日志记录器
from shared.utils.logger import get_normal_logger, get_exception_logger, get_test_logger
from core.analyzer.base_analyzer import (
    DetectionAnalyzer, 
    SegmentationAnalyzer, 
    TrackingAnalyzer
)
from core.analyzer.tracking.tracker import Tracker, TrackerType # 确保导入TrackerType
from core.config import settings
from core.exceptions import ModelLoadException, ProcessingException, ResourceNotFoundException

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()

# 尝试导入YOLOE特定库
try:
    from yoloe.models.yoloe_predictor import YOLOEExecutor
    from yoloe.utils.config_utils import merge_cfg_file, merge_cfg_options
    from yoloe.utils.general_utils import get_img_tensor, get_model_info, xywh2xyxy, postprocess, 비디오_프레임_추론
    yoloe_available = True
    normal_logger.info("YOLOE相关库成功导入。")
except ImportError as e:
    yoloe_available = False
    exception_logger.warning(f"YOLOE相关库导入失败，YOLOE分析器功能将不可用。错误: {e}")
    # 定义一些占位符，以避免在yoloe_available为False时出现NameError
    class YOLOEExecutor:
        def __init__(self, *args, **kwargs): 
            exception_logger.error("YOLOEExecutor尝试在库未成功导入时初始化。")
            raise ImportError("YOLOE库未成功导入，无法使用YOLOEExecutor。")
    def merge_cfg_file(*args, **kwargs): pass
    def merge_cfg_options(*args, **kwargs): pass
    def get_img_tensor(*args, **kwargs): pass
    def get_model_info(*args, **kwargs): pass
    def xywh2xyxy(*args, **kwargs): pass
    def postprocess(*args, **kwargs): pass
    def 비디오_프레임_추론(*args, **kwargs): pass # 韩文函数名占位符


class YOLOEBaseAnalyzer: # 创建一个共同的基类处理YOLOE模型加载和通用逻辑
    """YOLOE分析器的通用基类，处理模型加载和配置。"""
    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0, 
                 yolo_version: int = 0,  # YOLOE可能不严格区分yolo_version
                 device: str = "auto", **kwargs):
        if not yoloe_available:
            msg = "YOLOE库未初始化或导入失败，无法创建YOLOE分析器。"
            exception_logger.error(msg)
            raise ModelLoadException(msg)

        self.model_code = model_code
        self.engine_type = engine_type # YOLOE主要基于PyTorch，此参数可能影响不大
        self.device_str = device
        self.executor: Optional[YOLOEExecutor] = None
        self.model_config: Optional[Dict[str, Any]] = None
        self.task_type: str = "检测" # 子类中会覆盖
        self.current_model_name: Optional[str] = None # 用于存储实际加载的模型文件名

        # 从kwargs获取YOLOE特定配置
        self.config_file = kwargs.get("yoloe_config_file", self._get_default_config_file())
        self.model_weights_path = kwargs.get("yoloe_weights_path") # 优先使用指定的权重路径
        self.opts = kwargs.get("yoloe_opts", [])

        normal_logger.info(f"YOLOE基础分析器初始化开始: 模型代码={model_code}, 配置文件={self.config_file}, 设备={device}")
        test_logger.info(f"[初始化-YOLOEBase] 模型代码={model_code}, 配置文件={self.config_file}")

        if model_code or self.model_weights_path: # 如果有模型代码或直接权重路径，则尝试加载
            asyncio.create_task(self.load_model(model_code))
    
    def _get_default_config_file(self) -> str:
        # 根据任务类型（子类中定义）选择默认配置文件
        # 例如: "yoloe/configs/yoloe_l_crowdhuman_method1_train.py"
        # 基类中返回一个通用或错误提示
        default_cfg = f"yoloe/configs/yoloe_s_{self.task_type.lower()}_coco.py" # 假设的默认结构
        # 实际的默认配置文件路径需要根据YOLOE项目结构确定
        # 这里用一个占位符，实际应指向一个真实存在的默认配置文件
        placeholder_default_config = os.path.join(settings.STORAGE.BASE_DIR, settings.STORAGE.MODEL_DIR, "yoloe", "configs", "default_yoloe_config.py")
        if os.path.exists(placeholder_default_config):
            normal_logger.info(f"使用占位符默认配置文件: {placeholder_default_config}")
            return placeholder_default_config
        normal_logger.warning(f"无法确定YOLOE的默认配置文件路径，请通过yoloe_config_file参数指定。尝试使用: {default_cfg}")
        return default_cfg # 返回一个理论上的路径，load_model会检查是否存在

    async def load_model(self, model_code: Optional[str] = None) -> bool:
        log_prefix = f"[模型加载-YOLOEBase] 模型代码={model_code or self.model_code}"
        test_logger.info(f"{log_prefix} | 开始加载YOLOE模型...")
        if not yoloe_available:
            test_logger.error(f"{log_prefix} | YOLOE库不可用，加载失败。")
            return False
        try:
            # 1. 确定配置文件路径
            config_path = self.config_file
            if not os.path.isabs(config_path):
                # 配置文件通常位于yoloe项目内或模型目录内
                # 尝试在 data/models/yoloe/configs/ 或 yoloe/configs/ 下查找
                yoloe_root_candidates = [ 
                    Path(settings.STORAGE.BASE_DIR) / settings.STORAGE.MODEL_DIR / "yoloe",
                    Path(".") / "yoloe" # 假设yoloe代码库在项目根目录的yoloe文件夹下
                ]
                found_cfg = False
                for root_candidate in yoloe_root_candidates:
                    abs_cfg_path = root_candidate / "configs" / os.path.basename(config_path)
                    if abs_cfg_path.exists():
                        config_path = str(abs_cfg_path)
                        found_cfg = True
                        break
                if not found_cfg and not Path(config_path).exists(): # 如果相对路径也不存在
                    msg = f"YOLOE配置文件 '{self.config_file}' (尝试的绝对路径 '{config_path}') 未找到。"
                    exception_logger.error(msg)
                    test_logger.error(f"{log_prefix} | {msg}")
                    raise ModelLoadException(msg)
            
            test_logger.info(f"{log_prefix} | 使用配置文件: {config_path}")

            # 2. 确定模型权重路径
            weights_path = self.model_weights_path # 优先使用直接指定的权重路径
            if not weights_path:
                if model_code:
                    # 权重文件通常在 data/models/{model_code}/best.pth 或类似
                    # YOLOE的权重文件通常是 .pth 或 .pt
                    model_dir = Path(settings.STORAGE.BASE_DIR) / settings.STORAGE.MODEL_DIR / model_code
                    possible_weights_names = ["best.pth", "model.pth", f"{model_code}.pth", "best.pt", f"{model_code}.pt"]
                    found_weights = False
                    for fname in possible_weights_names:
                        if (model_dir / fname).exists():
                            weights_path = str(model_dir / fname)
                            found_weights = True
                            break
                    if not found_weights:
                        msg = f"在目录 {model_dir} 中找不到YOLOE的权重文件 (尝试了 {possible_weights_names})。"
                        exception_logger.error(msg)
                        test_logger.error(f"{log_prefix} | {msg}")
                        raise ModelLoadException(msg)
                else:
                    msg = "未提供model_code且未指定yoloe_weights_path，无法加载YOLOE模型。"
                    exception_logger.error(msg)
                    test_logger.error(f"{log_prefix} | {msg}")
                    raise ModelLoadException(msg)
            
            test_logger.info(f"{log_prefix} | 使用权重文件: {weights_path}")
            self.current_model_name = os.path.basename(weights_path)

            # 3. 合并配置
            cfg = merge_cfg_file(config_path)
            if self.opts:
                cfg = merge_cfg_options(cfg, self.opts)
            test_logger.info(f"{log_prefix} | 配置已合并。任务类型配置: {cfg.dataset.task_type if hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'task_type') else '未知'}")

            # 4. 确定设备
            if self.device_str == "auto":
                device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device_to_use = self.device_str
            test_logger.info(f"{log_prefix} | 将在设备上加载模型: {device_to_use}")

            # 5. 创建YOLOEExecutor实例
            load_start_time = time.time()
            self.executor = YOLOEExecutor(cfg, weights_path, device=device_to_use)
            load_time = time.time() - load_start_time
            self.model_config = cfg # 保存解析后的配置

            normal_logger.info(f"YOLOE模型 '{self.current_model_name}' 从 '{weights_path}' 加载成功到设备 '{device_to_use}'。耗时: {load_time:.2f}秒。")
            test_logger.info(f"{log_prefix} | 模型 '{self.current_model_name}' 加载成功。耗时: {load_time:.2f}秒。")
            return True

        except ModelLoadException as mle: # 已记录的异常，直接抛出
            exception_logger.error(f"YOLOE模型加载期间发生已知错误: {str(mle)}") # 确保也记录到exception_logger
            raise
        except FileNotFoundError as fnfe:
            msg = f"YOLOE模型加载失败: 文件未找到 - {str(fnfe)}"
            exception_logger.error(msg)
            test_logger.error(f"{log_prefix} | {msg}")
            raise ModelLoadException(msg)
        except Exception as e:
            exception_logger.exception(f"YOLOE模型 '{model_code or self.model_code}' 加载过程中发生未知错误")
            test_logger.error(f"{log_prefix} | 加载时发生未知错误: {str(e)}")
            raise ModelLoadException(f"加载YOLOE模型时发生未知错误: {str(e)}")

    def _get_common_yoloe_model_info(self) -> Dict[str, Any]:
        if not self.executor or not self.model_config:
            return {"loaded": False, "model_name": None, "config_file": self.config_file}
        
        model_name = self.current_model_name or (self.model_config.weight.split('/')[-1] if hasattr(self.model_config, 'weight') and self.model_config.weight else "未知YOLOE模型")
        input_size = self.model_config.test_image_size if hasattr(self.model_config, 'test_image_size') else "未知"
        num_classes = self.model_config.data_cfg.num_classes if hasattr(self.model_config, 'data_cfg') and hasattr(self.model_config.data_cfg, 'num_classes') else "未知"

        return {
            "loaded": True,
            "model_name": model_name,
            "config_file": self.config_file,
            "device": self.executor.device.type if self.executor and self.executor.device else self.device_str,
            "input_size": input_size,
            "num_classes": num_classes,
            "task_type": self.task_type,
            "engine_type": "YOLOE_PyTorch" # YOLOE通常基于PyTorch
        }

    def release(self):
        log_prefix = f"[资源释放-YOLOEBase] 模型名={self.current_model_name or '未知'}"
        normal_logger.info(f"开始释放YOLOE分析器资源: {self.current_model_name or self.model_code}")
        test_logger.info(log_prefix)
        if self.executor:
            # YOLOEExecutor可能没有显式的release方法，依赖Python的垃圾回收
            # 清理对模型的引用
            if hasattr(self.executor, 'model'):
                del self.executor.model
            del self.executor
            self.executor = None
            normal_logger.info("YOLOEExecutor实例已清除。")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            normal_logger.info("CUDA缓存已为YOLOE分析器清空。")
        normal_logger.info("YOLOE分析器资源已释放。")
        test_logger.info(f"{log_prefix} | 释放完成。")

    # 通用辅助方法 (如果YOLOE有特定的图像预处理，应在此处或子类中实现)
    async def _preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if not yoloe_available:
            raise RuntimeError("YOLOE库不可用，无法预处理图像。")
        # YOLOE的get_img_tensor通常处理了大部分预处理
        # image: BGR numpy array
        # cfg.test_image_size 是 (h, w)
        img_size = self.model_config.test_image_size if self.model_config and hasattr(self.model_config, 'test_image_size') else (640, 640) # 提供默认值
        img_tensor, ratio = get_img_tensor(image, img_size) # get_img_tensor 返回处理后的图像和缩放比例
        return img_tensor.to(self.executor.device), ratio

    # 通用结果图像保存方法，可被子类复用或覆盖
    async def _save_result_image(self, 
                                 image: np.ndarray, 
                                 results_for_filename: List[Dict], 
                                 task_name: Optional[str],
                                 log_prefix: str,
                                 image_type_suffix: str = "yoloe_detection") -> Optional[str]:
        try:
            if image is None or image.size == 0:
                test_logger.warning(f"{log_prefix} | 保存YOLOE结果图片失败: 输入图像为空或大小为0")
                return None

            current_dir = os.getcwd()
            results_base_dir = os.path.join(current_dir, "results")
            os.makedirs(results_base_dir, exist_ok=True)
            
            date_str = datetime.now().strftime("%Y%m%d")
            type_specific_dir = os.path.join(results_base_dir, image_type_suffix, date_str)
            os.makedirs(type_specific_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            task_prefix_name = f"{task_name if task_name and task_name != '未命名任务' else image_type_suffix}_"
            
            classes_info_str = ""
            if results_for_filename:
                class_counts = {}
                for item in results_for_filename:
                    cls_name = item.get("class_name", "未知")
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                classes_info_str = "_".join([f"{cls}_{count}" for cls, count in class_counts.items()])
                classes_info_str = f"_{classes_info_str}_" if classes_info_str else "_"
            else:
                 classes_info_str = f"_no_results_"

            filename = f"{task_prefix_name}{classes_info_str.strip('_')}_{timestamp}.jpg"
            file_path = os.path.join(type_specific_dir, filename)

            success = cv2.imwrite(file_path, image)
            if success and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                test_logger.info(f"{log_prefix} | YOLOE结果图片已保存: {file_path}, 大小: {os.path.getsize(file_path)/1024:.1f}KB")
                return os.path.join("results", image_type_suffix, date_str, filename)
            else:
                test_logger.warning(f"{log_prefix} | 保存YOLOE结果图片失败或文件无效: {file_path}")
                return None
        except Exception as e:
            exception_logger.exception(f"任务 {task_name} 保存YOLOE结果图片过程中发生严重错误")
            test_logger.info(f"{log_prefix} | 保存YOLOE结果图片时发生错误: {str(e)}")
            return None

# --- YOLOE Detection Analyzer ---
class YOLOEDetectionAnalyzer(YOLOEBaseAnalyzer, DetectionAnalyzer):
    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0, 
                 yolo_version: int = 0, device: str = "auto", **kwargs):
        self.task_type = "检测" # 在调用super().__init__之前设置
        super().__init__(model_code, engine_type, yolo_version, device, **kwargs)
        normal_logger.info(f"YOLOE目标检测分析器已初始化。")
        test_logger.info(f"[初始化] YOLOE检测分析器 (YOLOEDetectionAnalyzer) 模型代码: {model_code}")

    async def detect(self, image: np.ndarray, 
                     confidence: Optional[float] = None,
                     iou_threshold: Optional[float] = None, # YOLOE的postprocess可能使用不同的NMS参数
                     classes: Optional[List[int]] = None, # YOLOE可能通过其配置文件过滤类别
                     task_name: Optional[str] = "YOLOE检测任务",
                     **kwargs) -> Dict[str, Any]:
        log_prefix = f"[YOLOE检测] 任务={task_name}, 模型={self.current_model_name or '未知'}"
        test_logger.info(f"{log_prefix} | 开始检测图像，尺寸: {image.shape if image is not None else '无图像'}")
        if not self.executor or not self.model_config:
            msg = "YOLOE执行器或模型配置未初始化。"
            exception_logger.error(msg)
            test_logger.error(f"{log_prefix} | {msg}")
            raise ModelLoadException(msg)

        start_time = time.time()
        annotated_image_bytes = None
        detections = []
        
        # 获取YOLOE特定参数或使用默认/已配置值
        conf_thres = confidence if confidence is not None else (self.model_config.test_conf if hasattr(self.model_config, 'test_conf') else 0.001) # YOLOE通常用test_conf
        nms_thres = iou_threshold if iou_threshold is not None else (self.model_config.nms_conf_thres if hasattr(self.model_config, 'nms_conf_thres') else 0.7) # YOLOE的NMS阈值
        test_logger.info(f"{log_prefix} | 使用参数: 置信度阈值(test_conf)={conf_thres}, NMS阈值={nms_thres}")

        try:
            img_tensor, ratio = await self._preprocess_image(image) # (h_origin, w_origin)
            preprocess_time = (time.time() - start_time) * 1000
            
            infer_start_time = time.time()
            # YOLOE的executor.inference可能返回原始输出，需要后处理
            # 或者YOLOE的推理接口可能像 비디오_프레임_추론 直接返回处理好的结果
            # 此处假设executor.inference返回的是需要后处理的原始preds
            # 如果有直接的推理+后处理接口，应使用那个
            raw_preds = self.executor.inference(img_tensor) 
            inference_time = (time.time() - infer_start_time) * 1000

            post_start_time = time.time()
            # YOLOE的postprocess函数参数： preds, cfg, ratio, dwdh=None, pad_image=False
            # ratio通常是 (ratio_h, ratio_w)
            # dwdh 是letterbox的padding (dw, dh)
            # 需要确保从_preprocess_image返回的ratio格式正确，或者调整这里的调用
            # 假设 ratio 是 (ratio_h, ratio_w) 或单个值表示等比例缩放
            if isinstance(ratio, tuple) and len(ratio) == 2:
                ratio_h, ratio_w = ratio
            else: # 假设ratio是单个值
                ratio_h = ratio_w = ratio 

            # YOLOE的postprocess函数可能需要原始图像尺寸来正确缩放边界框
            # 如果get_img_tensor的ratio是单个值，可能需要做如下处理
            # 或者确保get_img_tensor返回(ratio_h, ratio_w)
            # dwdh的计算通常在letterbox函数内部完成，如果get_img_tensor做了letterbox，它应该返回dwdh
            # 这里简化处理，假设postprocess能正确处理。
            # YOLOE的postprocess通常需要 preds, cfg, ratio_h, ratio_w
            # 检查YOLOE的postprocess签名，确保参数匹配
            # preds, cfg, ratio_h, ratio_w, conf_thres=None, nms_thres=None, 
            # class_agnostic=False, labels=(), pad_offset=0, max_candidates=3000, max_detections=300
            processed_results = postprocess(
                raw_preds, self.model_config, ratio_h, ratio_w, 
                conf_thres=conf_thres, 
                nms_thres=nms_thres,
                # classes=classes, # YOLOE的postprocess可能不直接接受类别过滤，通常在config中设置
                max_detections=self.model_config.get("max_detections", 300) # 从配置或默认
            )
            # processed_results 的结构通常是 [batch_idx, x1, y1, x2, y2, score, class_idx]
            postprocess_time = (time.time() - post_start_time) * 1000

            # 转换结果格式
            if processed_results is not None and len(processed_results) > 0:
                for res in processed_results:
                    # 假设res的顺序是 x1,y1,x2,y2,score,class_id (移除了batch_idx)
                    # 如果postprocess返回的包含batch_idx，需要 res[1:]
                    # 具体结构取决于YOLOE的postprocess实现
                    res_data = res[1:] if len(res) == 7 else res # 假设带batch_idx时长度为7
                    if len(res_data) == 6:
                        x1, y1, x2, y2, score, class_id = res_data
                        class_names = self.model_config.data_cfg.class_names if hasattr(self.model_config, 'data_cfg') else [f"类{i}" for i in range(int(class_id)+1)]
                        detections.append({
                            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                            "confidence": float(score),
                            "class_id": int(class_id),
                            "class_name": class_names[int(class_id)] if int(class_id) < len(class_names) else "未知类别"
                        })
            test_logger.info(f"{log_prefix} | 检测到目标数: {len(detections)}")

            # 绘制和保存图像 (如果需要)
            save_images = kwargs.get("save_images", False)
            if save_images and detections:
                # YOLOE 可能有自己的绘图工具，或者需要手动绘制
                # annotated_image = self._draw_yoloe_detections(image.copy(), detections) # 假设有此方法
                # temp_annotated_image_bytes = await self._encode_result_image(annotated_image)
                # final_results["annotated_image_bytes"] = temp_annotated_image_bytes
                # saved_path = await self._save_result_image(annotated_image, detections, task_name, log_prefix, "yoloe_detection")
                # if saved_path: test_logger.info(f"{log_prefix} | YOLOE检测结果图片已保存: {saved_path}")
                pass # 暂时跳过绘图和保存，集中于核心逻辑

            return {
                "detections": detections,
                "pre_process_time": preprocess_time,
                "inference_time": inference_time,
                "post_process_time": postprocess_time,
                "annotated_image_bytes": annotated_image_bytes 
            }

        except Exception as e:
            exception_logger.exception(f"YOLOE检测任务 '{task_name}' 执行失败")
            test_logger.error(f"{log_prefix} | 检测失败: {str(e)}")
            return {"detections": [], "pre_process_time": 0, "inference_time": 0, "post_process_time": 0, "annotated_image_bytes": None}
    
    async def process_video_frame(self, frame: np.ndarray, frame_index: int, task_name: Optional[str] = "YOLOE视频帧检测", **kwargs) -> Dict[str, Any]:
        log_prefix = f"[YOLOE帧处理] 任务={task_name}, 模型={self.current_model_name or '未知'}, 帧={frame_index}"
        test_logger.info(f"{log_prefix} | YOLOEDetectionAnalyzer 开始处理视频帧")
        kwargs['task_name'] = task_name
        results = await self.detect(frame, **kwargs)
        results["frame_index"] = frame_index
        test_logger.info(f"{log_prefix} | 处理完成, 检测到目标数: {len(results.get('detections', []))}")
        return results

    @property
    def model_info(self) -> Dict[str, Any]:
        info = self._get_common_yoloe_model_info()
        test_logger.info(f"[模型信息] YOLOEDetectionAnalyzer: {json.dumps(info, ensure_ascii=False)}")
        return info

# --- YOLOE Segmentation Analyzer ---
class YOLOESegmentationAnalyzer(YOLOEBaseAnalyzer, SegmentationAnalyzer):
    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0, 
                 yolo_version: int = 0, device: str = "auto", **kwargs):
        self.task_type = "分割" # 在调用super().__init__之前设置
        super().__init__(model_code, engine_type, yolo_version, device, **kwargs)
        normal_logger.info(f"YOLOE图像分割分析器已初始化。")
        test_logger.info(f"[初始化] YOLOE分割分析器 (YOLOESegmentationAnalyzer) 模型代码: {model_code}")

    async def segment(self, image: np.ndarray, 
                      confidence: Optional[float] = None,
                      iou_threshold: Optional[float] = None, 
                      classes: Optional[List[int]] = None, 
                      retina_masks: bool = True, # YOLOE分割可能也用此参数
                      task_name: Optional[str] = "YOLOE分割任务",
                      **kwargs) -> Dict[str, Any]:
        log_prefix = f"[YOLOE分割] 任务={task_name}, 模型={self.current_model_name or '未知'}"
        test_logger.info(f"{log_prefix} | 开始分割图像，尺寸: {image.shape if image is not None else '无图像'}")
        if not self.executor or not self.model_config:
            msg = "YOLOE执行器或模型配置未初始化 (分割)。"
            exception_logger.error(msg)
            test_logger.error(f"{log_prefix} | {msg}")
            raise ModelLoadException(msg)

        start_time = time.time()
        annotated_image_bytes = None
        segmentations = [] # 存储分割结果

        conf_thres = confidence if confidence is not None else (self.model_config.test_conf if hasattr(self.model_config, 'test_conf') else 0.001)
        nms_thres = iou_threshold if iou_threshold is not None else (self.model_config.nms_conf_thres if hasattr(self.model_config, 'nms_conf_thres') else 0.7)
        test_logger.info(f"{log_prefix} | 使用参数: 置信度阈值={conf_thres}, NMS阈值={nms_thres}, RetinaMasks={retina_masks}")

        try:
            img_tensor, ratio = await self._preprocess_image(image)
            preprocess_time = (time.time() - start_time) * 1000

            infer_start_time = time.time()
            # 假设YOLOE的分割推理也是通过 executor.inference
            # 并且其后处理函数能处理分割任务的原始输出
            # YOLOE的分割任务可能在配置文件中由 task_type='seg' 指定
            raw_preds = self.executor.inference(img_tensor)
            inference_time = (time.time() - infer_start_time) * 1000

            post_start_time = time.time()
            if isinstance(ratio, tuple) and len(ratio) == 2:
                ratio_h, ratio_w = ratio
            else: 
                ratio_h = ratio_w = ratio 
            
            # 调用postprocess进行分割后处理
            # 需要确认YOLOE的postprocess是否能直接输出分割掩码，或需要特定参数
            # processed_results的结构可能包含掩码信息
            # YOLOE的官方postprocess可能主要针对检测，分割的后处理可能不同或需要额外步骤
            # 假设 postprocess 返回的结构也包含分割信息，或者需要另一个特定的后处理函数
            # 如果YOLOE的postprocess不直接支持分割，需要调用其分割专用后处理或手动解析raw_preds
            # 为简化，我们先假设postprocess能处理，但标记这里可能需要调整
            normal_logger.info(f"{log_prefix} | 警告: YOLOE分割的postprocess逻辑可能需要特定实现。当前使用通用postprocess。")
            test_logger.info(f"{log_prefix} | 警告: YOLOE分割的postprocess逻辑可能需要特定实现。")

            processed_results = postprocess(
                raw_preds, self.model_config, ratio_h, ratio_w,
                conf_thres=conf_thres, 
                nms_thres=nms_thres,
                # retina_masks=retina_masks, # postprocess函数可能没有这个参数
                # classes=classes,
                max_detections=self.model_config.get("max_detections_seg", 100) # 分割可能用不同最大数量
            )
            postprocess_time = (time.time() - post_start_time) * 1000

            # 转换分割结果
            # 这里的转换逻辑高度依赖YOLOE分割任务的输出格式
            # 通常包括bbox, score, class_id,以及mask数据 (可能是轮廓点或二值图)
            if processed_results is not None and len(processed_results) > 0:
                # 假设processed_results每个条目除了检测信息外，还有掩码信息
                # 例如: res = [x1,y1,x2,y2,score,class_id, mask_data_or_contours]
                # 这个转换逻辑需要根据YOLOE的实际输出来适配
                for res in processed_results: # 这是一个示例结构
                    res_data = res[1:] if len(res) > 7 else res # 假设带batch_idx时长度>6
                    if len(res_data) >= 6: # 至少有bbox, score, class_id
                        x1, y1, x2, y2, score, class_id_val = res_data[:6]
                        class_names_list = self.model_config.data_cfg.class_names if hasattr(self.model_config, 'data_cfg') else []
                        class_name_val = class_names_list[int(class_id_val)] if int(class_id_val) < len(class_names_list) else "未知类别"
                        
                        mask_info = None # 示例，实际掩码处理会更复杂
                        # if len(res_data) > 6: mask_info = res_data[6] # 假设掩码数据在第七个元素
                        
                        segmentations.append({
                            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                            "confidence": float(score),
                            "class_id": int(class_id_val),
                            "class_name": class_name_val,
                            "mask": mask_info # 存储掩码数据 (具体格式待定)
                        })
            test_logger.info(f"{log_prefix} | 分割到实例数: {len(segmentations)}")
            
            # 绘制和保存 (与检测类似，但绘图函数可能需要画掩码)
            # save_images = kwargs.get("save_images", False)
            # if save_images and segmentations:
            #     annotated_image = self._draw_yoloe_segmentations(image.copy(), segmentations) # 假设有此方法
            #     annotated_image_bytes = await self._encode_result_image(annotated_image)
            #     saved_path = await self._save_result_image(annotated_image, segmentations, task_name, log_prefix, "yoloe_segmentation")
            #     if saved_path: test_logger.info(f"{log_prefix} | YOLOE分割结果图片已保存: {saved_path}")

            return {
                "segmentations": segmentations,
                "pre_process_time": preprocess_time,
                "inference_time": inference_time,
                "post_process_time": postprocess_time,
                "annotated_image_bytes": annotated_image_bytes
            }

        except Exception as e:
            exception_logger.exception(f"YOLOE分割任务 '{task_name}' 执行失败")
            test_logger.error(f"{log_prefix} | 分割失败: {str(e)}")
            return {"segmentations": [], "pre_process_time": 0, "inference_time": 0, "post_process_time": 0, "annotated_image_bytes": None}

    async def process_video_frame(self, frame: np.ndarray, frame_index: int, task_name: Optional[str] = "YOLOE视频帧分割", **kwargs) -> Dict[str, Any]:
        log_prefix = f"[YOLOE分割帧处理] 任务={task_name}, 模型={self.current_model_name or '未知'}, 帧={frame_index}"
        test_logger.info(f"{log_prefix} | YOLOESegmentationAnalyzer 开始处理视频帧")
        kwargs['task_name'] = task_name
        results = await self.segment(frame, **kwargs)
        results["frame_index"] = frame_index
        test_logger.info(f"{log_prefix} | 处理完成, 分割到实例数: {len(results.get('segmentations', []))}")
        return results

    @property
    def model_info(self) -> Dict[str, Any]:
        info = self._get_common_yoloe_model_info()
        info["task_type"] = "分割" # 明确指定任务类型
        test_logger.info(f"[模型信息] YOLOESegmentationAnalyzer: {json.dumps(info, ensure_ascii=False)}")
        return info

# --- YOLOE Tracking Analyzer ---
class YOLOETrackingAnalyzer(YOLOEBaseAnalyzer, TrackingAnalyzer):
    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0, 
                 yolo_version: int = 0, device: str = "auto",
                 tracker_type_name: str = "sort", # 从字符串接收跟踪器类型
                  **kwargs):
        self.task_type = "跟踪" # 在调用super().__init__之前设置
        YOLOEBaseAnalyzer.__init__(self, model_code, engine_type, yolo_version, device, **kwargs)
        
        try:
            tracker_type_enum = TrackerType[tracker_type_name.upper()]
        except KeyError:
            exception_logger.error(f"无效的YOLOE跟踪器类型名称(Tracker): '{tracker_type_name}'. 将使用默认的SORT。")
            tracker_type_enum = TrackerType.SORT
        # 初始化TrackingAnalyzer部分，它会创建self.tracker
        # 确保将tracker_type的int值传递给父类
        TrackingAnalyzer.__init__(self, model_code=None, # 模型加载由YOLOEBaseAnalyzer处理
                                  tracker_type=tracker_type_enum.value, 
                                  # 其他跟踪器参数如max_age等从kwargs传递给TrackingAnalyzer
                                  max_age=kwargs.get("max_age", 30),
                                  min_hits=kwargs.get("min_hits", 3),
                                  iou_threshold=kwargs.get("iou_threshold", 0.3),
                                  device=device, # 传递device给DeepSORT等可能需要GPU的跟踪器
                                  model_path=kwargs.get("deepsort_model_path"), # DeepSORT的ReID模型路径
                                  **kwargs)
        normal_logger.info(f"YOLOE目标跟踪分析器已初始化。检测模型: {model_code}, 跟踪器类型: {tracker_type_enum.name}")
        test_logger.info(f"[初始化] YOLOE跟踪分析器 (YOLOETrackingAnalyzer) 检测模型: {model_code}, 跟踪器: {tracker_type_enum.name}")

    async def track(self, image: np.ndarray, 
                    confidence: Optional[float] = None,
                    iou_threshold: Optional[float] = None,
                    classes: Optional[List[int]] = None,
                    task_name: Optional[str] = "YOLOE跟踪任务",
                    **kwargs) -> Dict[str, Any]:
        log_prefix = f"[YOLOE跟踪] 任务={task_name}, 模型={self.current_model_name or '未知'}"
        test_logger.info(f"{log_prefix} | 开始处理图像进行检测和跟踪")
        if not self.executor or not self.model_config:
            msg = "YOLOE执行器或模型配置未初始化 (跟踪)。"
            exception_logger.error(msg)
            test_logger.error(f"{log_prefix} | {msg}")
            raise ModelLoadException(msg)
        if not self.tracker:
            msg = "内部跟踪器未初始化 (跟踪)。"
            exception_logger.error(msg)
            test_logger.error(f"{log_prefix} | {msg}")
            raise ProcessingException(msg)

        start_time = time.time()
        # 1. 使用YOLOE进行检测 (调用YOLOEDetectionAnalyzer的detect逻辑，或直接实现)
        # 为避免重复代码，这里可以创建一个临时的YOLOEDetectionAnalyzer实例或调用一个共享的检测方法
        # 但更简单的方式是直接复用YOLOEDetectionAnalyzer的detect实现的核心部分
        detections_for_tracker_np = np.empty((0,6)) # x1,y1,x2,y2,score,class_id
        detection_module_results = {} # 存储检测模块的原始时间和结果
        
        conf_thres_detect = confidence if confidence is not None else (self.model_config.test_conf if hasattr(self.model_config, 'test_conf') else 0.001)
        nms_thres_detect = iou_threshold if iou_threshold is not None else (self.model_config.nms_conf_thres if hasattr(self.model_config, 'nms_conf_thres') else 0.7)

        try:
            img_tensor, ratio = await self._preprocess_image(image)
            detection_module_results["pre_process_time"] = (time.time() - start_time) * 1000
            
            infer_start_time = time.time()
            raw_preds = self.executor.inference(img_tensor)
            detection_module_results["inference_time"] = (time.time() - infer_start_time) * 1000

            post_start_time = time.time()
            if isinstance(ratio, tuple) and len(ratio) == 2: ratio_h, ratio_w = ratio
            else: ratio_h = ratio_w = ratio
            
            processed_detect_results = postprocess(
                raw_preds, self.model_config, ratio_h, ratio_w,
                conf_thres=conf_thres_detect, nms_thres=nms_thres_detect,
                max_detections=self.model_config.get("max_detections", 300)
            )
            detection_module_results["post_process_time"] = (time.time() - post_start_time) * 1000

            temp_detections = []
            if processed_detect_results is not None and len(processed_detect_results) > 0:
                class_names_list = self.model_config.data_cfg.class_names if hasattr(self.model_config, 'data_cfg') else []
                for res in processed_detect_results:
                    res_data = res[1:] if len(res) == 7 else res
                    if len(res_data) == 6:
                        x1, y1, x2, y2, score, class_id_val = res_data
                        # YOLOE的postprocess直接返回原始检测，无需再转换
                        temp_detections.append([float(x1), float(y1), float(x2), float(y2), float(score), int(class_id_val)])
                        # 存储一份带类别名的检测结果，用于返回给调用者
                        class_name_val = class_names_list[int(class_id_val)] if int(class_id_val) < len(class_names_list) else "未知类别"
                        if "detections" not in detection_module_results: detection_module_results["detections"] = []
                        detection_module_results["detections"].append({
                             "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                             "confidence": float(score), "class_id": int(class_id_val), "class_name": class_name_val
                        })
            if temp_detections:
                detections_for_tracker_np = np.array(temp_detections)
            test_logger.info(f"{log_prefix} | YOLOE检测步骤完成，检测到 {len(detections_for_tracker_np)} 个目标。")
        except Exception as e:
            exception_logger.exception(f"YOLOE跟踪任务 '{task_name}' 的检测步骤失败")
            test_logger.error(f"{log_prefix} | 检测步骤失败: {str(e)}")
            # 即使检测失败，也继续尝试更新跟踪器（使用空检测），以保持跟踪器状态的连续性
            detections_for_tracker_np = np.empty((0,6))
            if "detections" not in detection_module_results: detection_module_results["detections"] = []
            if "pre_process_time" not in detection_module_results: detection_module_results["pre_process_time"] = (time.time() - start_time) * 1000 # 估算
            if "inference_time" not in detection_module_results: detection_module_results["inference_time"] = 0
            if "post_process_time" not in detection_module_results: detection_module_results["post_process_time"] = 0

        # 2. 更新跟踪器
        tracking_start_time = time.time()
        actual_tracker_type_for_check = self.tracker.tracker_type.value if isinstance(self.tracker.tracker_type, Enum) else self.tracker.tracker_type
        original_image_for_tracker = image if actual_tracker_type_for_check == TrackerType.DEEP_SORT.value else None
        tracked_objects_list = self.tracker.update(detections_for_tracker_np, original_image=original_image_for_tracker)
        tracking_time = (time.time() - tracking_start_time) * 1000
        test_logger.info(f"{log_prefix} | 跟踪器更新完成，得到 {len(tracked_objects_list)} 个跟踪ID。耗时: {tracking_time:.2f}ms")
        
        # 3. 构建返回结果
        final_results = {
            "detections": detection_module_results.get("detections", []),
            "tracking_results": tracked_objects_list,
            "pre_process_time": detection_module_results.get("pre_process_time", 0),
            "inference_time": detection_module_results.get("inference_time", 0),
            "post_process_time": detection_module_results.get("post_process_time", 0),
            "tracking_update_time": tracking_time, # 单独的跟踪器更新时间
            "annotated_image_bytes": None,
            "counts": self.tracker.get_counts() if hasattr(self.tracker, 'get_counts') else {},
        }

        # 绘制和保存逻辑 (可选)
        # save_images = kwargs.get("save_images", False)
        # if save_images and tracked_objects_list:
        #     try:
        #         # YOLOE的绘制需要区分检测框和跟踪ID
        #         # annotated_image = self._draw_yoloe_tracks(image.copy(), tracked_objects_list, detection_module_results.get("detections", []))
        #         # final_results["annotated_image_bytes"] = await self._encode_result_image(annotated_image)
        #         # await self._save_result_image(annotated_image, tracked_objects_list, task_name, log_prefix, "yoloe_tracking")
        #     except Exception as e:
        #         exception_logger.exception(f"任务 {task_name} 绘制或保存YOLOE跟踪图像时出错。")

        total_processing_time = (time.time() - start_time) * 1000
        test_logger.info(f"{log_prefix} | YOLOE跟踪处理总耗时: {total_processing_time:.2f}ms")
        return final_results

    async def process_video_frame(self, frame: np.ndarray, frame_index: int, task_name: Optional[str] = "YOLOE视频帧跟踪", **kwargs) -> Dict[str, Any]:
        log_prefix = f"[YOLOE跟踪帧处理] 任务={task_name}, 模型={self.current_model_name or '未知'}, 帧={frame_index}"
        test_logger.info(f"{log_prefix} | YOLOETrackingAnalyzer 开始处理视频帧")
        kwargs['task_name'] = task_name
        results = await self.track(frame, **kwargs) # 调用自身的track方法
        results["frame_index"] = frame_index
        test_logger.info(f"{log_prefix} | 处理完成, 跟踪到ID数: {len(results.get('tracking_results', []))}")
        return results

    @property
    def model_info(self) -> Dict[str, Any]:
        common_info = self._get_common_yoloe_model_info()
        common_info["task_type"] = "跟踪"
        
        actual_tracker_type_name = "未知"
        # self.tracker 由 TrackingAnalyzer.__init__ 设置
        if hasattr(self, 'tracker') and self.tracker and hasattr(self.tracker, 'tracker_type') and isinstance(self.tracker.tracker_type, TrackerType):
            actual_tracker_type_name = self.tracker.tracker_type.name
        elif hasattr(self, 'tracker_type') and isinstance(self.tracker_type, TrackerType): # 从父类 TrackingAnalyzer 继承的 tracker_type
            actual_tracker_type_name = self.tracker_type.name
        elif hasattr(self, 'tracker_type') and isinstance(self.tracker_type, int):
            try: actual_tracker_type_name = TrackerType(self.tracker_type).name
            except ValueError: pass
        
        common_info["tracker_type"] = actual_tracker_type_name
        test_logger.info(f"[模型信息] YOLOETrackingAnalyzer: {json.dumps(common_info, ensure_ascii=False)}")
        return common_info

    # 子类可以覆盖 release 方法以添加特定的清理逻辑，但要确保调用 super().release()
    # def release(self):
    #     super().release()
    #     # YOLOE跟踪器特定的清理，如果除了YOLOEBaseAnalyzer和TrackingAnalyzer之外还有其他资源
    #     normal_logger.info("YOLOE跟踪分析器特定资源已释放（如果有）。")
