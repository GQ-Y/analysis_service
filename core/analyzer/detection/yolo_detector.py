"""
YOLO检测器模块
实现基于YOLOv8、YOLOv7、YOLOv5、YOLOX、YOLOE的目标检测功能
"""
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
from shared.utils.logger import get_normal_logger, get_exception_logger, get_test_logger, get_analysis_logger
from core.config import settings
import time
import asyncio
from PIL import Image
import colorsys
from datetime import datetime
from core.exceptions import ModelLoadException
import io
import json
from core.analyzer.model_loader import ModelLoader

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger() # 获取测试日志记录器
analysis_logger = get_analysis_logger() # 获取分析日志记录器

class YOLODetector:
    """YOLO检测器实现"""

    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0,
                 yolo_version: int = 0, device: str = "auto"):
        """
        初始化检测器

        Args:
            model_code: 模型代码，如果提供则立即加载模型
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            device: 推理设备
        """
        self.model = None
        self.current_model_code = model_code
        self.engine_type = engine_type
        self.yolo_version = yolo_version
        self.device = device

        # 默认配置
        self.default_confidence = settings.ANALYSIS.confidence
        self.default_iou = settings.ANALYSIS.iou
        self.default_max_det = settings.ANALYSIS.max_det

        # 设置保存目录
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.results_dir = self.project_root / settings.OUTPUT.save_dir

        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)

        # 增强初始化日志
        normal_logger.info(f"初始化YOLO检测器，使用设备: {self.device}")
        normal_logger.info(f"默认置信度阈值: {self.default_confidence}")
        normal_logger.info(f"结果保存目录: {self.results_dir}")
        
        # 添加分析日志
        test_logger.info(f"[初始化] YOLO检测器: 模型={model_code}, 设备={self.device}")
        test_logger.info(f"[初始化] YOLO检测器默认参数: 置信度={self.default_confidence}, IOU={self.default_iou}, 最大检测数={self.default_max_det}")

        # 如果提供了model_code，立即加载模型
        if model_code:
            asyncio.create_task(self.load_model(model_code))

    async def get_model_path(self, model_code: str) -> str:
        """获取模型路径

        Args:
            model_code: 模型代码,例如'model-gcc'

        Returns:
            str: 本地模型文件路径

        Raises:
            Exception: 当找不到模型时抛出异常
        """
        try:
            # 使用ModelLoader获取模型路径
            return await ModelLoader.get_model_path(model_code, self.yolo_version)
        except Exception as e:
            exception_logger.error(f"获取模型路径时出错: {str(e)}")
            # 尝试查找默认模型
            return await self._find_default_model(model_code)

    async def _find_default_model(self, model_code: str) -> str:
        """
        查找默认模型

        Args:
            model_code: 模型代码

        Returns:
            str: 默认模型路径

        Raises:
            Exception: 当找不到默认模型时
        """
        normal_logger.info(f"尝试查找默认模型替代 {model_code}...")
        test_logger.info(f"[模型查找] 尝试为 {model_code} 查找默认替代模型")

        # 检查通用模型目录
        models_dir = os.path.join("data", "models")

        # 尝试使用通用YOLOv8模型
        yolo_dirs = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
        for yolo_dir in yolo_dirs:
            yolo_path = os.path.join(models_dir, yolo_dir)
            if os.path.exists(yolo_path):
                pt_files = [f for f in os.listdir(yolo_path) if f.endswith('.pt')]
                if pt_files:
                    model_path = os.path.join(yolo_path, pt_files[0])
                    normal_logger.warning(f"未找到{model_code}模型，使用替代模型: {model_path}")
                    test_logger.info(f"[模型查找] {model_code} 未找到，使用替代: {model_path}")
                    return model_path

        # 如果找不到任何模型，抛出异常
        exception_logger.error(f"无法找到任何可用的模型，请确保至少有一个模型文件在data/models目录下。检查模型代码: {model_code}")
        test_logger.info(f"[模型查找] {model_code} 未找到任何备用模型")
        raise Exception(f"无法找到任何可用的模型，请确保至少有一个模型文件在data/models目录下")

    async def load_model(self, model_code: str, max_retries: int = 3):
        """
        加载模型

        Args:
            model_code: 模型代码
            max_retries: 最大重试次数

        Raises:
            ModelLoadException: 当模型加载失败时
        """
        retry_count = 0
        last_error = None

        # 记录开始加载模型
        test_logger.info(f"[模型加载] 开始加载模型: {model_code}")
        
        while retry_count < max_retries:
            try:
                # 获取模型路径
                model_path = await self.get_model_path(model_code)
                normal_logger.info(f"正在加载模型: {model_path}")
                test_logger.info(f"[模型加载] 模型路径: {model_path}")

                # 记录加载开始时间
                load_start_time = time.time()
                
                # 加载模型
                self.model = YOLO(model_path)
                # 确定设备
                if self.device == "auto":
                    self.current_device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self.current_device = self.device
                self.model.to(self.current_device)
                normal_logger.info(f"模型已移至设备: {self.current_device}")

                # 记录加载耗时
                load_time = time.time() - load_start_time
                test_logger.info(f"[模型加载] 模型 {model_code} 加载耗时: {load_time:.2f}秒")

                # 设置模型参数
                self.model.conf = self.default_confidence
                self.model.iou = self.default_iou
                self.model.max_det = self.default_max_det
                
                # 特别记录model-gcc模型的详细信息
                if "gcc" in model_code.lower():
                    model_info = {
                        "model_code": model_code,
                        "model_path": model_path,
                        "device": str(self.current_device),
                        "confidence": self.model.conf,
                        "iou": self.model.iou,
                        "max_det": self.model.max_det,
                        "model_type": type(self.model).__name__,
                    }
                    
                    # 记录模型类别
                    if hasattr(self.model, "names"):
                        model_info["classes"] = self.model.names
                        class_count = len(self.model.names)
                        analysis_logger.info(f"[AI测试] model-gcc模型支持的类别数量: {class_count}")
                        analysis_logger.info(f"[AI测试] model-gcc模型类别列表: {list(self.model.names.values())[:10]}{'...' if class_count > 10 else ''}")
                    
                    # 记录模型尺寸
                    if hasattr(self.model, "model") and hasattr(self.model.model, "yaml"):
                        if "imgsz" in self.model.model.yaml:
                            model_info["input_size"] = self.model.model.yaml["imgsz"]
                            analysis_logger.info(f"[AI测试] model-gcc模型输入尺寸: {self.model.model.yaml['imgsz']}")
                    
                    analysis_logger.info(f"[AI测试] model-gcc模型加载详情: {json.dumps(model_info, indent=2)}")

                # 更新当前模型代码
                self.current_model_code = model_code

                normal_logger.info(f"模型加载成功: {model_code}")
                analysis_logger.info(f"[AI测试] 模型成功加载: {model_code}")
                
                # 记录可用显存信息(CUDA)
                if torch.cuda.is_available():
                    free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)  # 转换为GB
                    total_mem = torch.cuda.mem_get_info()[1] / (1024 ** 3)  # 转换为GB
                    used_mem = total_mem - free_mem
                    analysis_logger.info(f"[AI测试] 模型加载后GPU内存使用: {used_mem:.2f}GB/{total_mem:.2f}GB (已用/总共)")
                
                return  # 成功加载，退出函数

            except Exception as e:
                last_error = e
                retry_count += 1
                normal_logger.warning(f"模型加载失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                analysis_logger.error(f"[AI测试] 模型加载失败: {model_code}, 错误: {str(e)}")

                if retry_count < max_retries:
                    # 等待一段时间后重试
                    await asyncio.sleep(1.0)
                    normal_logger.info(f"正在重试加载模型: {model_code}")
                    analysis_logger.info(f"[AI测试] 正在重试加载模型: {model_code} (第{retry_count}次)")
                else:
                    # 达到最大重试次数，记录错误并抛出异常
                    normal_logger.error(f"模型加载失败，已达到最大重试次数: {str(last_error)}")
                    import traceback
                    error_trace = traceback.format_exc()
                    exception_logger.exception(error_trace)
                    analysis_logger.error(f"[AI测试] 模型加载失败，已达到最大重试次数: {str(last_error)}")
                    analysis_logger.error(f"[AI测试] 错误堆栈: {error_trace}")

                    # 确保模型为None
                    self.model = None
                    self.current_model_code = None

                    raise ModelLoadException(f"模型加载失败: {str(last_error)}")

    def _get_color_by_id(self, obj_id: int) -> Tuple[int, int, int]:
        """根据对象ID生成固定的颜色

        Args:
            obj_id: 对象ID

        Returns:
            Tuple[int, int, int]: RGB颜色值
        """
        # 使用黄金比例法生成不同的色相值
        golden_ratio = 0.618033988749895
        hue = (obj_id * golden_ratio) % 1.0

        # 转换HSV到RGB（固定饱和度和明度以获得鲜艳的颜色）
        rgb = tuple(round(x * 255) for x in colorsys.hsv_to_rgb(hue, 0.8, 0.95))
        return rgb

    async def _encode_result_image(self, image: np.ndarray) -> bytes:
        """将结果图像编码为字节流

        Args:
            image: 需要编码的图像数组

        Returns:
            编码后的图像字节流
        """
        try:
            # 将图像从BGR转换为RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 将numpy数组转换为PIL图像
            pil_image = Image.fromarray(rgb_image)

            # 创建一个字节流缓冲区
            buffer = io.BytesIO()

            # 将图像保存为JPEG格式到缓冲区
            pil_image.save(buffer, format='JPEG')

            # 获取字节流
            image_bytes = buffer.getvalue()

            return image_bytes
        except Exception as e:
            exception_logger.exception(f"编码结果图像时出错: {str(e)}")
            return b""

    async def detect(self,
                     image: np.ndarray,
                     verbose: bool = False,
                     roi: Optional[Dict[str, Any]] = None,
                     roi_type: int = 1,
                     classes: Optional[List[int]] = None,
                     confidence: Optional[float] = None,
                     iou_threshold: Optional[float] = None,
                     **kwargs
                     ) -> Dict[str, Any]:
        """对输入图像进行目标检测

        Args:
            image: BGR格式的输入图像
            verbose: 是否打印详细日志，默认False
            roi: 感兴趣区域配置
            roi_type: ROI类型 (1=矩形, 2=多边形, 3=线段)
            classes: 需要检测的类别列表
            confidence: 置信度阈值
            iou_threshold: IoU阈值
            **kwargs: 其他参数

        Returns:
            包含检测结果和图像数据的字典:
            - detections: 检测结果列表
            - pre_process_time: 预处理时间 (ms)
            - inference_time: 推理时间 (ms)
            - post_process_time: 后处理时间 (ms)
            - annotated_image_bytes: 标注后图像的 JPEG 字节流
        """
        start_time = time.time()
        pre_process_time_ms = 0
        inference_time_ms = 0
        post_process_time_ms = 0
        annotated_image_bytes = None
        
        # 记录检测任务信息
        task_name = kwargs.get("task_name", "未命名任务")
        
        # 记录低阈值检测
        if confidence is not None:
            if confidence <= 0.05:
                analysis_logger.info(f"[AI测试] 执行低阈值检测: task={task_name}, confidence={confidence}")
            else:
                analysis_logger.info(f"[AI测试] 执行检测: task={task_name}, confidence={confidence}")

        try:
            # 检查模型是否已加载
            if self.model is None:
                normal_logger.error("模型未加载，无法执行检测")
                analysis_logger.error(f"[AI测试] 执行检测失败: task={task_name}, 错误=模型未加载")
                return {
                    "detections": [],
                    "pre_process_time": 0,
                    "inference_time": 0,
                    "post_process_time": 0,
                    "annotated_image_bytes": None
                }

            # 设置模型参数
            if confidence is not None:
                self.model.conf = confidence
                analysis_logger.info(f"[AI测试] 设置检测阈值: task={task_name}, confidence={confidence}")
            if iou_threshold is not None:
                self.model.iou = iou_threshold
                analysis_logger.info(f"[AI测试] 设置IoU阈值: task={task_name}, iou_threshold={iou_threshold}")
            if classes is not None:
                self.model.classes = classes
                analysis_logger.info(f"[AI测试] 设置检测类别: task={task_name}, classes={classes}")

            # 记录图像信息
            if image is not None:
                analysis_logger.info(f"[AI测试] 待分析图像: task={task_name}, shape={image.shape}")
            
            # 记录检测开始
            detect_start_time = time.time()
            
            # 运行检测
            results = self.model(image, verbose=verbose)
            
            # 记录检测用时
            detect_time = time.time() - detect_start_time
            
            # 获取计时信息
            if results and hasattr(results[0], 'speed'):
                speed_info = results[0].speed
                pre_process_time_ms = speed_info.get('preprocess', 0)
                inference_time_ms = speed_info.get('inference', 0)
                post_process_time_ms = speed_info.get('postprocess', 0)
                analysis_logger.info(f"[AI测试] 检测性能指标: task={task_name}, 预处理={pre_process_time_ms}ms, 推理={inference_time_ms}ms, 后处理={post_process_time_ms}ms")

            # 解析检测结果
            detections = await self._parse_results(results)
            
            # 记录检测结果
            if len(detections) > 0:
                analysis_logger.info(f"[AI测试] 检测结果: task={task_name}, 检测到{len(detections)}个目标")
                
                # 统计类别分布
                class_counts = {}
                for det in detections:
                    class_name = det.get("class_name", "unknown")
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
                
                analysis_logger.info(f"[AI测试] 类别分布: task={task_name}, {json.dumps(class_counts)}")
                
                # 记录前5个检测结果的详细信息
                for i, det in enumerate(detections[:5]):
                    analysis_logger.info(f"[AI测试] 检测结果[{i+1}]: task={task_name}, 类别={det.get('class_name')}, 置信度={det.get('confidence'):.4f}, 边界框={det.get('bbox')}")
            else:
                analysis_logger.info(f"[AI测试] 检测结果: task={task_name}, 未检测到目标")

            # 如果有ROI配置，进行ROI过滤
            if roi:
                height, width = image.shape[:2]
                filtered_before = len(detections)
                detections = self._filter_by_roi(detections, roi, roi_type, height, width)
                filtered_after = len(detections)
                
                if filtered_before != filtered_after:
                    analysis_logger.info(f"[AI测试] ROI过滤: task={task_name}, 过滤前={filtered_before}个目标, 过滤后={filtered_after}个目标")

            # --- 总是生成和编码标注图像 ---
            try:
                annotated_image = results[0].plot() # 使用ultralytics自带的plot
                is_success, buffer = cv2.imencode(".jpg", annotated_image)
                if not is_success:
                    normal_logger.warning("标注图像编码失败")
                    analysis_logger.warning(f"[AI测试] 标注图像编码失败: task={task_name}")
                else:
                    annotated_image_bytes = buffer.tobytes()

                    # 检查是否需要保存图片
                    save_images = kwargs.get("save_images", False)

                    # 只在检测到目标时才保存图片
                    if save_images and len(detections) > 0:
                        # 保存图片
                        saved_path = await self._save_result_image(annotated_image, detections, task_name)
                        if saved_path:
                            analysis_logger.info(f"[AI测试] 保存分析结果图片: task={task_name}, path={saved_path}")

            except Exception as plot_err:
                exception_logger.exception(f"绘制或编码标注图像时出错: {plot_err}")
                analysis_logger.error(f"[AI测试] 生成标注图像失败: task={task_name}, 错误={str(plot_err)}")
            # --- 结束图像处理 ---

            # 计算总耗时
            total_time_ms = (time.time() - start_time) * 1000
            analysis_logger.info(f"[AI测试] 检测完成: task={task_name}, 共{len(detections)}个目标, 总耗时{total_time_ms:.2f}ms")

            # 构建返回字典 (始终包含 annotated_image_bytes，即使为 None)
            return_data = {
                "detections": detections,
                "pre_process_time": pre_process_time_ms,
                "inference_time": inference_time_ms,
                "post_process_time": post_process_time_ms,
                "annotated_image_bytes": annotated_image_bytes # 可能为 None
            }

            return return_data

        except Exception as e:
            exception_logger.exception(f"检测失败: {str(e)}")
            import traceback
            error_trace = traceback.format_exc()
            exception_logger.error(error_trace)
            analysis_logger.error(f"[AI测试] 检测执行失败: task={task_name}, 错误={str(e)}")
            analysis_logger.error(f"[AI测试] 错误堆栈: {error_trace}")
            return {
                "detections": [],
                "pre_process_time": 0,
                "inference_time": 0,
                "post_process_time": 0,
                "annotated_image_bytes": None # 错误时也返回 None
            }

    def _filter_by_roi(self, detections: List[Dict], roi: Dict, roi_type: int, img_height: int, img_width: int) -> List[Dict]:
        """
        根据ROI过滤检测结果

        Args:
            detections: 检测结果列表
            roi: ROI配置
            roi_type: ROI类型 (1=矩形, 2=多边形, 3=线段)
            img_height: 图像高度
            img_width: 图像宽度

        Returns:
            过滤后的检测结果列表
        """
        try:
            if not roi:
                return detections

            # 初始化过滤后的检测结果列表
            filtered_detections = []

            # 将ROI坐标转换为像素坐标
            pixel_roi = {}
            normalized = roi.get("normalized", True)

            if roi_type == 0:  # 无ROI类型
                # 对于无ROI类型，我们仍然需要初始化pixel_roi，但它不会被使用
                pixel_roi = {
                    "x1": 0,
                    "y1": 0,
                    "x2": img_width,
                    "y2": img_height
                }

            elif roi_type == 1:  # 矩形ROI
                # 从coordinates获取坐标点
                if "coordinates" in roi:
                    points = roi["coordinates"]
                    if len(points) >= 2:
                        if normalized:
                            pixel_roi = {
                                "x1": int(points[0][0] * img_width),
                                "y1": int(points[0][1] * img_height),
                                "x2": int(points[1][0] * img_width),
                                "y2": int(points[1][1] * img_height)
                            }
                        else:
                            pixel_roi = {
                                "x1": int(points[0][0]),
                                "y1": int(points[0][1]),
                                "x2": int(points[1][0]),
                                "y2": int(points[1][1])
                            }
                # 兼容旧格式
                elif all(k in roi for k in ["x1", "y1", "x2", "y2"]):
                    if normalized:
                        pixel_roi = {
                            "x1": int(roi["x1"] * img_width),
                            "y1": int(roi["y1"] * img_height),
                            "x2": int(roi["x2"] * img_width),
                            "y2": int(roi["y2"] * img_height)
                        }
                    else:
                        pixel_roi = {
                            "x1": int(roi["x1"]),
                            "y1": int(roi["y1"]),
                            "x2": int(roi["x2"]),
                            "y2": int(roi["y2"])
                        }
                else:
                    normal_logger.error("无效的矩形ROI格式")
                    return detections

            elif roi_type in [2, 3]:  # 多边形或线段ROI
                if "points" in roi:
                    points = roi["points"]
                elif "coordinates" in roi:
                    points = roi["coordinates"]
                else:
                    normal_logger.error("无效的多边形/线段ROI格式")
                    return detections

                if normalized:
                    pixel_roi["points"] = [
                        (int(p[0] * img_width), int(p[1] * img_height))
                        for p in points
                    ]
                else:
                    pixel_roi["points"] = [
                        (int(p[0]), int(p[1]))
                        for p in points
                    ]

            # 处理每个检测结果
            for detection in detections:
                bbox = detection["bbox"]
                # 转换边界框为像素坐标
                if bbox.get("x1") < 1 and bbox.get("y1") < 1 and bbox.get("x2") < 1 and bbox.get("y2") < 1:
                    # 如果边界框是归一化坐标（0-1范围），转换为像素坐标
                    pixel_bbox = {
                        "x1": int(bbox["x1"] * img_width),
                        "y1": int(bbox["y1"] * img_height),
                        "x2": int(bbox["x2"] * img_width),
                        "y2": int(bbox["y2"] * img_height)
                    }
                else:
                    pixel_bbox = bbox

                # 计算检测框面积
                det_area = (pixel_bbox["x2"] - pixel_bbox["x1"]) * (pixel_bbox["y2"] - pixel_bbox["y1"])

                # 根据ROI类型进行过滤
                if roi_type == 0:  # 无ROI类型，直接保留所有目标
                    filtered_detections.append(detection)

                elif roi_type == 1:  # 矩形ROI
                    # 计算与ROI的重叠区域
                    overlap_x1 = max(pixel_roi["x1"], pixel_bbox["x1"])
                    overlap_y1 = max(pixel_roi["y1"], pixel_bbox["y1"])
                    overlap_x2 = min(pixel_roi["x2"], pixel_bbox["x2"])
                    overlap_y2 = min(pixel_roi["y2"], pixel_bbox["y2"])

                    # 检查是否有重叠
                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        # 计算重叠区域的面积
                        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                        # 检查是否有50%以上的目标在ROI内
                        if overlap_area / det_area >= 0.5:
                            filtered_detections.append(detection)

                elif roi_type == 2:  # 多边形ROI
                    # 获取检测框的中心点
                    center_x = (pixel_bbox["x1"] + pixel_bbox["x2"]) / 2
                    center_y = (pixel_bbox["y1"] + pixel_bbox["y2"]) / 2

                    # 检查点是否在多边形内
                    if self._point_in_polygon((center_x, center_y), pixel_roi["points"]):
                        filtered_detections.append(detection)

                elif roi_type == 3:  # 线段ROI
                    if len(pixel_roi["points"]) >= 2:
                        # 获取线段的两个端点
                        p1, p2 = pixel_roi["points"][:2]

                        # 检测框的四个边
                        box_edges = [
                            ((pixel_bbox["x1"], pixel_bbox["y1"]), (pixel_bbox["x2"], pixel_bbox["y1"])),  # 上边
                            ((pixel_bbox["x2"], pixel_bbox["y1"]), (pixel_bbox["x2"], pixel_bbox["y2"])),  # 右边
                            ((pixel_bbox["x2"], pixel_bbox["y2"]), (pixel_bbox["x1"], pixel_bbox["y2"])),  # 下边
                            ((pixel_bbox["x1"], pixel_bbox["y2"]), (pixel_bbox["x1"], pixel_bbox["y1"]))   # 左边
                        ]

                        # 检查线段是否与任何边相交
                        for edge in box_edges:
                            if self._line_segments_intersect(p1, p2, edge[0], edge[1]):
                                filtered_detections.append(detection)
                                break

            # 返回过滤后的检测结果
            return filtered_detections

        except Exception as e:
            exception_logger.exception(f"根据ROI过滤检测结果失败: {str(e)}")
            import traceback
            exception_logger.error(traceback.format_exc())
            analysis_logger.error(f"ROI过滤失败: {str(e)}")
            # 如果过滤失败，返回原始检测结果
            return detections

    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """
        判断点是否在多边形内(射线法)

        Args:
            point: 待判断的点(x,y)
            polygon: 多边形顶点列表[(x1,y1), (x2,y2),...]

        Returns:
            bool: 点是否在多边形内
        """
        x, y = point
        inside = False

        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def _line_segments_intersect(self, p1, p2, p3, p4):
        """
        检查两条线段是否相交

        Args:
            p1, p2: 第一条线段的两个端点
            p3, p4: 第二条线段的两个端点

        Returns:
            bool: 两条线段是否相交
        """
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # 共线
            return 1 if val > 0 else 2  # 顺时针或逆时针

        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

        # 计算四个方向
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)

        # 一般情况下的相交
        if o1 != o2 and o3 != o4:
            return True

        # 特殊情况
        if o1 == 0 and on_segment(p1, p3, p2):
            return True
        if o2 == 0 and on_segment(p1, p4, p2):
            return True
        if o3 == 0 and on_segment(p3, p1, p4):
            return True
        if o4 == 0 and on_segment(p3, p2, p4):
            return True

        return False

    async def _save_result_image(self, image: np.ndarray, detections: List[Dict], task_name: Optional[str] = None) -> str:
        """保存带有检测结果的图片"""
        try:
            # 检查图像是否为空
            if image is None:
                return None

            # 直接使用传入的图像（已经包含了检测结果的标注）
            result_image = image

            # 获取当前工作目录
            current_dir = os.getcwd()

            # 确保results目录存在
            results_dir = os.path.join(current_dir, "results")
            os.makedirs(results_dir, exist_ok=True)

            # 确保每天的结果保存在单独的目录中
            date_str = datetime.now().strftime("%Y%m%d")
            date_dir = os.path.join(results_dir, date_str)
            os.makedirs(date_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
            task_prefix = f"{task_name}_" if task_name else ""

            # 添加检测到的目标类别信息到文件名
            classes_info = ""
            if detections and len(detections) > 0:
                # 获取所有检测到的类别
                classes = [det.get("class_name", "unknown") for det in detections]
                # 统计每个类别的数量
                class_counts = {}
                for cls in classes:
                    if cls in class_counts:
                        class_counts[cls] += 1
                    else:
                        class_counts[cls] = 1
                # 生成类别信息字符串，例如：person_2_car_1
                classes_info = "_".join([f"{cls}_{count}" for cls, count in class_counts.items()])
                classes_info = f"{classes_info}_"

            filename = f"{task_prefix}{classes_info}{timestamp}.jpg"

            # 完整的文件路径
            file_path = os.path.join(date_dir, filename)

            # 保存图片
            try:
                # 检查图像是否为空
                if result_image is None or result_image.size == 0:
                    return None

                # 尝试使用cv2保存
                success = cv2.imwrite(file_path, result_image)

                if not success:
                    # 尝试使用PIL保存
                    try:
                        from PIL import Image
                        pil_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                        pil_image.save(file_path)
                    except Exception:
                        # 尝试直接写入文件
                        try:
                            _, buffer = cv2.imencode(".jpg", result_image)
                            with open(file_path, "wb") as f:
                                f.write(buffer)
                        except Exception:
                            return None

                # 检查文件是否存在
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    # 检查文件大小是否正常
                    if file_size == 0:
                        analysis_logger.warning(f"[AI测试] 保存的结果图片大小为0: task={task_name}, path={file_path}")
                        return None
                    analysis_logger.info(f"[AI测试] 结果图片已保存: task={task_name}, size={file_size/1024:.1f}KB, path={file_path}")
                else:
                    analysis_logger.warning(f"[AI测试] 结果图片保存失败: task={task_name}, path={file_path}")
                    return None
            except Exception as e:
                analysis_logger.error(f"[AI测试] 保存结果图片异常: task={task_name}, 错误={str(e)}")
                return None

            # 返回相对路径
            relative_path = os.path.join("results", date_str, filename)
            return relative_path

        except Exception as e:
            analysis_logger.error(f"[AI测试] 保存结果图片异常: task={task_name}, 错误={str(e)}")
            return None

    async def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image: 原始图像
            detections: 检测结果列表

        Returns:
            np.ndarray: 绘制了检测结果的图像
        """
        try:
            # 创建图像副本
            result_image = image.copy()

            # 绘制每个检测结果
            for det in detections:
                bbox = det["bbox"]
                label = det.get("label", "")
                confidence = det.get("confidence", 0)
                class_id = det.get("class_id", 0)

                # 获取类别颜色
                color = self._get_color_by_id(class_id)

                # 绘制边界框
                x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                x2, y2 = int(bbox["x2"]), int(bbox["y2"])
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

                # 绘制标签
                text = f"{label}: {confidence:.2f}" if label else f"Class {class_id}: {confidence:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                # 确保标签在图像内
                text_x = x1
                text_y = y1 - 5 if y1 - 5 > text_size[1] else y1 + text_size[1]

                # 绘制文本背景
                cv2.rectangle(
                    result_image,
                    (text_x, text_y - text_size[1]),
                    (text_x + text_size[0], text_y),
                    color,
                    -1
                )

                # 绘制文本
                cv2.putText(
                    result_image,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )

            return result_image

        except Exception as e:
            exception_logger.exception(f"绘制检测结果时出错: {str(e)}")
            return image.copy()

    async def _parse_results(self, results) -> List[Dict[str, Any]]:
        """解析YOLO检测结果

        Args:
            results: YOLO检测结果

        Returns:
            List[Dict[str, Any]]: 解析后的检测结果列表
        """
        try:
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = result.names[cls]

                    detection = {
                        "bbox": {
                            "x1": float(bbox[0]),
                            "y1": float(bbox[1]),
                            "x2": float(bbox[2]),
                            "y2": float(bbox[3])
                        },
                        "confidence": conf,
                        "class_id": cls,
                        "class_name": name
                    }
                    detections.append(detection)
            
            # 记录低阈值检测的详细结果
            if len(detections) > 0 and any(d["confidence"] <= 0.1 for d in detections):
                low_conf_detections = [d for d in detections if d["confidence"] <= 0.1]
                if low_conf_detections:
                    analysis_logger.info(f"[AI测试] 低阈值检测结果: 检测到{len(low_conf_detections)}个低置信度目标 (置信度<=0.1)")
                    # 按置信度从高到低排序
                    low_conf_detections.sort(key=lambda x: x["confidence"], reverse=True)
                    # 记录低置信度检测结果的详细信息（最多5个）
                    for i, det in enumerate(low_conf_detections[:5]):
                        analysis_logger.info(f"[AI测试] 低置信度检测[{i+1}]: 类别={det['class_name']}, 置信度={det['confidence']:.4f}")

            return detections

        except Exception as e:
            exception_logger.exception(f"解析检测结果失败: {str(e)}")
            return []

    async def process_video_frame(self, frame: np.ndarray, frame_index: int, **kwargs) -> Dict[str, Any]:
        """
        处理视频帧

        Args:
            frame: 视频帧
            frame_index: 帧索引
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        # 调用detect方法处理帧
        result = await self.detect(frame, **kwargs)

        # 添加帧索引
        result["frame_index"] = frame_index

        return result

    @property
    def model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            Dict[str, Any]: 模型信息
        """
        if not self.model:
            return {
                "loaded": False,
                "model_code": None
            }

        return {
            "loaded": True,
            "model_code": self.current_model_code,
            "device": str(self.device),
            "confidence": self.model.conf if hasattr(self.model, "conf") else self.default_confidence,
            "iou_threshold": self.model.iou if hasattr(self.model, "iou") else self.default_iou,
            "max_detections": self.model.max_det if hasattr(self.model, "max_det") else self.default_max_det
        }

    def release(self) -> None:
        """释放模型和资源"""
        # 清除模型实例和CUDA缓存
        torch.cuda.empty_cache()
        self.model = None
        
        normal_logger.info("YOLO检测器资源已释放")