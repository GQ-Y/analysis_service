"""
YOLOE分析器实现
提供基于YOLOE的目标检测、分割和跟踪功能
支持文本提示、图像提示和无提示推理
"""
import os
import time
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

from shared.utils.logger import setup_logger
from core.analyzer.base_analyzer import DetectionAnalyzer, SegmentationAnalyzer, TrackingAnalyzer
from core.analyzer.model_loader import ModelLoader

logger = setup_logger(__name__)


def log_detections_if_found(result, boxes):
    """
    只在检测到目标时记录检测结果日志

    Args:
        result: YOLO检测结果
        boxes: 检测到的边界框
    """
    if len(boxes) > 0:
        # 构建检测结果日志
        detection_info = []
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = result.names.get(cls_id, f"class_{cls_id}")
            detection_info.append(f"{cls_name}: {conf:.2f}")

        # 输出检测结果日志
        logger.info(f"检测到 {len(boxes)} 个目标: {', '.join(detection_info)}")


def log_detections(result, boxes):
    """
    记录检测结果日志

    Args:
        result: YOLO检测结果
        boxes: 检测到的边界框
    """
    if len(boxes) > 0:
        # 构建检测结果日志
        detection_info = []
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = result.names.get(cls_id, f"class_{cls_id}")
            detection_info.append(f"{cls_name}: {conf:.2f}")

        # 输出检测结果日志
        logger.info(f"检测到 {len(boxes)} 个目标: {', '.join(detection_info)}")

class YOLOEBaseAnalyzer:
    """YOLOE基础分析器"""

    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0,
                 yolo_version: int = 0, device: str = "auto", **kwargs):
        """
        初始化YOLOE基础分析器

        Args:
            model_code: 模型代码
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            device: 推理设备
            **kwargs: 其他参数
        """
        self.model = None
        self.current_model_code = None
        self.engine_type = engine_type
        self.yolo_version = yolo_version
        self.device = device

        # YOLOE特有参数
        self.prompt_type = kwargs.get("prompt_type", 3)  # 1=文本提示, 2=图像提示, 3=无提示
        self.text_prompt = kwargs.get("text_prompt", [])
        self.visual_prompt = kwargs.get("visual_prompt", {})

        # 内部使用0表示无提示，但API使用3表示无提示
        self._internal_prompt_type = self.prompt_type
        if self.prompt_type == 3:
            self._internal_prompt_type = 0
            logger.info("使用无提示模式 (prompt_type=3)")

        logger.info(f"初始化YOLOE分析器: 提示类型={self._get_prompt_type_name()}")

        # 如果提供了model_code，立即加载模型
        if model_code:
            # 注意：这里不能直接使用await，因为__init__不是异步方法
            # 创建一个异步任务来加载模型
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环已经在运行，创建一个任务
                    asyncio.create_task(self.load_model(model_code))
                else:
                    # 如果事件循环没有运行，直接运行直到完成
                    loop.run_until_complete(self.load_model(model_code))
            except Exception as e:
                logger.error(f"初始化时加载模型失败: {str(e)}")

    def _get_prompt_type_name(self) -> str:
        """获取提示类型名称"""
        prompt_types = {
            1: "文本提示",
            2: "图像提示",
            3: "无提示"
        }
        return prompt_types.get(self.prompt_type, f"未知提示类型({self.prompt_type})")

    def _prepare_prompt(self, image: np.ndarray) -> Dict[str, Any]:
        """
        准备提示信息

        Args:
            image: 输入图像

        Returns:
            Dict[str, Any]: 提示信息
        """
        prompt_data = {}

        # 根据提示类型准备提示信息
        if self._internal_prompt_type == 1 and self.text_prompt:  # 文本提示
            prompt_data["text"] = self.text_prompt
            logger.debug(f"使用文本提示: {self.text_prompt}")

        elif self._internal_prompt_type == 2 and self.visual_prompt:  # 图像提示
            # 处理视觉提示
            height, width = image.shape[:2]
            visual_type = self.visual_prompt.get("type", 0)

            if visual_type == 0:  # 点
                points = self.visual_prompt.get("points", [])
                if points:
                    # 转换为像素坐标
                    pixel_points = []
                    for point in points:
                        x = int(point["x"] * width)
                        y = int(point["y"] * height)
                        pixel_points.append((x, y))
                    prompt_data["points"] = pixel_points

            elif visual_type == 1:  # 框
                points = self.visual_prompt.get("points", [])
                if len(points) >= 2:
                    # 转换为像素坐标
                    x1 = int(points[0]["x"] * width)
                    y1 = int(points[0]["y"] * height)
                    x2 = int(points[1]["x"] * width)
                    y2 = int(points[1]["y"] * height)
                    prompt_data["box"] = [x1, y1, x2, y2]

            elif visual_type == 2:  # 多边形
                points = self.visual_prompt.get("points", [])
                if points:
                    # 转换为像素坐标
                    pixel_points = []
                    for point in points:
                        x = int(point["x"] * width)
                        y = int(point["y"] * height)
                        pixel_points.append((x, y))
                    prompt_data["polygon"] = pixel_points

                    # 是否作为掩码使用
                    if self.visual_prompt.get("use_as_mask", False):
                        # 创建掩码
                        mask = np.zeros((height, width), dtype=np.uint8)
                        points_array = np.array(pixel_points, dtype=np.int32)
                        cv2.fillPoly(mask, [points_array], 255)
                        prompt_data["mask"] = mask

            logger.debug(f"使用视觉提示: 类型={visual_type}, 数据={prompt_data}")

        return prompt_data


class YOLOEDetectionAnalyzer(DetectionAnalyzer, YOLOEBaseAnalyzer):
    """YOLOE检测分析器"""

    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0,
                 yolo_version: int = 0, device: str = "auto", **kwargs):
        """
        初始化YOLOE检测分析器

        Args:
            model_code: 模型代码
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            device: 推理设备
            **kwargs: 其他参数
        """
        DetectionAnalyzer.__init__(self, model_code, engine_type, yolo_version, device)
        YOLOEBaseAnalyzer.__init__(self, model_code, engine_type, yolo_version, device, **kwargs)

    async def load_model(self, model_code: str) -> bool:
        """
        加载YOLOE模型

        Args:
            model_code: 模型代码

        Returns:
            bool: 是否成功加载模型
        """
        try:
            # 使用ModelLoader加载模型
            self.model = await ModelLoader.load_model(
                model_code,
                self.engine_type,
                self.yolo_version,
                self.device
            )

            # 更新当前模型代码
            self.current_model_code = model_code

            logger.info(f"YOLOE检测模型加载成功: {model_code}")
            return True

        except Exception as e:
            logger.error(f"YOLOE检测模型加载失败: {str(e)}")
            return False

    async def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        使用YOLOE模型检测图像

        Args:
            image: 输入图像
            **kwargs: 其他参数
                - confidence: 置信度阈值
                - iou_threshold: IOU阈值
                - classes: 类别过滤列表
                - roi: 感兴趣区域 [x1, y1, x2, y2]
                - max_detections: 最大检测数量

        Returns:
            Dict[str, Any]: 检测结果
        """
        start_time = time.time()

        try:
            if self.model is None:
                logger.error("模型未加载")
                return {
                    "detections": [],
                    "pre_process_time": 0,
                    "inference_time": 0,
                    "post_process_time": 0,
                    "annotated_image_bytes": None
                }

            # 获取参数
            confidence = kwargs.get("confidence", 0.25)
            iou_threshold = kwargs.get("iou_threshold", 0.45)
            classes = kwargs.get("classes", None)
            roi = kwargs.get("roi", None)
            max_detections = kwargs.get("max_detections", 100)

            # 预处理开始时间
            pre_process_start = time.time()

            # 准备提示信息
            prompt_data = self._prepare_prompt(image)

            # 处理ROI
            if roi is not None:
                # 如果ROI是字典格式
                if isinstance(roi, dict) and all(k in roi for k in ["x1", "y1", "x2", "y2"]):
                    height, width = image.shape[:2]
                    x1 = int(roi["x1"] * width)
                    y1 = int(roi["y1"] * height)
                    x2 = int(roi["x2"] * width)
                    y2 = int(roi["y2"] * height)
                    roi = [x1, y1, x2, y2]

                # 裁剪图像到ROI
                if isinstance(roi, list) and len(roi) == 4:
                    x1, y1, x2, y2 = roi
                    # 确保坐标在图像范围内
                    height, width = image.shape[:2]
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))

                    # 裁剪图像
                    roi_image = image[y1:y2, x1:x2]
                    if roi_image.size == 0:
                        logger.warning(f"ROI裁剪后图像为空: {roi}")
                        roi_image = image
                        roi = None
                    else:
                        image = roi_image
            else:
                roi = None

            # 预处理时间
            pre_process_time = time.time() - pre_process_start

            # 推理开始时间
            inference_start = time.time()

            # 执行检测
            # 根据提示类型选择不同的检测方法
            if self._internal_prompt_type == 0:  # 无提示
                # 使用标准检测
                results = self.model(
                    image,
                    conf=confidence,
                    iou=iou_threshold,
                    classes=classes,
                    max_det=max_detections,
                    verbose=False  # 禁用自动打印
                )
            elif self._internal_prompt_type == 1 and self.text_prompt:  # 文本提示
                # 使用文本提示检测
                results = self.model(
                    image,
                    conf=confidence,
                    iou=iou_threshold,
                    classes=classes,
                    max_det=max_detections,
                    text_prompt=self.text_prompt,
                    verbose=False  # 禁用自动打印
                )
            elif self._internal_prompt_type == 2 and self.visual_prompt:  # 图像提示
                # 使用视觉提示检测
                if "mask" in prompt_data:
                    # 使用掩码提示
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        mask_prompt=prompt_data["mask"],
                        verbose=False  # 禁用自动打印
                    )
                elif "box" in prompt_data:
                    # 使用框提示
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        box_prompt=prompt_data["box"],
                        verbose=False  # 禁用自动打印
                    )
                elif "points" in prompt_data:
                    # 使用点提示
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        point_prompt=prompt_data["points"],
                        verbose=False  # 禁用自动打印
                    )
                else:
                    # 没有有效的视觉提示，使用标准检测
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        verbose=False  # 禁用自动打印
                    )
            else:
                # 默认使用标准检测
                results = self.model(
                    image,
                    conf=confidence,
                    iou=iou_threshold,
                    classes=classes,
                    max_det=max_detections,
                    verbose=False  # 禁用自动打印
                )

            # 推理时间
            inference_time = time.time() - inference_start

            # 后处理开始时间
            post_process_start = time.time()

            # 处理检测结果
            detections = []

            # 获取结果
            if results and len(results) > 0:
                # 获取第一帧结果
                result = results[0]

                # 获取边界框
                boxes = result.boxes

                # 只在检测到目标时记录日志
                log_detections_if_found(result, boxes)

                # 处理每个检测结果
                for i in range(len(boxes)):
                    # 获取边界框坐标
                    box = boxes.xyxy[i].cpu().numpy()

                    # 如果使用了ROI，调整坐标
                    if roi is not None:
                        box[0] += roi[0]
                        box[1] += roi[1]
                        box[2] += roi[0]
                        box[3] += roi[1]

                    # 获取置信度
                    conf = float(boxes.conf[i].cpu().numpy())

                    # 获取类别ID和名称
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = result.names.get(cls_id, f"class_{cls_id}")

                    # 创建检测结果
                    detection = {
                        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name
                    }

                    detections.append(detection)

            # 后处理时间
            post_process_time = time.time() - post_process_start

            # 总时间
            total_time = time.time() - start_time

            # 返回结果
            return {
                "detections": detections,
                "pre_process_time": pre_process_time * 1000,  # 转换为毫秒
                "inference_time": inference_time * 1000,  # 转换为毫秒
                "post_process_time": post_process_time * 1000,  # 转换为毫秒
                "total_time": total_time * 1000,  # 转换为毫秒
                "annotated_image_bytes": None
            }

        except Exception as e:
            logger.error(f"YOLOE检测失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "detections": [],
                "pre_process_time": 0,
                "inference_time": 0,
                "post_process_time": 0,
                "total_time": (time.time() - start_time) * 1000,
                "annotated_image_bytes": None
            }

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
            "engine_type": self.engine_type,
            "yolo_version": self.yolo_version,
            "device": self.device,
            "prompt_type": self.prompt_type
        }

    def release(self) -> None:
        """释放资源"""
        self.model = None
        logger.info("YOLOE检测器资源已释放")


class YOLOESegmentationAnalyzer(SegmentationAnalyzer, YOLOEBaseAnalyzer):
    """YOLOE分割分析器"""

    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0,
                 yolo_version: int = 0, device: str = "auto", **kwargs):
        """
        初始化YOLOE分割分析器

        Args:
            model_code: 模型代码
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            device: 推理设备
            **kwargs: 其他参数
        """
        SegmentationAnalyzer.__init__(self, model_code, engine_type, yolo_version, device)
        YOLOEBaseAnalyzer.__init__(self, model_code, engine_type, yolo_version, device, **kwargs)

    async def load_model(self, model_code: str) -> bool:
        """
        加载YOLOE模型

        Args:
            model_code: 模型代码

        Returns:
            bool: 是否成功加载模型
        """
        try:
            # 使用ModelLoader加载模型
            self.model = await ModelLoader.load_model(
                model_code,
                self.engine_type,
                self.yolo_version,
                self.device
            )

            # 更新当前模型代码
            self.current_model_code = model_code

            logger.info(f"YOLOE分割模型加载成功: {model_code}")
            return True

        except Exception as e:
            logger.error(f"YOLOE分割模型加载失败: {str(e)}")
            return False

    async def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        使用YOLOE模型分割图像

        Args:
            image: 输入图像
            **kwargs: 其他参数
                - confidence: 置信度阈值
                - iou_threshold: IOU阈值
                - classes: 类别过滤列表
                - roi: 感兴趣区域 [x1, y1, x2, y2]
                - max_detections: 最大检测数量

        Returns:
            Dict[str, Any]: 分割结果
        """
        start_time = time.time()

        try:
            if self.model is None:
                logger.error("模型未加载")
                return {
                    "segmentations": [],
                    "pre_process_time": 0,
                    "inference_time": 0,
                    "post_process_time": 0,
                    "annotated_image_bytes": None
                }

            # 获取参数
            confidence = kwargs.get("confidence", 0.25)
            iou_threshold = kwargs.get("iou_threshold", 0.45)
            classes = kwargs.get("classes", None)
            roi = kwargs.get("roi", None)
            max_detections = kwargs.get("max_detections", 100)

            # 预处理开始时间
            pre_process_start = time.time()

            # 准备提示信息
            prompt_data = self._prepare_prompt(image)

            # 处理ROI
            if roi is not None:
                # 如果ROI是字典格式
                if isinstance(roi, dict) and all(k in roi for k in ["x1", "y1", "x2", "y2"]):
                    height, width = image.shape[:2]
                    x1 = int(roi["x1"] * width)
                    y1 = int(roi["y1"] * height)
                    x2 = int(roi["x2"] * width)
                    y2 = int(roi["y2"] * height)
                    roi = [x1, y1, x2, y2]

                # 裁剪图像到ROI
                if isinstance(roi, list) and len(roi) == 4:
                    x1, y1, x2, y2 = roi
                    # 确保坐标在图像范围内
                    height, width = image.shape[:2]
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))

                    # 裁剪图像
                    roi_image = image[y1:y2, x1:x2]
                    if roi_image.size == 0:
                        logger.warning(f"ROI裁剪后图像为空: {roi}")
                        roi_image = image
                        roi = None
                    else:
                        image = roi_image
            else:
                roi = None

            # 预处理时间
            pre_process_time = time.time() - pre_process_start

            # 推理开始时间
            inference_start = time.time()

            # 执行分割
            # 根据提示类型选择不同的分割方法
            if self._internal_prompt_type == 0:  # 无提示
                # 使用标准分割
                results = self.model(
                    image,
                    conf=confidence,
                    iou=iou_threshold,
                    classes=classes,
                    max_det=max_detections,
                    retina_masks=True,  # 使用高精度掩码
                    verbose=False  # 禁用自动打印
                )
            elif self._internal_prompt_type == 1 and self.text_prompt:  # 文本提示
                # 使用文本提示分割
                results = self.model(
                    image,
                    conf=confidence,
                    iou=iou_threshold,
                    classes=classes,
                    max_det=max_detections,
                    retina_masks=True,
                    text_prompt=self.text_prompt,
                    verbose=False  # 禁用自动打印
                )
            elif self._internal_prompt_type == 2 and self.visual_prompt:  # 图像提示
                # 使用视觉提示分割
                if "mask" in prompt_data:
                    # 使用掩码提示
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        retina_masks=True,
                        mask_prompt=prompt_data["mask"],
                        verbose=False  # 禁用自动打印
                    )
                elif "box" in prompt_data:
                    # 使用框提示
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        retina_masks=True,
                        box_prompt=prompt_data["box"],
                        verbose=False  # 禁用自动打印
                    )
                elif "points" in prompt_data:
                    # 使用点提示
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        retina_masks=True,
                        point_prompt=prompt_data["points"],
                        verbose=False  # 禁用自动打印
                    )
                else:
                    # 没有有效的视觉提示，使用标准分割
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        retina_masks=True,
                        verbose=False  # 禁用自动打印
                    )
            else:
                # 默认使用标准分割
                results = self.model(
                    image,
                    conf=confidence,
                    iou=iou_threshold,
                    classes=classes,
                    max_det=max_detections,
                    retina_masks=True,
                    verbose=False  # 禁用自动打印
                )

            # 推理时间
            inference_time = time.time() - inference_start

            # 后处理开始时间
            post_process_start = time.time()

            # 处理分割结果
            segmentations = []

            # 获取结果
            if results and len(results) > 0:
                # 获取第一帧结果
                result = results[0]

                # 检查是否有掩码
                if hasattr(result, "masks") and result.masks is not None:
                    # 获取掩码
                    masks = result.masks

                    # 获取边界框
                    boxes = result.boxes

                    # 只在检测到目标时记录日志
                    log_detections_if_found(result, boxes)

                    # 处理每个分割结果
                    for i in range(len(masks)):
                        # 获取掩码数据
                        mask = masks.data[i].cpu().numpy()

                        # 如果使用了ROI，调整掩码
                        if roi is not None:
                            # 创建完整大小的掩码
                            full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                            # 将ROI区域的掩码复制到完整掩码中
                            full_mask[roi[1]:roi[3], roi[0]:roi[2]] = mask
                            mask = full_mask

                        # 获取边界框坐标
                        box = boxes.xyxy[i].cpu().numpy()

                        # 如果使用了ROI，调整坐标
                        if roi is not None:
                            box[0] += roi[0]
                            box[1] += roi[1]
                            box[2] += roi[0]
                            box[3] += roi[1]

                        # 获取置信度
                        conf = float(boxes.conf[i].cpu().numpy())

                        # 获取类别ID和名称
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        cls_name = result.names.get(cls_id, f"class_{cls_id}")

                        # 创建分割结果
                        segmentation = {
                            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                            "confidence": conf,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "mask": mask.tolist() if isinstance(mask, np.ndarray) else None
                        }

                        segmentations.append(segmentation)
                else:
                    # 没有掩码，使用边界框
                    boxes = result.boxes

                    # 只在检测到目标时记录日志
                    log_detections_if_found(result, boxes)

                    # 处理每个检测结果
                    for i in range(len(boxes)):
                        # 获取边界框坐标
                        box = boxes.xyxy[i].cpu().numpy()

                        # 如果使用了ROI，调整坐标
                        if roi is not None:
                            box[0] += roi[0]
                            box[1] += roi[1]
                            box[2] += roi[0]
                            box[3] += roi[1]

                        # 获取置信度
                        conf = float(boxes.conf[i].cpu().numpy())

                        # 获取类别ID和名称
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        cls_name = result.names.get(cls_id, f"class_{cls_id}")

                        # 创建分割结果（无掩码）
                        segmentation = {
                            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                            "confidence": conf,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "mask": None
                        }

                        segmentations.append(segmentation)

            # 后处理时间
            post_process_time = time.time() - post_process_start

            # 总时间
            total_time = time.time() - start_time

            # 返回结果
            return {
                "segmentations": segmentations,
                "pre_process_time": pre_process_time * 1000,  # 转换为毫秒
                "inference_time": inference_time * 1000,  # 转换为毫秒
                "post_process_time": post_process_time * 1000,  # 转换为毫秒
                "total_time": total_time * 1000,  # 转换为毫秒
                "annotated_image_bytes": None
            }

        except Exception as e:
            logger.error(f"YOLOE分割失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "segmentations": [],
                "pre_process_time": 0,
                "inference_time": 0,
                "post_process_time": 0,
                "total_time": (time.time() - start_time) * 1000,
                "annotated_image_bytes": None
            }

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
            "engine_type": self.engine_type,
            "yolo_version": self.yolo_version,
            "device": self.device,
            "prompt_type": self.prompt_type
        }

    def release(self) -> None:
        """释放资源"""
        self.model = None
        logger.info("YOLOE分割器资源已释放")


class YOLOETrackingAnalyzer(TrackingAnalyzer, YOLOEBaseAnalyzer):
    """YOLOE跟踪分析器"""

    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0,
                 yolo_version: int = 0, device: str = "auto",
                 tracker_type: int = 0, **kwargs):
        """
        初始化YOLOE跟踪分析器

        Args:
            model_code: 模型代码
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            device: 推理设备
            tracker_type: 跟踪器类型
            **kwargs: 其他参数
        """
        TrackingAnalyzer.__init__(self, model_code, engine_type, yolo_version, device, tracker_type, **kwargs)
        YOLOEBaseAnalyzer.__init__(self, model_code, engine_type, yolo_version, device, **kwargs)

    async def load_model(self, model_code: str) -> bool:
        """
        加载YOLOE模型

        Args:
            model_code: 模型代码

        Returns:
            bool: 是否成功加载模型
        """
        try:
            # 使用ModelLoader加载模型
            self.model = await ModelLoader.load_model(
                model_code,
                self.engine_type,
                self.yolo_version,
                self.device
            )

            # 更新当前模型代码
            self.current_model_code = model_code

            logger.info(f"YOLOE跟踪模型加载成功: {model_code}")
            return True

        except Exception as e:
            logger.error(f"YOLOE跟踪模型加载失败: {str(e)}")
            return False

    async def detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        使用YOLOE模型检测并跟踪图像

        Args:
            image: 输入图像
            **kwargs: 其他参数
                - confidence: 置信度阈值
                - iou_threshold: IOU阈值
                - classes: 类别过滤列表
                - roi: 感兴趣区域 [x1, y1, x2, y2]
                - max_detections: 最大检测数量

        Returns:
            Dict[str, Any]: 跟踪结果
        """
        start_time = time.time()

        try:
            if self.model is None:
                logger.error("模型未加载")
                return {
                    "detections": [],
                    "tracked_objects": [],
                    "pre_process_time": 0,
                    "inference_time": 0,
                    "post_process_time": 0,
                    "annotated_image_bytes": None
                }

            # 获取参数
            confidence = kwargs.get("confidence", 0.25)
            iou_threshold = kwargs.get("iou_threshold", 0.45)
            classes = kwargs.get("classes", None)
            roi = kwargs.get("roi", None)
            max_detections = kwargs.get("max_detections", 100)

            # 预处理开始时间
            pre_process_start = time.time()

            # 准备提示信息
            prompt_data = self._prepare_prompt(image)

            # 处理ROI
            if roi is not None:
                # 如果ROI是字典格式
                if isinstance(roi, dict) and all(k in roi for k in ["x1", "y1", "x2", "y2"]):
                    height, width = image.shape[:2]
                    x1 = int(roi["x1"] * width)
                    y1 = int(roi["y1"] * height)
                    x2 = int(roi["x2"] * width)
                    y2 = int(roi["y2"] * height)
                    roi = [x1, y1, x2, y2]

                # 裁剪图像到ROI
                if isinstance(roi, list) and len(roi) == 4:
                    x1, y1, x2, y2 = roi
                    # 确保坐标在图像范围内
                    height, width = image.shape[:2]
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))

                    # 裁剪图像
                    roi_image = image[y1:y2, x1:x2]
                    if roi_image.size == 0:
                        logger.warning(f"ROI裁剪后图像为空: {roi}")
                        roi_image = image
                        roi = None
                    else:
                        image = roi_image
            else:
                roi = None

            # 预处理时间
            pre_process_time = time.time() - pre_process_start

            # 推理开始时间
            inference_start = time.time()

            # 执行检测
            # 根据提示类型选择不同的检测方法
            if self._internal_prompt_type == 0:  # 无提示
                # 使用标准检测
                results = self.model(
                    image,
                    conf=confidence,
                    iou=iou_threshold,
                    classes=classes,
                    max_det=max_detections,
                    verbose=False
                )
            elif self._internal_prompt_type == 1 and self.text_prompt:  # 文本提示
                # 使用文本提示检测
                results = self.model(
                    image,
                    conf=confidence,
                    iou=iou_threshold,
                    classes=classes,
                    max_det=max_detections,
                    text_prompt=self.text_prompt,
                    verbose=False
                )
            elif self._internal_prompt_type == 2 and self.visual_prompt:  # 图像提示
                # 使用视觉提示检测
                if "mask" in prompt_data:
                    # 使用掩码提示
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        mask_prompt=prompt_data["mask"],
                        verbose=False
                    )
                elif "box" in prompt_data:
                    # 使用框提示
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        box_prompt=prompt_data["box"],
                        verbose=False
                    )
                elif "points" in prompt_data:
                    # 使用点提示
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        point_prompt=prompt_data["points"],
                        verbose=False
                    )
                else:
                    # 没有有效的视觉提示，使用标准检测
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        verbose=False
                    )
            else:
                # 默认使用标准检测
                results = self.model(
                    image,
                    conf=confidence,
                    iou=iou_threshold,
                    classes=classes,
                    max_det=max_detections,
                    verbose=False
                )

            # 推理时间
            inference_time = time.time() - inference_start

            # 后处理开始时间
            post_process_start = time.time()

            # 处理检测结果
            detections = []

            # 获取结果
            if results and len(results) > 0:
                # 获取第一帧结果
                result = results[0]

                # 获取边界框
                boxes = result.boxes

                # 只在检测到目标时记录日志
                log_detections_if_found(result, boxes)

                # 处理每个检测结果
                for i in range(len(boxes)):
                    # 获取边界框坐标
                    box = boxes.xyxy[i].cpu().numpy()

                    # 如果使用了ROI，调整坐标
                    if roi is not None:
                        box[0] += roi[0]
                        box[1] += roi[1]
                        box[2] += roi[0]
                        box[3] += roi[1]

                    # 获取置信度
                    conf = float(boxes.conf[i].cpu().numpy())

                    # 获取类别ID和名称
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = result.names.get(cls_id, f"class_{cls_id}")

                    # 创建检测结果
                    detection = {
                        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name
                    }

                    detections.append(detection)

            # 使用跟踪器更新跟踪结果
            tracked_objects = []
            if hasattr(self, "tracker") and self.tracker:
                # 将检测结果转换为跟踪器所需的格式
                tracker_detections = []
                for det in detections:
                    bbox = det["bbox"]
                    tracker_detections.append({
                        "bbox": bbox,
                        "score": det["confidence"],
                        "class_id": det["class_id"],
                        "class_name": det["class_name"]
                    })

                # 更新跟踪器
                tracked_objects = self.tracker.update(tracker_detections)

            # 后处理时间
            post_process_time = time.time() - post_process_start

            # 总时间
            total_time = time.time() - start_time

            # 返回结果
            return {
                "detections": detections,
                "tracked_objects": tracked_objects,
                "pre_process_time": pre_process_time * 1000,  # 转换为毫秒
                "inference_time": inference_time * 1000,  # 转换为毫秒
                "post_process_time": post_process_time * 1000,  # 转换为毫秒
                "total_time": total_time * 1000,  # 转换为毫秒
                "annotated_image_bytes": None
            }

        except Exception as e:
            logger.error(f"YOLOE跟踪失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "detections": [],
                "tracked_objects": [],
                "pre_process_time": 0,
                "inference_time": 0,
                "post_process_time": 0,
                "total_time": (time.time() - start_time) * 1000,
                "annotated_image_bytes": None
            }

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
            "engine_type": self.engine_type,
            "yolo_version": self.yolo_version,
            "device": self.device,
            "prompt_type": self.prompt_type,
            "tracker_type": self.tracker_type if hasattr(self, "tracker_type") else None
        }

    def release(self) -> None:
        """释放资源"""
        self.model = None
        if hasattr(self, "tracker") and self.tracker:
            # 释放跟踪器资源
            self.tracker = None
        logger.info("YOLOE跟踪器资源已释放")
