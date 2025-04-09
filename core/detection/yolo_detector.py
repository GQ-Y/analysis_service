"""
YOLO检测器模块
实现基于YOLOv8的物体检测功能
"""
import os
import base64
from pathlib import Path
import cv2
import numpy as np
import aiohttp
import torch
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Union, Tuple
from shared.utils.logger import setup_logger
from core.config import settings
import time
import asyncio
from PIL import Image, ImageDraw, ImageFont
import colorsys
from datetime import datetime
from core.exceptions import (
    InvalidInputException,
    ModelLoadException,
    ProcessingException,
    ResourceNotFoundException
)

logger = setup_logger(__name__)

class YOLODetector:
    """YOLOv8检测器实现"""
    
    def __init__(self):
        """初始化检测器"""
        self.model = None
        self.current_model_code = None
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.ANALYSIS.device != "cpu" else "cpu")
        
        # 默认配置
        self.default_confidence = settings.ANALYSIS.confidence
        self.default_iou = settings.ANALYSIS.iou
        self.default_max_det = settings.ANALYSIS.max_det
        
        # 设置保存目录
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.results_dir = self.project_root / settings.OUTPUT.save_dir
        
        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"初始化YOLO检测器，使用设备: {self.device}")
        logger.info(f"默认置信度阈值: {self.default_confidence}")
        logger.info(f"结果保存目录: {self.results_dir}")
        
    async def get_model_path(self, model_code: str) -> str:
        """获取模型路径
        
        Args:
            model_code: 模型代码,例如'model-gcc'
            
        Returns:
            str: 本地模型文件路径
            
        Raises:
            Exception: 当模型下载或保存失败时抛出异常
        """
        try:
            # 检查本地缓存
            cache_dir = os.path.join("data", "models", model_code)
            model_path = os.path.join(cache_dir, "best.pt")
            
            if os.path.exists(model_path):
                logger.info(f"找到本地缓存模型: {model_path}")
                return model_path
            
            # 本地不存在,从模型服务下载
            logger.info(f"本地未找到模型 {model_code},准备从模型服务下载...")
            
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

    async def load_model(self, model_code: str):
        """加载模型"""
        try:
            # 获取模型路径
            model_path = await self.get_model_path(model_code)
            logger.info(f"正在加载模型: {model_path}")
            
            # 加载模型
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # 设置模型参数
            self.model.conf = self.default_confidence
            self.model.iou = self.default_iou
            self.model.max_det = self.default_max_det
            
            # 更新当前模型代码
            self.current_model_code = model_code
            
            logger.info(f"模型加载成功: {model_code}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise ModelLoadException(f"模型加载失败: {str(e)}")

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

    async def _encode_result_image(
        self,
        image: np.ndarray,
        detections: List[Dict],
        return_image: bool = False
    ) -> Union[str, np.ndarray, None]:
        """将检测结果绘制到图片上"""
        try:
            # 复制图片以免修改原图
            result_image = image.copy()
            
            # 使用 PIL 处理图片，以支持中文
            img_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 加载字体
            try:
                # 尝试加载系统中文字体
                font_size = 16  # 字体大小
                font_paths = [
                    # macOS 系统字体
                    "/System/Library/Fonts/STHeiti Light.ttc",  # 华文细黑
                    "/System/Library/Fonts/STHeiti Medium.ttc", # 华文中黑
                    "/System/Library/Fonts/PingFang.ttc",       # 苹方
                    "/System/Library/Fonts/Hiragino Sans GB.ttc", # 冬青黑体
                    
                    # Windows 系统字体
                    "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑
                    "C:/Windows/Fonts/simsun.ttc",   # 宋体
                    "C:/Windows/Fonts/simhei.ttf",   # 黑体
                    
                    # Linux 系统字体
                    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    
                    # 项目本地字体（作为后备）
                    "fonts/simhei.ttf"
                ]
                
                font = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        try:
                            font = ImageFont.truetype(font_path, font_size)
                            logger.debug(f"成功加载字体: {font_path}")
                            break
                        except Exception as e:
                            logger.debug(f"尝试加载字体失败 {font_path}: {str(e)}")
                            continue
                
                if font is None:
                    logger.warning("未找到合适的中文字体，使用默认字体")
                    # 使用 PIL 默认字体
                    font = ImageFont.load_default()
                    
            except Exception as e:
                logger.warning(f"加载字体失败，使用默认字体: {str(e)}")
                font = ImageFont.load_default()
            
            def draw_detection(det: Dict, level: int = 0):
                """递归绘制检测框及其子目标
                
                Args:
                    det: 检测结果
                    level: 嵌套层级，用于确定颜色
                """
                bbox = det['bbox']
                x1, y1 = bbox['x1'], bbox['y1']
                x2, y2 = bbox['x2'], bbox['y2']
                
                # 根据对象ID或层级选择颜色
                if "obj_id" in det:
                    box_color = self._get_color_by_id(det["obj_id"])
                else:
                    # 根据层级选择不同的颜色
                    colors = [
                        (0, 255, 0),   # 绿色 - 父级
                        (255, 0, 0),   # 红色 - 一级子目标
                        (0, 0, 255),   # 蓝色 - 二级子目标
                        (255, 255, 0)  # 黄色 - 更深层级
                    ]
                    box_color = colors[min(level, len(colors) - 1)]
                
                # 绘制边界框
                draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=3)  # 加粗边框
                
                # 准备标签文本
                label = f"{det['class_name']} {det['confidence']:.2f}"
                if "obj_id" in det:
                    label += f" ID:{det['obj_id']}"
                
                # 计算文本大小
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # 确保标签不会超出图片顶部
                label_y = max(y1 - text_height - 4, 0)
                
                # 绘制标签背景
                background_shape = [(x1, label_y), (x1 + text_width + 4, label_y + text_height + 4)]
                draw.rectangle(background_shape, fill=box_color)
                
                # 绘制文本
                text_position = (x1 + 2, label_y + 2)
                draw.text(
                    text_position,
                    label,
                    font=font,
                    fill=(255, 255, 255)  # 白色文字
                )
                
                # 递归处理子目标
                for child in det.get('children', []):
                    draw_detection(child, level + 1)
            
            # 处理所有顶层检测结果
            for det in detections:
                draw_detection(det)
            
            # 转换回OpenCV格式
            result_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            if return_image:
                return result_image
            
            try:
                # 将图片编码为base64，使用较高的图片质量
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                _, buffer = cv2.imencode('.jpg', result_image, encode_params)
                if buffer is None:
                    logger.error("图片编码失败")
                    return None
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                logger.debug(f"成功生成base64图片，长度: {len(image_base64)}")
                return image_base64
            
            except Exception as e:
                logger.error(f"图片编码为base64失败: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"处理结果图片失败: {str(e)}")
            return None

    async def detect(self, image, config: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """执行检测
        
        Args:
            image: 输入图片
            config: 检测配置，包含以下可选参数：
                - confidence: 置信度阈值
                - iou: IoU阈值
                - classes: 需要检测的类别ID列表
                - roi: 感兴趣区域，用于过滤检测结果，不再用于裁剪原始图像
                - roi_type: ROI类型（1=矩形，2=多边形，3=直线，4=圆形）
                - imgsz: 输入图片大小
                - nested_detection: 是否进行嵌套检测
        """
        try:
            if self.model is None:
                model_code = config.get("model_code", self.current_model_code)
                if not model_code:
                    raise Exception("未指定模型代码")
                await self.load_model(model_code)
            
            # 使用配置参数或默认值
            config = config or {}
            conf = config.get('confidence', self.default_confidence)
            iou = config.get('iou', self.default_iou)
            classes = config.get('classes', None)
            roi = config.get('roi', None)
            imgsz = config.get('imgsz', None)
            nested_detection = config.get('nested_detection', False)
            
            # 确保置信度和IoU阈值有效
            if conf is None:
                conf = self.default_confidence
            if iou is None:
                iou = self.default_iou
                
            logger.info(f"检测配置 - 置信度: {conf}, IoU: {iou}, 类别: {classes}, ROI: {roi}, 图片大小: {imgsz}, 嵌套检测: {nested_detection}")
            
            # 处理图片大小
            if imgsz:
                resized_image = cv2.resize(image, (imgsz, imgsz))
            else:
                resized_image = image.copy()
            
            # 执行推理 - 在整个图像上进行检测
            results = self.model(
                resized_image,
                conf=conf,
                iou=iou,
                classes=classes
            )
            
            # 处理检测结果
            h, w = image.shape[:2]  # 原始图像尺寸，用于ROI过滤
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = result.names[cls]
                    
                    # 如果检测是在调整大小后的图像上进行的，需要调整坐标
                    if imgsz:
                        scale_x = w / imgsz
                        scale_y = h / imgsz
                        bbox[0] *= scale_x
                        bbox[1] *= scale_y
                        bbox[2] *= scale_x
                        bbox[3] *= scale_y
                    
                    detection = {
                        "bbox": {
                            "x1": float(bbox[0]),
                            "y1": float(bbox[1]),
                            "x2": float(bbox[2]),
                            "y2": float(bbox[3])
                        },
                        "confidence": conf,
                        "class_id": cls,
                        "class_name": name,
                        "area": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),  # 计算面积
                        "parent_idx": None,  # 用于存储父目标的索引
                        "children": []  # 用于存储子目标列表
                    }
                    detections.append(detection)
            
            # 处理嵌套检测
            if nested_detection and len(detections) > 1:
                logger.info("开始处理嵌套检测...")
                
                # 按面积从大到小排序
                detections.sort(key=lambda x: x['area'], reverse=True)
                
                # 检查嵌套关系
                for i, parent in enumerate(detections):
                    parent_bbox = parent['bbox']
                    
                    # 检查其他目标是否在当前目标内部
                    for j, child in enumerate(detections):
                        if i != j:  # 不与自己比较
                            child_bbox = child['bbox']
                            
                            # 计算重叠区域
                            overlap_x1 = max(parent_bbox['x1'], child_bbox['x1'])
                            overlap_y1 = max(parent_bbox['y1'], child_bbox['y1'])
                            overlap_x2 = min(parent_bbox['x2'], child_bbox['x2'])
                            overlap_y2 = min(parent_bbox['y2'], child_bbox['y2'])
                            
                            # 如果有重叠
                            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                                # 计算重叠区域面积
                                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                                child_area = (child_bbox['x2'] - child_bbox['x1']) * (child_bbox['y2'] - child_bbox['y1'])
                                
                                # 如果子目标的90%以上区域在父目标内部
                                if overlap_area / child_area > 0.9:
                                    child['parent_idx'] = i
                                    parent['children'].append(child)
                
                # 只保留没有父目标的检测结果
                detections = [det for det in detections if det['parent_idx'] is None]
            
            # 如果设置了ROI，根据ROI类型过滤检测结果
            if roi:
                # 默认为矩形ROI (roi_type=1)
                roi_type = config.get('roi_type', 1)
                # 优先使用roi对象中的roi_type，这是经过规范化处理的
                roi_type_from_roi = roi.get("roi_type", None)
                if roi_type_from_roi is not None:
                    roi_type = roi_type_from_roi
                detections = self._filter_by_roi(detections, roi, roi_type, h, w)
            
            return detections
                    
        except Exception as e:
            logger.error(f"检测失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise ProcessingException(f"检测失败: {str(e)}")

    def _filter_by_roi(self, detections, roi, roi_type, img_height, img_width):
        """
        根据ROI过滤检测结果
        
        Args:
            detections: 检测结果列表
            roi: ROI配置
            roi_type: ROI类型 (1=矩形, 2=多边形, 3=直线, 4=圆形)
            img_height: 图像高度
            img_width: 图像宽度
            
        Returns:
            过滤后的检测结果列表
        """
        try:
            if not roi:
                return detections
                
            # 准备原始图像尺寸
            image_size = (img_width, img_height)
            filtered_detections = []
            
            # 将归一化的ROI转换为像素坐标
            pixel_roi = {}
            if roi.get("normalized", True):
                for key, value in roi.items():
                    if key == "x1" or key == "x2":
                        pixel_roi[key] = int(value * img_width)
                    elif key == "y1" or key == "y2":
                        pixel_roi[key] = int(value * img_height)
                    elif key == "center" and isinstance(value, list) and len(value) == 2:
                        pixel_roi[key] = [int(value[0] * img_width), int(value[1] * img_height)]
                    elif key == "radius":
                        # 对于圆形ROI，使用平均半径
                        pixel_roi[key] = int(value * (img_width + img_height) / 2)
                    else:
                        pixel_roi[key] = value
            else:
                pixel_roi = roi
            
            # 根据ROI类型过滤
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
                if roi_type == 1:  # 矩形ROI
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
                    # 使用矩形包围盒近似，实际项目可能需要更准确的多边形重叠计算
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
                
                elif roi_type == 3:  # 直线ROI
                    # 对于直线ROI，只要边界框与直线有任何交点，就保留此检测结果
                    # 从roi中获取点，优先使用规范化的points列表
                    points = roi.get("points", [])
                    if not points and "points" in roi:
                        points = roi["points"]
                    
                    if len(points) >= 2:  # 至少需要两个点才能形成线段
                        # 获取线段的两个端点
                        if isinstance(points[0], dict) and 'x' in points[0] and 'y' in points[0]:
                            p1 = (int(points[0]['x'] * img_width), int(points[0]['y'] * img_height))
                            p2 = (int(points[1]['x'] * img_width), int(points[1]['y'] * img_height))
                        else:
                            p1 = (int(points[0][0] * img_width), int(points[0][1] * img_height))
                            p2 = (int(points[1][0] * img_width), int(points[1][1] * img_height))
                        
                        # 检查线段是否与检测框相交
                        # 简化处理：检查线段是否与检测框的任意边相交
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
                
                elif roi_type == 4:  # 圆形ROI
                    # 对于圆形ROI，只要有50%以上的目标在圆内，就保留此检测结果
                        if len(points) >= 2:  # 至少需要两个点才能形成线段
                            # 获取线段的两个端点
                            if isinstance(points[0], dict) and 'x' in points[0] and 'y' in points[0]:
                                p1 = (int(points[0]['x'] * img_width), int(points[0]['y'] * img_height))
                                p2 = (int(points[1]['x'] * img_width), int(points[1]['y'] * img_height))
                            else:
                                p1 = (int(points[0][0] * img_width), int(points[0][1] * img_height))
                                p2 = (int(points[1][0] * img_width), int(points[1][1] * img_height))
                            
                            # 检查线段是否与检测框相交
                            # 简化处理：检查线段是否与检测框的任意边相交
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
                
                elif roi_type == 4:  # 圆形ROI
                    # 对于圆形ROI，只要有50%以上的目标在圆内，就保留此检测结果
                    if "center" in pixel_roi and "radius" in pixel_roi:
                        center = pixel_roi["center"]
                        radius = pixel_roi["radius"]
                        
                        # 检测框的中心点
                        box_center_x = (pixel_bbox["x1"] + pixel_bbox["x2"]) / 2
                        box_center_y = (pixel_bbox["y1"] + pixel_bbox["y2"]) / 2
                        
                        # 计算中心点到圆心的距离
                        distance = ((box_center_x - center[0]) ** 2 + (box_center_y - center[1]) ** 2) ** 0.5
                        
                        # 如果检测框中心在圆内，并且检测框面积的50%以上在圆内
                        # 这里使用简化逻辑，实际可能需要更精确的圆与矩形重叠计算
                        if distance <= radius:
                            # 假设有50%以上的目标在圆内
                            filtered_detections.append(detection)
            
            return filtered_detections
                
        except Exception as e:
            logger.error(f"根据ROI过滤检测结果失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 如果过滤失败，返回原始检测结果
            return detections
    
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
            # 生成带检测结果的图片
            logger.info("开始生成检测结果图片...")
            result_image = await self._encode_result_image(image, detections, return_image=True)
            if result_image is None:
                logger.error("生成检测结果图片失败")
                return None
                
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_prefix = f"{task_name}_" if task_name else ""
            filename = f"{task_prefix}{timestamp}.jpg"
            
            # 确保每天的结果保存在单独的目录中
            date_dir = self.results_dir / datetime.now().strftime("%Y%m%d")
            os.makedirs(date_dir, exist_ok=True)
            
            # 保存图片
            file_path = date_dir / filename
            success = cv2.imwrite(str(file_path), result_image)
            if not success:
                logger.error("保存图片失败")
                return None
            
            # 返回相对于项目根目录的路径
            relative_path = file_path.relative_to(self.project_root)
            logger.info(f"图片已保存: {relative_path}")
            return str(relative_path)
            
        except Exception as e:
            logger.error(f"保存结果图片失败: {str(e)}")
            return None 

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """绘制检测结果到图像上
        
        Args:
            image: 输入图像
            detections: 检测结果列表
            
        Returns:
            np.ndarray: 带有检测框和标签的图像
        """
        try:
            # 创建异步事件循环来调用_encode_result_image方法
            loop = asyncio.new_event_loop()
            result_image = loop.run_until_complete(
                self._encode_result_image(image, detections, return_image=True)
            )
            loop.close()
            
            # 检查结果
            if result_image is None:
                logger.error("生成检测结果图像失败")
                return image  # 返回原始图像
                
            return result_image
        except Exception as e:
            logger.error(f"绘制检测结果时出错: {str(e)}")
            logger.exception(e)
            return image  # 发生错误时返回原始图像 