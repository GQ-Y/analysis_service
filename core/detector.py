"""
检测器模块
封装YOLO模型的检测功能
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
import random
from datetime import datetime
from loguru import logger
import httpx
import colorsys
from core.tracker import create_tracker, BaseTracker
from core.redis_manager import RedisManager
from core.task_queue import TaskQueue, TaskStatus
from core.exceptions import (
    InvalidInputException,
    ModelLoadException,
    ProcessingException,
    ResourceNotFoundException
)
import uuid

logger = setup_logger(__name__)

class CallbackData:
    """标准回调数据结构"""
    def __init__(self, 
                 camera_device_type: int = 1,
                 camera_device_stream_url: str = "",
                 camera_device_status: int = 1,
                 camera_device_group: str = "",
                 camera_device_gps: str = "",
                 camera_device_id: int = 0,
                 camera_device_name: str = "",
                 algorithm_id: int = 0,
                 algorithm_name: str = "",
                 algorithm_name_en: str = "",
                 data_id: str = "",
                 parameter: Dict = None,
                 picture: str = "",
                 src_url: str = "",
                 alarm_url: str = "",
                 task_id: int = 0,
                 camera_id: int = 0,
                 camera_url: str = "",
                 camera_name: str = "",
                 timestamp: int = 0,
                 image_width: int = 1920,
                 image_height: int = 1080,
                 src_pic_data: str = "",
                 src_pic_name: str = "",
                 alarm_pic_name: str = "",
                 src: str = "",
                 alarm: str = "",
                 alarm_pic_data: str = "",
                 other: str = "",
                 result: str = "",
                 extra_info: List = None,
                 result_data: Dict = None,
                 degree: int = 3):
        
        self.camera_device_type = camera_device_type
        self.camera_device_stream_url = camera_device_stream_url
        self.camera_device_status = camera_device_status
        self.camera_device_group = camera_device_group
        self.camera_device_gps = camera_device_gps
        self.camera_device_id = camera_device_id
        self.camera_device_name = camera_device_name
        self.algorithm_id = algorithm_id
        self.algorithm_name = algorithm_name
        self.algorithm_name_en = algorithm_name_en
        self.data_id = data_id
        self.parameter = parameter or {}
        self.picture = picture
        self.src_url = src_url
        self.alarm_url = alarm_url
        self.task_id = task_id
        self.camera_id = camera_id
        self.camera_url = camera_url
        self.camera_name = camera_name
        self.timestamp = timestamp or int(time.time())
        self.image_width = image_width
        self.image_height = image_height
        self.src_pic_data = src_pic_data
        self.src_pic_name = src_pic_name
        self.alarm_pic_name = alarm_pic_name
        self.src = src
        self.alarm = alarm
        self.alarm_pic_data = alarm_pic_data
        self.other = other
        self.result = result
        self.extra_info = extra_info or []
        self.result_data = result_data or {}
        self.degree = degree

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "cameraDeviceType": self.camera_device_type,
            "cameraDeviceStreamUrl": self.camera_device_stream_url,
            "cameraDeviceStatus": self.camera_device_status,
            "cameraDeviceGroup": self.camera_device_group,
            "cameraDeviceGps": self.camera_device_gps,
            "cameraDeviceId": self.camera_device_id,
            "cameraDeviceName": self.camera_device_name,
            "algorithmId": self.algorithm_id,
            "algorithmName": self.algorithm_name,
            "algorithmNameEn": self.algorithm_name_en,
            "dataID": self.data_id,
            "parameter": self.parameter,
            "picture": self.picture,
            "srcUrl": self.src_url,
            "alarmUrl": self.alarm_url,
            "taskId": self.task_id,
            "cameraId": self.camera_id,
            "cameraUrl": self.camera_url,
            "cameraName": self.camera_name,
            "timestamp": self.timestamp,
            "imageWidth": self.image_width,
            "imageHeight": self.image_height,
            "srcPicData": self.src_pic_data,
            "srcPicName": self.src_pic_name,
            "alarmPicName": self.alarm_pic_name,
            "src": self.src,
            "alarm": self.alarm,
            "alarmPicData": self.alarm_pic_data,
            "other": self.other,
            "result": self.result,
            "extraInfo": self.extra_info,
            "resultData": self.result_data,
            "degree": self.degree
        }

class YOLODetector:
    """YOLO检测器"""
    
    def __init__(self):
        """初始化检测器"""
        self.model = None
        self.current_model_code = None
        self.tracker: Optional[BaseTracker] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.ANALYSIS_DEVICE != "cpu" else "cpu")
        
        # Redis相关
        self.redis = RedisManager()
        self.task_queue = TaskQueue()
        
        # 模型服务配置
        self.model_service_url = settings.MODEL_SERVICE_URL
        self.api_prefix = settings.MODEL_SERVICE_API_PREFIX
        
        # 默认配置
        self.default_confidence = settings.ANALYSIS_CONFIDENCE
        self.default_iou = settings.ANALYSIS_IOU
        self.default_max_det = settings.ANALYSIS_MAX_DET
        
        # 设置保存目录
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.results_dir = self.project_root / settings.OUTPUT_SAVE_DIR
        
        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"Model service URL: {self.model_service_url}")
        logger.info(f"Model service API prefix: {self.api_prefix}")
        logger.info(f"Default confidence: {self.default_confidence}")
        logger.info(f"Results directory: {self.results_dir}")
        
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
            api_url = f"{self.model_service_url}{self.api_prefix}/models/download?code={model_code}"
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
            logger.info(f"Loading model from: {model_path}")
            
            # 加载模型
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # 设置模型参数
            self.model.conf = self.default_confidence
            self.model.iou = self.default_iou
            self.model.max_det = self.default_max_det
            
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    async def _download_image(self, url: str) -> Optional[np.ndarray]:
        """下载图片并转换为 numpy 数组
        
        支持以下格式：
        - HTTP/HTTPS URL
        - Base64编码的图片数据（以 'data:image/' 开头）
        - Blob URL（以 'blob:' 开头）
        """
        try:
            # 处理 Base64 编码的图片数据
            if url.startswith('data:image/'):
                try:
                    # 提取实际的 base64 数据
                    base64_data = url.split(',')[1]
                    image_data = base64.b64decode(base64_data)
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        logger.error(f"Failed to decode base64 image data")
                        return None
                    return img
                except Exception as e:
                    logger.error(f"Error processing base64 image: {str(e)}")
                    return None

            # 处理 HTTP/HTTPS URL
            elif url.startswith(('http://', 'https://')):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as response:
                            if response.status == 200:
                                image_data = await response.read()
                                nparr = np.frombuffer(image_data, np.uint8)
                                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                if img is None:
                                    logger.error(f"Failed to decode image from URL: {url}")
                                    return None
                                return img
                            else:
                                logger.error(f"Failed to download image from URL: {url}, status: {response.status}")
                                return None
                except Exception as e:
                    logger.error(f"Error downloading image from URL {url}: {str(e)}")
                    return None

            # 处理 Blob URL
            elif url.startswith('blob:'):
                try:
                    # 从请求中获取 blob 数据
                    # 注意：这里需要前端将 blob 数据转换为 base64 或直接上传文件
                    # 因为后端无法直接访问浏览器的 blob URL
                    logger.error(f"Blob URL is not supported directly. Please convert to base64 or upload file: {url}")
                    return None
                except Exception as e:
                    logger.error(f"Error processing blob URL {url}: {str(e)}")
                    return None

            else:
                logger.error(f"Unsupported image URL format: {url}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error processing image {url}: {str(e)}")
            return None

    def _get_color_by_id(self, track_id: int) -> Tuple[int, int, int]:
        """根据跟踪ID生成固定的颜色
        
        Args:
            track_id: 跟踪ID
            
        Returns:
            Tuple[int, int, int]: RGB颜色值
        """
        # 使用黄金比例法生成不同的色相值
        golden_ratio = 0.618033988749895
        hue = (track_id * golden_ratio) % 1.0
        
        # 转换HSV到RGB（固定饱和度和明度以获得鲜艳的颜色）
        rgb = tuple(round(x * 255) for x in colorsys.hsv_to_rgb(hue, 0.8, 0.95))
        return rgb

    async def _encode_result_image(
        self,
        image: np.ndarray,
        detections: List[Dict],
        return_image: bool = False,
        draw_tracks: bool = False,  # 新增: 是否绘制轨迹
        draw_track_ids: bool = False  # 新增: 是否绘制跟踪ID
    ) -> Union[str, np.ndarray, None]:
        """将检测结果绘制到图片上"""
        try:
            # 复制图片以免修改原图
            result_image = image.copy()
            
            # 使用 PIL 处理图片，以支持中文
            img_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 加载中文字体
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
                            logger.info(f"成功加载字体: {font_path}")
                            break
                        except Exception as e:
                            logger.debug(f"尝试加载字体失败 {font_path}: {str(e)}")
                            continue
                
                if font is None:
                    logger.warning("未找到合适的中文字体，使用默认字体")
                    # 使用 PIL 默认字体，但增加字体大小以提高可读性
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
                nonlocal draw, img_pil  # 声明使用外部的draw和img_pil变量
                
                bbox = det['bbox']
                x1, y1 = bbox['x1'], bbox['y1']
                x2, y2 = bbox['x2'], bbox['y2']
                
                # 根据跟踪ID或层级选择颜色
                if "track_id" in det:
                    box_color = self._get_color_by_id(det["track_id"])
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
                label_parts = []
                label_parts.append(f"{det['class_name']} {det['confidence']:.2f}")
                if draw_track_ids and "track_id" in det:
                    label_parts.append(f"ID:{det['track_id']}")
                label = " | ".join(label_parts)
                
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
                
                # 如果启用了轨迹绘制，且有轨迹信息
                if draw_tracks and det.get("track_info", {}).get("trajectory"):
                    # 先转换回OpenCV格式处理轨迹
                    temp_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    
                    trajectory = det["track_info"]["trajectory"]
                    # 只绘制最近的N个点
                    max_trajectory_points = 30
                    if len(trajectory) > 1:
                        points = trajectory[-max_trajectory_points:]
                        for i in range(len(points) - 1):
                            pt1 = points[i]
                            pt2 = points[i + 1]
                            # 计算轨迹线的中心点
                            pt1_center = (
                                int((pt1[0] + pt1[2]) / 2),
                                int((pt1[1] + pt1[3]) / 2)
                            )
                            pt2_center = (
                                int((pt2[0] + pt2[2]) / 2),
                                int((pt2[1] + pt2[3]) / 2)
                            )
                            # 绘制轨迹线，使用半透明效果
                            alpha = 0.5
                            overlay = temp_image.copy()
                            cv2.line(
                                overlay,
                                pt1_center,
                                pt2_center,
                                box_color,
                                2
                            )
                            cv2.addWeighted(
                                overlay,
                                alpha,
                                temp_image,
                                1 - alpha,
                                0,
                                temp_image
                            )
                    
                    # 转换回PIL格式
                    img_pil = Image.fromarray(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
            
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
                logger.error(f"图片编码为base64失败: {str(e)}", exc_info=True)
                return None
                
        except Exception as e:
            logger.error(f"处理结果图片失败: {str(e)}", exc_info=True)
            return None

    async def detect(self, image, config: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """执行检测
        
        Args:
            image: 输入图片
            config: 检测配置，包含以下可选参数：
                - confidence: 置信度阈值
                - iou: IoU阈值
                - classes: 需要检测的类别ID列表
                - roi: 感兴趣区域，格式为{x1, y1, x2, y2}，值为0-1的归一化坐标
                - imgsz: 输入图片大小
                - nested_detection: 是否进行嵌套检测
        """
        try:
            if self.model is None:
                model_code = self.current_model_code
                if not model_code:
                    raise Exception("No model code specified")
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
            
            # 处理ROI
            original_shape = None
            if roi:
                h, w = image.shape[:2]
                original_shape = (h, w)
                x1 = int(roi['x1'] * w)
                y1 = int(roi['y1'] * h)
                x2 = int(roi['x2'] * w)
                y2 = int(roi['y2'] * h)
                image = image[y1:y2, x1:x2]
            
            # 处理图片大小
            if imgsz:
                image = cv2.resize(image, (imgsz, imgsz))
            
            # 执行推理
            results = self.model(
                image,
                conf=conf,
                iou=iou,
                classes=classes
            )
            
            # 处理检测结果
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = result.names[cls]
                    
                    # 如果使用了ROI，需要调整坐标
                    if roi and original_shape:
                        h, w = original_shape
                        bbox[0] = bbox[0] / w * (roi['x2'] - roi['x1']) * w + roi['x1'] * w
                        bbox[1] = bbox[1] / h * (roi['y2'] - roi['y1']) * h + roi['y1'] * h
                        bbox[2] = bbox[2] / w * (roi['x2'] - roi['x1']) * w + roi['x1'] * w
                        bbox[3] = bbox[3] / h * (roi['y2'] - roi['y1']) * h + roi['y1'] * h
                    
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
            
            return detections
                    
        except Exception as e:
            logger.error(f"检测失败: {str(e)}", exc_info=True)
            raise

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
            logger.error(f"保存结果图片失败: {str(e)}", exc_info=True)
            return None

    async def detect_images(
        self,
        model_code: str,
        image_urls: List[str],
        callback_urls: str = None,
        is_base64: bool = False,
        config: Optional[Dict] = None,
        task_name: Optional[str] = None,
        enable_callback: bool = True,
        save_result: bool = False
    ) -> Dict[str, Any]:
        """检测图片"""
        if not model_code:
            raise ValueError("No model code specified")
            
        task_id = f"img_{int(time.time() * 1000)}"
        try:
            # 初始化任务信息
            task_info = {
                'id': task_id,
                'task_name': task_name,
                'model_code': model_code,
                'image_urls': image_urls,
                'callback_urls': callback_urls,
                'enable_callback': enable_callback,
                'save_result': save_result,
                'config': config,
                'status': TaskStatus.PROCESSING,
                'start_time': datetime.now().isoformat(),
                'type': 'image'
            }
            
            # 保存任务信息到Redis
            await self.task_queue.add_task(task_info)
            
            # 加载模型
            await self.load_model(model_code)
            
            results = []
            for url in image_urls:
                # 下载图片
                image = await self._download_image(url)
                if image is None:
                    continue
                    
                # 执行检测
                detections = await self.detect(image, config=config)
                
                # 处理结果图
                result_image = None
                if is_base64:
                    result_image = await self._encode_result_image(image, detections)
                    
                # 保存结果
                saved_path = None
                if save_result:
                    logger.info("尝试保存检测结果图片...")
                    saved_path = await self._save_result_image(image, detections, task_name)
                    if saved_path:
                        logger.info(f"成功保存检测结果图片，路径: {saved_path}")
                    else:
                        logger.error("保存检测结果图片失败")
                    
                result_dict = {
                    'image_url': url,
                    'detections': detections,
                    'result_image': result_image,
                    'saved_path': saved_path
                }
                results.append(result_dict)
            
            # 更新任务状态和结果
            task_info.update({
                'status': TaskStatus.COMPLETED,
                'end_time': datetime.now().isoformat(),
                'results': results
            })
            await self._save_task_result(task_id, task_info)
            
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Image detection failed: {str(e)}", exc_info=True)
            await self._fail_task(task_id, str(e))
            raise

    async def start_stream_analysis(
        self,
        task_id: str,
        model_code: str,
        stream_url: str,
        callback_urls: Optional[str] = None,
        system_callback_url: Optional[str] = None,  # 新增：系统回调URL
        config: Optional[Dict[str, Any]] = None,
        task_name: Optional[str] = None,
        enable_callback: bool = False,
        save_result: bool = False,
        analysis_type: str = "detection"
    ) -> Dict[str, Any]:
        """启动流分析任务
        
        Args:
            task_id: 任务ID
            model_code: 模型代码
            stream_url: 流URL
            callback_urls: 回调地址，多个用逗号分隔
            system_callback_url: 系统回调URL，用于系统级回调
            config: 分析配置
            task_name: 任务名称
            enable_callback: 是否启用回调
            save_result: 是否保存结果
            analysis_type: 分析类型
            
        Returns:
            Dict[str, Any]: 任务信息
        """
        try:
            logger.info(f"YOLODetector.start_stream_analysis - 开始启动任务: task_id={task_id}")
            logger.info(f"YOLODetector.start_stream_analysis - 参数: model_code={model_code}, stream_url={stream_url}")
            logger.info(f"YOLODetector.start_stream_analysis - 回调: system={system_callback_url}, user={callback_urls}, enabled={enable_callback}")
            logger.info(f"YOLODetector.start_stream_analysis - 配置: task_name={task_name}, save_result={save_result}, analysis_type={analysis_type}")
            
            if config:
                logger.info(f"YOLODetector.start_stream_analysis - 任务配置: {config}")
            
            # 生成任务ID
            if not task_id:
                task_id = f"str_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                logger.info(f"YOLODetector.start_stream_analysis - 生成任务ID: {task_id}")
            
            # 检查任务是否已存在
            existing_task = await self._get_task_info(task_id)
            if existing_task:
                # 如果任务已存在但状态是完成或失败，则重置任务
                # 修复：使用TaskStatus.FAILED替代不存在的TaskStatus.ERROR
                if existing_task.get("status") in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED]:
                    logger.info(f"YOLODetector.start_stream_analysis - 任务 {task_id} 已存在但已完成或失败，重置任务")
                # 否则返回已存在的任务信息
                else:
                    logger.info(f"YOLODetector.start_stream_analysis - 任务 {task_id} 已存在，返回已存在的任务信息: {existing_task}")
                    return existing_task
            else:
                logger.info(f"YOLODetector.start_stream_analysis - 任务 {task_id} 不存在，将创建新任务")
            
            # 创建任务并保存信息
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            task_info = {
                "task_id": task_id,
                "model_code": model_code,
                "stream_url": stream_url,
                "callback_urls": callback_urls,
                "system_callback_url": system_callback_url,  # 保存系统回调URL
                "config": config,
                "task_name": task_name,
                "enable_callback": enable_callback,
                "save_result": save_result,
                "status": TaskStatus.PROCESSING,
                "start_time": start_time,
                "progress": 0,
                "analysis_type": analysis_type,
                "frame_count": 0,
                "detection_count": 0,
                "frame_width": 0,
                "frame_height": 0
            }
            
            # 保存任务信息
            logger.info(f"YOLODetector.start_stream_analysis - 保存任务信息到Redis")
            try:
                await self._update_task_info(task_id, task_info)
                logger.info(f"YOLODetector.start_stream_analysis - 任务信息保存成功")
            except Exception as e:
                logger.error(f"YOLODetector.start_stream_analysis - 保存任务信息失败: {str(e)}")
                raise
            
            # 启动处理任务
            logger.info(f"YOLODetector.start_stream_analysis - 创建异步任务处理流")
            asyncio.create_task(self._process_stream_analysis(task_id))
            
            logger.info(f"YOLODetector.start_stream_analysis - 流分析任务启动成功: task_id={task_id}")
            
            # 检查任务信息是否正确保存
            try:
                saved_task = await self._get_task_info(task_id)
                if saved_task:
                    logger.info(f"YOLODetector.start_stream_analysis - 验证任务已保存: {saved_task}")
                else:
                    logger.warning(f"YOLODetector.start_stream_analysis - 警告：验证时未找到任务 {task_id}")
            except Exception as e:
                logger.error(f"YOLODetector.start_stream_analysis - 验证任务保存时出错: {str(e)}")
            
            return task_info
            
        except Exception as e:
            logger.error(f"YOLODetector.start_stream_analysis - 启动流分析任务失败: {str(e)}", exc_info=True)
            raise

    async def _process_stream_analysis(
        self,
        task_id: str
    ) -> None:
        """处理流分析任务"""
        try:
            # 获取任务信息
            task_info = await self._get_task_info(task_id)
            if not task_info:
                logger.error(f"任务 {task_id} 不存在")
                return
            
            # 获取任务参数
            model_code = task_info["model_code"]
            stream_url = task_info["stream_url"]
            callback_urls = task_info.get("callback_urls")
            system_callback_url = task_info.get("system_callback_url")
            enable_callback = task_info.get("enable_callback", False)
            save_result = task_info.get("save_result", False)
            config = task_info.get("config", {})
            task_name = task_info.get("task_name")
            analysis_type = task_info.get("analysis_type", "detection")
            
            # 确保配置是字典
            if config is None:
                config = {}
                
            # 设置默认参数
            config.setdefault("confidence", self.default_confidence)
            config.setdefault("iou", self.default_iou)
            config.setdefault("max_det", self.default_max_det)
            
            # 加载模型
            await self.load_model(model_code)
            
            # 打开流
            logger.info(f"开始处理流 {stream_url}")
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                logger.error(f"无法打开流: {stream_url}")
                task_info["status"] = TaskStatus.FAILED
                task_info["error_message"] = f"无法打开流: {stream_url}"
                await self._update_task_info(task_id, task_info)
                return
            
            # 获取流信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25  # 默认为25fps
            
            # 更新任务信息
            task_info["frame_width"] = width
            task_info["frame_height"] = height
            task_info["fps"] = fps
            await self._update_task_info(task_id, task_info)
            
            # 设置帧处理计数器
            frame_count = 0
            last_process_time = time.time()
            process_interval = 1.0  # 默认处理间隔，单位秒
            
            # 设置回调参数
            callback_interval = 10  # 回调间隔，单位帧
            last_callback_frame = 0
            
            # 检测结果缓存
            last_detections = []
            
            # 根据分析类型初始化相关组件
            if analysis_type == "tracking":
                tracker_type = config.get("tracker_type", "sort")
                self.tracker = create_tracker(tracker_type)
                logger.info(f"启用目标跟踪，跟踪器类型: {tracker_type}")
            
            # 主循环
            while not await self._should_stop(task_id):
                try:
                    # 读取一帧
                    ret, frame = cap.read()
                    if not ret:
                        # 对于大多数流，读取失败通常意味着流结束或出现错误
                        # 重新打开流继续处理
                        logger.warning(f"读取帧失败，尝试重新打开流: {stream_url}")
                        cap.release()
                        await asyncio.sleep(2)  # 等待一下再重试
                        cap = cv2.VideoCapture(stream_url)
                        if not cap.isOpened():
                            logger.error(f"重新打开流失败: {stream_url}")
                            break
                        continue
                    
                    # 更新帧计数
                    frame_count += 1
                    task_info["frame_count"] = frame_count
                    
                    # 控制处理频率
                    current_time = time.time()
                    if current_time - last_process_time < process_interval and frame_count > 1:
                        continue
                    
                    last_process_time = current_time
                    
                    # 执行检测
                    detections = await self._process_frame(frame, self.model, config)
                    
                    # 处理不同类型的ROI
                    roi_type = config.get("roi_type", 0)
                    roi = config.get("roi")
                    
                    # 如果配置了ROI，过滤检测结果
                    if roi and roi_type > 0:
                        # 矩形ROI
                        if roi_type == 1 and all(k in roi for k in ["x1", "y1", "x2", "y2"]):
                            x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
                            # 转换为像素坐标
                            x1_px, y1_px = int(x1 * width), int(y1 * height)
                            x2_px, y2_px = int(x2 * width), int(y2 * height)
                            # 过滤检测结果
                            filtered_detections = []
                            for det in detections:
                                bbox = det["bbox"]
                                # 计算中心点
                                cx = bbox["x"] + bbox["width"] / 2
                                cy = bbox["y"] + bbox["height"] / 2
                                # 检查是否在ROI内
                                if x1_px <= cx <= x2_px and y1_px <= cy <= y2_px:
                                    filtered_detections.append(det)
                            detections = filtered_detections
                            
                        # 多边形ROI
                        elif roi_type == 2 and "points" in roi:
                            # 转换为像素坐标
                            points = [(int(p[0] * width), int(p[1] * height)) for p in roi["points"]]
                            points_array = np.array(points, np.int32)
                            points_array = points_array.reshape((-1, 1, 2))
                            
                            # 过滤检测结果
                            filtered_detections = []
                            for det in detections:
                                bbox = det["bbox"]
                                # 计算中心点
                                cx = int(bbox["x"] + bbox["width"] / 2)
                                cy = int(bbox["y"] + bbox["height"] / 2)
                                # 检查点是否在多边形内
                                result = cv2.pointPolygonTest(points_array, (cx, cy), False)
                                if result >= 0:  # 点在多边形内或在边界上
                                    filtered_detections.append(det)
                            detections = filtered_detections
                            
                        # 线段ROI
                        elif roi_type == 3 and "points" in roi and len(roi["points"]) == 2:
                            # 转换为像素坐标
                            points = [(int(p[0] * width), int(p[1] * height)) for p in roi["points"]]
                            line_start, line_end = points
                            line_vec = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
                            line_length = np.linalg.norm(line_vec)
                            
                            # 过滤检测结果 - 对于线段，检测目标是否与线段相交
                            filtered_detections = []
                            for det in detections:
                                bbox = det["bbox"]
                                # 定义边界框的四个角点
                                box_left = bbox["x"]
                                box_top = bbox["y"]
                                box_right = box_left + bbox["width"]
                                box_bottom = box_top + bbox["height"]
                                
                                # 检查线段是否与边界框相交
                                # 使用简化的相交检测：检查线段的两个端点是否在边界框的两侧
                                intersects = False
                                
                                # 计算点到线段的距离
                                def point_to_line_distance(point, line_start, line_end):
                                    # 线段向量
                                    line_vec = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
                                    # 点到线段起点的向量
                                    point_vec = np.array([point[0] - line_start[0], point[1] - line_start[1]])
                                    # 计算点在线段上的投影长度
                                    line_length = np.linalg.norm(line_vec)
                                    projection = np.dot(point_vec, line_vec) / line_length
                                    
                                    # 如果投影在线段外，返回到端点的距离
                                    if projection < 0:
                                        return np.linalg.norm(point_vec)
                                    elif projection > line_length:
                                        return np.linalg.norm(np.array([point[0] - line_end[0], point[1] - line_end[1]]))
                                    
                                    # 如果投影在线段上，计算点到线的垂直距离
                                    unit_line_vec = line_vec / line_length
                                    # 垂直向量
                                    perp_vec = np.array([-unit_line_vec[1], unit_line_vec[0]])
                                    # 垂直距离
                                    distance = abs(np.dot(point_vec, perp_vec))
                                    return distance
                                
                                # 计算边界框中心到线段的距离
                                center = (box_left + box_right) / 2, (box_top + box_bottom) / 2
                                distance = point_to_line_distance(center, line_start, line_end)
                                
                                # 如果距离小于阈值，认为相交
                                threshold = (bbox["width"] + bbox["height"]) / 4  # 使用边界框尺寸的平均值的一半作为阈值
                                if distance < threshold:
                                    filtered_detections.append(det)
                                    
                            detections = filtered_detections
                    
                    # 更新检测计数
                    task_info["detection_count"] = len(detections)
                    
                    # 是否需要执行用户回调
                    need_user_callback = enable_callback and callback_urls and (
                        frame_count - last_callback_frame >= callback_interval or 
                        frame_count == 1  # 第一帧始终回调
                    )
                    
                    # 是否需要执行系统回调（始终需要，除非未指定系统回调URL）
                    need_system_callback = system_callback_url is not None and (
                        frame_count - last_callback_frame >= callback_interval or 
                        frame_count == 1  # 第一帧始终回调
                    )
                    
                    # 结果图片
                    result_image = None
                    
                    # 保存结果或需要回调时，绘制结果
                    if save_result or need_user_callback or need_system_callback:
                        result_image = await self._encode_result_image(
                            frame, 
                            detections,
                            return_image=True,
                            draw_tracks=analysis_type == "tracking",
                            draw_track_ids=analysis_type == "tracking"
                        )
                    
                    # 保存结果图片
                    saved_path = None
                    if save_result and result_image is not None:
                        saved_path = await self._save_result_image(result_image, detections, task_name or task_id)
                    
                    # 如果需要回调
                    if need_user_callback or need_system_callback:
                        last_callback_frame = frame_count
                        base64_image = None
                        
                        # 转换图片为base64
                        if result_image is not None:
                            _, buffer = cv2.imencode('.jpg', result_image)
                            base64_image = base64.b64encode(buffer).decode('utf-8')
                        
                        # 准备回调数据
                        callback_data = CallbackData(
                            camera_device_stream_url=stream_url,
                            camera_device_name=task_name or task_id,
                            camera_device_id=0,
                            algorithm_name=model_code,
                            algorithm_id=0,
                            data_id=task_id,
                            task_id=int(task_id.split('_')[-1], 16) if task_id.split('_')[-1].isalnum() else 0,
                            camera_url=stream_url,
                            camera_name=task_name or task_id,
                            timestamp=int(time.time()),
                            image_width=width,
                            image_height=height,
                            src_pic_data=base64_image,
                            alarm_pic_data=base64_image,
                            parameter=config,
                            result_data={
                                "detections": detections,
                                "task_id": task_id,
                                "frame_index": frame_count
                            },
                            extra_info=detections
                        )
                        
                        # 执行系统回调
                        system_callback_success = True
                        if need_system_callback:
                            try:
                                logger.info(f"发送系统级回调到 {system_callback_url}")
                                system_callback_success = await self._send_callback(system_callback_url, callback_data.to_dict())
                                
                                if not system_callback_success:
                                    logger.error(f"系统级回调失败! URL: {system_callback_url}")
                                    # 停止任务，因为系统回调是必须的
                                    logger.error(f"系统级回调失败，停止任务 {task_id}")
                                    task_info["status"] = TaskStatus.STOPPING
                                    task_info["error_message"] = "系统级回调失败，任务停止"
                                    await self._update_task_info(task_id, task_info)
                                    break
                            except Exception as e:
                                logger.error(f"系统级回调异常: {str(e)}")
                                # 停止任务，因为系统回调是必须的
                                logger.error(f"系统级回调异常，停止任务 {task_id}")
                                task_info["status"] = TaskStatus.STOPPING
                                task_info["error_message"] = f"系统级回调异常: {str(e)}"
                                await self._update_task_info(task_id, task_info)
                                break
                        
                        # 执行用户回调（仅当系统回调成功时）
                        if need_user_callback and system_callback_success:
                            try:
                                await self._send_callback(callback_urls, callback_data.to_dict())
                            except Exception as e:
                                logger.error(f"用户回调异常: {str(e)}")
                                # 用户回调失败不影响任务继续执行
                    
                    # 缓存检测结果
                    last_detections = detections
                    
                    # 更新任务信息
                    task_info["last_detections"] = detections
                    task_info["last_update_time"] = datetime.now().isoformat()
                    await self._update_task_info(task_id, task_info)
                    
                except Exception as e:
                    logger.error(f"处理帧时出错: {str(e)}", exc_info=True)
                    continue
            
            # 任务完成
            cap.release()
            logger.info(f"流分析任务 {task_id} 已停止")
            
            # 更新任务状态
            task_info["status"] = TaskStatus.COMPLETED
            task_info["end_time"] = datetime.now().isoformat()
            await self._update_task_info(task_id, task_info)
            
        except Exception as e:
            logger.error(f"流分析任务 {task_id} 处理时出错: {str(e)}", exc_info=True)
            
            # 更新任务状态为失败
            task_info = await self._get_task_info(task_id)
            if task_info:
                task_info["status"] = TaskStatus.FAILED
                task_info["error_message"] = str(e)
                task_info["end_time"] = datetime.now().isoformat()
                await self._update_task_info(task_id, task_info)

    async def stop_stream_analysis(self, task_id: str):
        """停止视频流分析"""
        try:
            # 创建一个全局字典来跟踪强制停止的任务
            if not hasattr(self, '_force_stop_tasks'):
                self._force_stop_tasks = set()
            
            # 将任务ID添加到强制停止集合中
            self._force_stop_tasks.add(task_id)
            
            # 更新任务状态为停止中
            await self._update_task_info(task_id, {'status': TaskStatus.STOPPING})
            
            # 使用事件循环的callater方法在短时间后尝试清理资源
            asyncio.get_event_loop().call_later(3, lambda: asyncio.create_task(self._force_clean_task(task_id)))
            
            # 等待任务实际停止
            max_wait = 10  # 最大等待10秒
            while max_wait > 0:
                task_info = await self._get_task_info(task_id)
                if not task_info or task_info.get('status') in [TaskStatus.STOPPED, TaskStatus.FAILED, TaskStatus.COMPLETED]:
                    break
                await asyncio.sleep(1)
                max_wait -= 1
            
            # 如果任务仍在运行，更强硬地停止任务
            if max_wait == 0:
                logger.warning(f"任务 {task_id} 停止超时，强制停止")
                # 更新任务状态为已停止
                await self._update_task_info(task_id, {
                    'status': TaskStatus.STOPPED,
                    'error_message': '任务强制停止',
                    'end_time': datetime.now().isoformat()
                })
            
            logger.info(f"任务 {task_id} 已停止")
            
        except Exception as e:
            logger.error(f"停止任务 {task_id} 失败: {str(e)}")
            # 确保任务状态被更新为已停止
            await self._update_task_info(task_id, {
                'status': TaskStatus.STOPPED,
                'error_message': f'停止失败: {str(e)}',
                'end_time': datetime.now().isoformat()
            })
        finally:
            # 从强制停止集合中移除任务ID
            if hasattr(self, '_force_stop_tasks') and task_id in self._force_stop_tasks:
                self._force_stop_tasks.remove(task_id)

    async def _force_clean_task(self, task_id: str):
        """强制清理任务资源"""
        try:
            logger.info(f"强制清理任务 {task_id} 资源")
            # 这里可以添加关闭视频捕获、关闭网络连接等清理操作
            # 更新任务状态为已停止
            await self._update_task_info(task_id, {
                'status': TaskStatus.STOPPED,
                'error_message': '任务资源已强制清理',
                'end_time': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"强制清理任务 {task_id} 资源失败: {str(e)}")

    async def _should_stop(self, task_id: str) -> bool:
        """检查任务是否应该停止"""
        try:
            # 首先检查强制停止列表
            if hasattr(self, '_force_stop_tasks') and task_id in self._force_stop_tasks:
                logger.info(f"任务 {task_id} 在强制停止列表中")
                return True
            
            # 获取任务信息
            task_info = await self._get_task_info(task_id)
            if not task_info:
                return True
            
            # 检查任务状态
            status = task_info.get("status")
            return status in [TaskStatus.STOPPING, TaskStatus.CANCELLED, TaskStatus.STOPPED]
            
        except Exception as e:
            logger.error(f"检查任务状态失败: {str(e)}", exc_info=True)
            return True

    async def start_video_analysis(
        self,
        task_id: str,
        model_code: str,
        video_url: str,
        callback_urls: Optional[str] = None,
        config: Optional[Dict] = None,
        task_name: Optional[str] = None,
        enable_callback: bool = True,
        save_result: bool = False,
        enable_tracking: bool = False,
        tracking_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """开始视频分析任务"""
        try:
            # 初始化任务信息
            task_info = {
                'id': task_id,
                'task_name': task_name,
                'model_code': model_code,
                'stream_url': video_url,
                'callback_urls': callback_urls,
                'enable_callback': enable_callback,
                'save_result': save_result,
                'config': config,
                'status': TaskStatus.PROCESSING,
                'start_time': datetime.now().isoformat(),
                'type': 'video',
                'enable_tracking': enable_tracking,
                'tracking_config': tracking_config
            }
            
            # 保存任务信息到Redis
            await self.task_queue.add_task(task_info)
            
            # 启动异步处理任务
            asyncio.create_task(self._process_video_analysis(
                task_id=task_id,
                model_code=model_code,
                video_url=video_url,
                callback_urls=callback_urls,
                config=config,
                task_name=task_name,
                enable_callback=enable_callback,
                save_result=save_result,
                enable_tracking=enable_tracking,
                tracking_config=tracking_config
            ))
            
            return await self._get_task_info(task_id)
            
        except Exception as e:
            logger.error(f"启动视频分析任务失败: {str(e)}", exc_info=True)
            await self._fail_task(task_id, str(e))
            raise

    async def _get_task_info(self, task_id: str) -> Optional[Dict]:
        """从Redis获取任务信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[Dict]: 任务信息或None
        """
        try:
            # 步骤1: 尝试直接从Redis中获取任务信息
            task_key = f"task:{task_id}"
            logger.debug(f"尝试从Redis键 {task_key} 获取任务信息")
            
            task_info = await self.redis.get_value(task_key, as_json=True)
            
            # 如果找到了，直接返回
            if task_info:
                logger.debug(f"在Redis中找到任务信息: {task_id}")
                return task_info
                
            # 步骤2: 如果直接获取失败，尝试从TaskQueue中获取
            logger.debug(f"在Redis直接键中未找到任务，尝试从TaskQueue中获取: {task_id}")
            task_info_from_queue = await self.task_queue.get_task(task_id)
            
            if task_info_from_queue:
                logger.debug(f"在TaskQueue中找到任务信息: {task_id}")
                
                # 将任务信息同步回Redis直接键，便于将来查询
                await self._update_task_info(task_id, task_info_from_queue)
                
                return task_info_from_queue
                
            # 步骤3: 如果都找不到，再尝试其他可能的ID格式（兼容不同命名方式）
            if not task_id.startswith("task:"):
                alt_task_id = f"task:{task_id}"
                logger.debug(f"尝试使用替代任务ID格式: {alt_task_id}")
                alt_task_info = await self.redis.get_value(alt_task_id, as_json=True)
                
                if alt_task_info:
                    logger.debug(f"使用替代ID格式找到任务: {alt_task_id}")
                    
                    # 同步到标准格式
                    await self._update_task_info(task_id, alt_task_info)
                    
                    return alt_task_info
                    
            # 任务未找到
            logger.warning(f"任务 {task_id} 在所有存储位置均未找到")
            return None
            
        except Exception as e:
            logger.error(f"获取任务信息时出错: {str(e)}", exc_info=True)
            return None

    async def _update_task_info(self, task_id: str, info: Dict[str, Any]):
        """更新任务信息，同时更新到Redis和TaskQueue
        
        Args:
            task_id: 任务ID
            info: 要更新的任务信息
        """
        try:
            logger.debug(f"更新任务信息: {task_id}")
            
            # 步骤1: 获取当前任务信息
            current_info = await self._get_task_info(task_id)
            
            if current_info:
                # 如果任务已存在，更新它
                logger.debug(f"更新现有任务: {task_id}")
                current_info.update(info)
                update_data = current_info
            else:
                # 如果任务不存在，使用提供的信息创建它
                logger.debug(f"创建新任务: {task_id}")
                update_data = info
                
                # 确保基本字段存在
                if "id" not in update_data and "task_id" not in update_data:
                    update_data["id"] = task_id
                    update_data["task_id"] = task_id
                if "created_at" not in update_data:
                    update_data["created_at"] = datetime.now().isoformat()
            
            # 步骤2: 直接更新Redis键
            task_key = f"task:{task_id}"
            await self.redis.set_value(task_key, update_data)
            logger.debug(f"已更新Redis键: {task_key}")
            
            # 步骤3: 通过TaskQueue更新
            # 确保任务有status字段
            if "status" not in update_data:
                update_data["status"] = TaskStatus.PROCESSING
            
            # 使用add_task更新任务队列中的任务
            try:
                await self.task_queue.add_task(update_data, priority=0, task_id=task_id)
                logger.debug(f"已通过TaskQueue更新任务: {task_id}")
            except Exception as e:
                logger.warning(f"通过TaskQueue更新任务失败: {str(e)}")
            
            # 步骤4: 如果包含状态更新，调用专门的状态更新方法
            if 'status' in info:
                try:
                    await self.task_queue.update_task_status(task_id, info['status'], update_data)
                    logger.debug(f"已更新任务状态: {task_id} -> {info['status']}")
                except Exception as e:
                    logger.warning(f"更新任务状态失败: {str(e)}")
            
            logger.debug(f"任务 {task_id} 信息已更新")

        except Exception as e:
            logger.error(f"更新任务 {task_id} 信息失败: {str(e)}", exc_info=True)

    async def _save_task_result(self, task_id: str, result: Dict):
        """保存任务结果到Redis"""
        await self.task_queue.complete_task(task_id, result)

    async def _fail_task(self, task_id: str, error: str):
        """标记任务失败"""
        await self.task_queue.fail_task(task_id, error)

    async def _process_video_analysis(
        self,
        task_id: str,
        model_code: str,
        video_url: str,
        callback_urls: Optional[str] = None,
        config: Optional[Dict] = None,
        task_name: Optional[str] = None,
        enable_callback: bool = True,
        save_result: bool = False,
        enable_tracking: bool = False,
        tracking_config: Optional[Dict] = None
    ):
        """实际的视频处理逻辑"""
        cap = None
        video_writer = None
        local_video_path = None
        start_time = time.time()
        
        try:
            logger.info(f"开始处理视频分析任务: {task_id}")
            
            # 获取任务信息
            task_info = await self._get_task_info(task_id)
            if not task_info:
                raise Exception(f"任务 {task_id} 不存在")
                
            # 更新任务状态为处理中
            task_info['status'] = TaskStatus.PROCESSING
            task_info['process_start_time'] = datetime.now().isoformat()
            await self._update_task_info(task_id, task_info)
            
            # 加载模型
            if not self.model:
                await self.load_model(model_code)
            
            # 初始化跟踪器（如果启用）
            if enable_tracking:
                tracking_config = tracking_config or {}
                self.tracker = create_tracker(
                    tracker_type=tracking_config.get("tracker_type", "sort"),
                    max_age=tracking_config.get("max_age", 30),
                    min_hits=tracking_config.get("min_hits", 3),
                    iou_threshold=tracking_config.get("iou_threshold", 0.3)
                )
                logger.info(f"初始化跟踪器: {tracking_config.get('tracker_type', 'sort')}")
            
            # 使用配置参数或默认值
            config_dict = config.dict() if hasattr(config, 'dict') else (config or {})
            conf = config_dict.get('confidence', self.default_confidence)
            iou = config_dict.get('iou', self.default_iou)
            
            # 确保置信度和IoU阈值有效
            if conf is None:
                conf = self.default_confidence
            if iou is None:
                iou = self.default_iou
            
            config_dict['confidence'] = conf
            config_dict['iou'] = iou
            
            # 下载视频
            try:
                # 生成本地文件路径
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"video_{timestamp}.mp4"
                videos_dir = self.project_root / "data" / "videos" / "temp"
                os.makedirs(videos_dir, exist_ok=True)
                local_video_path = str(videos_dir / filename)
                
                logger.info(f"开始下载视频: {video_url}")
                logger.info(f"保存到: {local_video_path}")
                
                # 使用 httpx 下载视频
                async with httpx.AsyncClient() as client:
                    async with client.stream("GET", video_url) as response:
                        response.raise_for_status()
                        
                        # 打开本地文件
                        with open(local_video_path, "wb") as f:
                            # 分块下载
                            total_size = 0
                            async for chunk in response.aiter_bytes():
                                f.write(chunk)
                                total_size += len(chunk)
                                # 更新下载进度
                                task_info['download_progress'] = total_size
                                await self._update_task_info(task_id, task_info)
                
                logger.info(f"视频下载完成: {local_video_path}")
                
            except Exception as e:
                logger.error(f"下载视频失败: {str(e)}")
                if os.path.exists(local_video_path):
                    os.remove(local_video_path)
                raise
            
            # 打开视频
            cap = cv2.VideoCapture(local_video_path)
            if not cap.isOpened():
                raise Exception(f"无法打开视频: {local_video_path}")
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = 3  # 每3帧检测一次
            
            # 更新任务信息
            task_info.update({
                'video_info': {
                    'fps': fps,
                    'width': frame_width,
                    'height': frame_height,
                    'total_frames': total_frames,
                    'frame_interval': frame_interval
                },
                'progress': 0,
                'processed_frames': 0
            })
            await self._update_task_info(task_id, task_info)
            
            logger.info(f"视频信息 - FPS: {fps}, 尺寸: {frame_width}x{frame_height}, 总帧数: {total_frames}")
            
            # 如果需要保存结果，创建视频写入器
            saved_path = None
            relative_saved_path = None
            if save_result:
                # 生成保存路径
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                task_prefix = f"{task_name}_" if task_name else ""
                filename = f"{task_prefix}{timestamp}.mp4"
                
                # 确保每天的结果保存在单独的目录中
                date_dir = self.results_dir / datetime.now().strftime("%Y%m%d")
                os.makedirs(date_dir, exist_ok=True)
                
                # 完整的保存路径
                saved_path = str(date_dir / filename)
                relative_saved_path = str(Path(saved_path).relative_to(self.project_root))
                
                # 创建视频写入器
                if os.name == 'nt':  # Windows
                    fourcc = cv2.VideoWriter_fourcc(*'H264')
                else:  # macOS/Linux
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                
                video_writer = cv2.VideoWriter(
                    saved_path,
                    fourcc,
                    fps,
                    (frame_width, frame_height)
                )
                
                if not video_writer.isOpened():
                    logger.error("无法创建视频写入器，尝试使用其他编码格式")
                    video_writer.release()
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        saved_path,
                        fourcc,
                        fps,
                        (frame_width, frame_height)
                    )
            
            frame_count = 0
            processed_count = 0
            last_progress_time = time.time()
            last_detections = None
            frames_buffer = []
            tracking_start_time = time.time()
            total_tracking_time = 0
            
            # 检查任务是否应该停止
            async def should_stop():
                task_info = await self._get_task_info(task_id)
                return task_info.get('status') == TaskStatus.STOPPING or task_info.get('status') == TaskStatus.CANCELLED
            
            while True:
                # 检查是否需要停止
                if await should_stop():
                    logger.info(f"任务 {task_id} 收到停止信号")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                frames_buffer.append(frame.copy())
                
                # 按间隔处理帧
                if frame_count % frame_interval == 0:
                    processed_count += 1
                    current_time = time.time()
                    
                    try:
                        # 执行检测
                        detections = await self.detect(frame, config=config_dict)
                        
                        # 如果启用了跟踪，更新跟踪状态
                        if enable_tracking and self.tracker:
                            tracking_start = time.time()
                            tracked_objects = self.tracker.update(detections)
                            tracking_time = time.time() - tracking_start
                            total_tracking_time += tracking_time
                            
                            # 更新检测结果
                            for det, track in zip(detections, tracked_objects):
                                det.update(track.to_dict())
                            
                            # 更新跟踪统计信息
                            tracking_stats = {
                                'total_tracks': self.tracker.next_track_id - 1,
                                'active_tracks': len([t for t in tracked_objects if t.time_since_update == 0]),
                                'avg_track_length': sum(t.age for t in tracked_objects) / len(tracked_objects) if tracked_objects else 0,
                                'tracking_fps': processed_count / total_tracking_time if total_tracking_time > 0 else 0
                            }
                            task_info['tracking_stats'] = tracking_stats
                        
                        last_detections = detections
                        
                        # 更新进度
                        progress = (frame_count / total_frames) * 100
                        task_info.update({
                            'progress': round(progress, 2),
                            'processed_frames': frame_count,
                            'total_frames': total_frames,
                            'current_detections': detections,
                            'last_update_time': datetime.now().isoformat(),
                            'video_info': {
                                'total_frames': total_frames,
                                'fps': cap.get(cv2.CAP_PROP_FPS),
                                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            }
                        })
                        await self._update_task_info(task_id, task_info)
                        
                        # 每秒最多更新一次进度日志
                        if current_time - last_progress_time >= 1.0:
                            logger.info(f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})")
                            last_progress_time = current_time
                        
                        # 发送回调
                        if enable_callback and callback_urls and detections:
                            callback_data = {
                                "task_id": task_id,
                                "task_name": task_name,
                                "frame_index": frame_count,
                                "total_frames": total_frames,
                                "progress": progress,
                                "detections": detections,
                                "tracking_enabled": enable_tracking,
                                "tracking_stats": task_info.get('tracking_stats'),
                                "timestamp": current_time
                            }
                            # 将回调数据缓存到Redis
                            await self.redis.hset_dict(
                                f"task:{task_id}:callbacks",
                                str(int(current_time * 1000)),
                                callback_data
                            )
                            # 发送回调
                            await self._send_callbacks(callback_urls, callback_data)
                        
                        # 处理结果帧
                        if save_result and video_writer is not None:
                            for buffered_frame in frames_buffer:
                                result_frame = await self._encode_result_image(
                                    buffered_frame,
                                    last_detections,
                                    return_image=True,
                                    draw_tracks=enable_tracking and tracking_config.get('visualization', {}).get('show_tracks', True),
                                    draw_track_ids=enable_tracking and tracking_config.get('visualization', {}).get('show_track_ids', True)
                                )
                                if result_frame is not None:
                                    video_writer.write(result_frame)
                            frames_buffer = []
                            
                    except Exception as e:
                        logger.error(f"处理第 {frame_count} 帧时出错: {str(e)}")
                        continue
                
                # 每处理5帧后让出控制权
                if frame_count % 5 == 0:
                    await asyncio.sleep(0.01)
            
            # 处理剩余帧
            if save_result and video_writer is not None and frames_buffer:
                for buffered_frame in frames_buffer:
                    if last_detections is not None:
                        result_frame = await self._encode_result_image(
                            buffered_frame,
                            last_detections,
                            return_image=True,
                            draw_tracks=enable_tracking and tracking_config.get('visualization', {}).get('show_tracks', True),
                            draw_track_ids=enable_tracking and tracking_config.get('visualization', {}).get('show_track_ids', True)
                        )
                        if result_frame is not None:
                            video_writer.write(result_frame)
            
            # 更新最终状态
            end_time = time.time()
            analysis_duration = end_time - start_time
            
            final_status = TaskStatus.COMPLETED
            if await should_stop():
                final_status = TaskStatus.CANCELLED
            
            task_info.update({
                'status': final_status,
                'end_time': datetime.now().isoformat(),
                'analysis_duration': analysis_duration,
                'saved_path': relative_saved_path if save_result else None,
                'final_progress': 100 if final_status == TaskStatus.COMPLETED else round((frame_count / total_frames) * 100, 2)
            })
            
            if final_status == TaskStatus.COMPLETED:
                await self._save_task_result(task_id, task_info)
            else:
                await self._update_task_info(task_id, task_info)
            
            logger.info(f"视频分析完成: {task_id}")
            logger.info(f"- 总帧数: {total_frames}")
            logger.info(f"- 处理帧数: {frame_count}")
            logger.info(f"- 分析耗时: {analysis_duration:.2f}秒")
            if enable_tracking:
                logger.info(f"- 跟踪目标数: {task_info.get('tracking_stats', {}).get('total_tracks', 0)}")
                logger.info(f"- 跟踪处理帧率: {task_info.get('tracking_stats', {}).get('tracking_fps', 0):.2f} FPS")
            
            return task_info
            
        except Exception as e:
            logger.error(f"视频分析失败: {str(e)}", exc_info=True)
            await self._fail_task(task_id, str(e))
            raise
            
        finally:
            # 清理资源
            if cap is not None:
                cap.release()
            if video_writer is not None:
                video_writer.release()
            if local_video_path and os.path.exists(local_video_path):
                try:
                    os.remove(local_video_path)
                except Exception as e:
                    logger.error(f"删除临时视频文件失败: {str(e)}")
            
            # 重置跟踪器
            self.tracker = None

    async def get_video_task_status(self, task_id: str) -> Optional[Dict]:
        """获取视频分析任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[Dict]: 任务状态信息，包含进度、检测结果等
        """
        try:
            # 从Redis获取任务信息
            task_info = await self._get_task_info(task_id)
            if not task_info:
                logger.warning(f"任务不存在: {task_id}")
                return None
                
            # 获取回调数据
            callback_data = {}
            if task_info.get('enable_callback'):
                callback_key = f"task:{task_id}:callbacks"
                callback_data = await self.redis.hgetall(callback_key)
            
            # 构造完整的状态信息
            status_info = {
                'task_id': task_id,
                'task_name': task_info.get('task_name'),
                'status': task_info.get('status'),
                'progress': task_info.get('progress', 0),
                'processed_frames': task_info.get('processed_frames', 0),
                'total_frames': task_info.get('video_info', {}).get('total_frames', 0),
                'start_time': task_info.get('start_time'),
                'end_time': task_info.get('end_time'),
                'analysis_duration': task_info.get('analysis_duration'),
                'saved_path': task_info.get('saved_path'),
                'error_message': task_info.get('error'),
                'current_detections': task_info.get('current_detections', []),
                'tracking_stats': task_info.get('tracking_stats'),
                'callback_data': callback_data
            }
            
            return status_info
            
        except Exception as e:
            logger.error(f"获取任务状态失败: {str(e)}", exc_info=True)
            return None

    async def _process_frame(
        self,
        frame: np.ndarray,
        model: YOLO,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """处理单帧图像"""
        try:
            # 使用配置参数或默认值
            conf = config.get('confidence', self.default_confidence)
            iou = config.get('iou', self.default_iou)
            classes = config.get('classes', None)
            roi = config.get('roi', None)
            imgsz = config.get('imgsz', None)
            
            # 保存原始图像尺寸
            original_shape = frame.shape[:2]  # (height, width)
            
            # 如果指定了ROI，裁剪图像
            if roi:
                h, w = frame.shape[:2]
                x1 = int(roi['x1'] * w)
                y1 = int(roi['y1'] * h)
                x2 = int(roi['x2'] * w)
                y2 = int(roi['y2'] * h)
                frame = frame[y1:y2, x1:x2]
            
            # 处理图片大小
            if imgsz:
                frame = cv2.resize(frame, (imgsz, imgsz))
            
            # 执行推理
            results = model(
                frame,
                conf=conf,
                iou=iou,
                classes=classes
            )
            
            # 处理检测结果
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = result.names[cls]
                    
                    # 如果使用了ROI，需要调整坐标
                    if roi and original_shape:
                        h, w = original_shape
                        bbox[0] = bbox[0] / w * (roi['x2'] - roi['x1']) * w + roi['x1'] * w
                        bbox[1] = bbox[1] / h * (roi['y2'] - roi['y1']) * h + roi['y1'] * h
                        bbox[2] = bbox[2] / w * (roi['x2'] - roi['x1']) * w + roi['x1'] * w
                        bbox[3] = bbox[3] / h * (roi['y2'] - roi['y1']) * h + roi['y1'] * h
                    
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
            
            return detections
            
        except Exception as e:
            logger.error(f"处理帧失败: {str(e)}", exc_info=True)
            raise ProcessingException(f"处理帧失败: {str(e)}")

    async def _send_callback(self, callback_urls: str, data: Dict[str, Any]) -> bool:
        """发送回调数据
        
        Args:
            callback_urls: 回调URL，可以是单个URL或多个URL用逗号分隔
            data: 回调数据
            
        Returns:
            bool: 是否发送成功
        """
        if not callback_urls:
            return False
        
        # 处理单个URL的情况
        if ',' not in callback_urls:
            url = callback_urls.strip()
            if not url:
                return False
                
            try:
                logger.debug(f"发送回调到 {url}")
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        url,
                        json=data,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        logger.debug(f"回调成功: {url}")
                        return True
                    else:
                        logger.warning(f"回调失败: {url}, 状态码: {response.status_code}")
                        return False
                        
            except Exception as e:
                logger.error(f"发送回调时出错: {url}, {str(e)}")
                return False
        
        # 处理多个URL的情况
        urls = callback_urls.split(',')
        success_count = 0
        
        for url in urls:
            url = url.strip()
            if not url:
                continue
                
            try:
                logger.debug(f"发送回调到 {url}")
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        url,
                        json=data,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        logger.debug(f"回调成功: {url}")
                        success_count += 1
                    else:
                        logger.warning(f"回调失败: {url}, 状态码: {response.status_code}")
                        
            except Exception as e:
                logger.error(f"发送回调时出错: {url}, {str(e)}")
                
        # 如果任一URL回调成功，则认为回调成功
        return success_count > 0
