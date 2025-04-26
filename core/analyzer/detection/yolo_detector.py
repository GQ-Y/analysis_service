"""
YOLO检测器模块
实现基于YOLOv8、YOLOv7、YOLOv5、YOLOX、YOLOE的目标检测功能
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
import functools
import io

logger = setup_logger(__name__)

class YOLODetector:
    """YOLO检测器实现"""
    
    def __init__(self, model_code: Optional[str] = None):
        """
        初始化检测器
        
        Args:
            model_code: 模型代码，如果提供则立即加载模型
        """
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
        detection_results: List[Dict],
        return_image: bool = False
    ) -> Union[str, np.ndarray, None]:
        """将检测结果绘制到图片上，并压缩到合适大小
        
        Args:
            image: 原始图像数据
            detection_results: 检测结果列表，包含边界框、类别等信息
            return_image: 是否返回图像数据而非base64编码
            
        Returns:
            Union[str, np.ndarray, None]: base64编码的图像，或图像数据，或None（出错时）
        """
        try:
            # 复制图片以免修改原图
            result_image = image.copy()
            
            # 使用PIL进行绘制，更灵活的处理中文和文本绘制
            img_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 尝试加载字体
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
                    try:
                        if os.path.exists(font_path):
                            font = ImageFont.truetype(font_path, font_size)
                            logger.debug(f"成功加载字体: {font_path}")
                            break
                    except Exception as font_err:
                        logger.debug(f"加载字体失败: {font_path}, 错误: {font_err}")
                
                # 如果所有字体都失败，使用默认字体
                if font is None:
                    logger.warning("无法加载中文字体，将使用默认字体")
                    font = ImageFont.load_default()
            except Exception as e:
                logger.warning(f"加载字体出错: {str(e)}，将使用默认字体")
                font = ImageFont.load_default()
            
            # 绘制检测结果
            for i, det in enumerate(detection_results):
                try:
                    # 获取检测框
                    bbox = det.get("bbox", {})
                    if not bbox:
                        continue
                        
                    x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
                    x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 获取类别信息
                    class_name = det.get("class_name", "unknown")
                    confidence = det.get("confidence", 0.0)
                    
                    # 生成不同颜色
                    class_id = det.get("class_id", i)
                    color = self._get_color_by_id(class_id)
                    
                    # 绘制边界框
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    # 准备标签文本
                    label = f"{class_name} {confidence:.2f}"
                    
                    # 获取文本大小 - 兼容不同版本的PIL
                    try:
                        # 新版PIL使用textbbox
                        text_bbox = draw.textbbox((0, 0), label, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    except AttributeError:
                        # 旧版PIL使用getsize
                        try:
                            text_width, text_height = font.getsize(label)
                        except AttributeError:
                            # 更旧版本
                            text_width, text_height = draw.textsize(label, font=font)
                    
                    # 确定标签位置
                    label_y = max(0, y1 - text_height - 2) if y1 - text_height - 2 > 0 else y1
                    
                    # 绘制标签背景
                    background_shape = [(x1, label_y), (x1 + text_width + 4, label_y + text_height + 4)]
                    draw.rectangle(background_shape, fill=color)
                    
                    # 绘制文本
                    text_position = (x1 + 2, label_y + 2)
                    draw.text(
                        text_position,
                        label,
                        font=font,
                        fill=(255, 255, 255)  # 白色文字
                    )
                except Exception as det_err:
                    logger.warning(f"绘制检测框 {i} 出错: {det_err}")
            
            # 转换回OpenCV格式
            result_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            if return_image:
                return result_image
            
            # 图像压缩函数 - 保持目标大小为300KB
            def compress_image(img, max_size_kb=200, initial_quality=85):
                """压缩图像到指定大小"""
                logger.debug(f"开始压缩图像，目标大小: {max_size_kb}KB")
                quality = initial_quality
                min_quality = 15  # 最低质量阈值
                
                # 将OpenCV图像转为PIL图像
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                # 降低分辨率
                img_width, img_height = img_pil.size
                # 如果图像尺寸过大，先调整大小
                max_dim = 1280  # 更低的最大宽度或高度以减小初始图像大小
                if img_width > max_dim or img_height > max_dim:
                    scale = max_dim / max(img_width, img_height)
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
                    logger.debug(f"图像初始调整大小: {img_width}x{img_height} -> {new_width}x{new_height}")
                
                # 尝试不同的质量级别进行压缩，直到大小合适或达到最低质量
                while quality >= min_quality:
                    buffer = io.BytesIO()
                    img_pil.save(buffer, format='JPEG', quality=quality, optimize=True, subsampling=2)
                    size_kb = buffer.tell() / 1024
                    
                    logger.debug(f"压缩质量: {quality}, 大小: {size_kb:.2f}KB")
                    
                    if size_kb <= max_size_kb:
                        logger.info(f"图像已压缩: 质量={quality}, 大小={size_kb:.2f}KB")
                        return buffer.getvalue()
                    
                    # 降低质量幅度更大，加快压缩
                    if size_kb > max_size_kb * 2:  # 如果当前大小超过目标的2倍
                        quality -= 15
                    else:
                        quality -= 8  # 否则使用较小的步长
                
                # 如果降低质量还不够，尝试逐步减小图像尺寸
                current_width, current_height = img_pil.size
                scale_factor = 0.75  # 每次缩小到原尺寸的75%
                
                while quality <= min_quality and size_kb > max_size_kb and min(current_width, current_height) > 320:
                    # 降低分辨率
                    new_width = int(current_width * scale_factor)
                    new_height = int(current_height * scale_factor)
                    resized_img = img_pil.resize((new_width, new_height), Image.LANCZOS)
                    
                    # 压缩
                    buffer = io.BytesIO()
                    resized_img.save(buffer, format='JPEG', quality=min_quality, optimize=True, subsampling=2)
                    size_kb = buffer.tell() / 1024
                    
                    logger.debug(f"调整大小: {current_width}x{current_height} -> {new_width}x{new_height}, 大小: {size_kb:.2f}KB")
                    
                    if size_kb <= max_size_kb:
                        logger.info(f"图像已压缩: 尺寸={new_width}x{new_height}, 质量={min_quality}, 大小={size_kb:.2f}KB")
                        return buffer.getvalue()
                    
                    # 更新当前尺寸
                    current_width, current_height = new_width, new_height
                    img_pil = resized_img
                
                # 如果所有方法都失败，使用最小尺寸和最低质量的结果
                logger.warning(f"图像压缩到最小尺寸和质量仍超过目标大小: {size_kb:.2f}KB > {max_size_kb}KB")
                return buffer.getvalue()
            
            try:
                # 将图片编码为JPEG并压缩到约300KB
                image_bytes = compress_image(result_image)
                image_size_kb = len(image_bytes) / 1024
                logger.info(f"检测结果图像压缩后大小: {image_size_kb:.2f}KB")
                
                # Base64编码
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                logger.debug(f"成功生成base64图片，长度: {len(image_base64)}")
                return image_base64
            
            except Exception as e:
                logger.error(f"图片编码为base64失败: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"处理检测结果图片失败: {str(e)}")
            return None

    async def detect(self, 
                     image: np.ndarray, 
                     verbose: bool = False,
                     generate_image: bool = False
                     ) -> Dict[str, Any]:
        """对输入图像进行目标检测
        
        Args:
            image: BGR格式的输入图像
            verbose: 是否打印详细日志，默认False
            generate_image: 是否生成标注图像，默认False
            
        Returns:
            包含检测结果和图像数据的字典:
            - detections: 检测结果列表
            - pre_process_time: 预处理时间 (ms)
            - inference_time: 推理时间 (ms)
            - post_process_time: 后处理时间 (ms)
            - annotated_image: 标注后图像的 base64 字符串 (仅当generate_image=True时)
        """
        start_time = time.time()
        pre_process_time_ms = 0
        inference_time_ms = 0
        post_process_time_ms = 0
        annotated_image = None  # 改为字符串而非字节流

        try:
            # 运行检测
            logger.debug("【调试】开始执行目标检测")
            results = self.model(image, verbose=verbose)
            logger.debug("【调试】模型推理完成")
            
            # 获取计时信息
            if results and hasattr(results[0], 'speed'):
                speed_info = results[0].speed
                pre_process_time_ms = speed_info.get('preprocess', 0)
                inference_time_ms = speed_info.get('inference', 0)
                post_process_time_ms = speed_info.get('postprocess', 0)
            
            # 解析检测结果
            logger.debug("【调试】开始解析检测结果")
            detections = await self._parse_results(results)
            logger.debug(f"【调试】检测到 {len(detections)} 个目标")
            
            # 只有在请求时才生成和编码标注图像
            if generate_image and detections:
                try:
                    logger.debug("【调试】开始生成和压缩标注图像")
                    annotated_image_np = results[0].plot() # 使用ultralytics自带的plot
                    # 使用压缩方法编码图像 - 将图像压缩到约300KB
                    annotated_image = await self._encode_result_image(annotated_image_np, detections, False)
                    if annotated_image:
                        kb_size = len(base64.b64decode(annotated_image)) / 1024
                        logger.debug(f"【调试】标注图像已压缩，大小: {kb_size:.2f}KB")
                        if kb_size > 350:  # 如果超过预期，记录警告
                            logger.warning(f"标注图像大小({kb_size:.2f}KB)超过目标大小(300KB)")
                except Exception as plot_err:
                    logger.error(f"绘制或编码标注图像时出错: {plot_err}")
                    import traceback
                    logger.error(f"详细错误: {traceback.format_exc()}")
            elif generate_image and not detections:
                logger.debug("【调试】未检测到目标，不生成标注图像")

            total_time = (time.time() - start_time) * 1000
            logger.debug(f"【调试】检测总耗时: {total_time:.2f}ms")

            # 构建返回字典 (字段名改为annotated_image)
            return_data = {
                "detections": detections,
                "pre_process_time": pre_process_time_ms,
                "inference_time": inference_time_ms,
                "post_process_time": post_process_time_ms,
                "annotated_image": annotated_image  # 改为字符串
            }
                
            return return_data
            
        except Exception as e:
            logger.error(f"检测失败: {str(e)}")
            return {
                "detections": [],
                "pre_process_time": 0,
                "inference_time": 0,
                "post_process_time": 0,
                "annotated_image": None  # 错误时返回None
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
                
            # 准备原始图像尺寸
            image_size = (img_width, img_height)
            filtered_detections = []
            
            # 将ROI坐标转换为像素坐标
            pixel_roi = {}
            normalized = roi.get("normalized", True)
            
            if roi_type == 1:  # 矩形ROI
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
                    logger.error("无效的矩形ROI格式")
                    return detections
                    
            elif roi_type in [2, 3]:  # 多边形或线段ROI
                if "points" in roi:
                    points = roi["points"]
                elif "coordinates" in roi:
                    points = roi["coordinates"]
                else:
                    logger.error("无效的多边形/线段ROI格式")
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
            
            return filtered_detections
                
        except Exception as e:
            logger.error(f"根据ROI过滤检测结果失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
            # 生成带检测结果的图片
            logger.info("开始生成检测结果图片...")
            result_image = await self._encode_result_image(image, detections, True)
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
            logger.error(f"绘制检测结果时出错: {str(e)}")
            logger.error(str(e), exc_info=True)
            return image 

    async def _parse_results(self, results) -> List[Dict[str, Any]]:
        """解析YOLO检测结果
        
        Args:
            results: YOLO检测结果
            
        Returns:
            检测结果列表，每个检测包含:
            - bbox: 边界框坐标
            - confidence: 置信度
            - class_id: 类别ID
            - class_name: 类别名称
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
                    
            return detections
            
        except Exception as e:
            logger.error(f"解析检测结果失败: {str(e)}")
            logger.error(str(e), exc_info=True)
            return [] 