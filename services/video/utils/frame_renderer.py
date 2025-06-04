"""
帧渲染工具
用于在视频帧上渲染分析结果（检测框、跟踪ID等）
支持中英文字体渲染
"""
from typing import Dict, Any, Optional, List
import cv2
import numpy as np
from datetime import datetime
import os
import platform
from PIL import Image, ImageDraw, ImageFont
import logging

from core.task_management.utils.status import TaskStatus
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)


class FontManager:
    """字体管理器，负责加载和管理系统字体"""
    
    def __init__(self):
        self.fonts = {}
        self._load_system_fonts()
    
    def _load_system_fonts(self):
        """加载系统字体"""
        try:
            system = platform.system()
            
            # 定义不同系统的字体路径
            font_paths = {
                'Darwin': [  # macOS
                    '/System/Library/Fonts/PingFang.ttc',
                    '/System/Library/Fonts/STHeiti Medium.ttc',
                    '/System/Library/Fonts/Arial.ttf',
                    '/System/Library/Fonts/Helvetica.ttc',
                ],
                'Windows': [  # Windows
                    'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
                    'C:/Windows/Fonts/simsun.ttc',  # 宋体
                    'C:/Windows/Fonts/arial.ttf',
                    'C:/Windows/Fonts/tahoma.ttf',
                ],
                'Linux': [  # Linux
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                ]
            }
            
            # 获取当前系统的字体路径
            paths = font_paths.get(system, [])
            
            # 尝试加载字体
            for font_path in paths:
                if os.path.exists(font_path):
                    try:
                        # 加载不同大小的字体
                        for size in [12, 16, 20, 24, 28, 32]:
                            font_key = f"{os.path.basename(font_path)}_{size}"
                            self.fonts[font_key] = ImageFont.truetype(font_path, size)
                        
                        break  # 成功加载一个字体后就够了
                    except Exception as e:
                        normal_logger.debug(f"加载字体失败 {font_path}: {str(e)}")
                        continue
            
            # 如果没有加载到任何系统字体，使用默认字体
            if not self.fonts:
                normal_logger.warning("未找到合适的系统字体，使用默认字体")
                for size in [12, 16, 20, 24, 28, 32]:
                    font_key = f"default_{size}"
                    self.fonts[font_key] = ImageFont.load_default()
                    
        except Exception as e:
            exception_logger.exception(f"字体管理器初始化失败: {str(e)}")
            # 使用默认字体作为后备
            for size in [12, 16, 20, 24, 28, 32]:
                font_key = f"default_{size}"
                self.fonts[font_key] = ImageFont.load_default()
    
    def get_font(self, size: int = 16) -> ImageFont.ImageFont:
        """
        获取指定大小的字体
        
        Args:
            size: 字体大小
            
        Returns:
            ImageFont.ImageFont: 字体对象
        """
        # 找到最接近的可用字体大小
        available_sizes = [12, 16, 20, 24, 28, 32]
        closest_size = min(available_sizes, key=lambda x: abs(x - size))
        
        # 尝试获取系统字体
        for font_key in self.fonts.keys():
            if font_key.endswith(f"_{closest_size}") and not font_key.startswith("default_"):
                return self.fonts[font_key]
        
        # 如果没有系统字体，使用默认字体
        default_key = f"default_{closest_size}"
        return self.fonts.get(default_key, ImageFont.load_default())


# 全局字体管理器实例
font_manager = FontManager()


class FrameRenderer:
    """帧渲染器类，支持中英文字体渲染"""

    @staticmethod
    def _put_chinese_text(img: np.ndarray, text: str, position: tuple, font_size: int = 16, 
                         color: tuple = (255, 255, 255), background_color: Optional[tuple] = None) -> np.ndarray:
        """
        在图像上绘制中英文文本
        
        Args:
            img: OpenCV图像（BGR格式）
            text: 要绘制的文本
            position: 文本位置 (x, y)
            font_size: 字体大小
            color: 文本颜色 (B, G, R)
            background_color: 背景颜色，如果为None则不绘制背景
            
        Returns:
            np.ndarray: 绘制文本后的图像
        """
        try:
            # 转换 OpenCV 图像到 PIL
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 获取字体
            font = font_manager.get_font(font_size)
            
            # 转换颜色格式（BGR to RGB）
            rgb_color = (color[2], color[1], color[0])
            
            # 如果需要背景色
            if background_color is not None:
                # 获取文本边界框
                bbox = draw.textbbox(position, text, font=font)
                bg_rgb = (background_color[2], background_color[1], background_color[0])
                # 绘制背景矩形
                draw.rectangle(bbox, fill=bg_rgb)
            
            # 绘制文本
            draw.text(position, text, font=font, fill=rgb_color)
            
            # 转换回 OpenCV 格式
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            normal_logger.error(f"绘制中文文本失败: {str(e)}")
            # 如果失败，回退到 OpenCV 的英文字体
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_size/20, color, 2)
            return img

    @staticmethod
    def render_analysis_results(frame: np.ndarray, analysis_result: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        在视频帧上渲染分析结果（目标框、跟踪ID等）

        Args:
            frame: 原始视频帧
            analysis_result: 分析结果字典

        Returns:
            np.ndarray: 渲染后的视频帧
        """
        if analysis_result is None:
            # 添加调试信息：没有分析结果
            debug_frame = frame.copy()
            debug_frame = FrameRenderer._put_chinese_text(
                debug_frame, "无分析结果", (10, 30), 20, (0, 0, 255)
            )
            return debug_frame

        # 如果已经有预处理的预览帧，直接使用它
        if "preview_frame" in analysis_result and analysis_result["preview_frame"] is not None:
            preview = analysis_result["preview_frame"]
            # 确保尺寸一致
            if preview.shape[:2] != frame.shape[:2]:
                preview = cv2.resize(preview, (frame.shape[1], frame.shape[0]))
            return preview

        rendered_frame = frame.copy()
        height, width = frame.shape[:2]

        # 添加调试信息：分析结果状态
        debug_info = f"分析键: {list(analysis_result.keys())}"
        rendered_frame = FrameRenderer._put_chinese_text(
            rendered_frame, debug_info, (10, height - 50), 14, (255, 255, 0)
        )

        try:
            # 渲染检测结果
            detections = analysis_result.get("detections", [])
            detection_count = len(detections)

            # 添加检测数量信息
            detection_text = f"检测数量: {detection_count}"
            rendered_frame = FrameRenderer._put_chinese_text(
                rendered_frame, detection_text, (10, 60), 18, (0, 255, 0)
            )

            for i, det in enumerate(detections):
                try:
                    # 获取边界框
                    bbox = det.get("bbox", [])
                    if not bbox:
                        # 尝试其他可能的字段名
                        bbox = det.get("bbox_pixels", det.get("box", []))
                        if not bbox:
                            continue

                    # 处理不同格式的边界框
                    if isinstance(bbox, dict):
                        if all(k in bbox for k in ['x1', 'y1', 'x2', 'y2']):
                            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                        elif all(k in bbox for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                            x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                        else:
                            continue
                    elif isinstance(bbox, list) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                    else:
                        continue

                    # 转换为像素坐标
                    try:
                        if isinstance(x1, (int, float)) and isinstance(y1, (int, float)) and isinstance(x2, (int, float)) and isinstance(y2, (int, float)):
                            if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                                # 归一化坐标，转换为像素坐标
                                x1, y1 = int(x1 * width), int(y1 * height)
                                x2, y2 = int(x2 * width), int(y2 * height)
                            else:
                                # 已经是像素坐标
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        else:
                            continue
                    except (ValueError, TypeError):
                        continue

                    # 确保坐标在有效范围内
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width-1, x2), min(height-1, y2)

                    if x1 >= x2 or y1 >= y2:
                        continue

                    # 获取类别和置信度
                    class_name = det.get("class_name", det.get("class", "未知"))
                    confidence = float(det.get("confidence", det.get("score", 0)))

                    # 确定边界框颜色（基于类名）
                    # 使用一个简单的哈希函数来为不同类别生成不同的颜色
                    color_hash = hash(class_name) % 0xFFFFFF
                    r = (color_hash & 0xFF0000) >> 16
                    g = (color_hash & 0x00FF00) >> 8
                    b = color_hash & 0x0000FF
                    color = (b, g, r)  # OpenCV使用BGR顺序

                    # 绘制边界框
                    cv2.rectangle(rendered_frame, (x1, y1), (x2, y2), color, 2)

                    # 绘制标签（支持中文）
                    label = f"{class_name}: {confidence:.2f}"
                    # 使用支持中文的文本渲染
                    rendered_frame = FrameRenderer._put_chinese_text(
                        rendered_frame, label, (x1, y1 - 25), 16, (255, 255, 255), color
                    )

                except Exception as e:
                    # 忽略单个检测结果的渲染错误
                    continue

            # 渲染跟踪结果
            tracked_objects = analysis_result.get("tracked_objects", [])
            for track in tracked_objects:
                try:
                    # 获取边界框
                    bbox = track.get("bbox", [])
                    if not bbox:
                        # 尝试其他可能的字段名
                        bbox = track.get("bbox_pixels", track.get("box", []))
                        if not bbox:
                            continue

                    # 处理边界框（与检测结果类似）
                    if isinstance(bbox, dict):
                        if all(k in bbox for k in ['x1', 'y1', 'x2', 'y2']):
                            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                        elif all(k in bbox for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                            x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                        else:
                            continue
                    elif isinstance(bbox, list) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                    else:
                        continue

                    # 转换为像素坐标
                    try:
                        if isinstance(x1, (int, float)) and isinstance(y1, (int, float)) and isinstance(x2, (int, float)) and isinstance(y2, (int, float)):
                            if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                                # 归一化坐标，转换为像素坐标
                                x1, y1 = int(x1 * width), int(y1 * height)
                                x2, y2 = int(x2 * width), int(y2 * height)
                            else:
                                # 已经是像素坐标
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        else:
                            continue
                    except (ValueError, TypeError):
                        continue

                    # 确保坐标在有效范围内
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width-1, x2), min(height-1, y2)

                    if x1 >= x2 or y1 >= y2:
                        continue

                    # 获取跟踪ID
                    track_id = track.get("track_id", track.get("id", "未知"))

                    # 绘制跟踪边界框（使用不同颜色）
                    color = (255, 0, 0)  # 蓝色（BGR顺序）
                    cv2.rectangle(rendered_frame, (x1, y1), (x2, y2), color, 2)

                    # 绘制跟踪ID（支持中文）
                    track_label = f"ID: {track_id}"
                    rendered_frame = FrameRenderer._put_chinese_text(
                        rendered_frame, track_label, (x1, y2 + 20), 16, color
                    )

                except Exception as e:
                    # 忽略单个跟踪结果的渲染错误
                    continue

            # 添加视频信息到帧的右下角
            task_id = analysis_result.get('task_id', '无')
            frame_index = analysis_result.get('frame_index', 0)
            info_text = f"任务: {task_id} | 帧: {frame_index}"
            rendered_frame = FrameRenderer._put_chinese_text(
                rendered_frame, info_text, (10, height - 10), 14, (0, 255, 255)
            )

        except Exception as e:
            normal_logger.error(f"渲染分析结果时出错: {str(e)}")

        return rendered_frame
        
    @staticmethod
    def add_status_info(frame: np.ndarray, status, elapsed_seconds: float) -> np.ndarray:
        """
        添加状态信息到帧上
        
        Args:
            frame: 视频帧
            status: 任务状态
            elapsed_seconds: 已运行时间（秒）
            
        Returns:
            np.ndarray: 添加状态信息后的帧
        """
        h, w = frame.shape[:2]

        # 状态映射
        status_map = {
            TaskStatus.WAITING: "等待中",
            TaskStatus.PROCESSING: "处理中",
            TaskStatus.COMPLETED: "已完成",
            TaskStatus.FAILED: "失败",
            TaskStatus.STOPPED: "已停止"
        }

        status_text = status_map.get(status, f"未知状态({status})")

        # 根据状态设置颜色
        if status == TaskStatus.PROCESSING:
            color = (0, 255, 0)  # 绿色
        elif status == TaskStatus.WAITING:
            color = (255, 165, 0)  # 橙色
        elif status in [TaskStatus.FAILED, TaskStatus.STOPPED]:
            color = (0, 0, 255)  # 红色
        else:
            color = (255, 255, 255)  # 白色

        # 格式化运行时间
        minutes, seconds = divmod(elapsed_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h-70), (300, h-10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 添加状态文本（支持中文）
        status_label = f"状态: {status_text}"
        frame = FrameRenderer._put_chinese_text(
            frame, status_label, (20, h-50), 18, color
        )

        # 添加运行时间
        runtime_label = f"运行时间: {time_str}"
        frame = FrameRenderer._put_chinese_text(
            frame, runtime_label, (20, h-20), 18, (255, 255, 255)
        )

        return frame 