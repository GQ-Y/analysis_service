"""
优化的帧渲染器
专门为直播流场景优化，减少卡顿和性能瓶颈
"""
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np
from datetime import datetime
import threading
import time
from collections import deque

from shared.utils.logger import get_normal_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)


class OptimizedFrameRenderer:
    """优化的帧渲染器 - 专为直播流设计"""
    
    def __init__(self):
        self.font_scale = 0.6
        self.thickness = 2
        self.line_type = cv2.LINE_AA
        
        # 颜色缓存 - 避免重复计算颜色
        self.color_cache = {}
        
        # 文字缓存 - 缓存渲染后的文字图像
        self.text_cache = {}
        self.text_cache_max_size = 100
        
        # 渲染统计
        self.render_times = deque(maxlen=100)  # 保留最近100次渲染时间
        self.lock = threading.Lock()
        
        # 性能开关
        self.enable_text_rendering = True  # 是否启用文字渲染
        self.enable_debug_info = False     # 是否显示调试信息
        self.max_detections_render = 50   # 最大渲染目标数量
        self.use_chinese_fonts = False    # 是否使用中文字体（性能模式控制）
        
        normal_logger.info("优化帧渲染器初始化完成")
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        获取类别颜色（带缓存）
        
        Args:
            class_name: 类别名称
            
        Returns:
            Tuple[int, int, int]: BGR颜色值
        """
        if class_name not in self.color_cache:
            # 生成稳定的颜色
            color_hash = hash(class_name) % 0xFFFFFF
            r = (color_hash & 0xFF0000) >> 16
            g = (color_hash & 0x00FF00) >> 8
            b = color_hash & 0x0000FF
            
            # 确保颜色亮度足够
            if r + g + b < 200:
                r, g, b = min(255, r + 100), min(255, g + 100), min(255, b + 100)
            
            self.color_cache[class_name] = (b, g, r)  # OpenCV使用BGR
        
        return self.color_cache[class_name]
    
    def parse_bbox(self, det: Dict[str, Any], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        """
        快速解析边界框坐标
        
        Args:
            det: 检测结果
            width: 图像宽度
            height: 图像高度
            
        Returns:
            Optional[Tuple[int, int, int, int]]: (x1, y1, x2, y2) 或 None
        """
        # 优先使用bbox_pixels字段（YOLO输出）
        bbox = det.get("bbox_pixels") or det.get("bbox") or det.get("box")
        if not bbox:
            return None
        
        try:
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            elif isinstance(bbox, dict):
                if all(k in bbox for k in ['x1', 'y1', 'x2', 'y2']):
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                else:
                    return None
            else:
                return None
            
            # 坐标转换
            if all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
                if all(coord <= 1.0 for coord in [x1, y1, x2, y2]):
                    # 归一化坐标
                    x1, y1 = int(x1 * width), int(y1 * height)
                    x2, y2 = int(x2 * width), int(y2 * height)
                else:
                    # 像素坐标
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 边界检查
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width-1, x2), min(height-1, y2)
                
                if x1 < x2 and y1 < y2:
                    return (x1, y1, x2, y2)
            
        except (ValueError, TypeError, KeyError):
            pass
        
        return None
    
    def render_text_optimized(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                             color: Tuple[int, int, int]) -> None:
        """
        优化的文字渲染（直接修改原图）
        
        Args:
            frame: 图像帧
            text: 文字内容
            position: 位置 (x, y)
            color: 颜色 (B, G, R)
        """
        if not self.enable_text_rendering:
            return
        
        # 缓存键
        cache_key = f"{text}_{self.font_scale}_{color}"
        
        if self.use_chinese_fonts:
            # 高质量模式：使用中文字体（较慢但支持中文）
            if cache_key not in self.text_cache:
                # 只有高质量模式才使用中文渲染
                try:
                    from services.video.utils.frame_renderer import FrameRenderer
                    # 创建一个小的临时图像来渲染文字
                    temp_frame = np.zeros((100, 400, 3), dtype=np.uint8)
                    rendered_temp = FrameRenderer._put_chinese_text(
                        temp_frame, text, (5, 30), int(self.font_scale * 20), color
                    )
                    self.text_cache[cache_key] = rendered_temp
                    
                    # 限制缓存大小
                    if len(self.text_cache) > self.text_cache_max_size:
                        # 移除最旧的缓存项
                        oldest_key = next(iter(self.text_cache))
                        del self.text_cache[oldest_key]
                        
                except Exception as e:
                    # 如果中文渲染失败，回退到OpenCV
                    self.use_chinese_fonts = False
                    normal_logger.debug(f"中文渲染失败，回退到OpenCV: {e}")
        
        if not self.use_chinese_fonts:
            # 性能模式：使用OpenCV原生字体（快速但只支持英文）
            cv2.putText(frame, text, position, 
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 
                       color, 1, self.line_type)

    def render_detection_fast(self, frame: np.ndarray, det: Dict[str, Any], 
                            bbox: Tuple[int, int, int, int]) -> None:
        """
        快速渲染单个检测结果（直接修改原图）
        
        Args:
            frame: 图像帧
            det: 检测结果
            bbox: 边界框坐标 (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bbox
        
        # 获取类别信息
        class_name = det.get("class_name", det.get("class", "obj"))
        confidence = float(det.get("confidence", det.get("score", 0)))
        
        # 获取颜色
        color = self.get_class_color(class_name)
        
        # 绘制边界框 - 使用更粗的线条提高可见性
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
        
        # 优化的文字渲染
        if self.enable_text_rendering:
            # 简化的标签，减少处理开销
            if confidence >= 0.01:
                label = f"{class_name}:{confidence:.2f}"
            else:
                label = class_name
            
            # 计算文字位置
            text_x, text_y = x1, y1 - 10
            if text_y < 20:
                text_y = y1 + 20
            
            # 绘制文字背景（可选）
            if self.use_chinese_fonts:
                # 高质量模式：绘制背景
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                           self.font_scale, 1)[0]
                cv2.rectangle(frame, 
                             (text_x - 2, text_y - label_size[1] - 4),
                             (text_x + label_size[0] + 2, text_y + 4),
                             color, -1)
                text_color = (255, 255, 255)
            else:
                text_color = color
            
            # 使用优化的文字渲染
            self.render_text_optimized(frame, label, (text_x, text_y), text_color)
    
    def render_analysis_results_optimized(self, frame: np.ndarray, 
                                        analysis_result: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        优化的分析结果渲染
        
        Args:
            frame: 原始帧
            analysis_result: 分析结果
            
        Returns:
            np.ndarray: 渲染后的帧
        """
        start_time = time.perf_counter()
        
        # 如果没有分析结果，直接返回原帧
        if not analysis_result:
            return frame
        
        # 如果有预处理的预览帧，优先使用
        if "preview_frame" in analysis_result and analysis_result["preview_frame"] is not None:
            preview = analysis_result["preview_frame"]
            if preview.shape[:2] != frame.shape[:2]:
                preview = cv2.resize(preview, (frame.shape[1], frame.shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
            return preview
        
        # 获取帧尺寸
        height, width = frame.shape[:2]
        
        # 创建结果帧的副本（仅一次复制）
        result_frame = frame.copy()
        
        # 处理检测结果
        detections = analysis_result.get("detections", [])
        detection_count = len(detections)
        
        # 限制渲染的目标数量，避免性能问题
        max_render = min(detection_count, self.max_detections_render)
        
        # 按置信度排序，优先渲染高置信度目标
        if detection_count > max_render:
            detections = sorted(detections, 
                              key=lambda x: float(x.get("confidence", 0)), 
                              reverse=True)[:max_render]
        
        # 批量渲染检测结果
        rendered_count = 0
        for det in detections:
            bbox = self.parse_bbox(det, width, height)
            if bbox:
                self.render_detection_fast(result_frame, det, bbox)
                rendered_count += 1
        
        # 可选的调试信息
        if self.enable_debug_info:
            info_text = f"检测: {rendered_count}/{detection_count}"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 记录渲染时间
        render_time = (time.perf_counter() - start_time) * 1000
        with self.lock:
            self.render_times.append(render_time)
        
        return result_frame
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        获取性能统计信息
        
        Returns:
            Dict[str, float]: 性能统计
        """
        with self.lock:
            if not self.render_times:
                return {"avg_render_time_ms": 0, "max_render_time_ms": 0, "min_render_time_ms": 0}
            
            times = list(self.render_times)
            return {
                "avg_render_time_ms": sum(times) / len(times),
                "max_render_time_ms": max(times),
                "min_render_time_ms": min(times),
                "render_count": len(times)
            }
    
    def set_performance_mode(self, mode: str):
        """
        设置性能模式
        
        Args:
            mode: 性能模式 - "high_quality", "balanced", "high_performance"
        """
        if mode == "high_quality":
            self.enable_text_rendering = True
            self.enable_debug_info = True
            self.max_detections_render = 100
            self.thickness = 2
            self.use_chinese_fonts = True
        elif mode == "balanced":
            self.enable_text_rendering = True
            self.enable_debug_info = False
            self.max_detections_render = 50
            self.thickness = 2
            self.use_chinese_fonts = False
        elif mode == "high_performance":
            self.enable_text_rendering = False  # 关闭文字渲染
            self.enable_debug_info = False
            self.max_detections_render = 30
            self.thickness = 1
            self.use_chinese_fonts = False
        
        normal_logger.info(f"帧渲染器性能模式设置为: {mode}")

    def render_analysis_results(self, frame: np.ndarray, 
                              analysis_result: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        兼容性方法：渲染分析结果（调用优化版本）
        
        Args:
            frame: 原始帧
            analysis_result: 分析结果
            
        Returns:
            np.ndarray: 渲染后的帧
        """
        return self.render_analysis_results_optimized(frame, analysis_result)

    def add_status_info(self, frame: np.ndarray, status: str, elapsed_seconds: float) -> np.ndarray:
        """
        在帧上添加状态信息
        
        Args:
            frame: 原始帧
            status: 状态信息
            elapsed_seconds: 运行时间（秒）
            
        Returns:
            np.ndarray: 添加状态信息后的帧
        """
        # 创建结果帧的副本
        result_frame = frame.copy()
        
        # 格式化状态信息
        status_text = f"状态: {status}"
        time_text = f"运行时间: {elapsed_seconds:.1f}s"
        
        # 在左上角添加状态信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)  # 绿色
        thickness = 2
        
        # 添加半透明背景
        text_height = 50
        cv2.rectangle(result_frame, (0, 0), (300, text_height), (0, 0, 0), -1)
        cv2.rectangle(result_frame, (0, 0), (300, text_height), (255, 255, 255), 1)
        
        # 添加状态文本
        cv2.putText(result_frame, status_text, (10, 20), 
                   font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(result_frame, time_text, (10, 40), 
                   font, font_scale, color, thickness, cv2.LINE_AA)
        
        return result_frame

    @staticmethod 
    def _put_chinese_text(img: np.ndarray, text: str, position: tuple, font_size: int = 16, 
                         color: tuple = (255, 255, 255), background_color: Optional[tuple] = None) -> np.ndarray:
        """
        兼容性方法：优化的中文文本渲染
        
        在高性能模式下，这个方法会回退到OpenCV的英文字体以提高性能
        
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
        # 检查全局渲染器的性能模式
        if optimized_renderer and not optimized_renderer.use_chinese_fonts:
            # 高性能模式：使用OpenCV原生字体
            result_img = img.copy()
            
            # 绘制背景（如果需要）
            if background_color is not None:
                # 估算文本大小
                font_scale = font_size / 20.0
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                x, y = position
                cv2.rectangle(result_img, 
                             (x - 2, y - text_size[1] - 4),
                             (x + text_size[0] + 2, y + 4),
                             background_color, -1)
            
            # 绘制文本
            cv2.putText(result_img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_size / 20.0, color, 2, cv2.LINE_AA)
            return result_img
        else:
            # 高质量模式：使用原始的中文渲染
            try:
                from services.video.utils.frame_renderer import FrameRenderer
                return FrameRenderer._put_chinese_text(img, text, position, font_size, color, background_color)
            except Exception as e:
                normal_logger.debug(f"中文渲染失败，回退到OpenCV: {e}")
                # 回退到OpenCV渲染
                result_img = img.copy()
                cv2.putText(result_img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                           font_size / 20.0, color, 2, cv2.LINE_AA)
                return result_img


# 全局优化渲染器实例
optimized_renderer = OptimizedFrameRenderer() 