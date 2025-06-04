"""
YOLO检测分析器模块 - 重构版
基于Ultralytics YOLO的对象检测分析器
"""
import os
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from core.analyzer.base_analyzer import DetectionAnalyzer
from core.analyzer.detection.yolo_detector import YOLODetector
from core.analyzer.registry import register_analyzer
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

@register_analyzer("detection")
class YOLODetectionAnalyzer(DetectionAnalyzer):
    """YOLO检测分析器 - 重构版，支持插件化加载"""

    def __init__(self, model_code: Optional[str] = None, device: str = "auto", **kwargs):
        """
        初始化YOLO检测分析器
        
        Args:
            model_code: 模型代码
            device: 推理设备 ("cpu", "cuda", "auto")
            **kwargs: 其他参数，包括：
                - custom_weights_path: 自定义权重路径
                - half_precision: 是否使用半精度
                - confidence: 置信度阈值
                - iou_threshold: IoU阈值
                - max_detections: 最大检测目标数
                - classes: 类别列表
                - nested_detection: 是否启用嵌套检测
        """
        # 先初始化检测器，这样在父类调用load_model时detector已经存在
        self.detector = YOLODetector(model_code, device, **kwargs)
        
        # 检测参数
        self.confidence = kwargs.get("confidence", 0.25)
        self.iou_threshold = kwargs.get("iou_threshold", 0.45)
        self.max_detections = kwargs.get("max_detections", 100)
        self.classes = kwargs.get("classes")
        self.nested_detection = kwargs.get("nested_detection", False)
        
        # 检测统计
        self._detection_count = 0
        self._total_detection_time = 0
        self._frame_count = 0
        
        # 初始化current_model_code属性
        self.current_model_code = model_code
        
        # 调用父类初始化
        super().__init__(model_code, device, **kwargs)
        
        normal_logger.info(f"YOLO检测分析器初始化: 置信度={self.confidence}, IoU阈值={self.iou_threshold}")
        if self.nested_detection:
            normal_logger.info("启用嵌套检测")

    async def load_model(self, model_code: str) -> bool:
        """
        加载模型
        
        Args:
            model_code: 模型代码
            
        Returns:
            bool: 是否成功加载模型
        """
        # 如果模型已加载且代码相同，则直接返回成功
        if self.loaded and hasattr(self, 'current_model_code') and self.current_model_code == model_code:
            normal_logger.info(f"模型 {model_code} 已为YOLO检测分析器加载。")
            return True

        try:
            # 加载检测器模型
            success = await self.detector.load_model(model_code)
            if success:
                self.current_model_code = model_code
                self.loaded = True  #  关键修改：设置 loaded 标志为 True
                normal_logger.info(f"YOLO检测分析器成功加载模型: {model_code}")
            else:
                self.loaded = False # 确保如果加载失败，loaded 状态也明确
            return success
        except Exception as e:
            exception_logger.exception(f"YOLO检测分析器加载模型失败: {e}")
            self.loaded = False # 发生异常时也标记为未加载
            return False

    async def detect(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        检测图像中的目标，并整合来自 YOLODetector 的详细结果。

        Args:
            frame: 输入图像
            **kwargs: 其他参数，传递给 YOLODetector.detect，包括：
                - confidence: 置信度阈值
                - iou_threshold: IoU阈值
                - max_detections: 最大检测目标数
                - classes: 类别列表
                - return_image: 是否返回标注后的图像
                - imgsz: 推理图像尺寸

        Returns:
            Dict[str, Any]: 来自 YOLODetector.detect 的完整结果，可能包含额外的分析器级别统计。
                           预期结构见 YOLODetector.detect 文档字符串。
        """
        analyzer_level_start_time = time.perf_counter()
        frame_count_for_log = self._frame_count + 1 # 用于日志，_frame_count 在 process_video_frame 中更新
        
        normal_logger.info(f"YOLODetectionAnalyzer: 开始检测第 {frame_count_for_log} 帧 (粗略计数), 帧大小: {frame.shape}")
        
        if not self.loaded:
            normal_logger.warning(f"YOLODetectionAnalyzer: 模型 {self.current_model_code or '未指定'} 未加载，无法执行检测。")
            analyzer_level_end_time = time.perf_counter()
            return {
                "success": False,
                "error": f"模型 {self.current_model_code or '未指定'} 未加载",
                "detections": [],
                "applied_config": {},
                "image_info": {"original_height": frame.shape[0], "original_width": frame.shape[1]},
                "timing_stats": {
                    "analyzer_detect_call_time_ms": (analyzer_level_end_time - analyzer_level_start_time) * 1000
                },
                "annotated_image_base64": None
            }
        
        try:
            # 调用底层 YOLO 检测器的 detect 方法
            # 它会返回包含 "success", "detections", "applied_config", "image_info", "timing_stats", "annotated_image_base64"
            detector_result = await self.detector.detect(
                frame,
                **kwargs  # 将所有相关参数传递下去
            )
            
            analyzer_level_end_time = time.perf_counter()
            analyzer_detect_call_duration_ms = (analyzer_level_end_time - analyzer_level_start_time) * 1000

            if not detector_result.get("success", False):
                normal_logger.warning(f"YOLODetectionAnalyzer: 底层检测器报告失败: {detector_result.get('error', '未知错误')}")
                # 即使失败，也尝试添加 analyzer 级别的计时
                if "timing_stats" not in detector_result or detector_result["timing_stats"] is None:
                    detector_result["timing_stats"] = {}
                detector_result["timing_stats"]["analyzer_detect_call_time_ms"] = analyzer_detect_call_duration_ms
                return detector_result
            
            # 更新分析器级别的统计 (如果需要的话，但现在 detector_result["timing_stats"] 已经很详细了)
            # self._detection_count 在 process_video_frame 中处理
            # self._total_detection_time 也可以在 process_video_frame 中累加 analyzer_detect_call_duration_ms
            
            # 将 analyzer_detect_call_time_ms 添加到底层检测器的 timing_stats 中
            # 这是 YOLODetectionAnalyzer.detect 方法本身的开销，不包括YOLODetector.detect内部的计时
            if "timing_stats" not in detector_result or detector_result["timing_stats"] is None:
                 detector_result["timing_stats"] = {}
            detector_result["timing_stats"]["analyzer_detect_call_time_ms"] = analyzer_detect_call_duration_ms

            num_detections = len(detector_result.get("detections", []))
            normal_logger.info(f"YOLODetectionAnalyzer: 成功处理第 {frame_count_for_log} 帧 (粗略计数)。耗时 (detect call): {analyzer_detect_call_duration_ms:.2f} ms。检测到 {num_detections} 个目标。")
            if num_detections > 0:
                # 日志截断，避免过长的输出
                log_dets = detector_result["detections"][:3]
                normal_logger.info(f"YOLODetectionAnalyzer: 部分检测结果: {log_dets}")
            
            # 直接返回从 YOLODetector 获得的完整结果，其中已包含详细信息
            # 如果启用了嵌套检测，则处理嵌套关系
            if self.nested_detection and detector_result.get("detections"):
                normal_logger.info(f"YOLODetectionAnalyzer: 启用嵌套检测，处理 {len(detector_result['detections'])} 个检测结果")
                try:
                    # 转换检测结果格式用于嵌套检测处理
                    detections_for_nested = []
                    for det in detector_result["detections"]:
                        # 使用像素坐标bbox进行嵌套检测
                        if "bbox_pixels" in det:
                            bbox = det["bbox_pixels"]  # [x1, y1, x2, y2]
                            nested_det = {
                                "bbox": bbox,
                                "class_id": det.get("class_id", 0),
                                "class_name": det.get("class_name", "unknown"),
                                "confidence": det.get("confidence", 0),
                                "id": det.get("id", 0)
                            }
                            detections_for_nested.append(nested_det)
                    
                    # 处理嵌套检测关系
                    if detections_for_nested:
                        processed_nested = self._process_nested_detections(detections_for_nested)
                        
                        # 将嵌套关系信息合并回原始检测结果
                        for i, det in enumerate(detector_result["detections"]):
                            if i < len(processed_nested):
                                det["contains"] = processed_nested[i].get("contains", [])
                                det["contained_by"] = processed_nested[i].get("contained_by", [])
                                
                        normal_logger.info(f"YOLODetectionAnalyzer: 嵌套检测处理完成")
                    
                except Exception as e:
                    exception_logger.exception(f"YOLODetectionAnalyzer: 嵌套检测处理失败: {str(e)}")
                    # 嵌套检测失败不影响主要检测结果
            
            return detector_result
            
        except Exception as e:
            exception_logger.exception(f"YOLODetectionAnalyzer 在执行 detect 时发生严重错误: {e}")
            analyzer_level_end_time = time.perf_counter()
            return {
                "success": False,
                "error": f"YOLODetectionAnalyzer 内部错误: {str(e)}",
                "detections": [],
                "applied_config": kwargs, # 返回传入的配置作为参考
                "image_info": {"original_height": frame.shape[0], "original_width": frame.shape[1]},
                "timing_stats": {
                    "analyzer_detect_call_time_ms": (analyzer_level_end_time - analyzer_level_start_time) * 1000
                },
                "annotated_image_base64": None
            }

    async def process_video_frame(self, frame: np.ndarray, frame_index: int, **kwargs) -> Dict[str, Any]:
        """
        处理视频帧，调用 detect 方法，并添加帧特定信息和分析器处理总耗时。
        
        Args:
            frame: 视频帧
            frame_index: 帧索引
            **kwargs: 其他参数，传递给 detect 方法
            
        Returns:
            Dict[str, Any]: 包含完整检测结果以及帧信息的字典。
                           新增 timing_stats.analyzer_process_video_frame_time_ms
        """
        process_frame_start_time = time.perf_counter()
        self._frame_count += 1 # 实际的帧计数器
        
        # 执行核心检测逻辑
        # kwargs 包含了如 confidence, iou_threshold, classes, return_image, imgsz 等
        # 这些参数应该从任务配置中传递下来
        detection_result = await self.detect(frame, **kwargs)
        
        # 添加帧相关的固定信息到结果中
        detection_result["frame_info"] = {
            "frame_index_in_stream": frame_index, # 原始帧序号
            "analyzer_frame_counter": self._frame_count # 分析器处理的第几帧
        }

        # 添加 YOLODetectionAnalyzer.process_video_frame 本身的耗时
        process_frame_end_time = time.perf_counter()
        process_video_frame_duration_ms = (process_frame_end_time - process_frame_start_time) * 1000
        
        if "timing_stats" not in detection_result or detection_result["timing_stats"] is None:
            detection_result["timing_stats"] = {}
        detection_result["timing_stats"]["analyzer_process_video_frame_time_ms"] = process_video_frame_duration_ms

        # 更新分析器级别的累计总耗时 (可选)
        # self._total_detection_time += process_video_frame_duration_ms 
        # (如果需要计算平均值等，现在 detector_result["timing_stats"] 已经很详细了)
        
        return detection_result

    def _process_nested_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理嵌套检测关系
        
        Args:
            detections: 检测结果列表
            
        Returns:
            List[Dict[str, Any]]: 处理后的检测结果
        """
        # 遍历所有检测结果，计算IoU并确定嵌套关系
        for i, det_i in enumerate(detections):
            det_i["contains"] = []
            det_i["contained_by"] = []
            
            x1_i, y1_i, x2_i, y2_i = det_i["bbox"]
            area_i = (x2_i - x1_i) * (y2_i - y1_i)
            
            for j, det_j in enumerate(detections):
                if i == j:
                    continue
                    
                x1_j, y1_j, x2_j, y2_j = det_j["bbox"]
                
                # 计算重叠区域
                x1_inter = max(x1_i, x1_j)
                y1_inter = max(y1_i, y1_j)
                x2_inter = min(x2_i, x2_j)
                y2_inter = min(y2_i, y2_j)
                
                if x1_inter < x2_inter and y1_inter < y2_inter:
                    area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                    area_j = (x2_j - x1_j) * (y2_j - y1_j)
                    
                    # 判断嵌套关系
                    if area_inter / area_j > 0.95:  # j基本被i包含
                        det_i["contains"].append(j)
                        det_j["contained_by"].append(i)
                        
        return detections

    @property
    def model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        base_info = {
            "analyzer_type": "detection",
            "analyzer_name": self.__class__.__name__,
            "model_code": self.current_model_code,
            "device": self.device,
            "half_precision": self.half_precision,
            "custom_weights_path": self.custom_weights_path,
            "confidence": self.confidence,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "nested_detection": self.nested_detection
        }
        
        # 添加检测器信息
        if hasattr(self.detector, "model_info"):
            detector_info = self.detector.model_info
            base_info.update(detector_info)
            
        return base_info

    def release(self) -> None:
        """释放资源"""
        if self.detector:
            self.detector.release()
            normal_logger.info("YOLO检测分析器已释放资源")
