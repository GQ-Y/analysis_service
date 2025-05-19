"""
YOLO跟踪器模块
结合YOLO检测器和跟踪器实现目标跟踪功能
"""
import os
import time
import json
import asyncio
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from enum import Enum

from shared.utils.logger import get_normal_logger, get_exception_logger, get_test_logger
from core.analyzer.base_analyzer import TrackingAnalyzer
from core.analyzer.detection.yolo_detector import YOLODetector
from core.analyzer.tracking.tracker import Tracker, TrackerType
from core.exceptions import ModelLoadException, ProcessingException

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()

class YOLOTracker(TrackingAnalyzer):
    """YOLO目标跟踪器实现"""
    
    def __init__(self, model_code: Optional[str] = None, engine_type: int = 0, 
                 yolo_version: int = 0, device: str = "auto", 
                 tracker_type_name: str = "sort", **kwargs):
        """
        初始化YOLO跟踪器
        
        Args:
            model_code: 模型代码
            engine_type: 推理引擎类型
            yolo_version: YOLO版本
            device: 推理设备
            tracker_type: 跟踪器类型
            **kwargs: 其他参数
        """
        # 将tracker_type_name转换为TrackerType枚举
        try:
            tracker_type_enum = TrackerType[tracker_type_name.upper()]
        except KeyError:
            exception_logger.error(f"无效的跟踪器类型名称: '{tracker_type_name}'. 将使用默认的SORT。")
            tracker_type_enum = TrackerType.SORT
        
        # 调用父类的初始化方法，传递TrackerType枚举成员的value
        # 父类的__init__期望的是int类型的tracker_type
        super().__init__(model_code, engine_type, yolo_version, device, 
                         tracker_type=tracker_type_enum.value, **kwargs)
        
        self.detector = YOLODetector(model_code, engine_type, yolo_version, device)
        
        # self.tracker_type 在父类中已经是 TrackerType 枚举 (或应该是)。
        # 如果父类中 self.tracker_type 存储的是int, 那么日志中需要转换回名称
        # 假设父类也正确处理了 tracker_type (例如，在 _init_tracker 中使用它)
        current_tracker_type_name = tracker_type_enum.name 
        if hasattr(self, 'tracker_type') and isinstance(self.tracker_type, TrackerType):
             current_tracker_type_name = self.tracker_type.name
        elif hasattr(self, 'tracker_type') and isinstance(self.tracker_type, int):
            # 如果父类存的是int, 尝试从TrackerType反向查找名称
            try:
                current_tracker_type_name = TrackerType(self.tracker_type).name
            except ValueError:
                current_tracker_type_name = f"未知类型值({self.tracker_type})"

        normal_logger.info(f"YOLO跟踪器已初始化: 模型={model_code}, 检测器设备={self.detector.device}, 跟踪器类型={current_tracker_type_name}")
        test_logger.info(f"[初始化] YOLO跟踪器 (YOLOTracker): 模型={model_code}, 跟踪器={current_tracker_type_name}")
    
    async def load_model(self, model_code: str) -> bool:
        """
        加载模型
        
        Args:
            model_code: 模型代码
            
        Returns:
            bool: 是否成功加载模型
        """
        log_prefix = f"[模型加载-YOLOTracker] 模型={model_code}"
        test_logger.info(f"{log_prefix} | 开始加载检测模型...")
        try:
            await self.detector.load_model(model_code)
            self.current_model_code = self.detector.current_model_code 
            normal_logger.info(f"YOLO跟踪器成功加载检测模型: {self.current_model_code}")
            test_logger.info(f"{log_prefix} | 检测模型 {self.current_model_code} 加载成功。")
            
            actual_tracker_type_for_log = self.tracker_type.name if isinstance(self.tracker_type, Enum) else str(self.tracker_type)
            if actual_tracker_type_for_log == TrackerType.DEEP_SORT.name and hasattr(self.tracker, 'extractor'):
                test_logger.info(f"{log_prefix} | DeepSORT ReID模型应已在跟踪器初始化时加载。")
            return True
        except ModelLoadException as mle:
            exception_logger.exception(f"YOLO跟踪器加载检测模型 {model_code} 失败: {str(mle)}")
            test_logger.info(f"{log_prefix} | 检测模型 {model_code} 加载失败: {str(mle)}")
            return False
        except Exception as e:
            exception_logger.exception(f"YOLO跟踪器加载检测模型 {model_code} 时发生未知错误")
            test_logger.info(f"{log_prefix} | 检测模型 {model_code} 加载时发生未知错误: {str(e)}")
            return False
    
    async def detect(self, image: np.ndarray, task_name: Optional[str] = "未命名跟踪检测任务", **kwargs) -> Dict[str, Any]:
        """
        对输入图像进行目标检测和跟踪
        
        Args:
            image: BGR格式的输入图像
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 跟踪结果
        """
        log_prefix = f"[跟踪处理-YOLOTracker] 任务={task_name}, 模型={self.current_model_code or '未知'}"
        test_logger.info(f"{log_prefix} | 开始处理图像进行检测和跟踪")

        if self.detector.model is None:
            exception_logger.error(f"检测模型未加载，无法为任务 {task_name} 执行跟踪。")
            test_logger.info(f"{log_prefix} | 失败: 检测模型未加载")
            raise ProcessingException("检测模型未加载，无法执行跟踪。")

        detection_results = await self.detector.detect(image, task_name=f"{task_name}_detection_step", **kwargs)
        detections_for_tracker = []
        if detection_results and detection_results.get("detections"):
            for det in detection_results["detections"]:
                bbox = det["bbox"]
                x1 = bbox.get("x1")
                y1 = bbox.get("y1")
                x2 = bbox.get("x2")
                y2 = bbox.get("y2")
                score = det["confidence"]
                class_id = det["class_id"]
                if all(v is not None for v in [x1, y1, x2, y2, score, class_id]):
                    detections_for_tracker.append([x1, y1, x2, y2, score, class_id])
                else:
                    test_logger.warning(f"{log_prefix} | 检测结果中存在不完整数据，跳过: {det}")
        
        detections_np = np.array(detections_for_tracker)
        test_logger.info(f"{log_prefix} | 检测步骤完成，检测到 {len(detections_np)} 个目标送入跟踪器。")

        actual_tracker_type_for_check = self.tracker_type.value if isinstance(self.tracker_type, Enum) else self.tracker_type
        original_image_for_tracker = image if actual_tracker_type_for_check == TrackerType.DEEP_SORT.value else None
        tracked_objects_list = self.tracker.update(detections_np, original_image=original_image_for_tracker)
        test_logger.info(f"{log_prefix} | 跟踪器更新完成，得到 {len(tracked_objects_list)} 个跟踪ID。")

        final_results = {
            "detections": detection_results.get("detections", []),
            "tracking_results": tracked_objects_list,
            "pre_process_time": detection_results.get("pre_process_time", 0),
            "inference_time": detection_results.get("inference_time", 0),
            "post_process_time": detection_results.get("post_process_time", 0),
            "annotated_image_bytes": None, 
            "counts": self.tracker.get_counts() if hasattr(self.tracker, 'get_counts') else {},
        }

        save_images = kwargs.get("save_images", False)
        if save_images and tracked_objects_list:
            try:
                annotated_image_with_tracks = await self.draw_tracked_objects(image.copy(), tracked_objects_list)
                if hasattr(self.detector, '_encode_result_image'):
                     final_results["annotated_image_bytes"] = await self.detector._encode_result_image(annotated_image_with_tracks)
            except Exception as e:
                exception_logger.exception(f"为任务 {task_name} 绘制或保存跟踪图像时出错。")
                test_logger.info(f"{log_prefix} | 绘制或保存跟踪图像失败: {str(e)}")

        test_logger.info(f"{log_prefix} | 跟踪处理完成。")
        return final_results
    
    async def process_video_frame(self, frame: np.ndarray, frame_index: int, task_name: Optional[str] = "视频帧跟踪任务", **kwargs) -> Dict[str, Any]:
        """
        处理视频帧
        
        Args:
            frame: 视频帧
            frame_index: 帧索引
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        log_prefix = f"[跟踪帧处理] 任务={task_name}, 模型={self.current_model_code or '未知'}, 帧={frame_index}"
        test_logger.info(f"{log_prefix} | YOLOTracker 开始处理视频帧")
        kwargs['task_name'] = task_name
        results = await self.detect(frame, **kwargs)
        results["frame_index"] = frame_index
        test_logger.info(f"{log_prefix} | YOLOTracker 处理视频帧完成, 跟踪到ID数: {len(results.get('tracking_results', []))}")
        return results
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        detector_info = self.detector.model_info if self.detector else {"loaded": False}
        
        # 确定实际的跟踪器类型名称用于日志和返回
        actual_tracker_type_name = "未知"
        if hasattr(self.tracker, 'tracker_type') and isinstance(self.tracker.tracker_type, TrackerType):
            actual_tracker_type_name = self.tracker.tracker_type.name
        elif isinstance(self.tracker_type, TrackerType): # 如果 self.tracker_type 是枚举
            actual_tracker_type_name = self.tracker_type.name
        elif isinstance(self.tracker_type, int): # 如果是整数
            try:
                actual_tracker_type_name = TrackerType(self.tracker_type).name
            except ValueError:
                pass #保持"未知"
        
        tracker_info = {
            "tracker_type": actual_tracker_type_name,
            "max_age": self.tracker.max_age if self.tracker and hasattr(self.tracker, 'max_age') else self.max_age,
            "min_hits": self.tracker.min_hits if self.tracker and hasattr(self.tracker, 'min_hits') else self.min_hits,
            "iou_threshold": self.tracker.iou_threshold if self.tracker and hasattr(self.tracker, 'iou_threshold') else self.iou_threshold
        }
        combined_info = {
            "detector": detector_info,
            "tracker": tracker_info
        }
        test_logger.info(f"[模型信息] YOLOTracker 当前信息: {json.dumps(combined_info, ensure_ascii=False)}")
        return combined_info
    
    def release(self) -> None:
        """释放资源"""
        normal_logger.info(f"开始释放YOLOTracker资源 (模型: {self.current_model_code or '无'})...")
        test_logger.info(f"[资源释放] 开始释放YOLOTracker (模型: {self.current_model_code or '无'})")
        if self.detector:
            self.detector.release()
            normal_logger.info("内部YOLO检测器资源已释放。")
        if self.tracker:
            self.tracker = None 
            normal_logger.info("跟踪器实例已清除引用。")
        normal_logger.info("YOLOTracker资源释放完毕。")
        test_logger.info("[资源释放] YOLOTracker资源释放完成。")

    async def draw_tracked_objects(self, image: np.ndarray, tracked_objects: List[Dict[str, Any]]) -> np.ndarray:
        """在图像上绘制跟踪结果（包括ID和边界框）"""
        img_copy = image.copy()
        try:
            for obj in tracked_objects:
                track_id = obj.get("track_id", -1)
                bbox = obj.get("bbox") 
                class_id = obj.get("class_id", -1)
                confidence = obj.get("confidence", 0.0)
                class_name = obj.get("class_name", f"ID:{class_id}" if class_id != -1 else "未知")
                speed_kmh = obj.get("speed_kmh")

                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    color = self.detector._get_color_by_id(track_id) 
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"ID:{track_id} {class_name} ({confidence:.2f})"
                    if speed_kmh is not None:
                        label += f" {speed_kmh:.1f}km/h"
                    
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # 确保标签背景和文本在图像上方，如果空间不足则在下方
                    label_bg_y1 = y1 - text_height - baseline - 2
                    label_bg_y2 = y1 - baseline + 2
                    label_txt_y = y1 - baseline - 2
                    if label_bg_y1 < 0: # 如果上方空间不足
                        label_bg_y1 = y2 + baseline
                        label_bg_y2 = y2 + text_height + baseline + 4
                        label_txt_y = y2 + text_height + baseline

                    cv2.rectangle(img_copy, (x1, label_bg_y1), (x1 + text_width + 2, label_bg_y2), color, -1)
                    cv2.putText(img_copy, label, (x1 + 1, label_txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            return img_copy
        except Exception as e:
            exception_logger.exception("绘制跟踪对象时出错")
            return image 

    # async def _save_tracking_image(self, image, tracked_objects, task_name, log_prefix):
    # # ... 实现保存逻辑 ...
    # # 例如:
    #     try:
    #         if image is None:
    #             test_logger.warning(f"{log_prefix} | 保存跟踪图像失败: 输入图像为空")
    #             return None
    #         current_dir = os.getcwd()
    #         results_dir = os.path.join(current_dir, "results", "tracking") # 为跟踪结果创建单独子目录
    #         os.makedirs(results_dir, exist_ok=True)
    #         date_str = datetime.now().strftime("%Y%m%d")
    #         date_dir = os.path.join(results_dir, date_str)
    #         os.makedirs(date_dir, exist_ok=True)
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    #         task_prefix_name = f"{task_name if task_name and task_name != '未命名跟踪检测任务' else 'tracking'}_"
    #         # 文件名可以包含跟踪到的ID数量等信息
    #         filename = f"{task_prefix_name}tracked_{len(tracked_objects)}_{timestamp}.jpg"
    #         file_path = os.path.join(date_dir, filename)
    #         success = cv2.imwrite(file_path, image)
    #         if success and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    #             test_logger.info(f"{log_prefix} | 跟踪结果图像已保存: {file_path}")
    #             return os.path.join("results", "tracking", date_str, filename)
    #         else:
    #             test_logger.warning(f"{log_prefix} | 保存跟踪结果图像失败或文件无效: {file_path}")
    #             return None
    #     except Exception as e:
    #         exception_logger.exception(f"任务 {task_name} 保存跟踪图像时出错。")
    #         return None

# 注意：父类TrackingAnalyzer中的_init_tracker方法也需要适配新的日志记录器。
# 如果它在base_analyzer.py中，该文件已更新。
# 如果它在其他地方或被此文件覆盖，需要确保那里的日志也已更新。
