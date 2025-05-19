"""
目标跟踪器模块
实现基于SORT、ByteTrack、DeepSORT等算法的目标跟踪功能
"""
import os
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple, Union
from shared.utils.logger import get_normal_logger, get_exception_logger, get_test_logger
from core.config import settings
import time
from collections import defaultdict, OrderedDict, deque
import math
from enum import Enum

# 使用新的日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()

class TrackerType(Enum):
    """跟踪器类型枚举"""
    SORT = 0
    BYTE_TRACK = 1
    DEEP_SORT = 2

class Tracker:
    """目标跟踪器类"""

    def __init__(self, tracker_type: TrackerType = TrackerType.SORT,
                 max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3,
                 use_byte: bool = False, # ByteTrack特有参数，现在整合到tracker_type
                 model_path: Optional[str] = None, # DeepSORT模型路径
                 config_path: Optional[str] = None, # DeepSORT配置路径
                 device: str = "auto"):
        """
        初始化跟踪器

        Args:
            tracker_type: 跟踪器类型
            max_age: 最大失活帧数
            min_hits: 最小命中次数
            iou_threshold: IoU阈值
            model_path: DeepSORT模型路径
            config_path: DeepSORT配置路径
            device: 推理设备
        """
        self.tracker_type = tracker_type
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.device = device

        self.tracker = None
        self._init_tracker(model_path, config_path)

        # 计数相关
        self.counting_line: Optional[List[Tuple[int, int]]] = None
        self.counting_enabled: bool = False
        self.crossed_ids: set = set()
        self.counts: Dict[str, int] = {"in": 0, "out": 0, "total": 0} # 方向可以更通用

        # 速度估计相关
        self.speed_estimation_enabled: bool = False
        self.track_history: Dict[int, deque] = OrderedDict() # 存储轨迹点和时间戳
        self.pixels_per_meter: float = 100.0
        self.fps: float = 25.0

        normal_logger.info(f"跟踪器已初始化: 类型={tracker_type.name}, 最大失活帧数={max_age}, 最小命中数={min_hits}, IoU阈值={iou_threshold}")
        test_logger.info(f"[初始化] 跟踪器: 类型={tracker_type.name}, 最大失活帧数={max_age}, 最小命中数={min_hits}, IoU阈值={iou_threshold}")

    def _init_tracker(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """根据类型初始化具体的跟踪算法"""
        try:
            if self.tracker_type == TrackerType.SORT:
                from .sort_tracker import Sort
                self.tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_threshold)
                normal_logger.info("SORT跟踪器已初始化。")
                test_logger.info("[初始化] SORT跟踪器创建成功。")
            elif self.tracker_type == TrackerType.BYTE_TRACK:
                from .byte_tracker import BYTETracker
                # ByteTrack有自己的参数，可以从外部传入或使用默认值
                # 例如: track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30
                self.tracker = BYTETracker(track_thresh=0.5, track_buffer=self.max_age, 
                                           match_thresh=self.iou_threshold, frame_rate=30)
                normal_logger.info("ByteTrack跟踪器已初始化。")
                test_logger.info("[初始化] ByteTrack跟踪器创建成功。")
            elif self.tracker_type == TrackerType.DEEP_SORT:
                # DeepSORT的初始化可能更复杂，需要模型路径等
                try:
                    from deep_sort_pytorch.utils.parser import get_config
                    from deep_sort_pytorch.deep_sort import DeepSort
                    
                    cfg = get_config()
                    if config_path and os.path.exists(config_path):
                        cfg.merge_from_file(config_path)
                    else:
                        normal_logger.warning(f"DeepSORT配置文件 {config_path} 未找到或未提供，使用默认配置。")
                    
                    # 确定设备ID
                    device_id = -1 # CPU
                    if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()):
                        if torch.cuda.is_available():
                            device_id = 0 # 假设使用第一个GPU
                        else:
                            normal_logger.warning("请求CUDA设备但CUDA不可用，DeepSORT将使用CPU。")
                    
                    # 使用更灵活的方式查找模型文件
                    reid_model_path = model_path
                    if not reid_model_path or not os.path.exists(reid_model_path):
                        normal_logger.warning(f"DeepSORT Reid模型路径 '{reid_model_path}' 无效或未提供。尝试查找默认模型 ckpt.t7")
                        # 尝试在常见位置或项目特定位置查找
                        default_model_name = "ckpt.t7"
                        possible_paths = [
                            os.path.join("data", "models", "deepsort", default_model_name),
                            os.path.join(".", default_model_name) # 当前目录
                        ]
                        for p_path in possible_paths:
                            if os.path.exists(p_path):
                                reid_model_path = p_path
                                normal_logger.info(f"找到默认DeepSORT Reid模型: {reid_model_path}")
                                break
                        if not reid_model_path or not os.path.exists(reid_model_path):
                            exception_logger.error("无法找到DeepSORT Reid模型 (ckpt.t7)。请确保模型文件存在。")
                            raise FileNotFoundError("DeepSORT Reid模型文件未找到。")

                    self.tracker = DeepSort(
                        reid_model_path, 
                        max_dist=cfg.DEEPSORT.MAX_DIST, 
                        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=self.max_age, 
                        n_init=cfg.DEEPSORT.N_INIT, 
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=(device_id != -1)
                    )
                    normal_logger.info(f"DeepSORT跟踪器已初始化，ReID模型: {reid_model_path}, 设备ID: {device_id}")
                    test_logger.info("[初始化] DeepSORT跟踪器创建成功。")
                except ImportError:
                    exception_logger.exception("DeepSORT相关库 (deep_sort_pytorch) 未安装，无法使用DeepSORT跟踪器。")
                    raise ImportError("DeepSORT相关库未安装。")
                except Exception as e:
                    exception_logger.exception(f"初始化DeepSORT跟踪器失败: {str(e)}")
                    raise
            else:
                exception_logger.error(f"不支持的跟踪器类型: {self.tracker_type}")
                raise ValueError(f"不支持的跟踪器类型: {self.tracker_type}")
        except Exception as e:
            exception_logger.exception(f"初始化跟踪器 ({self.tracker_type.name}) 失败")
            # 可以选择抛出或回退到SORT
            if self.tracker_type != TrackerType.SORT:
                normal_logger.warning(f"由于初始化失败，回退到SORT跟踪器。")
                test_logger.info(f"[初始化] {self.tracker_type.name} 初始化失败，回退到SORT。")
                self.tracker_type = TrackerType.SORT
                self._init_tracker() # 重新尝试用SORT初始化
            else:
                raise # 如果SORT也失败，则抛出异常

    def update(self, detections: np.ndarray, original_image: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """更新跟踪器状态"""
        if self.tracker is None:
            exception_logger.error("跟踪器未初始化，无法更新。")
            return []

        # 确保detections是 (N, 5) 或 (N, 6) 的numpy数组 [x1, y1, x2, y2, score, (class_id)]
        if not isinstance(detections, np.ndarray) or detections.ndim != 2 or detections.shape[1] < 5:
            # exception_logger.warning(f"无效的检测数据格式: {detections}. 应为 (N, 5+) numpy数组。跳过此帧跟踪。")
            # test_logger.info(f"[跟踪更新] 无效检测数据格式，跳过此帧。数据: {detections}")
            # 即使检测为空，也应该调用update，让跟踪器知道时间流逝
            if detections.shape[0] == 0: # 如果是空检测，创建一个兼容形状的空数组
                 pass # SORT 和 ByteTrack 应该能处理空检测的np.array([])
            else:
                exception_logger.warning(f"无效的检测数据格式，形状: {detections.shape if isinstance(detections, np.ndarray) else type(detections)}. 跳过此帧跟踪。")
                test_logger.info(f"[跟踪更新] 无效检测数据格式，跳过此帧。")
                return []

        tracked_objects = []
        log_prefix = f"[跟踪更新] 类型={self.tracker_type.name}"
        test_logger.info(f"{log_prefix} | 输入检测数: {len(detections)}")

        try:
            if self.tracker_type == TrackerType.SORT:
                # SORT 需要 [x1, y1, x2, y2, score]
                tracks = self.tracker.update(detections[:, :5]) # 取前5列
                for track in tracks:
                    x1, y1, x2, y2, track_id = track[:5]
                    # SORT不直接提供类别和置信度，需要从原始检测中匹配或单独处理
                    tracked_objects.append({
                        "track_id": int(track_id),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": detections[0, 4] if len(detections) > 0 and len(track) < 6 else (track[5] if len(track) > 5 else 0.0), # 尝试获取置信度
                        "class_id": int(detections[0, 5]) if len(detections) > 0 and detections.shape[1] > 5 and len(track) < 7 else (int(track[6]) if len(track) > 6 else -1) # 尝试获取类别
                    })
            elif self.tracker_type == TrackerType.BYTE_TRACK:
                # ByteTrack 需要 [x1, y1, x2, y2, score, class_id]
                # 如果detections只有5列，ByteTrack内部可能有默认处理或需要特定输入
                # ByteTracker.update期望的是一个包含bboxes, scores, cls_ids, ids的对象或元组
                # 这里我们假设detections是(N,6) [x1,y1,x2,y2,score,cls_id]
                # ByteTracker的update方法返回的是一个包含多个属性的列表，每个元素是一个轨迹对象
                # output_stracks = self.tracker.update(detections)
                # ByteTrack 官方示例通常是 self.tracker.update(dets_xyxy, scores, clss_ids)
                # 我们需要将 numpy 数组拆分
                if detections.shape[1] >= 6:
                    dets_xyxy = detections[:, 0:4]
                    scores = detections[:, 4]
                    clss_ids = detections[:, 5].astype(int)
                elif detections.shape[1] == 5:
                    dets_xyxy = detections[:, 0:4]
                    scores = detections[:, 4]
                    clss_ids = np.array([0] * len(detections)) # 如果没有类别，默认为0
                else: # 空检测
                    dets_xyxy = np.empty((0, 4))
                    scores = np.empty((0,))
                    clss_ids = np.empty((0,), dtype=int)

                online_targets = self.tracker.update(dets_xyxy, scores, clss_ids)
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    cls_id = t.cls_id if hasattr(t, 'cls_id') else (clss_ids[0] if len(clss_ids)>0 else -1) # 尝试获取类别
                    score = t.score if hasattr(t, 'score') else 0.0
                    tracked_objects.append({
                        "track_id": int(tid),
                        "bbox": [float(tlwh[0]), float(tlwh[1]), float(tlwh[0] + tlwh[2]), float(tlwh[1] + tlwh[3])],
                        "confidence": float(score),
                        "class_id": int(cls_id)
                    })
            elif self.tracker_type == TrackerType.DEEP_SORT:
                # DeepSORT需要 [x,y,w,h], confidence, class, (feature)
                # 原始图像用于特征提取
                if original_image is None:
                    exception_logger.warning("DeepSORT需要原始图像进行特征提取，但未提供。跟踪可能不准确。")
                    # 不提供图像，DeepSORT将仅基于运动模型
                    # DeepSORT的update输入是 xywhs, confs, clss, image
                xywhs = detections[:, :4] # 假设前4列是xywh或可以通过转换得到
                # 这里需要将 xyxy 转换为 xywh
                # x_center = (detections[:, 0] + detections[:, 2]) / 2
                # y_center = (detections[:, 1] + detections[:, 3]) / 2
                # width = detections[:, 2] - detections[:, 0]
                # height = detections[:, 3] - detections[:, 1]
                # xywhs = np.stack([x_center, y_center, width, height], axis=1)
                # 但是DeepSort内部似乎更期望 xyxy, 所以我们传递detections[:,:4]
                
                confs = detections[:, 4]
                clss = detections[:, 5].astype(int) if detections.shape[1] >= 6 else np.array([0] * len(detections))
                
                outputs = self.tracker.update(detections[:,:4], confs, clss, original_image)
                # outputs 是 [x1,y1,x2,y2,track_id,class_id,conf]
                for output in outputs:
                    x1, y1, x2, y2, track_id, class_id, conf = output
                    tracked_objects.append({
                        "track_id": int(track_id),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(conf),
                        "class_id": int(class_id)
                    })
        except Exception as e:
            exception_logger.exception(f"跟踪器 ({self.tracker_type.name}) 更新时发生错误")
            test_logger.info(f"{log_prefix} | 更新失败: {str(e)}")
            return [] # 发生错误时返回空列表

        test_logger.info(f"{log_prefix} | 输出跟踪目标数: {len(tracked_objects)}")
        if tracked_objects:
            test_logger.info(f"{log_prefix} | 前3个跟踪目标: {tracked_objects[:3]}")

        # 更新计数和速度（如果启用）
        if self.counting_enabled and self.counting_line:
            self._update_counting(tracked_objects)
        
        if self.speed_estimation_enabled:
            self._update_speed(tracked_objects)
            # 将速度信息附加到tracked_objects
            for obj in tracked_objects:
                if obj["track_id"] in self.track_history and self.track_history[obj["track_id"]][-1].get("speed_kmh") is not None:
                    obj["speed_kmh"] = self.track_history[obj["track_id"]][-1]["speed_kmh"]

        return tracked_objects

    def _update_counting(self, tracked_objects: List[Dict[str, Any]]):
        """更新目标计数"""
        if not self.counting_line or len(self.counting_line) < 2:
            # exception_logger.warning("计数线未正确设置，无法进行计数。") # 避免过于频繁的日志
            return

        log_prefix = f"[计数更新]"
        line_p1 = self.counting_line[0]
        line_p2 = self.counting_line[1]

        for obj in tracked_objects:
            track_id = obj["track_id"]
            bbox = obj["bbox"]  # [x1, y1, x2, y2]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            current_pos = (center_x, center_y)

            # 获取上一帧的位置 (如果存在)
            # 确保 track_history 是 Dict[int, deque] 类型
            if track_id in self.track_history and len(self.track_history[track_id]) > 0:
                # 从deque的末尾获取最新的历史记录，再从中获取位置
                last_record = self.track_history[track_id][-1]
                # 假设历史记录中存储了 'center_pos' 键
                prev_pos = last_record.get("center_pos") 
            else:
                prev_pos = None
            
            # 更新轨迹历史（即使不用于速度估计，也可用于计数判断方向）
            # 只保留必要的点以避免内存过大，例如保留最近2帧的位置
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=2) 
            self.track_history[track_id].append({"center_pos": current_pos, "timestamp": time.time()})
            # 如果deque满了，最早的点会被自动移除

            if prev_pos and track_id not in self.crossed_ids:
                if self._check_line_crossing(prev_pos, current_pos, line_p1, line_p2):
                    self.crossed_ids.add(track_id)
                    self.counts["total"] += 1
                    
                    # 判断穿越方向 (简单示例: 基于y坐标变化)
                    # 需要根据线的方向和场景进行调整
                    # 例如，如果线是水平的，可以看y坐标变化
                    # 如果线是垂直的，可以看x坐标变化
                    # 此处简化为基于prev_pos和current_pos相对于线的简单判断
                    # 更鲁棒的方法是计算 prev_pos->current_pos向量 与 线的法向量 的点积
                    line_vec = (line_p2[0] - line_p1[0], line_p2[1] - line_p1[1])
                    # 假设法向量指向"进入"方向 (可调整)
                    # 对于水平线 (y1=y2), 法向量 (0, 1) 或 (0, -1)
                    # 对于垂直线 (x1=x2), 法向量 (1, 0) 或 (-1, 0)
                    # 此处使用简化的方向判断
                    if current_pos[1] < prev_pos[1]: # 向上或向左 (取决于线)
                        self.counts["in"] += 1 # 或者 "up" / "left"
                        test_logger.info(f"{log_prefix} | 目标 {track_id} 进入/向上/向左 越过计数线。当前总数: {self.counts['total']}, 进入: {self.counts['in']}")
                    else: # 向下或向右
                        self.counts["out"] += 1 # 或者 "down" / "right"
                        test_logger.info(f"{log_prefix} | 目标 {track_id} 离开/向下/向右 越过计数线。当前总数: {self.counts['total']}, 离开: {self.counts['out']}")
                    
    def _check_line_crossing(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                               line_p1: Tuple[int, int], line_p2: Tuple[int, int]) -> bool:
        """检查线段(p1, p2)是否与线段(line_p1, line_p2)相交"""
        # 使用向量叉乘的方法判断线段相交
        def cross_product(v1, v2):
            return v1[0] * v2[1] - v1[1] * v2[0]

        # 将点转换为向量
        vec_p1_line_p1 = (line_p1[0] - p1[0], line_p1[1] - p1[1])
        vec_p1_line_p2 = (line_p2[0] - p1[0], line_p2[1] - p1[1])
        vec_p1_p2 = (p2[0] - p1[0], p2[1] - p1[1])

        vec_line_p1_p1 = (p1[0] - line_p1[0], p1[1] - line_p1[1])
        vec_line_p1_p2 = (p2[0] - line_p1[0], p2[1] - line_p1[1])
        vec_line_p1_line_p2 = (line_p2[0] - line_p1[0], line_p2[1] - line_p1[1])

        # 检查叉乘结果的符号
        cp1 = cross_product(vec_p1_p2, vec_p1_line_p1)
        cp2 = cross_product(vec_p1_p2, vec_p1_line_p2)
        cp3 = cross_product(vec_line_p1_line_p2, vec_line_p1_p1)
        cp4 = cross_product(vec_line_p1_line_p2, vec_line_p1_p2)

        # 如果符号不同，则线段相交
        if ((cp1 * cp2) < 0) and ((cp3 * cp4) < 0):
            return True
        # 此处可以添加共线情况的处理，但为简化，暂时忽略
        return False

    def set_counting_line(self, line_points: List[Tuple[Union[int, float], Union[int, float]]], 
                          image_width: Optional[int] = None, image_height: Optional[int] = None, 
                          enabled: bool = True):
        """设置计数线，坐标可以是绝对像素值或相对比例"""
        if len(line_points) != 2:
            exception_logger.error("计数线必须由两个点定义。")
            self.counting_line = None
            self.counting_enabled = False
            return

        p1, p2 = line_points[0], line_points[1]
        final_line = []

        # 检查坐标是否是比例 (0.0-1.0)
        is_relative = all(0.0 <= val <= 1.0 for point in line_points for val in point)

        if is_relative:
            if image_width is None or image_height is None:
                exception_logger.error("使用相对坐标设置计数线时，必须提供图像的宽和高。")
                self.counting_line = None
                self.counting_enabled = False
                return
            final_line.append((int(p1[0] * image_width), int(p1[1] * image_height)))
            final_line.append((int(p2[0] * image_width), int(p2[1] * image_height)))
        else:
            final_line.append((int(p1[0]), int(p1[1])))
            final_line.append((int(p2[0]), int(p2[1])))
        
        self.counting_line = final_line
        self.counting_enabled = enabled
        self.crossed_ids.clear() # 重置已计数ID
        self.counts = {"in": 0, "out": 0, "total": 0} # 重置计数器
        normal_logger.info(f"计数线已设置: {self.counting_line}, 状态: {'启用' if enabled else '禁用'}")
        test_logger.info(f"[计数设置] 计数线: {self.counting_line}, 状态: {'启用' if enabled else '禁用'}")

    def get_counts(self) -> Dict[str, int]:
        """获取当前计数"""
        return self.counts

    def enable_speed_estimation(self, enabled: bool = True, pixels_per_meter: float = 100.0, fps: float = 25.0):
        """启用或禁用速度估计"""
        self.speed_estimation_enabled = enabled
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps # Tracker内部的帧率，用于计算
        self.track_history.clear() # 重置历史轨迹
        if enabled:
            normal_logger.info(f"速度估计已启用: 每米像素数={pixels_per_meter}, FPS={self.fps}")
            test_logger.info(f"[速度设置] 速度估计已启用: PPM={pixels_per_meter}, FPS={self.fps}")
        else:
            normal_logger.info("速度估计已禁用。")
            test_logger.info("[速度设置] 速度估计已禁用。")

    def _update_speed(self, tracked_objects: List[Dict[str, Any]]):
        """更新目标速度估计"""
        if not self.speed_estimation_enabled or self.fps <= 0 or self.pixels_per_meter <= 0:
            return

        log_prefix = "[速度更新]"
        current_time = time.time()

        for obj in tracked_objects:
            track_id = obj["track_id"]
            bbox = obj["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            current_pos_time = {"pos": (center_x, center_y), "time": current_time}

            if track_id not in self.track_history:
                # 增加maxlen以存储更多历史点用于速度计算，例如5个点
                self.track_history[track_id] = deque(maxlen=5) 
            
            self.track_history[track_id].append(current_pos_time)

            if len(self.track_history[track_id]) >= 2:
                # 使用最近的两个点（或更多点进行平滑）来计算速度
                prev_pt_data = self.track_history[track_id][-2] # 上一个点
                curr_pt_data = self.track_history[track_id][-1] # 当前点

                dist_pixels = math.sqrt((curr_pt_data["pos"][0] - prev_pt_data["pos"][0])**2 + 
                                      (curr_pt_data["pos"][1] - prev_pt_data["pos"][1])**2)
                time_diff_seconds = curr_pt_data["time"] - prev_pt_data["time"]

                if time_diff_seconds > 0: # 避免除以零
                    speed_pixels_per_second = dist_pixels / time_diff_seconds
                    speed_meters_per_second = speed_pixels_per_second / self.pixels_per_meter
                    speed_kmh = speed_meters_per_second * 3.6
                    
                    # 将计算出的速度存储在最新的历史记录点中
                    self.track_history[track_id][-1]["speed_kmh"] = round(speed_kmh, 2)
                    test_logger.info(f"{log_prefix} 目标ID {track_id}: 速度 {speed_kmh:.2f} km/h (像素距离: {dist_pixels:.2f}, 时间差: {time_diff_seconds:.3f}s)")
                # else:
                    # test_logger.warning(f"{log_prefix} 目标ID {track_id}: 时间差为0，无法计算速度。")

    def reset(self):
        """重置跟踪器状态"""
        self._init_tracker() # 重新初始化跟踪算法实例
        self.crossed_ids.clear()
        self.counts = {"in": 0, "out": 0, "total": 0}
        self.track_history.clear()
        normal_logger.info("跟踪器状态已重置。")
        test_logger.info("[重置] 跟踪器状态已重置。")
