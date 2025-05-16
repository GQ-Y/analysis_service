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


def log_detections_if_found(result, boxes, prompt_free_class_names=None, internal_prompt_type=None, result_names_override=None):
    """
    记录检测结果日志，只在检测到目标时记录详细信息

    Args:
        result: YOLO检测结果
        boxes: 检测到的边界框
        prompt_free_class_names: 无提示模式下的类别名称列表
        internal_prompt_type: 内部提示类型
        result_names_override: 备用类别名称字典，用于回退机制
    """
    if boxes is None:
        logger.warning("边界框对象为None，无法记录检测结果")
        return

    # 只在检测到目标时记录详细信息
    if len(boxes) > 0:
        # 构建检测结果日志
        detection_info = []
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())

            # 在无提示模式下，按优先级顺序尝试获取类别名称
            if internal_prompt_type == 0:
                # 1. 首先从model.names获取
                if hasattr(result, "model") and hasattr(result.model, "names") and isinstance(result.model.names, dict) and cls_id in result.model.names:
                    cls_name = result.model.names[cls_id]
                # 2. 如果model.names中没有，但有result_names_override，则从result_names_override获取
                elif result_names_override and cls_id in result_names_override:
                    cls_name = result_names_override[cls_id]
                # 3. 如果result_names_override中没有，但有prompt_free_class_names，则从prompt_free_class_names获取
                elif prompt_free_class_names and 0 <= cls_id < len(prompt_free_class_names):
                    cls_name = prompt_free_class_names[cls_id]
                # 4. 最后从result.names获取或使用默认名称
                else:
                    cls_name = result.names.get(cls_id, f"class_{cls_id}")
            # 对于文本提示模式，使用提供的文本提示类别名称
            elif internal_prompt_type == 1 and hasattr(result, "text_prompt") and result.text_prompt and 0 <= cls_id < len(result.text_prompt):
                cls_name = result.text_prompt[cls_id]
            # 否则使用result.names
            else:
                cls_name = result.names.get(cls_id, f"class_{cls_id}")

            detection_info.append(f"{cls_name}: {conf:.2f}")

        # 只在检测到目标时输出检测结果日志
        logger.info(f"检测到 {len(boxes)} 个目标: {', '.join(detection_info)}")
    # 不再打印未检测到目标的日志，减少日志量


def log_detections(result, boxes, prompt_free_class_names=None, internal_prompt_type=None):
    """
    记录检测结果日志

    Args:
        result: YOLO检测结果
        boxes: 检测到的边界框
        prompt_free_class_names: 无提示模式下的类别名称列表
        internal_prompt_type: 内部提示类型
    """
    if len(boxes) > 0:
        # 构建检测结果日志
        detection_info = []
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())

            # 获取类别名称，优先使用model.names
            if hasattr(result, "model") and hasattr(result.model, "names") and cls_id in result.model.names:
                cls_name = result.model.names[cls_id]
            # 如果是无提示模式，使用prompt_free_class_names
            elif internal_prompt_type == 0 and prompt_free_class_names and 0 <= cls_id < len(prompt_free_class_names):
                cls_name = prompt_free_class_names[cls_id]
            # 否则使用result.names
            else:
                cls_name = result.names.get(cls_id, f"class_{cls_id}")

            detection_info.append(f"{cls_name}: {conf:.2f}")

        # 输出检测结果日志
        logger.info(f"检测到 {len(boxes)} 个目标: {', '.join(detection_info)}")


def custom_plot_detections(image, result, boxes, prompt_free_class_names=None, internal_prompt_type=None, result_names_override=None):
    """
    自定义绘制检测结果，确保在无提示模式下使用正确的类别名称

    Args:
        image: 原始图像
        result: YOLO检测结果
        boxes: 检测到的边界框
        prompt_free_class_names: 无提示模式下的类别名称列表
        internal_prompt_type: 内部提示类型
        result_names_override: 备用类别名称字典，用于回退机制

    Returns:
        np.ndarray: 绘制了检测结果的图像
    """
    try:
        # 创建图像副本
        result_image = image.copy()

        # 绘制每个检测结果
        for i in range(len(boxes)):
            # 获取边界框坐标
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # 获取置信度
            conf = float(boxes.conf[i].cpu().numpy())

            # 获取类别ID
            cls_id = int(boxes.cls[i].cpu().numpy())

            # 在无提示模式下，按优先级顺序尝试获取类别名称
            if internal_prompt_type == 0:
                # 1. 首先从model.names获取
                if hasattr(result, "model") and hasattr(result.model, "names") and isinstance(result.model.names, dict) and cls_id in result.model.names:
                    cls_name = result.model.names[cls_id]
                # 2. 如果model.names中没有，但有result_names_override，则从result_names_override获取
                elif result_names_override and cls_id in result_names_override:
                    cls_name = result_names_override[cls_id]
                # 3. 如果result_names_override中没有，但有prompt_free_class_names，则从prompt_free_class_names获取
                elif prompt_free_class_names and 0 <= cls_id < len(prompt_free_class_names):
                    cls_name = prompt_free_class_names[cls_id]
                # 4. 最后从result.names获取或使用默认名称
                else:
                    cls_name = result.names.get(cls_id, f"class_{cls_id}")
            # 对于文本提示模式，使用提供的文本提示类别名称
            elif internal_prompt_type == 1 and hasattr(result, "text_prompt") and result.text_prompt and 0 <= cls_id < len(result.text_prompt):
                cls_name = result.text_prompt[cls_id]
            # 否则使用result.names
            else:
                cls_name = result.names.get(cls_id, f"class_{cls_id}")

            # 获取类别颜色 (使用类别ID的哈希值生成颜色)
            color_h = (cls_id * 50) % 360  # 色调
            color_s = 0.8  # 饱和度
            color_v = 0.8  # 亮度

            # 将HSV转换为BGR
            import colorsys
            rgb = colorsys.hsv_to_rgb(color_h / 360, color_s, color_v)
            color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR格式

            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{cls_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)

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
                label,
                (text_x, text_y - 2),
                font,
                font_scale,
                (255, 255, 255),  # 白色文字
                thickness
            )

        return result_image

    except Exception as e:
        logger.error(f"自定义绘制检测结果失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # 如果失败，返回原始图像
        return image

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
        self.model_code = model_code  # 保存model_code

        # YOLOE特有参数
        self.prompt_type = kwargs.get("prompt_type", 3)  # 1=文本提示, 2=图像提示, 3=无提示
        self.text_prompt = kwargs.get("text_prompt", [])
        self.visual_prompt = kwargs.get("visual_prompt", {})

        # 缓存文本嵌入，避免重复计算
        self._text_pe_cache = None
        self._text_prompt_cache = None
        self._text_embedding_set = False

        # 无提示模式的类别名称
        self._prompt_free_class_names = []
        # 回退机制使用的类别名称字典
        self._result_names_override = None

        self._load_prompt_free_class_names()

        # 内部使用0表示无提示，但API使用3表示无提示
        self._internal_prompt_type = self.prompt_type
        if self.prompt_type == 3:
            self._internal_prompt_type = 0
            logger.info("使用无提示模式 (prompt_type=3)")

        logger.info(f"初始化YOLOE分析器: 提示类型={self._get_prompt_type_name()}")

    def _load_prompt_free_class_names(self):
        """
        加载无提示模式的类别名称
        """
        try:
            # 尝试从多个位置加载tag_list.txt
            tag_list_paths = [
                "tag_list.txt",
                "data/tag_list.txt",
                "data/models/tag_list.txt",
                "models/tag_list.txt",
                "tools/ram_tag_list.txt",  # 官方示例路径
                "../tag_list.txt",  # 向上一级目录查找
                "../data/tag_list.txt"
            ]

            # 查找第一个存在的文件
            tag_list_path = None
            for path in tag_list_paths:
                if os.path.exists(path):
                    tag_list_path = path
                    break

            if tag_list_path:
                # 读取类别名称文件
                with open(tag_list_path, "r", encoding="utf-8") as f:
                    self._prompt_free_class_names = [line.strip() for line in f.readlines()]

                # 确保没有空类别名称并去重
                self._prompt_free_class_names = [name for name in self._prompt_free_class_names if name]
                if len(self._prompt_free_class_names) != len(set(self._prompt_free_class_names)):
                    # 存在重复类别，记录并去重
                    original_count = len(self._prompt_free_class_names)
                    self._prompt_free_class_names = list(dict.fromkeys(self._prompt_free_class_names))  # 保持顺序的去重
                    logger.warning(f"类别列表存在重复项：原始数量 {original_count}，去重后 {len(self._prompt_free_class_names)}")

                logger.info(f"已从 {tag_list_path} 加载无提示模式类别名称: {len(self._prompt_free_class_names)}个类别")

                # 打印前几个类别名称进行确认
                if self._prompt_free_class_names:
                    preview = self._prompt_free_class_names[:5]
                    logger.info(f"前几个类别名称: {preview}")

                    # 检查是否有非法类别名称（空字符串或只有空格的字符串）
                    invalid_names = [i for i, name in enumerate(self._prompt_free_class_names) if not name or name.isspace()]
                    if invalid_names:
                        logger.warning(f"发现 {len(invalid_names)} 个非法类别名称（空或仅含空格），位置: {invalid_names[:10]}...")
                        # 移除非法类别名称
                        self._prompt_free_class_names = [name for name in self._prompt_free_class_names if name and not name.isspace()]
                        logger.info(f"清理后的类别数量: {len(self._prompt_free_class_names)}")
            else:
                logger.warning(f"无提示模式类别名称文件不存在，尝试了以下路径: {', '.join(tag_list_paths)}")
                # 尝试创建一个简单的默认类别列表
                self._prompt_free_class_names = ["object", "person", "animal", "vehicle", "furniture",
                                               "electronics", "food", "plant", "clothing", "building"]
                logger.warning(f"使用默认类别列表: {self._prompt_free_class_names}")
        except Exception as e:
            logger.error(f"加载无提示模式类别名称失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 确保有默认值
            self._prompt_free_class_names = ["object", "person", "animal", "vehicle", "furniture"]
            logger.warning(f"使用错误恢复类别列表: {self._prompt_free_class_names}")

        # 如果提供了model_code，立即加载模型
        if self.model_code:
            # 注意：这里不能直接使用await，因为__init__不是异步方法
            # 创建一个异步任务来加载模型
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环已经在运行，创建一个任务
                    asyncio.create_task(self.load_model(self.model_code))
                else:
                    # 如果事件循环没有运行，直接运行直到完成
                    loop.run_until_complete(self.load_model(self.model_code))
            except Exception as e:
                logger.error(f"初始化时加载模型失败: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())



    def _get_prompt_type_name(self) -> str:
        """获取提示类型名称"""
        prompt_types = {
            1: "文本提示",
            2: "图像提示",
            3: "无提示"
        }
        return prompt_types.get(self.prompt_type, f"未知提示类型({self.prompt_type})")

    def _setup_yoloe_prompt_free_mode(self) -> bool:
        """
        按照官方示例正确设置YOLOE模型的无提示模式

        参考：https://github.com/THU-MIG/yoloe
        """
        try:
            import torch

            # 记录模型类型信息，用于调试
            model_type = type(self.model).__name__
            logger.info(f"模型类型: {model_type}")

            # 检查是否有类别名称
            if not self._prompt_free_class_names or len(self._prompt_free_class_names) == 0:
                logger.warning("无提示模式类别名称为空，无法设置YOLOE模型词汇表")
                return False

            # 记录类别名称数量和前几个类别
            logger.info(f"无提示模式类别名称数量: {len(self._prompt_free_class_names)}")
            preview_classes = self._prompt_free_class_names[:5] if len(self._prompt_free_class_names) > 5 else self._prompt_free_class_names
            logger.info(f"前几个类别: {preview_classes}")

            # 1. 设置模型为无提示模式
            if hasattr(self.model, "prompt_free"):
                self.model.prompt_free = True
                logger.info("已设置模型prompt_free属性为True")

            # 2. 模型names属性处理 - 检查是否可写
            try:
                # 尝试获取当前names属性
                names_dict = {}
                if hasattr(self.model, "names"):
                    if isinstance(self.model.names, dict):
                        # 获取当前names并复制
                        names_dict = dict(self.model.names)
                        logger.info(f"获取到现有model.names: {names_dict if len(names_dict) < 10 else '(太多不显示)'}")

                # 创建一个新的names字典
                for i, name in enumerate(self._prompt_free_class_names):
                    names_dict[i] = name

                # 尝试设置names属性
                try:
                    self.model.names = names_dict
                    logger.info(f"通过直接赋值方式设置model.names成功，共 {len(names_dict)} 个类别")
                except AttributeError:
                    # 如果直接赋值失败，尝试使用__dict__方式设置
                    try:
                        self.model.__dict__['names'] = names_dict
                        logger.info(f"通过__dict__方式设置model.names成功，共 {len(names_dict)} 个类别")
                    except Exception as dict_err:
                        # 如果也失败，尝试查找模型内部结构
                        logger.warning(f"使用__dict__设置model.names失败: {str(dict_err)}")
                        logger.info("尝试查找模型内部结构...")

                        # 尝试找到names实际位置
                        if hasattr(self.model, "model"):
                            # 有些模型封装在model.model中
                            inner_model = self.model.model
                            try:
                                inner_model.names = names_dict
                                logger.info(f"通过model.model.names设置成功，共 {len(names_dict)} 个类别")
                            except AttributeError:
                                # 尝试更深层的model结构
                                if hasattr(inner_model, "model"):
                                    deepest_model = inner_model.model
                                    try:
                                        deepest_model.names = names_dict
                                        logger.info(f"通过model.model.model.names设置成功，共 {len(names_dict)} 个类别")
                                    except AttributeError:
                                        # 如果还是无法设置，记录找到的属性以便调试
                                        if hasattr(deepest_model, "__dict__"):
                                            logger.info(f"model.model.model属性列表: {list(deepest_model.__dict__.keys())[:10]}")
                                        logger.warning("无法在模型结构中找到可设置的names属性")
                                else:
                                    if hasattr(inner_model, "__dict__"):
                                        logger.info(f"model.model属性列表: {list(inner_model.__dict__.keys())[:10]}")
                                    logger.warning("模型结构中没有更深层的model属性")
                        else:
                            logger.warning("模型没有内部model属性，无法进一步设置names")

                # 验证是否成功设置names
                if hasattr(self.model, "names"):
                    names_set = self.model.names
                    # 检查第一个类别是否正确设置
                    if 0 in names_set and names_set[0] == self._prompt_free_class_names[0]:
                        logger.info("验证names设置成功")
                        # 输出几个示例以确认
                        preview = {i: names_set[i] for i in range(min(10, len(names_set)))}
                        logger.info(f"设置后的类别ID映射示例: {preview}")
                    else:
                        logger.warning(f"names设置可能不正确: 首个类别为 {names_set.get(0, 'None')}")
            except Exception as names_err:
                logger.error(f"设置names属性时出错: {str(names_err)}")

            # 3. 如果模型有head，设置head参数
            if hasattr(self.model, "model") and hasattr(self.model.model, "model"):
                try:
                    head = self.model.model.model[-1]
                    head_type = type(head).__name__
                    logger.info(f"模型head类型: {head_type}")

                    # 设置融合状态
                    if hasattr(head, "is_fused"):
                        head.is_fused = True
                        logger.info("已设置模型head为融合状态")

                    # 设置较低的置信度阈值
                    if hasattr(head, "conf"):
                        head.conf = 0.001  # 与官方保持一致
                        logger.info("已设置模型head的置信度阈值为0.001")

                    # 设置较高的最大检测数量
                    if hasattr(head, "max_det"):
                        head.max_det = 1000
                        logger.info("已设置模型head的最大检测数量为1000")
                except Exception as e:
                    logger.warning(f"设置模型head参数时出错: {str(e)}")

            # 标记为已设置，避免重复操作
            self._text_embedding_set = True

            logger.info("成功完成无提示模式设置")
            return True

        except Exception as e:
            logger.error(f"设置YOLOE无提示模式失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

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

            # 只在首次或文本提示变化时输出调试日志
            if not self._text_embedding_set or self._text_prompt_cache != self.text_prompt:
                logger.debug(f"使用文本提示: {self.text_prompt}")

                # 检查MobileCLIP模型文件是否存在
                mobileclip_path = "mobileclip_blt.pt"
                if not os.path.exists(mobileclip_path):
                    mobileclip_path = "data/models/mobileclip_blt.pt"
                    if not os.path.exists(mobileclip_path):
                        logger.warning(f"MobileCLIP模型文件不存在: {mobileclip_path}")
                    else:
                        logger.info(f"找到MobileCLIP模型文件: {mobileclip_path}")

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

    async def _save_result_image(self, image: np.ndarray, detections: List[Dict], task_name: Optional[str] = None) -> str:
        """
        保存带有检测结果的图片

        Args:
            image: 带有检测结果标注的图像
            detections: 检测结果列表
            task_name: 任务名称，用于文件名前缀

        Returns:
            str: 保存的图片路径，如果保存失败则返回None
        """
        try:
            # 检查图像是否为空
            if image is None or image.size == 0:
                logger.warning("保存图片失败：图像为空")
                return None

            # 获取当前工作目录
            current_dir = os.getcwd()

            # 确保results目录存在
            results_dir = os.path.join(current_dir, "results")
            os.makedirs(results_dir, exist_ok=True)

            # 确保每天的结果保存在单独的目录中
            date_str = datetime.now().strftime("%Y%m%d")
            date_dir = os.path.join(results_dir, date_str)
            os.makedirs(date_dir, exist_ok=True)

            # 生成类别信息字符串
            classes_set = set()
            for det in detections:
                if "class_name" in det:
                    classes_set.add(det["class_name"])

            # 最多显示3个类别
            classes_list = list(classes_set)[:3]
            if len(classes_set) > 3:
                classes_list.append("...")

            classes_info = "_".join(classes_list) + "_"

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
            task_prefix = f"{task_name}_" if task_name else ""
            filename = f"{task_prefix}{classes_info}{timestamp}.jpg"

            # 完整的文件路径
            file_path = os.path.join(date_dir, filename)

            # 保存图片
            try:
                # 尝试使用cv2保存
                success = cv2.imwrite(file_path, image)

                if not success:
                    # 尝试使用PIL保存
                    try:
                        from PIL import Image
                        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        pil_image.save(file_path)
                    except Exception as e:
                        # 尝试直接写入文件
                        try:
                            _, buffer = cv2.imencode(".jpg", image)
                            with open(file_path, "wb") as f:
                                f.write(buffer)
                        except Exception as e2:
                            logger.error(f"保存图片失败: {str(e2)}")
                            return None

                # 检查文件是否存在
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    # 检查文件大小是否正常
                    if file_size == 0:
                        logger.warning(f"保存的图片文件大小为0: {file_path}")
                        return None
                else:
                    logger.warning(f"保存图片后文件不存在: {file_path}")
                    return None
            except Exception as e:
                logger.error(f"保存图片时出错: {str(e)}")
                return None

            # 返回相对路径
            relative_path = os.path.join("results", date_str, filename)
            logger.info(f"检测结果图片已保存: {relative_path}")
            return relative_path

        except Exception as e:
            logger.error(f"保存结果图片失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None


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

        # 检查模型是否已加载
        if self.model is None:
            # 如果模型未加载，尝试重新加载
            if self.current_model_code:
                logger.warning(f"YOLOE模型未加载，尝试重新加载: {self.current_model_code}")
                await self.load_model(self.current_model_code)
            else:
                # 如果没有模型代码，无法加载模型
                logger.error("YOLOE模型未加载且没有模型代码，无法进行检测")
                return {
                    "detections": [],
                    "pre_process_time": 0,
                    "inference_time": 0,
                    "post_process_time": 0,
                    "annotated_image_bytes": None
                }

            # 再次检查模型是否加载成功
            if self.model is None:
                logger.error("YOLOE模型重新加载失败，无法进行检测")
                return {
                    "detections": [],
                    "pre_process_time": 0,
                    "inference_time": 0,
                    "post_process_time": 0,
                    "annotated_image_bytes": None
                }

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
                # 检查是否是YOLOE模型
                if hasattr(self.model, "set_classes") and hasattr(self.model, "get_text_pe"):
                    # 对于YOLOE模型，无提示模式需要特殊处理
                    if not self._text_embedding_set:
                        logger.info("检测到YOLOE模型，使用无提示全目标检测模式")

                        # 尝试设置词汇表和类别名称，确保模型处于无提示模式
                        try:
                            # 检查是否有类别名称
                            if self._prompt_free_class_names and len(self._prompt_free_class_names) > 0:
                                # 使用正确的方式设置YOLOE无提示模式
                                success = self._setup_yoloe_prompt_free_mode()
                                if success:
                                    logger.info(f"已设置YOLOE模型为无提示模式，使用 {len(self._prompt_free_class_names)} 个类别")
                                    # 在无提示模式下，使用更低的置信度阈值以捕获更多目标
                                    original_confidence = confidence
                                    confidence = min(confidence, 0.01)  # 使用低置信度阈值
                                    logger.info(f"在无提示模式下调整置信度阈值: {original_confidence} -> {confidence}")
                                else:
                                    logger.warning("使用新方法设置YOLOE无提示模式失败，尝试其他方法")

                                    # 使用稳定的方法创建临时names字典
                                    names_dict = {}

                                    # 添加所有类别到临时字典
                                    for i, name in enumerate(self._prompt_free_class_names):
                                        names_dict[i] = name

                                    logger.info(f"回退方法: 已准备好 {len(self._prompt_free_class_names)} 个类别")

                                    # 为结果对象动态添加names属性
                                    # 这样即使model.names不可设置，我们仍能在结果处理时使用这些类别名称
                                    self._result_names_override = names_dict

                                    # 记录几个示例类别
                                    preview = {i: names_dict[i] for i in range(min(5, len(names_dict)))}
                                    logger.info(f"回退方法: 类别ID映射示例: {preview}")

                                    # 尝试设置model.prompt_free属性
                                    if hasattr(self.model, "prompt_free"):
                                        self.model.prompt_free = True
                                        logger.info("回退方法: 已设置model.prompt_free=True")

                                    # 标记为已初始化
                                    self._text_embedding_set = True
                            else:
                                logger.warning("无提示模式类别名称为空，无法完全设置YOLOE模型配置")
                                # 尝试设置为无提示模式
                                if hasattr(self.model, "prompt_free"):
                                    self.model.prompt_free = True
                                    logger.info("已设置YOLOE模型为无提示模式（无类别名称）")

                            # 标记为已设置，避免重复操作
                            self._text_embedding_set = True
                        except Exception as e:
                            logger.error(f"设置YOLOE模型无提示模式失败: {str(e)}")
                            # 即使完全失败，我们也应该确保模型可以工作
                            if hasattr(self.model, "names") and self._prompt_free_class_names:
                                # 重新初始化names字典
                                self.model.names = {}
                                # 添加所有类别
                                for i, name in enumerate(self._prompt_free_class_names):
                                    self.model.names[i] = name
                                logger.info(f"错误恢复: 已添加 {len(self._prompt_free_class_names)} 个类别到model.names")

                            if hasattr(self.model, "prompt_free"):
                                self.model.prompt_free = True
                                logger.info("错误恢复: 已设置prompt_free=True")

                            # 标记为已设置，避免重复尝试
                            self._text_embedding_set = True

                # 使用标准检测
                logger.info(f"执行YOLOE检测: 置信度={confidence}, IoU阈值={iou_threshold}, 类别过滤={classes}, 最大检测数={max_detections}")
                results = self.model(
                    image,
                    conf=confidence,
                    iou=iou_threshold,
                    classes=classes,
                    max_det=max_detections,
                    verbose=False  # 禁用自动打印
                )
                logger.info(f"检测完成，结果类型: {type(results)}, 结果长度: {len(results) if results else 0}")
            elif self._internal_prompt_type == 1 and self.text_prompt:  # 文本提示
                # 使用文本提示检测
                # 根据YOLOE官方文档，需要先设置类别和文本嵌入
                try:
                    # 检查是否是YOLOE模型
                    is_yoloe_model = False
                    if hasattr(self.model, "set_classes") and hasattr(self.model, "get_text_pe"):
                        is_yoloe_model = True

                        # 只在首次检测或文本提示变化时输出日志
                        if not self._text_embedding_set:
                            logger.info(f"检测到YOLOE模型，使用文本提示模式")

                            # 检查MobileCLIP模型文件是否存在
                            mobileclip_path = "mobileclip_blt.pt"
                            if not os.path.exists(mobileclip_path):
                                mobileclip_path = "data/models/mobileclip_blt.pt"
                                if not os.path.exists(mobileclip_path):
                                    logger.warning(f"MobileCLIP模型文件不存在: {mobileclip_path}，文本提示功能可能无法正常工作")
                                else:
                                    logger.info(f"找到MobileCLIP模型文件: {mobileclip_path}")

                    if is_yoloe_model:
                        # 检查是否需要重新设置文本嵌入
                        need_setup = False

                        # 如果文本提示发生变化或尚未设置过
                        if not self._text_embedding_set or self._text_prompt_cache != self.text_prompt:
                            need_setup = True

                        if need_setup:
                            # 设置类别和文本嵌入
                            logger.info(f"使用YOLOE文本提示模式，设置类别: {self.text_prompt}")
                            try:
                                # 检查text_prompt是否为空
                                if not self.text_prompt:
                                    logger.warning("文本提示为空，无法设置YOLOE文本提示")
                                else:
                                    # 尝试获取文本嵌入
                                    logger.info(f"正在获取文本嵌入: {self.text_prompt}")
                                    text_pe = self.model.get_text_pe(self.text_prompt)

                                    # 检查文本嵌入是否为None
                                    if text_pe is None:
                                        logger.warning("获取到的文本嵌入为None，无法设置YOLOE文本提示")
                                    else:
                                        # 设置类别和文本嵌入
                                        logger.info(f"正在设置类别和文本嵌入: {len(self.text_prompt)}个类别")
                                        self.model.set_classes(self.text_prompt, text_pe)
                                        logger.info("成功设置YOLOE文本提示")

                                        # 更新缓存
                                        self._text_pe_cache = text_pe
                                        self._text_prompt_cache = self.text_prompt.copy() if isinstance(self.text_prompt, list) else self.text_prompt
                                        self._text_embedding_set = True
                            except Exception as e:
                                logger.error(f"设置YOLOE文本提示失败: {str(e)}")
                                import traceback
                                logger.error(traceback.format_exc())
                        else:
                            # 使用已缓存的文本嵌入，不输出日志
                            pass
                    else:
                        # 只在首次检测时输出警告
                        if not self._text_embedding_set:
                            logger.warning("当前模型不是YOLOE模型或不支持文本提示，将使用标准检测")

                    # 执行标准检测（不传递text_prompt参数）
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        verbose=False  # 禁用自动打印
                    )
                except Exception as e:
                    logger.error(f"YOLOE文本提示模式失败: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # 回退到标准检测
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_detections=max_detections,
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

                # 记录结果信息
                logger.info(f"检测结果信息: 类型={type(result)}, 属性={dir(result)[:10]}...")

                # 检查是否有boxes属性
                if hasattr(result, "boxes"):
                    # 获取边界框
                    boxes = result.boxes

                    # 记录boxes信息
                    logger.info(f"边界框信息: 类型={type(boxes)}, 数量={len(boxes) if boxes else 0}")

                    # 记录names信息
                    if hasattr(result, "names"):
                        logger.info(f"类别名称: {result.names}")

                    # 只在检测到目标时记录日志
                    log_detections_if_found(result, boxes, self._prompt_free_class_names, self._internal_prompt_type, self._result_names_override)
                else:
                    logger.warning("检测结果中没有边界框信息")

                # 处理检测结果
                if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
                    # 处理每个检测结果
                    for i in range(len(boxes)):
                        try:
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

                            # 在无提示模式下，按优先级顺序尝试获取类别名称
                            if self._internal_prompt_type == 0:
                                # 1. 首先从model.names获取
                                if hasattr(self.model, "names") and isinstance(self.model.names, dict) and cls_id in self.model.names:
                                    cls_name = self.model.names[cls_id]
                                    logger.debug(f"从model.names获取类别：ID={cls_id}, 名称={cls_name}")
                                # 2. 如果model.names中没有，但有_result_names_override，则从_result_names_override获取
                                elif self._result_names_override and cls_id in self._result_names_override:
                                    cls_name = self._result_names_override[cls_id]
                                    logger.debug(f"从_result_names_override获取类别：ID={cls_id}, 名称={cls_name}")
                                # 3. 如果_result_names_override中没有，但有_prompt_free_class_names，则从_prompt_free_class_names获取
                                elif self._prompt_free_class_names and 0 <= cls_id < len(self._prompt_free_class_names):
                                    cls_name = self._prompt_free_class_names[cls_id]
                                    logger.debug(f"从_prompt_free_class_names获取类别：ID={cls_id}, 名称={cls_name}")
                                # 4. 最后从result.names获取或使用默认名称
                                else:
                                    cls_name = result.names.get(cls_id, f"class_{cls_id}")
                                    logger.debug(f"从result.names获取类别：ID={cls_id}, 名称={cls_name}")
                            elif self._internal_prompt_type == 1 and self.text_prompt and 0 <= cls_id < len(self.text_prompt):
                                cls_name = self.text_prompt[cls_id]
                                logger.debug(f"文本提示类别映射: ID={cls_id}, 名称={cls_name}")
                            # 其他情况，使用模型的默认类别名称
                            else:
                                cls_name = result.names.get(cls_id, f"class_{cls_id}")
                                logger.debug(f"使用默认类别名称：ID={cls_id}, 名称={cls_name}")

                            # 创建检测结果
                            detection = {
                                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                                "confidence": conf,
                                "class_id": cls_id,
                                "class_name": cls_name
                            }

                            detections.append(detection)

                        except Exception as e:
                            logger.error(f"处理检测结果时出错: {str(e)}")
                            import traceback
                            logger.error(traceback.format_exc())
                else:
                    logger.info("没有检测到任何目标，返回空的检测结果列表")

            # 后处理时间
            post_process_time = time.time() - post_process_start

            # 总时间
            total_time = time.time() - start_time

            # 检查是否需要保存图片
            save_images = kwargs.get("save_images", False)
            annotated_image_bytes = None

            # 处理图像保存
            if results and len(results) > 0 and len(detections) > 0:
                try:
                    # 使用自定义绘图方法，确保在无提示模式下使用正确的类别名称
                    if self._internal_prompt_type == 0 and len(self._prompt_free_class_names) > 0:
                        # 在无提示模式下使用自定义绘图方法
                        annotated_image = custom_plot_detections(
                            image, result, boxes,
                            self._prompt_free_class_names,
                            self._internal_prompt_type,
                            self._result_names_override
                        )
                    else:
                        # 其他模式下使用ultralytics自带的plot方法
                        annotated_image = results[0].plot()

                    # 编码图像
                    is_success, buffer = cv2.imencode(".jpg", annotated_image)
                    if not is_success:
                        logger.warning("标注图像编码失败")
                    else:
                        annotated_image_bytes = buffer.tobytes()

                        # 只在检测到目标时才保存图片
                        if save_images:
                            # 获取任务名称
                            task_name = kwargs.get("task_name", None)

                            # 保存图片
                            await self._save_result_image(annotated_image, detections, task_name)
                except Exception as plot_err:
                    logger.error(f"绘制或编码标注图像时出错: {str(plot_err)}")
                    import traceback
                    logger.error(traceback.format_exc())

            # 返回结果
            return {
                "detections": detections,
                "pre_process_time": pre_process_time * 1000,  # 转换为毫秒
                "inference_time": inference_time * 1000,  # 转换为毫秒
                "post_process_time": post_process_time * 1000,  # 转换为毫秒
                "total_time": total_time * 1000,  # 转换为毫秒
                "annotated_image_bytes": annotated_image_bytes
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
                # 检查是否是YOLOE模型
                if hasattr(self.model, "set_classes") and hasattr(self.model, "get_text_pe"):
                    # 对于YOLOE模型，无提示模式需要特殊处理
                    if not self._text_embedding_set:
                        logger.info("检测到YOLOE模型，使用无提示全目标检测模式")

                        # 尝试设置词汇表和类别名称，确保模型处于无提示模式
                        try:
                            # 检查是否有类别名称
                            if self._prompt_free_class_names and len(self._prompt_free_class_names) > 0:
                                # 使用正确的方式设置YOLOE无提示模式
                                success = self._setup_yoloe_prompt_free_mode()
                                if success:
                                    logger.info(f"已设置YOLOE分割模型为无提示模式，使用 {len(self._prompt_free_class_names)} 个类别")
                                    # 在无提示模式下，使用更低的置信度阈值以捕获更多目标
                                    original_confidence = confidence
                                    confidence = min(confidence, 0.01)  # 使用低置信度阈值
                                    logger.info(f"在无提示模式下调整置信度阈值: {original_confidence} -> {confidence}")
                                else:
                                    logger.warning("使用新方法设置YOLOE分割模型无提示模式失败，尝试其他方法")

                                    # 使用稳定的方法创建临时names字典
                                    names_dict = {}

                                    # 添加所有类别到临时字典
                                    for i, name in enumerate(self._prompt_free_class_names):
                                        names_dict[i] = name

                                    logger.info(f"回退方法: 已准备好 {len(self._prompt_free_class_names)} 个类别")

                                    # 为结果对象动态添加names属性
                                    # 这样即使model.names不可设置，我们仍能在结果处理时使用这些类别名称
                                    self._result_names_override = names_dict

                                    # 记录几个示例类别
                                    preview = {i: names_dict[i] for i in range(min(5, len(names_dict)))}
                                    logger.info(f"回退方法: 类别ID映射示例: {preview}")

                                    # 尝试设置model.prompt_free属性
                                    if hasattr(self.model, "prompt_free"):
                                        self.model.prompt_free = True
                                        logger.info("回退方法: 已设置model.prompt_free=True")

                                    # 标记为已初始化
                                    self._text_embedding_set = True
                            else:
                                logger.warning("无提示模式类别名称为空，无法完全设置YOLOE模型配置")
                                # 尝试设置为无提示模式
                                if hasattr(self.model, "prompt_free"):
                                    self.model.prompt_free = True
                                    logger.info("已设置YOLOE模型为无提示模式（无类别名称）")

                            # 标记为已设置，避免重复操作
                            self._text_embedding_set = True
                        except Exception as e:
                            logger.error(f"设置YOLOE分割模型无提示模式失败: {str(e)}")
                            # 即使完全失败，我们也应该确保模型可以工作
                            if hasattr(self.model, "names") and self._prompt_free_class_names:
                                # 重新初始化names字典
                                self.model.names = {}
                                # 添加所有类别
                                for i, name in enumerate(self._prompt_free_class_names):
                                    self.model.names[i] = name
                                logger.info(f"错误恢复: 已添加 {len(self._prompt_free_class_names)} 个类别到model.names")

                            if hasattr(self.model, "prompt_free"):
                                self.model.prompt_free = True
                                logger.info("错误恢复: 已设置prompt_free=True")

                            # 标记为已设置，避免重复尝试
                            self._text_embedding_set = True

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
                # 根据YOLOE官方文档，需要先设置类别和文本嵌入
                try:
                    # 检查是否是YOLOE模型
                    is_yoloe_model = False
                    if hasattr(self.model, "set_classes") and hasattr(self.model, "get_text_pe"):
                        is_yoloe_model = True

                        # 只在首次检测或文本提示变化时输出日志
                        if not self._text_embedding_set:
                            logger.info(f"检测到YOLOE模型，使用文本提示模式")

                            # 检查MobileCLIP模型文件是否存在
                            mobileclip_path = "mobileclip_blt.pt"
                            if not os.path.exists(mobileclip_path):
                                mobileclip_path = "data/models/mobileclip_blt.pt"
                                if not os.path.exists(mobileclip_path):
                                    logger.warning(f"MobileCLIP模型文件不存在: {mobileclip_path}，文本提示功能可能无法正常工作")
                                else:
                                    logger.info(f"找到MobileCLIP模型文件: {mobileclip_path}")

                    if is_yoloe_model:
                        # 检查是否需要重新设置文本嵌入
                        need_setup = False

                        # 如果文本提示发生变化或尚未设置过
                        if not self._text_embedding_set or self._text_prompt_cache != self.text_prompt:
                            need_setup = True

                        if need_setup:
                            # 设置类别和文本嵌入
                            logger.info(f"使用YOLOE文本提示模式，设置类别: {self.text_prompt}")
                            try:
                                # 检查text_prompt是否为空
                                if not self.text_prompt:
                                    logger.warning("文本提示为空，无法设置YOLOE文本提示")
                                else:
                                    # 尝试获取文本嵌入
                                    logger.info(f"正在获取文本嵌入: {self.text_prompt}")
                                    text_pe = self.model.get_text_pe(self.text_prompt)

                                    # 检查文本嵌入是否为None
                                    if text_pe is None:
                                        logger.warning("获取到的文本嵌入为None，无法设置YOLOE文本提示")
                                    else:
                                        # 设置类别和文本嵌入
                                        logger.info(f"正在设置类别和文本嵌入: {len(self.text_prompt)}个类别")
                                        self.model.set_classes(self.text_prompt, text_pe)
                                        logger.info("成功设置YOLOE文本提示")

                                        # 更新缓存
                                        self._text_pe_cache = text_pe
                                        self._text_prompt_cache = self.text_prompt.copy() if isinstance(self.text_prompt, list) else self.text_prompt
                                        self._text_embedding_set = True
                            except Exception as e:
                                logger.error(f"设置YOLOE文本提示失败: {str(e)}")
                                import traceback
                                logger.error(traceback.format_exc())
                        else:
                            # 使用已缓存的文本嵌入，不输出日志
                            pass
                    else:
                        # 只在首次检测时输出警告
                        if not self._text_embedding_set:
                            logger.warning("当前模型不是YOLOE模型或不支持文本提示，将使用标准分割")

                    # 执行标准分割（不传递text_prompt参数）
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        retina_masks=True,
                        verbose=False  # 禁用自动打印
                    )
                except Exception as e:
                    logger.error(f"YOLOE文本提示模式失败: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # 回退到标准分割
                    results = self.model(
                        image,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=classes,
                        max_det=max_detections,
                        retina_masks=True,
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
                    log_detections_if_found(result, boxes, self._prompt_free_class_names, self._internal_prompt_type, self._result_names_override)

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

                        # 在无提示模式下，按优先级顺序尝试获取类别名称
                        if self._internal_prompt_type == 0:
                            # 1. 首先从model.names获取
                            if hasattr(self.model, "names") and isinstance(self.model.names, dict) and cls_id in self.model.names:
                                cls_name = self.model.names[cls_id]
                                logger.debug(f"从model.names获取类别：ID={cls_id}, 名称={cls_name}")
                            # 2. 如果model.names中没有，但有_result_names_override，则从_result_names_override获取
                            elif self._result_names_override and cls_id in self._result_names_override:
                                cls_name = self._result_names_override[cls_id]
                                logger.debug(f"从_result_names_override获取类别：ID={cls_id}, 名称={cls_name}")
                            # 3. 如果_result_names_override中没有，但有_prompt_free_class_names，则从_prompt_free_class_names获取
                            elif self._prompt_free_class_names and 0 <= cls_id < len(self._prompt_free_class_names):
                                cls_name = self._prompt_free_class_names[cls_id]
                                logger.debug(f"从_prompt_free_class_names获取类别：ID={cls_id}, 名称={cls_name}")
                            # 4. 最后从result.names获取或使用默认名称
                            else:
                                cls_name = result.names.get(cls_id, f"class_{cls_id}")
                                logger.debug(f"从result.names获取类别：ID={cls_id}, 名称={cls_name}")
                        elif self._internal_prompt_type == 1 and self.text_prompt and 0 <= cls_id < len(self.text_prompt):
                            cls_name = self.text_prompt[cls_id]
                            logger.debug(f"文本提示类别映射: ID={cls_id}, 名称={cls_name}")
                        # 其他情况，使用模型的默认类别名称
                        else:
                            cls_name = result.names.get(cls_id, f"class_{cls_id}")
                            logger.debug(f"使用默认类别名称：ID={cls_id}, 名称={cls_name}")

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
                    log_detections_if_found(result, boxes, self._prompt_free_class_names, self._internal_prompt_type, self._result_names_override)

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

                        # 在无提示模式下，按优先级顺序尝试获取类别名称
                        if self._internal_prompt_type == 0:
                            # 1. 首先从model.names获取
                            if hasattr(self.model, "names") and isinstance(self.model.names, dict) and cls_id in self.model.names:
                                cls_name = self.model.names[cls_id]
                                logger.debug(f"从model.names获取类别：ID={cls_id}, 名称={cls_name}")
                            # 2. 如果model.names中没有，但有_result_names_override，则从_result_names_override获取
                            elif self._result_names_override and cls_id in self._result_names_override:
                                cls_name = self._result_names_override[cls_id]
                                logger.debug(f"从_result_names_override获取类别：ID={cls_id}, 名称={cls_name}")
                            # 3. 如果_result_names_override中没有，但有_prompt_free_class_names，则从_prompt_free_class_names获取
                            elif self._prompt_free_class_names and 0 <= cls_id < len(self._prompt_free_class_names):
                                cls_name = self._prompt_free_class_names[cls_id]
                                logger.debug(f"从_prompt_free_class_names获取类别：ID={cls_id}, 名称={cls_name}")
                            # 4. 最后从result.names获取或使用默认名称
                            else:
                                cls_name = result.names.get(cls_id, f"class_{cls_id}")
                                logger.debug(f"从result.names获取类别：ID={cls_id}, 名称={cls_name}")
                        elif self._internal_prompt_type == 1 and self.text_prompt and 0 <= cls_id < len(self.text_prompt):
                            cls_name = self.text_prompt[cls_id]
                            logger.debug(f"文本提示类别映射: ID={cls_id}, 名称={cls_name}")
                        # 其他情况，使用模型的默认类别名称
                        else:
                            cls_name = result.names.get(cls_id, f"class_{cls_id}")
                            logger.debug(f"使用默认类别名称：ID={cls_id}, 名称={cls_name}")

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

            # 检查是否需要保存图片
            save_images = kwargs.get("save_images", False)
            annotated_image_bytes = None

            # 处理图像保存
            if results and len(results) > 0 and len(segmentations) > 0:
                try:
                    # 使用自定义绘图方法，确保在无提示模式下使用正确的类别名称
                    if self._internal_prompt_type == 0 and len(self._prompt_free_class_names) > 0 and hasattr(result, "boxes"):
                        # 在无提示模式下使用自定义绘图方法
                        annotated_image = custom_plot_detections(
                            image, result, result.boxes,
                            self._prompt_free_class_names,
                            self._internal_prompt_type,
                            self._result_names_override
                        )
                    else:
                        # 其他模式下使用ultralytics自带的plot方法
                        annotated_image = results[0].plot()

                    # 编码图像
                    is_success, buffer = cv2.imencode(".jpg", annotated_image)
                    if not is_success:
                        logger.warning("标注图像编码失败")
                    else:
                        annotated_image_bytes = buffer.tobytes()

                        # 只在检测到目标时才保存图片
                        if save_images:
                            # 获取任务名称
                            task_name = kwargs.get("task_name", None)

                            # 保存图片
                            await self._save_result_image(annotated_image, segmentations, task_name)
                except Exception as plot_err:
                    logger.error(f"绘制或编码标注图像时出错: {str(plot_err)}")
                    import traceback
                    logger.error(traceback.format_exc())

            # 返回结果
            return {
                "segmentations": segmentations,
                "pre_process_time": pre_process_time * 1000,  # 转换为毫秒
                "inference_time": inference_time * 1000,  # 转换为毫秒
                "post_process_time": post_process_time * 1000,  # 转换为毫秒
                "total_time": total_time * 1000,  # 转换为毫秒
                "annotated_image_bytes": annotated_image_bytes
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
                # 检查是否是YOLOE模型
                if hasattr(self.model, "set_classes") and hasattr(self.model, "get_text_pe"):
                    # 对于YOLOE模型，无提示模式需要特殊处理
                    if not self._text_embedding_set:
                        logger.info("检测到YOLOE模型，使用无提示全目标检测模式")

                        # 尝试重置类别设置，确保模型处于无提示模式
                        try:
                            # 对于YOLOE模型，需要设置为无提示模式
                            if hasattr(self.model, "reset_classes"):
                                self.model.reset_classes()
                                logger.info("已重置YOLOE模型类别设置，启用全目标检测")
                            elif hasattr(self.model, "model") and hasattr(self.model.model, "reset_classes"):
                                self.model.model.reset_classes()
                                logger.info("已重置YOLOE模型类别设置（内部模型），启用全目标检测")
                            # 如果没有reset_classes方法，尝试直接设置为无提示模式
                            elif hasattr(self.model, "prompt_free"):
                                self.model.prompt_free = True
                                logger.info("已设置YOLOE模型为无提示模式")
                            elif hasattr(self.model, "model") and hasattr(self.model.model, "prompt_free"):
                                self.model.model.prompt_free = True
                                logger.info("已设置YOLOE模型为无提示模式（内部模型）")
                            # 设置较低的置信度阈值，以便检测更多目标
                            if confidence > 0.1:
                                logger.info(f"在无提示模式下降低置信度阈值: {confidence} -> 0.05")
                                confidence = 0.05
                            # 标记为已设置，避免重复操作
                            self._text_embedding_set = True
                        except Exception as e:
                            logger.error(f"重置YOLOE模型类别设置失败: {str(e)}")

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
                # 检查是否需要设置文本嵌入
                if hasattr(self.model, "set_classes") and hasattr(self.model, "get_text_pe"):
                    # 检查是否需要重新设置文本嵌入
                    need_setup = False

                    # 如果文本提示发生变化或尚未设置过
                    if not self._text_embedding_set or self._text_prompt_cache != self.text_prompt:
                        need_setup = True

                    if need_setup:
                        try:
                            # 检查text_prompt是否为空
                            if not self.text_prompt:
                                logger.warning("文本提示为空，无法设置YOLOE文本提示")
                            else:
                                # 尝试获取文本嵌入
                                logger.info(f"正在获取文本嵌入: {self.text_prompt}")
                                text_pe = self.model.get_text_pe(self.text_prompt)

                                # 检查文本嵌入是否为None
                                if text_pe is None:
                                    logger.warning("获取到的文本嵌入为None，无法设置YOLOE文本提示")
                                else:
                                    # 设置类别和文本嵌入
                                    logger.info(f"正在设置类别和文本嵌入: {len(self.text_prompt)}个类别")
                                    self.model.set_classes(self.text_prompt, text_pe)
                                    logger.info("成功设置YOLOE文本提示")

                                    # 更新缓存
                                    self._text_pe_cache = text_pe
                                    self._text_prompt_cache = self.text_prompt.copy() if isinstance(self.text_prompt, list) else self.text_prompt
                                    self._text_embedding_set = True
                        except Exception as e:
                            logger.error(f"设置YOLOE文本提示失败: {str(e)}")
                            import traceback
                            logger.error(traceback.format_exc())

                # 执行标准检测（不传递text_prompt参数）
                results = self.model(
                    image,
                    conf=confidence,
                    iou=iou_threshold,
                    classes=classes,
                    max_det=max_detections,
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
                log_detections_if_found(result, boxes, self._prompt_free_class_names, self._internal_prompt_type, self._result_names_override)

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

                    # 在无提示模式下，按优先级顺序尝试获取类别名称
                    if self._internal_prompt_type == 0:
                        # 1. 首先从model.names获取
                        if hasattr(self.model, "names") and isinstance(self.model.names, dict) and cls_id in self.model.names:
                            cls_name = self.model.names[cls_id]
                            logger.debug(f"从model.names获取类别：ID={cls_id}, 名称={cls_name}")
                        # 2. 如果model.names中没有，但有_result_names_override，则从_result_names_override获取
                        elif self._result_names_override and cls_id in self._result_names_override:
                            cls_name = self._result_names_override[cls_id]
                            logger.debug(f"从_result_names_override获取类别：ID={cls_id}, 名称={cls_name}")
                        # 3. 如果_result_names_override中没有，但有_prompt_free_class_names，则从_prompt_free_class_names获取
                        elif self._prompt_free_class_names and 0 <= cls_id < len(self._prompt_free_class_names):
                            cls_name = self._prompt_free_class_names[cls_id]
                            logger.debug(f"从_prompt_free_class_names获取类别：ID={cls_id}, 名称={cls_name}")
                        # 4. 最后从result.names获取或使用默认名称
                        else:
                            cls_name = result.names.get(cls_id, f"class_{cls_id}")
                            logger.debug(f"从result.names获取类别：ID={cls_id}, 名称={cls_name}")
                    elif self._internal_prompt_type == 1 and self.text_prompt and 0 <= cls_id < len(self.text_prompt):
                        cls_name = self.text_prompt[cls_id]
                        logger.debug(f"文本提示类别映射: ID={cls_id}, 名称={cls_name}")
                    # 其他情况，使用模型的默认类别名称
                    else:
                        cls_name = result.names.get(cls_id, f"class_{cls_id}")
                        logger.debug(f"使用默认类别名称：ID={cls_id}, 名称={cls_name}")

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

            # 检查是否需要保存图片
            save_images = kwargs.get("save_images", False)
            annotated_image_bytes = None

            # 处理图像保存
            if results and len(results) > 0 and (len(detections) > 0 or len(tracked_objects) > 0):
                try:
                    # 使用自定义绘图方法，确保在无提示模式下使用正确的类别名称
                    if self._internal_prompt_type == 0 and len(self._prompt_free_class_names) > 0:
                        # 在无提示模式下使用自定义绘图方法
                        annotated_image = custom_plot_detections(
                            image, result, boxes,
                            self._prompt_free_class_names,
                            self._internal_prompt_type,
                            self._result_names_override
                        )
                    else:
                        # 其他模式下使用ultralytics自带的plot方法
                        annotated_image = results[0].plot()

                    # 编码图像
                    is_success, buffer = cv2.imencode(".jpg", annotated_image)
                    if not is_success:
                        logger.warning("标注图像编码失败")
                    else:
                        annotated_image_bytes = buffer.tobytes()

                        # 只在检测到目标时才保存图片
                        if save_images:
                            # 获取任务名称
                            task_name = kwargs.get("task_name", None)

                            # 保存图片
                            # 优先使用跟踪结果，如果没有则使用检测结果
                            results_to_save = tracked_objects if tracked_objects else detections
                            await self._save_result_image(annotated_image, results_to_save, task_name)
                except Exception as plot_err:
                    logger.error(f"绘制或编码标注图像时出错: {str(plot_err)}")
                    import traceback
                    logger.error(traceback.format_exc())

            # 返回结果
            return {
                "detections": detections,
                "tracked_objects": tracked_objects,
                "pre_process_time": pre_process_time * 1000,  # 转换为毫秒
                "inference_time": inference_time * 1000,  # 转换为毫秒
                "post_process_time": post_process_time * 1000,  # 转换为毫秒
                "total_time": total_time * 1000,  # 转换为毫秒
                "annotated_image_bytes": annotated_image_bytes
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
