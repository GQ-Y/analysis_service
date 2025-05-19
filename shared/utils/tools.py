"""
系统工具方法
提供各种系统级工具函数
"""
import uuid
import random
import platform
import socket
import psutil
import os
import json
from typing import Dict, Any, Optional, List, Union, Callable
from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

def get_mac_address() -> str:
    """
    获取MAC地址

    Returns:
        str: MAC地址
    """
    try:
        mac = uuid.getnode()
        mac_str = ':'.join(['{:02X}'.format((mac >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])
        return mac_str
    except Exception as e:
        exception_logger.exception(f"获取MAC地址失败: {str(e)}")
        # 使用随机生成的MAC地址
        mac = [random.randint(0x00, 0xff) for _ in range(6)]
        mac_str = ':'.join(['{:02X}'.format(x) for x in mac])
        return mac_str

def get_local_ip() -> str:
    """
    获取本地IP地址

    Returns:
        str: 本地IP地址
    """
    try:
        # 创建一个临时socket连接来获取本地IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        exception_logger.exception(f"获取本地IP地址失败: {e}")
        return "127.0.0.1"

def get_hostname() -> str:
    """
    获取主机名

    Returns:
        str: 主机名
    """
    return platform.node()

def get_system_info() -> Dict[str, Any]:
    """
    获取系统信息

    Returns:
        Dict[str, Any]: 系统信息
    """
    try:
        # 获取CPU信息
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()

        # 获取内存信息
        memory = psutil.virtual_memory()

        # 获取磁盘信息
        disk = psutil.disk_usage('/')

        # 获取网络信息
        net_io = psutil.net_io_counters()

        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "cpu": {
                "count": cpu_count,
                "frequency": {
                    "current": cpu_freq.current,
                    "min": cpu_freq.min,
                    "max": cpu_freq.max
                }
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            },
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        }
    except Exception as e:
        exception_logger.exception(f"获取系统信息失败: {e}")
        return {}

def get_resource_usage() -> Dict[str, Any]:
    """
    获取资源使用情况

    Returns:
        Dict[str, Any]: 资源使用情况
    """
    try:
        # 获取CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)

        # 获取内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / (1024 * 1024)  # 转换为MB

        # 获取磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used = disk.used / (1024 * 1024 * 1024)  # 转换为GB

        # 获取网络IO
        net_io = psutil.net_io_counters()
        net_sent = net_io.bytes_sent / (1024 * 1024)  # 转换为MB
        net_recv = net_io.bytes_recv / (1024 * 1024)  # 转换为MB

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_used": memory_used,
            "disk_percent": disk_percent,
            "disk_used": disk_used,
            "net_sent": net_sent,
            "net_recv": net_recv
        }
    except Exception as e:
        exception_logger.exception(f"获取资源使用情况失败: {e}")
        return {}

def get_local_models(models_dir: str = None) -> List[str]:
    """
    获取本地可用模型列表
    扫描models目录下的所有子目录，每个子目录名称即为model_code
    例如：
    - data/models/model-gcc/  -> 返回 "model-gcc"
    - data/models/yolo11n/    -> 返回 "yolo11n"

    Args:
        models_dir: 模型目录路径，如果为None则使用默认路径 data/models/

    Returns:
        List[str]: 模型代码列表，即models目录下的子目录名称列表
    """
    try:
        # 如果未指定模型目录，则使用默认路径
        if not models_dir:
            # 获取当前文件所在目录的上级目录（shared目录的上级）
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_dir = os.path.join(current_dir, "data", "models")

        # 检查目录是否存在
        if not os.path.exists(models_dir):
            normal_logger.warning(f"模型目录不存在: {models_dir}")
            return []

        # 获取所有子目录名称（model_code）
        model_codes = []
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            # 只返回目录名称（model_code），并排除以.开头的隐藏目录
            if os.path.isdir(item_path) and not item.startswith('.'):
                # 检查目录中是否包含模型文件（best.pt）
                if os.path.exists(os.path.join(item_path, "best.pt")):
                    model_codes.append(item)
                else:
                    normal_logger.warning(f"模型目录 {item} 中未找到模型文件best.pt")

        normal_logger.debug(f"找到本地模型代码: {model_codes}")
        return model_codes

    except Exception as e:
        exception_logger.exception(f"获取本地模型列表失败: {e}")
        return []

def pretty_print(
    title: str,
    data: Union[Dict[str, Any], List[Dict[str, Any]], List[str], str],
    logger_func: Optional[Callable] = None,
    indent: int = 2,
    prefix: str = "",
    suffix: str = "",
    separator: str = "=",
    separator_length: int = 50,
    show_border: bool = True
) -> None:
    """
    美化打印数据，支持字典、列表、字符串等多种数据类型

    Args:
        title: 标题
        data: 要打印的数据，可以是字典、列表或字符串
        logger_func: 日志函数，如果为None则使用print函数
        indent: 缩进空格数
        prefix: 每行前缀
        suffix: 每行后缀
        separator: 分隔符字符
        separator_length: 分隔符长度
        show_border: 是否显示边框
    """
    # 如果未提供日志函数，则使用print
    if logger_func is None:
        logger_func = print

    # 创建分隔线
    separator_line = separator * separator_length

    # 打印标题和上边框
    if show_border:
        logger_func(f"{prefix}{separator_line}{suffix}")
    logger_func(f"{prefix}【{title}】{suffix}")
    if show_border:
        logger_func(f"{prefix}{separator_line}{suffix}")

    # 根据数据类型进行格式化打印
    if isinstance(data, dict):
        # 打印字典数据
        for key, value in data.items():
            # 处理嵌套字典
            if isinstance(value, dict):
                logger_func(f"{prefix}{' ' * indent}● {key}:{suffix}")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (dict, list)):
                        sub_value_str = json.dumps(sub_value, ensure_ascii=False, indent=2)
                        logger_func(f"{prefix}{' ' * (indent*2)}○ {sub_key}: {sub_value_str}{suffix}")
                    else:
                        logger_func(f"{prefix}{' ' * (indent*2)}○ {sub_key}: {sub_value}{suffix}")
            # 处理嵌套列表
            elif isinstance(value, list):
                logger_func(f"{prefix}{' ' * indent}● {key}:{suffix}")
                for i, item in enumerate(value[:10]):  # 限制显示前10项
                    if isinstance(item, (dict, list)):
                        item_str = json.dumps(item, ensure_ascii=False, indent=2)
                        logger_func(f"{prefix}{' ' * (indent*2)}○ [{i}]: {item_str}{suffix}")
                    else:
                        logger_func(f"{prefix}{' ' * (indent*2)}○ [{i}]: {item}{suffix}")
                if len(value) > 10:
                    logger_func(f"{prefix}{' ' * (indent*2)}○ ... 还有 {len(value) - 10} 项未显示{suffix}")
            # 处理普通值
            else:
                logger_func(f"{prefix}{' ' * indent}● {key}: {value}{suffix}")

    elif isinstance(data, list):
        # 打印列表数据
        if all(isinstance(item, dict) for item in data):
            # 列表中都是字典
            for i, item in enumerate(data[:10]):  # 限制显示前10项
                logger_func(f"{prefix}{' ' * indent}● 项目 {i+1}:{suffix}")
                for key, value in item.items():
                    logger_func(f"{prefix}{' ' * (indent*2)}○ {key}: {value}{suffix}")
                if i < len(data) - 1 and i < 9:
                    logger_func(f"{prefix}{' ' * indent}------------------------{suffix}")
            if len(data) > 10:
                logger_func(f"{prefix}{' ' * indent}● ... 还有 {len(data) - 10} 项未显示{suffix}")
        else:
            # 普通列表
            for i, item in enumerate(data[:20]):  # 限制显示前20项
                logger_func(f"{prefix}{' ' * indent}● [{i}]: {item}{suffix}")
            if len(data) > 20:
                logger_func(f"{prefix}{' ' * indent}● ... 还有 {len(data) - 20} 项未显示{suffix}")

    elif isinstance(data, str):
        # 打印字符串数据，按行分割
        lines = data.split('\n')
        for line in lines:
            logger_func(f"{prefix}{' ' * indent}{line}{suffix}")

    else:
        # 其他类型直接打印
        logger_func(f"{prefix}{' ' * indent}{data}{suffix}")

    # 打印下边框
    if show_border:
        logger_func(f"{prefix}{separator_line}{suffix}")

def pretty_print_task_config(
    task_id: str,
    task_config: Dict[str, Any],
    logger_func: Optional[Callable] = None
) -> None:
    """
    美化打印任务配置

    Args:
        task_id: 任务ID
        task_config: 任务配置
        logger_func: 日志函数，如果为None则使用print函数
    """
    # 如果未提供日志函数，则使用print
    if logger_func is None:
        logger_func = print

    # 打印任务ID
    pretty_print(f"任务 {task_id} 配置参数详情", f"任务ID: {task_id}", logger_func)

    # 打印基本配置
    basic_config = {
        "流地址": task_config.get("stream_url", "未指定"),
        "分析间隔": f"{task_config.get('analysis_interval', 1)} 帧",
        "回调间隔": f"{task_config.get('callback_interval', '未指定')} 秒",
        "设备": task_config.get("device", "auto"),
        "帧率设置": task_config.get("frame_rate", "未指定")
    }
    pretty_print("基本配置", basic_config, logger_func)

    # 打印子任务配置
    subtask_config = task_config.get("subtask", {})
    subtask_data = {
        "分析类型": subtask_config.get("type", "detection"),
        "回调启用": subtask_config.get("callback", {}).get("enabled", False),
        "回调URL": subtask_config.get("callback", {}).get("url", "未指定")
    }
    pretty_print("子任务配置", subtask_data, logger_func)

    # 打印模型配置
    model_config = task_config.get("model", {})
    model_data = {
        "模型代码": model_config.get("code", "yolov8n.pt"),
        "置信度阈值": model_config.get("confidence", 0.5),
        "IoU阈值": model_config.get("iou_threshold", 0.45)
    }
    pretty_print("模型配置", model_data, logger_func)

    # 打印分析配置
    analysis_config = task_config.get("analysis", {})
    analysis_data = {}

    # ROI配置
    if "roi" in analysis_config:
        roi_type = analysis_config.get("roi_type", 0)
        roi_type_name = {
            0: "无ROI",
            1: "矩形",
            2: "多边形",
            3: "线段"
        }.get(roi_type, "未知")
        analysis_data["ROI类型"] = f"{roi_type_name} ({roi_type})"
        analysis_data["ROI设置"] = analysis_config.get("roi", "未指定")
    else:
        analysis_data["ROI"] = "未设置"

    # 检测类别
    if "classes" in analysis_config:
        analysis_data["检测类别"] = analysis_config.get("classes", [])

    # 跟踪配置
    if "tracking_type" in analysis_config:
        tracking_type = analysis_config.get("tracking_type", 0)
        analysis_data["跟踪类型"] = tracking_type

        if tracking_type > 0:
            analysis_data["最大跟踪数"] = analysis_config.get("max_tracks", "未指定")
            analysis_data["最大丢失时间"] = analysis_config.get("max_lost_time", "未指定")
            analysis_data["特征类型"] = analysis_config.get("feature_type", "未指定")

            if "related_cameras" in analysis_config:
                analysis_data["关联摄像头"] = analysis_config.get("related_cameras", [])

    # 计数和速度估计配置
    analysis_data["启用计数"] = analysis_config.get("counting_enabled", False)
    analysis_data["时间阈值"] = analysis_config.get("time_threshold", "未指定")
    analysis_data["速度估计"] = analysis_config.get("speed_estimation", False)

    if "object_filter" in analysis_config:
        analysis_data["对象过滤"] = analysis_config.get("object_filter", [])

    # 提示相关配置
    if "prompt_type" in analysis_config:
        prompt_type = analysis_config.get("prompt_type", 0)
        prompt_type_name = {
            0: "无提示",
            1: "文本提示",
            2: "视觉提示"
        }.get(prompt_type, "未知")
        analysis_data["提示类型"] = f"{prompt_type_name} ({prompt_type})"

        if prompt_type == 1 and "text_prompt" in analysis_config:
            analysis_data["文本提示"] = analysis_config.get("text_prompt", "未指定")
        elif prompt_type == 2 and "visual_prompt" in analysis_config:
            analysis_data["视觉提示"] = analysis_config.get("visual_prompt", "未指定")

    # 引擎和YOLO版本配置
    analysis_data["引擎类型"] = analysis_config.get("engine", task_config.get("engine", 0))
    analysis_data["YOLO版本"] = analysis_config.get("yolo_version", task_config.get("yolo_version", 0))

    # 其他分析配置
    for key, value in analysis_config.items():
        if key not in ["roi", "roi_type", "classes", "tracking_type", "max_tracks",
                      "max_lost_time", "feature_type", "related_cameras", "counting_enabled",
                      "time_threshold", "speed_estimation", "object_filter", "prompt_type",
                      "text_prompt", "visual_prompt", "engine", "yolo_version"]:
            analysis_data[key] = value

    pretty_print("分析配置", analysis_data, logger_func)

    # 打印结果配置
    result_config = task_config.get("result", {})
    result_data = {
        "保存图像": result_config.get("save_images", False),
        "返回Base64": result_config.get("return_base64", True)
    }

    if "alarm_recording" in result_config:
        alarm_recording = result_config.get("alarm_recording", {})
        result_data["报警录像"] = "已启用"
        result_data["报警前录像"] = f"{alarm_recording.get('before_seconds', 5)} 秒"
        result_data["报警后录像"] = f"{alarm_recording.get('after_seconds', 5)} 秒"

    pretty_print("结果配置", result_data, logger_func)

def pretty_print_detection_results(
    task_id: str,
    frame_count: int,
    processed_results: Dict[str, Any],
    detect_time: float,
    logger_func: Optional[Callable] = None
) -> None:
    """
    美化打印检测结果

    Args:
        task_id: 任务ID
        frame_count: 帧计数
        processed_results: 处理后的结果
        detect_time: 检测耗时
        logger_func: 日志函数，如果为None则使用print函数
    """
    # 如果未提供日志函数，则使用print
    if logger_func is None:
        logger_func = print

    # 检测结果
    if "detections" in processed_results:
        detection_count = len(processed_results["detections"])
        if detection_count > 0:
            # 基本信息
            detection_info = {
                "检测到目标数": detection_count,
                "检测耗时": f"{detect_time:.3f}秒",
                "预处理耗时": f"{processed_results['analysis_info']['pre_process_time']:.3f}秒",
                "推理耗时": f"{processed_results['analysis_info']['inference_time']:.3f}秒",
                "后处理耗时": f"{processed_results['analysis_info']['post_process_time']:.3f}秒"
            }

            pretty_print(f"任务 {task_id} 第 {frame_count} 帧检测结果", detection_info, logger_func)

            # 检测目标详情
            detections = []
            for i, det in enumerate(processed_results["detections"][:10]):  # 只显示前10个
                # 检查坐标是否在bbox字典中
                if 'bbox' in det:
                    bbox = det['bbox']
                    detections.append({
                        "目标ID": i+1,
                        "类别": det['class_name'],
                        "置信度": f"{det['confidence']:.2f}",
                        "位置": f"[{bbox['x1']:.1f},{bbox['y1']:.1f},{bbox['x2']:.1f},{bbox['y2']:.1f}]"
                    })
                else:
                    # 兼容直接包含坐标的情况
                    detections.append({
                        "目标ID": i+1,
                        "类别": det['class_name'],
                        "置信度": f"{det['confidence']:.2f}",
                        "位置": "未知"
                    })

            pretty_print("检测目标详情", detections, logger_func)

            if detection_count > 10:
                logger_func(f"... 还有 {detection_count - 10} 个目标未显示")
        else:
            logger_func(f"任务 {task_id}: 第 {frame_count} 帧未检测到目标, 耗时: {detect_time:.3f}秒")

    # 跟踪结果
    if "tracked_objects" in processed_results:
        tracked_count = len(processed_results["tracked_objects"])
        if tracked_count > 0:
            # 跟踪基本信息
            track_ids = [str(obj.get("track_id", "未知")) for obj in processed_results["tracked_objects"][:5]]
            more_ids = ""
            if tracked_count > 5:
                more_ids = f" 等共 {tracked_count} 个"

            tracking_info = {
                "跟踪到目标数": tracked_count,
                "跟踪ID列表": f"{', '.join(track_ids)}{more_ids}"
            }

            pretty_print(f"任务 {task_id} 第 {frame_count} 帧跟踪结果", tracking_info, logger_func)