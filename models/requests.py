from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field

class DetectionConfig(BaseModel):
    """检测配置"""
    confidence: Optional[float] = Field(
        None,
        description="置信度阈值",
        gt=0,
        lt=1,
        example=0.5
    )
    iou: Optional[float] = Field(
        None,
        description="IoU阈值",
        gt=0,
        lt=1,
        example=0.45
    )
    classes: Optional[List[int]] = Field(
        None,
        description="需要检测的类别ID列表",
        example=[0, 2]
    )
    roi_type: Optional[int] = Field(
        0,
        description="ROI类型: 0-无ROI, 1-矩形, 2-多边形, 3-线段",
        ge=0,
        le=3,
        example=1
    )
    roi: Optional[Dict[str, Any]] = Field(
        None,
        description="感兴趣区域，根据roi_type类型不同有不同定义。"
                    "矩形(type=1): {x1, y1, x2, y2}，值为0-1的归一化坐标; "
                    "多边形(type=2): {points: [[x1,y1], [x2,y2], ...]}，值为0-1的归一化坐标; "
                    "线段(type=3): {points: [[x1,y1], [x2,y2]]}, 值为0-1的归一化坐标",
        example={"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}
    )
    imgsz: Optional[int] = Field(
        None,
        description="输入图片大小",
        ge=32,
        le=4096,
        example=640
    )
    nested_detection: Optional[bool] = Field(
        False,
        description="是否进行嵌套检测（检查目标A是否在目标B内）"
    )

class TrackingConfig(BaseModel):
    """目标跟踪配置"""
    tracker_type: str = Field(
        "sort",
        description="跟踪器类型，支持 'sort'、'deepsort'、'bytetrack'",
        example="sort"
    )
    max_age: int = Field(
        30,
        description="目标消失后保持跟踪的最大帧数",
        ge=1,
        le=100,
        example=30
    )
    min_hits: int = Field(
        3,
        description="确认为有效目标所需的最小检测次数",
        ge=1,
        le=10,
        example=3
    )
    iou_threshold: float = Field(
        0.3,
        description="跟踪器的IOU阈值",
        gt=0,
        lt=1,
        example=0.3
    )
    visualization: Optional[Dict[str, bool]] = Field(
        None,
        description="可视化配置，控制跟踪结果的显示方式",
        example={
            "show_tracks": True,    # 显示轨迹
            "show_track_ids": True  # 显示跟踪ID
        }
    )

class ImageAnalysisRequest(BaseModel):
    """图片分析请求"""
    model_code: str = Field(
        ...,
        description="模型代码",
        example="model-gcc"
    )
    task_name: Optional[str] = Field(
        None,
        description="任务名称",
        example="行人检测-1"
    )
    image_urls: List[str] = Field(
        ...,
        description="图片URL列表",
        example=["http://example.com/image.jpg"]
    )
    callback_urls: Optional[str] = Field(
        None,
        description="回调地址，多个用逗号分隔。如果此字段为空，即使enable_callback为true也不会发送回调",
        example="http://callback1,http://callback2"
    )
    enable_callback: bool = Field(
        True,
        description="是否启用回调。注意：只有当此字段为true且callback_urls不为空时才会发送回调"
    )
    is_base64: bool = Field(
        False,
        description="是否返回base64编码的结果图片"
    )
    save_result: bool = Field(
        False,
        description="是否保存分析结果到本地。若为true，则在响应中返回保存的文件路径"
    )
    config: Optional[DetectionConfig] = Field(
        None,
        description="检测配置参数"
    )

    model_config = {"protected_namespaces": ()}

class VideoAnalysisRequest(BaseModel):
    """视频分析请求"""
    model_code: str = Field(
        ...,
        description="模型代码",
        example="model-gcc"
    )
    task_name: Optional[str] = Field(
        None,
        description="任务名称",
        example="视频分析-1"
    )
    video_url: str = Field(
        ...,
        description="视频URL",
        example="http://example.com/video.mp4"
    )
    callback_urls: Optional[str] = Field(
        None,
        description="回调地址，多个用逗号分隔。如果此字段为空，即使enable_callback为true也不会发送回调",
        example="http://callback1,http://callback2"
    )
    enable_callback: bool = Field(
        True,
        description="是否启用回调。注意：只有当此字段为true且callback_urls不为空时才会发送回调"
    )
    save_result: bool = Field(
        False,
        description="是否保存分析结果到本地。若为true，则在响应中返回保存的文件路径"
    )
    config: Optional[DetectionConfig] = Field(
        None,
        description="检测配置参数"
    )
    enable_tracking: bool = Field(
        False,
        description="是否启用目标跟踪"
    )
    tracking_config: Optional[TrackingConfig] = Field(
        None,
        description="目标跟踪配置参数，仅在enable_tracking为true时有效"
    )

    model_config = {"protected_namespaces": ()}

    @property
    def has_valid_video_source(self) -> bool:
        """检查是否有有效的视频源"""
        return bool(self.video_url)

    def model_post_init(self, __context) -> None:
        """验证视频源"""
        if not self.has_valid_video_source:
            raise ValueError("必须提供video_url")

class StreamTask(BaseModel):
    """单个流分析任务"""
    model_code: str = Field(
        ...,
        description="模型代码",
        example="model-gcc"
    )
    task_name: Optional[str] = Field(
        None,
        description="任务名称",
        example="流分析-1"
    )
    stream_url: str = Field(
        ...,
        description="流地址",
        example="rtsp://example.com/stream"
    )
    output_url: Optional[str] = Field(
        None,
        description="输出地址"
    )
    save_result: bool = Field(
        False,
        description="是否保存分析结果到本地。若为true，则在响应中返回保存的文件路径"
    )
    config: Optional[DetectionConfig] = Field(
        None,
        description="检测配置参数"
    )

    model_config = {"protected_namespaces": ()}

class StreamAnalysisRequest(BaseModel):
    """流分析请求"""
    tasks: List[StreamTask] = Field(
        ...,
        description="流分析任务列表",
        example=[{
            "model_code": "model-gcc",
            "stream_url": "rtsp://example.com/stream1",
            "output_url": "rtsp://example.com/output1",
            "config": {
                "confidence": 0.5,
                "iou": 0.45,
                "max_det": 100,
                "classes": [0, 1, 2],
                "roi": {
                    "x1": 0.1,
                    "y1": 0.1,
                    "x2": 0.9,
                    "y2": 0.9
                },
                "imgsz": 640,
                "nested_detection": True
            }
        }]
    )
    callback_urls: Optional[str] = Field(
        None,
        description="回调地址，多个用逗号分隔。如果此字段为空，即使enable_callback为true也不会发送回调",
        example="http://callback1,http://callback2"
    )
    enable_callback: bool = Field(
        True,
        description="是否启用回调。注意：只有当此字段为true且callback_urls不为空时才会发送回调"
    )
    analyze_interval: int = Field(
        1,
        description="分析间隔(秒)",
        example=1
    )
    alarm_interval: int = Field(
        60,
        description="报警间隔(秒)",
        example=60
    )
    random_interval: Tuple[int, int] = Field(
        (0, 0),
        description="随机延迟区间(秒)",
        example=(0, 0)
    )
    push_interval: int = Field(
        5,
        description="推送间隔(秒)",
        example=5
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tasks": [
                        {
                            "model_code": "model-gcc",
                            "stream_url": "rtsp://example.com/stream1",
                            "config": {
                                "confidence": 0.5,
                                "iou": 0.45,
                                "classes": [0, 2],
                                "roi": {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9},
                                "imgsz": 640,
                                "nested_detection": True
                            }
                        }
                    ],
                    "callback_urls": "http://127.0.0.1:8081",
                    "analyze_interval": 1,
                    "alarm_interval": 60,
                    "random_interval": (0, 0),
                    "push_interval": 5
                }
            ]
        }
    }

class BatchStreamTask(BaseModel):
    """批量流分析任务请求"""
    tasks: List[StreamTask] = Field(
        ...,
        description="流分析任务列表",
        min_items=1,
        example=[{
            "model_code": "model-gcc",
            "stream_url": "rtsp://example.com/stream1",
            "config": {
                "confidence": 0.5,
                "iou": 0.45,
                "classes": [0, 2],
                "roi": {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9},
                "imgsz": 640,
                "nested_detection": True
            }
        }]
    )
    callback_urls: Optional[str] = Field(
        None,
        description="回调地址，多个用逗号分隔。如果此字段为空，即使enable_callback为true也不会发送回调",
        example="http://callback1,http://callback2"
    )
    analyze_interval: int = Field(
        1,
        description="分析间隔(秒)",
        ge=1,
        example=1
    )
    alarm_interval: int = Field(
        60,
        description="报警间隔(秒)",
        ge=0,
        example=60
    )
    random_interval: Tuple[int, int] = Field(
        (0, 0),
        description="随机延迟区间(秒)",
        example=(0, 0)
    )
    push_interval: int = Field(
        5,
        description="推送间隔(秒)",
        ge=1,
        example=5
    )

    model_config = {"protected_namespaces": ()}

class TaskStatusRequest(BaseModel):
    """任务状态请求"""
    task_id: str = Field(
        ...,
        description="任务ID",
        example="vid_20240402_123456_abcd1234"
    )