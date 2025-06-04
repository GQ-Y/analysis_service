from typing import List, Optional, Tuple, Dict, Any, Literal
from pydantic import BaseModel, Field

class BaseAnalysisConfig(BaseModel):
    """分析配置基类 - 包含所有分析类型的公共参数"""
    confidence: Optional[float] = Field(
        None,
        description="置信度阈值",
        gt=0,
        lt=1,
        example=0.5
    )
    iou_threshold: Optional[float] = Field(
        None,
        description="IoU阈值",
        gt=0,
        lt=1,
        example=0.45
    )
    classes: Optional[List[str]] = Field(
        None,
        description="需要检测的类别列表",
        example=["person", "car", "truck"]
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
    image_size: Optional[Dict[str, int]] = Field(
        None,
        description="输入图像尺寸",
        example={"width": 640, "height": 640}
    )
    custom_weights_path: Optional[str] = Field(
        None,
        description="模型权重路径，支持HTTP/HTTPS URL、对象存储URL和本地路径",
        example="https://models.example.com/weights/yoloe-v8l-seg.pt"
    )
    device: Optional[int] = Field(
        1,
        description="推理设备：0=CPU, 1=GPU, 2=AUTO",
        ge=0,
        le=2,
        example=1
    )
    half_precision: Optional[bool] = Field(
        False,
        description="是否使用半精度(FP16)"
    )
    nms_type: Optional[int] = Field(
        0,
        description="NMS类型：0=默认, 1=软性, 2=加权, 3=DIoU NMS",
        ge=0,
        le=3,
        example=0
    )
    max_detections: Optional[int] = Field(
        100,
        description="最大检测目标数量",
        ge=1,
        le=1000,
        example=100
    )

class DetectionConfig(BaseAnalysisConfig):
    """目标检测配置 - 继承公共参数，添加检测专用参数"""
    nested_detection: Optional[bool] = Field(
        False,
        description="是否进行嵌套检测（检查目标A是否在目标B内）"
    )
    prompt_type: Optional[int] = Field(
        3,
        description="提示类型：1=文本提示, 2=视觉提示, 3=无提示",
        ge=1,
        le=3,
        example=3
    )
    text_prompt: Optional[List[str]] = Field(
        None,
        description="文本提示(关键词列表)",
        example=["安全帽", "工人", "反光背心", "车辆"]
    )
    visual_prompt: Optional[Dict[str, Any]] = Field(
        None,
        description="视觉提示信息",
        example={
            "type": 2,
            "points": [
                {"x": 0.2, "y": 0.3},
                {"x": 0.8, "y": 0.3},
                {"x": 0.7, "y": 0.7},
                {"x": 0.3, "y": 0.7}
            ],
            "line_width": 3,
            "use_as_mask": True
        }
    )
    object_filter: Optional[Dict[str, float]] = Field(
        None,
        description="目标过滤器",
        example={
            "min_size": 0.02,
            "max_size": 0.5
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
        example="yoloe"
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
    save_images: bool = Field(
        False,
        description="是否保存图像"
    )
    analysis_type: Optional[Literal["detection"]] = Field(
        "detection",
        description="分析类型：仅支持detection（目标检测）",
        example="detection"
    )
    frame_rate: Optional[int] = Field(
        10,
        description="帧率设置(fps)",
        ge=1,
        le=30,
        example=10
    )
    device: Optional[int] = Field(
        1,
        description="推理设备类型：0=CPU, 1=GPU, 2=AUTO",
        ge=0,
        le=2,
        example=1
    )
    enable_callback: bool = Field(
        False,
        description="是否启用回调"
    )
    callback_url: Optional[str] = Field(
        None,
        description="回调URL",
        example="http://example.com/api/callback"
    )
    enable_alarm_recording: bool = Field(
        False,
        description="是否启用报警录像"
    )
    alarm_recording_before: Optional[int] = Field(
        5,
        description="报警前录像时长(秒)",
        ge=0,
        le=60,
        example=5
    )
    alarm_recording_after: Optional[int] = Field(
        5,
        description="报警后录像时长(秒)",
        ge=0,
        le=60,
        example=5
    )
    stream_engine: Optional[Literal["auto", "opencv", "gstreamer"]] = Field(
        "auto",
        description="流处理引擎：auto=自动选择, opencv=OpenCV, gstreamer=GStreamer",
        example="auto"
    )
    enable_hardware_decode: bool = Field(
        False,
        description="是否启用硬件解码（仅GStreamer支持）"
    )
    low_latency: bool = Field(
        False,
        description="是否启用低延迟模式（推荐用于RTSP流）"
    )
    config: Optional[DetectionConfig] = Field(
        None,
        description="检测配置参数"
    )
    analysis_interval: int = Field(
        1,
        description="分析间隔(帧)，每隔多少帧分析一次",
        ge=1,
        example=5
    )
    callback_interval: Optional[int] = Field(
        None,
        description="回调间隔(秒)，同一目标每隔多少秒回调一次，0表示不限制",
        ge=0,
        example=5
    )
    test_markers: Optional[List[str]] = Field(
        None,
        description="测试标记，用于标识测试类型",
        example=["STREAM_RTSP_TEST"]
    )
    return_base64: Optional[bool] = Field(
        True,
        description="是否返回base64编码的图像"
    )

    model_config = {"protected_namespaces": ()}
    
    def model_post_init(self, __context) -> None:
        """验证分析类型"""
        if self.analysis_type and self.analysis_type != "detection":
            raise ValueError("analysis_type目前仅支持'detection'，其他分析类型将在后续版本中提供")

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
    callback_interval: Optional[int] = Field(
        None,
        description="回调间隔(秒)，同一目标每隔多少秒回调一次，0表示不限制",
        ge=0,
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

class VideoEncodingRequest(BaseModel):
    """视频编码请求"""
    task_id: str = Field(
        ...,
        description="任务ID",
        example="vid_20240402_123456_abcd1234"
    )
    enable_encoding: bool = Field(
        True,
        description="是否开启实时分析视频编码"
    )
    format: str = Field(
        "mp4",
        description="视频格式，支持'mp4'或'flv'",
        example="mp4"
    )
    quality: Optional[int] = Field(
        80,
        description="视频质量(1-100)",
        ge=1,
        le=100,
        example=80
    )
    width: Optional[int] = Field(
        None,
        description="视频宽度，为空则使用原始宽度",
        example=640
    )
    height: Optional[int] = Field(
        None,
        description="视频高度，为空则使用原始高度",
        example=480
    )
    fps: Optional[int] = Field(
        15,
        description="视频帧率",
        ge=1,
        le=30,
        example=15
    )

    model_config = {"protected_namespaces": ()}