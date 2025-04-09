"""
分析服务响应模型
"""
from typing import List, Dict, Any, Optional, TypeVar, Generic
from pydantic import BaseModel, Field
import uuid

T = TypeVar('T')

class BaseApiResponse(BaseModel, Generic[T]):
    """标准API响应模型"""
    requestId: str = Field(default_factory=lambda: str(uuid.uuid4()), description="请求ID")
    path: str = Field("", description="请求路径")
    success: bool = Field(True, description="请求是否成功")
    message: str = Field("请求成功", description="响应消息")
    code: int = Field(200, description="响应状态码")
    data: Optional[T] = Field(None, description="响应数据")

class BaseResponse(BaseModel):
    """基础响应模型"""
    requestId: str = Field(..., description="请求ID")
    path: str = Field(..., description="请求路径")
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    code: int = Field(..., description="响应码")
    data: Optional[Any] = Field(None, description="响应数据")

class SubTaskInfo(BaseModel):
    """子任务信息"""
    task_id: str = Field(..., description="任务ID")
    task_name: Optional[str] = Field(None, description="任务名称")
    status: int = Field(..., description="任务状态：0-等待中 1-运行中 2-已完成 -1-失败")
    stream_url: str = Field(..., description="视频流URL")
    output_url: Optional[str] = Field(None, description="输出URL")
    saved_path: Optional[str] = Field(None, description="保存路径")

class StreamBatchResponse(BaseModel):
    """批量流分析响应数据"""
    parent_task_id: str
    sub_tasks: List[SubTaskInfo]

class StreamResponse(BaseApiResponse[StreamBatchResponse]):
    """流分析响应"""
    pass

class DetectionResponse(BaseResponse):
    """检测响应"""
    data: Dict[str, Any] = {
        "detections": List[Dict[str, Any]],
        "result_image": Optional[str]
    }

class TaskStatusResponse(BaseResponse):
    """任务状态响应"""
    data: Dict[str, Any] = {
        "task_id": str,
        "status": str,
        "message": Optional[str]
    }

class HealthResponse(BaseResponse):
    """健康检查响应"""
    data: Dict[str, Any] = {
        "status": str,
        "version": str
    }

class DetectionResult(BaseModel):
    """检测结果"""
    track_id: Optional[int] = Field(None, description="跟踪ID")
    class_name: str = Field(..., description="类别名称")
    confidence: float = Field(..., description="置信度")
    bbox: Dict[str, float] = Field(..., description="边界框坐标")
    children: List["DetectionResult"] = Field(default_factory=list, description="嵌套检测的子目标列表")
    track_info: Optional[Dict[str, Any]] = Field(
        None,
        description="跟踪信息，包含轨迹、速度等",
        example={
            "trajectory": [(100, 100, 200, 200), (110, 110, 210, 210)],  # 历史轨迹点 [(x1,y1,x2,y2),...]
            "velocity": (10, 10),                                         # 速度向量 (dx,dy)
            "age": 10,                                                    # 跟踪持续帧数
            "time_since_update": 0                                        # 最后更新后经过的帧数
        }
    )

class ImageAnalysisData(BaseModel):
    """图像分析数据"""
    task_id: str = Field(..., description="任务ID")
    task_name: Optional[str] = Field(None, description="任务名称")
    image_url: str = Field(..., description="图像URL")
    saved_path: Optional[str] = Field(None, description="保存路径")
    objects: List[Dict] = Field(default_factory=list, description="检测到的目标列表")
    result_image: Optional[str] = Field(None, description="base64编码的结果图片")
    start_time: Optional[float] = Field(None, description="开始时间")
    end_time: Optional[float] = Field(None, description="结束时间")
    analysis_duration: Optional[float] = Field(None, description="分析耗时（秒）")

class VideoAnalysisData(BaseModel):
    """视频分析数据"""
    task_id: str = Field(..., description="任务ID")
    task_name: Optional[str] = Field(None, description="任务名称")
    status: int = Field(..., description="任务状态：0-等待中 1-运行中 2-已完成 -1-失败")
    video_url: str = Field(..., description="视频URL")
    saved_path: Optional[str] = Field(None, description="保存路径")
    start_time: Optional[float] = Field(None, description="开始时间")
    end_time: Optional[float] = Field(None, description="结束时间")
    analysis_duration: Optional[float] = Field(None, description="分析耗时（秒）")
    progress: Optional[float] = Field(None, description="处理进度（0-100）")
    total_frames: Optional[int] = Field(None, description="总帧数")
    processed_frames: Optional[int] = Field(None, description="已处理帧数")
    tracking_enabled: Optional[bool] = Field(None, description="是否启用了目标跟踪")
    tracking_stats: Optional[Dict[str, Any]] = Field(
        None,
        description="跟踪统计信息",
        example={
            "total_tracks": 100,           # 总跟踪目标数
            "active_tracks": 5,            # 当前活跃的跟踪目标数
            "avg_track_length": 25.5,      # 平均跟踪长度（帧数）
            "tracker_type": "sort",        # 使用的跟踪器类型
            "tracking_fps": 30.0           # 跟踪处理帧率
        }
    )

class StreamBatchData(BaseModel):
    """流批次数据"""
    batch_id: str = Field(..., description="批次ID")
    task_id: str = Field(..., description="任务ID")
    frame_id: int = Field(..., description="帧ID")
    timestamp: float = Field(..., description="时间戳")
    objects: List[Dict] = Field(default_factory=list, description="检测到的目标列表")
    image_url: Optional[str] = Field(None, description="图像URL")

class ResourceUsageData(BaseModel):
    """资源使用数据"""
    cpu_percent: float = Field(..., description="CPU使用率（%）")
    memory_percent: float = Field(..., description="内存使用率（%）")
    gpu_percent: Optional[float] = Field(None, description="GPU使用率（%）")
    gpu_memory_percent: Optional[float] = Field(None, description="GPU内存使用率（%）")
    disk_percent: float = Field(..., description="磁盘使用率（%）")
    running_tasks: int = Field(..., description="运行中的任务数")
    waiting_tasks: int = Field(..., description="等待中的任务数")

class StreamAnalysisResponse(BaseApiResponse[StreamBatchData]):
    """流分析响应"""
    pass

class ImageAnalysisResponse(BaseApiResponse[ImageAnalysisData]):
    """图像分析响应"""
    pass

class VideoAnalysisResponse(BaseApiResponse[VideoAnalysisData]):
    """视频分析响应"""
    pass

class ResourceStatusResponse(BaseApiResponse[ResourceUsageData]):
    """资源状态响应"""
    pass

# 向后兼容的类型别名
StreamResponse = StreamAnalysisResponse
BaseResponse = BaseApiResponse[Dict[str, Any]] 