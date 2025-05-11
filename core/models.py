"""
核心数据模型
定义分析服务的核心数据模型
"""
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class StandardResponse(BaseModel):
    """标准响应模型"""
    requestId: str = Field(..., description="请求ID")
    path: str = Field(..., description="请求路径")
    success: bool = Field(..., description="是否成功")
    message: str = Field("Success", description="响应消息")
    code: int = Field(200, description="状态码")
    data: Optional[Any] = Field(None, description="响应数据")
    timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000), description="时间戳")

    model_config = {
        "json_schema_extra": {
            "example": {
                "requestId": "550e8400-e29b-41d4-a716-446655440000",
                "path": "/api/v1/analyze",
                "success": True,
                "message": "Success",
                "code": 200,
                "data": {
                    "result": "分析结果"
                },
                "timestamp": 1616633599000
            }
        }
    }

class AnalysisType(str, Enum):
    """分析类型"""
    DETECTION = "detection"  # 目标检测
    SEGMENTATION = "segmentation"  # 实例分割
    TRACKING = "tracking"  # 目标跟踪
    CROSS_CAMERA = "cross_camera"  # 跨摄像头跟踪
    COUNTING = "counting"  # 计数分析

class RoiType(int, Enum):
    """ROI类型"""
    NONE = 0  # 无ROI
    RECTANGLE = 1  # 矩形ROI
    POLYGON = 2  # 多边形ROI
    LINE = 3  # 线段ROI

class AnalysisStatus(str, Enum):
    """分析状态"""
    PENDING = "pending"  # 等待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败

class BoundingBox(BaseModel):
    """边界框"""
    x: float = Field(..., description="左上角x坐标")
    y: float = Field(..., description="左上角y坐标")
    width: float = Field(..., description="宽度")
    height: float = Field(..., description="高度")

class DetectionResult(BaseModel):
    """检测结果"""
    class_id: int = Field(..., description="类别ID")
    class_name: str = Field(..., description="类别名称")
    confidence: float = Field(..., description="置信度")
    bbox: BoundingBox = Field(..., description="边界框")

class SegmentationResult(BaseModel):
    """分割结果"""
    class_id: int = Field(..., description="类别ID")
    class_name: str = Field(..., description="类别名称")
    confidence: float = Field(..., description="置信度")
    bbox: BoundingBox = Field(..., description="边界框")
    mask: List[List[float]] = Field(..., description="分割掩码")

class TrackingResult(BaseModel):
    """跟踪结果"""
    track_id: str = Field(..., description="跟踪ID")
    class_id: int = Field(..., description="类别ID")
    class_name: str = Field(..., description="类别名称")
    confidence: float = Field(..., description="置信度")
    bbox: BoundingBox = Field(..., description="边界框")
    frame_id: int = Field(..., description="帧ID")

class CrossCameraResult(BaseModel):
    """跨摄像头跟踪结果"""
    global_track_id: str = Field(..., description="全局跟踪ID")
    camera_id: str = Field(..., description="摄像头ID")
    track_id: str = Field(..., description="本地跟踪ID")
    class_id: int = Field(..., description="类别ID")
    class_name: str = Field(..., description="类别名称")
    confidence: float = Field(..., description="置信度")
    bbox: BoundingBox = Field(..., description="边界框")
    frame_id: int = Field(..., description="帧ID")
    timestamp: int = Field(..., description="时间戳")

class AnalysisResult(BaseModel):
    """分析结果模型"""
    task_id: str = Field(..., description="任务ID")
    timestamp: float = Field(..., description="时间戳")
    analysis_type: str = Field(..., description="分析类型")
    detection_results: Optional[List[Dict[str, Any]]] = Field(None, description="检测结果")
    segmentation_results: Optional[List[Dict[str, Any]]] = Field(None, description="分割结果")
    tracking_results: Optional[List[Dict[str, Any]]] = Field(None, description="跟踪结果")
    frame_info: Optional[Dict[str, Any]] = Field(None, description="帧信息")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    image_data: Optional[Dict[str, Any]] = Field(None, description="图像数据") 