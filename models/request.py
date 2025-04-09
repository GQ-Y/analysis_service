from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class StreamAnalysisRequest(BaseModel):
    """流分析请求"""
    model_code: str = Field(..., description="模型代码")
    task_name: str = Field(..., description="任务名称")
    stream_url: str = Field(..., description="流地址")
    callback_urls: Optional[str] = Field(None, description="回调地址，多个地址用逗号分隔")
    enable_callback: bool = Field(False, description="是否启用回调")
    save_result: bool = Field(False, description="是否保存结果")
    config: Optional[Dict[str, Any]] = Field(None, description="分析配置")
    
    model_config = {"protected_namespaces": ()}

    @validator("stream_url")
    def validate_stream_url(cls, v):
        """验证流地址"""
        if not v.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            raise ValueError("流地址必须以 rtsp://, rtmp://, http:// 或 https:// 开头")
        return v

    @validator("callback_urls")
    def validate_callback_urls(cls, v):
        """验证回调地址"""
        if v:
            urls = v.split(",")
            for url in urls:
                if not url.startswith(("http://", "https://")):
                    raise ValueError("回调地址必须以 http:// 或 https:// 开头")
        return v

    @validator("config")
    def validate_config(cls, v):
        """验证分析配置"""
        if v:
            # 验证置信度
            if "confidence" in v and not (0 <= v["confidence"] <= 1):
                raise ValueError("置信度必须在 0-1 之间")
                
            # 验证IOU
            if "iou" in v and not (0 <= v["iou"] <= 1):
                raise ValueError("IOU必须在 0-1 之间")
                
            # 验证类别
            if "classes" in v and not isinstance(v["classes"], list):
                raise ValueError("类别必须是列表")
                
            # 验证ROI
            if "roi" in v:
                roi = v["roi"]
                if not isinstance(roi, dict):
                    raise ValueError("ROI必须是字典")
                if not all(k in roi for k in ["x1", "y1", "x2", "y2"]):
                    raise ValueError("ROI必须包含 x1, y1, x2, y2")
                if not all(0 <= roi[k] <= 1 for k in ["x1", "y1", "x2", "y2"]):
                    raise ValueError("ROI坐标必须在 0-1 之间")
                if roi["x1"] >= roi["x2"] or roi["y1"] >= roi["y2"]:
                    raise ValueError("ROI坐标无效")
                    
            # 验证图像大小
            if "imgsz" in v and not isinstance(v["imgsz"], (int, list)):
                raise ValueError("图像大小必须是整数或列表")
                
            # 验证嵌套检测
            if "nested_detection" in v and not isinstance(v["nested_detection"], bool):
                raise ValueError("嵌套检测必须是布尔值")
                
        return v 