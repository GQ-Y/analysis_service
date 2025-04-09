"""
基础数据模型
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class BaseResponse(BaseModel):
    """基础响应模型"""
    status: str = "success"
    message: Optional[str] = None
    data: Optional[dict] = None

class ServiceInfo(BaseModel):
    """服务信息"""
    name: str = Field(..., description="服务名称")
    url: str = Field(..., description="服务URL")
    description: Optional[str] = Field(None, description="服务描述")
    version: Optional[str] = Field(None, description="服务版本")
    status: str = Field("unknown", description="服务状态")
    started_at: Optional[datetime] = Field(None, description="服务启动时间")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "api",
                "url": "http://localhost:8001",
                "description": "API服务",
                "version": "1.0.0",
                "status": "healthy",
                "started_at": "2024-12-12T00:00:00"
            }
        }