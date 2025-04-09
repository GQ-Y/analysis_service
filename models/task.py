"""
任务模型定义
"""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class TaskBase(BaseModel):
    """任务基础模型"""
    id: str = Field(..., description="任务ID")
    task_name: Optional[str] = Field(None, description="任务名称")
    model_code: str = Field(..., description="模型代码")
    stream_url: str = Field(..., description="流URL")
    callback_urls: Optional[str] = Field(None, description="回调地址")
    output_url: Optional[str] = Field(None, description="输出URL")
    analysis_type: Optional[str] = Field(None, description="分析类型")
    config: Optional[Dict[str, Any]] = Field(None, description="分析配置")
    enable_callback: bool = Field(False, description="是否启用回调")
    save_result: bool = Field(False, description="是否保存结果")
    status: int = Field(0, description="任务状态: 0-等待中, 1-运行中, 2-已完成, -1-失败")
    error_message: Optional[str] = Field(None, description="错误信息")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    stop_time: Optional[datetime] = Field(None, description="停止时间")
    duration: Optional[float] = Field(None, description="运行时长(分钟)")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    model_config = {"protected_namespaces": ()}

class QueueTask(BaseModel):
    """队列任务模型"""
    id: str = Field(..., description="队列任务ID")
    task_id: str = Field(..., description="关联任务ID")
    parent_task_id: Optional[str] = Field(None, description="父任务ID")
    status: int = Field(0, description="任务状态: 0-等待中, 1-运行中, 2-已完成, -1-失败")
    error_message: Optional[str] = Field(None, description="错误信息")
    priority: int = Field(0, description="优先级")
    retry_count: int = Field(0, description="重试次数")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "status": self.status,
            "error_message": self.error_message,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueueTask":
        """从字典创建实例"""
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "started_at" in data and isinstance(data["started_at"], str):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if "completed_at" in data and isinstance(data["completed_at"], str):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data) 