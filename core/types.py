"""
核心类型定义
"""
from typing import Dict, List, Optional, Any, Union, Tuple

class BoundingBox:
    """
    边界框类型
    用于表示检测框坐标
    """
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        """
        初始化边界框
        
        Args:
            x1: 左上角x坐标
            y1: 左上角y坐标
            x2: 右下角x坐标
            y2: 右下角y坐标
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        
    @property
    def width(self) -> int:
        """获取宽度"""
        return self.x2 - self.x1
        
    @property
    def height(self) -> int:
        """获取高度"""
        return self.y2 - self.y1
        
    @property
    def area(self) -> int:
        """获取面积"""
        return self.width * self.height
        
    @property
    def center(self) -> Tuple[float, float]:
        """获取中心点"""
        return (self.x1 + self.width / 2, self.y1 + self.height / 2)
        
    def to_dict(self) -> Dict[str, int]:
        """转换为字典"""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height
        }
        
    def __repr__(self) -> str:
        return f"BoundingBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})" 