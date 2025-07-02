"""
帧元数据定义模块
提供轻量级帧元数据结构（<100B）和高效序列化
"""
import time
import struct
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
import json

# 使用项目现有的日志系统
try:
    from shared.utils.logger import get_normal_logger
    logger = get_normal_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class FrameAnalysisStatus(IntEnum):
    """帧分析状态位掩码"""
    PENDING = 0x00        # 待分析
    PROCESSING = 0x01     # 分析中
    COMPLETED = 0x02      # 分析完成
    FAILED = 0x04         # 分析失败
    CACHED = 0x08         # 结果已缓存
    DETECTION = 0x10      # 包含检测结果
    TRACKING = 0x20       # 包含跟踪结果
    SEGMENTATION = 0x40   # 包含分割结果
    CLASSIFICATION = 0x80 # 包含分类结果


@dataclass
class FrameMetadata:
    """
    轻量级帧元数据（目标<100B）
    包含帧的基本信息、分析状态和结果索引
    """
    # 基础信息 (32B)
    frame_id: int = 0                    # 8B - 帧唯一ID
    timestamp: float = field(default_factory=time.time)  # 8B - 时间戳
    memory_block_ref: int = 0            # 8B - 内存块引用ID
    sequence_number: int = 0             # 8B - 序列号
    
    # 图像信息 (16B)
    width: int = 0                       # 4B - 图像宽度
    height: int = 0                      # 4B - 图像高度
    channels: int = 3                    # 4B - 图像通道数
    format_type: int = 0                 # 4B - 图像格式类型
    
    # 分析状态 (8B)
    analysis_status: int = FrameAnalysisStatus.PENDING  # 4B - 分析状态位掩码
    analysis_progress: int = 0           # 4B - 分析进度(0-100)
    
    # ROI信息 (16B)
    roi_count: int = 0                   # 4B - ROI数量
    roi_data_compressed: bytes = field(default_factory=lambda: b'\x00' * 12)  # 12B - 压缩ROI数据
    
    # 结果索引 (16B)
    detection_index: int = 0             # 8B - 检测结果索引（0表示无结果）
    result_data_size: int = 0            # 4B - 结果数据大小
    result_checksum: int = 0             # 4B - 结果校验和
    
    # 预留字段 (12B)
    reserved1: int = 0                   # 4B - 预留字段1
    reserved2: int = 0                   # 4B - 预留字段2
    reserved3: int = 0                   # 4B - 预留字段3
    
    # 总计: 100B
    
    def __post_init__(self):
        """初始化后处理"""
        if self.frame_id == 0:
            # 生成基于时间戳的帧ID
            self.frame_id = int(self.timestamp * 1000000) % (2**63)
    
    def set_analysis_status(self, status: FrameAnalysisStatus) -> None:
        """设置分析状态"""
        self.analysis_status |= status.value
        logger.debug(f"帧 {self.frame_id} 设置分析状态: {status.name}")
    
    def clear_analysis_status(self, status: FrameAnalysisStatus) -> None:
        """清除分析状态"""
        self.analysis_status &= ~status.value
        logger.debug(f"帧 {self.frame_id} 清除分析状态: {status.name}")
    
    def has_analysis_status(self, status: FrameAnalysisStatus) -> bool:
        """检查是否有指定分析状态"""
        return bool(self.analysis_status & status.value)
    
    def is_analysis_complete(self) -> bool:
        """检查分析是否完成"""
        return self.has_analysis_status(FrameAnalysisStatus.COMPLETED)
    
    def is_analysis_failed(self) -> bool:
        """检查分析是否失败"""
        return self.has_analysis_status(FrameAnalysisStatus.FAILED)
    
    def get_analysis_types(self) -> List[str]:
        """获取分析类型列表"""
        types = []
        if self.has_analysis_status(FrameAnalysisStatus.DETECTION):
            types.append("detection")
        if self.has_analysis_status(FrameAnalysisStatus.TRACKING):
            types.append("tracking")
        if self.has_analysis_status(FrameAnalysisStatus.SEGMENTATION):
            types.append("segmentation")
        if self.has_analysis_status(FrameAnalysisStatus.CLASSIFICATION):
            types.append("classification")
        return types
    
    def set_roi_data(self, rois: List[Tuple[int, int, int, int]]) -> None:
        """
        设置ROI数据（压缩存储）
        
        Args:
            rois: ROI列表，每个ROI为(x, y, width, height)
        """
        self.roi_count = len(rois)
        if rois:
            # 简单压缩：只存储前3个ROI的坐标
            compressed_data = bytearray(12)
            for i, (x, y, w, h) in enumerate(rois[:3]):
                # 每个ROI用4字节存储：x(1B) + y(1B) + w(1B) + h(1B)
                # 坐标归一化到0-255范围
                offset = i * 4
                compressed_data[offset] = min(255, max(0, x // 8))
                compressed_data[offset + 1] = min(255, max(0, y // 8))
                compressed_data[offset + 2] = min(255, max(0, w // 8))
                compressed_data[offset + 3] = min(255, max(0, h // 8))
            self.roi_data_compressed = bytes(compressed_data)
        else:
            self.roi_data_compressed = b'\x00' * 12
    
    def get_roi_data(self) -> List[Tuple[int, int, int, int]]:
        """
        获取ROI数据（解压缩）
        
        Returns:
            List[Tuple[int, int, int, int]]: ROI列表
        """
        rois = []
        if self.roi_count > 0 and len(self.roi_data_compressed) >= 12:
            for i in range(min(3, self.roi_count)):
                offset = i * 4
                x = self.roi_data_compressed[offset] * 8
                y = self.roi_data_compressed[offset + 1] * 8
                w = self.roi_data_compressed[offset + 2] * 8
                h = self.roi_data_compressed[offset + 3] * 8
                rois.append((x, y, w, h))
        return rois
    
    def serialize(self) -> bytes:
        """
        序列化为二进制数据
        
        Returns:
            bytes: 序列化后的二进制数据（100字节）
        """
        try:
            # 使用struct打包为固定100字节
            data = struct.pack(
                '<QdQQ'      # frame_id, timestamp, memory_block_ref, sequence_number (32B)
                'IIII'       # width, height, channels, format_type (16B)
                'II'         # analysis_status, analysis_progress (8B)
                'I12s'       # roi_count, roi_data_compressed (16B)
                'QII'        # detection_index, result_data_size, result_checksum (16B)
                'III',       # reserved1, reserved2, reserved3 (12B)
                
                self.frame_id,
                self.timestamp,
                self.memory_block_ref,
                self.sequence_number,
                
                self.width,
                self.height,
                self.channels,
                self.format_type,
                
                self.analysis_status,
                self.analysis_progress,
                
                self.roi_count,
                self.roi_data_compressed,
                
                self.detection_index,
                self.result_data_size,
                self.result_checksum,
                
                self.reserved1,
                self.reserved2,
                self.reserved3
            )
            
            logger.debug(f"帧 {self.frame_id} 序列化完成，大小: {len(data)}字节")
            return data
            
        except Exception as e:
            logger.error(f"帧 {self.frame_id} 序列化失败: {str(e)}")
            raise
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'FrameMetadata':
        """
        从二进制数据反序列化
        
        Args:
            data: 二进制数据
            
        Returns:
            FrameMetadata: 反序列化的帧元数据
        """
        try:
            if len(data) != 100:
                raise ValueError(f"数据长度错误: 期望100字节，实际{len(data)}字节")
            
            # 解包二进制数据
            unpacked = struct.unpack(
                '<QdQQ'      # frame_id, timestamp, memory_block_ref, sequence_number
                'IIII'       # width, height, channels, format_type
                'II'         # analysis_status, analysis_progress
                'I12s'       # roi_count, roi_data_compressed
                'QII'        # detection_index, result_data_size, result_checksum
                'III',       # reserved1, reserved2, reserved3
                data
            )
            
            metadata = cls(
                frame_id=unpacked[0],
                timestamp=unpacked[1],
                memory_block_ref=unpacked[2],
                sequence_number=unpacked[3],
                
                width=unpacked[4],
                height=unpacked[5],
                channels=unpacked[6],
                format_type=unpacked[7],
                
                analysis_status=unpacked[8],
                analysis_progress=unpacked[9],
                
                roi_count=unpacked[10],
                roi_data_compressed=unpacked[11],
                
                detection_index=unpacked[12],
                result_data_size=unpacked[13],
                result_checksum=unpacked[14],
                
                reserved1=unpacked[15],
                reserved2=unpacked[16],
                reserved3=unpacked[17]
            )
            
            logger.debug(f"帧 {metadata.frame_id} 反序列化完成")
            return metadata
            
        except Exception as e:
            logger.error(f"帧元数据反序列化失败: {str(e)}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典格式的元数据
        """
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "memory_block_ref": self.memory_block_ref,
            "sequence_number": self.sequence_number,
            
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "format_type": self.format_type,
            
            "analysis_status": self.analysis_status,
            "analysis_progress": self.analysis_progress,
            "analysis_types": self.get_analysis_types(),
            
            "roi_count": self.roi_count,
            "roi_data": self.get_roi_data(),
            
            "detection_index": self.detection_index,
            "result_data_size": self.result_data_size,
            "result_checksum": self.result_checksum,
            
            "is_complete": self.is_analysis_complete(),
            "is_failed": self.is_analysis_failed(),
        }
    
    def to_json(self) -> str:
        """
        转换为JSON格式
        
        Returns:
            str: JSON格式的元数据
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    def get_size_bytes(self) -> int:
        """
        获取元数据大小（字节）
        
        Returns:
            int: 元数据大小
        """
        return 100  # 固定100字节
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"FrameMetadata(id={self.frame_id}, "
                f"size={self.width}x{self.height}x{self.channels}, "
                f"status={self.analysis_status:08b}, "
                f"progress={self.analysis_progress}%)")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()
