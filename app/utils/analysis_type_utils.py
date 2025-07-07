#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: analysis_type_utils.py
作者: Assistant
创建日期: 2025-01-14
描述: 分析类型转换工具

提供数字和枚举之间的转换功能，支持向前和向后兼容。

本文件是分析服务项目的一部分。
"""

from typing import Optional, Dict, List
from app.models.base_model import AnalysisTypeEnum


# 分析类型数字映射表
ANALYSIS_TYPE_MAPPING = {
    1: AnalysisTypeEnum.DETECTION,          # 目标检测
    2: AnalysisTypeEnum.CLASSIFICATION,     # 图像分类
    3: AnalysisTypeEnum.SEGMENTATION,       # 图像分割  
    4: AnalysisTypeEnum.TRACKING,           # 目标跟踪
    5: AnalysisTypeEnum.POSE_ESTIMATION,    # 姿态估计
    6: AnalysisTypeEnum.FACE_RECOGNITION,   # 人脸识别
}

# 反向映射表（枚举到数字）
ANALYSIS_TYPE_REVERSE_MAPPING = {v: k for k, v in ANALYSIS_TYPE_MAPPING.items()}


def get_analysis_type_by_number(type_number: int) -> Optional[AnalysisTypeEnum]:
    """根据数字获取分析类型枚举
    
    Args:
        type_number: 分析类型数字
        
    Returns:
        Optional[AnalysisTypeEnum]: 分析类型枚举，如果找不到返回None
    """
    return ANALYSIS_TYPE_MAPPING.get(type_number)


def get_analysis_type_number(analysis_type: AnalysisTypeEnum) -> Optional[int]:
    """根据分析类型枚举获取数字
    
    Args:
        analysis_type: 分析类型枚举
        
    Returns:
        Optional[int]: 分析类型数字，如果找不到返回None
    """
    return ANALYSIS_TYPE_REVERSE_MAPPING.get(analysis_type)


def get_all_analysis_types() -> Dict[int, str]:
    """获取所有分析类型的映射
    
    Returns:
        Dict[int, str]: 数字到分析类型名称的映射
    """
    return {num: enum_type.value for num, enum_type in ANALYSIS_TYPE_MAPPING.items()}


def get_analysis_type_info() -> List[Dict[str, any]]:
    """获取分析类型详细信息
    
    Returns:
        List[Dict[str, any]]: 分析类型信息列表
    """
    info_list = []
    
    type_descriptions = {
        AnalysisTypeEnum.DETECTION: "检测图像中的目标对象，包括位置和类别",
        AnalysisTypeEnum.CLASSIFICATION: "对整个图像进行分类，识别图像内容",
        AnalysisTypeEnum.SEGMENTATION: "对图像进行像素级分割，识别每个像素的类别",
        AnalysisTypeEnum.TRACKING: "跟踪视频中的目标对象，维持目标身份",
        AnalysisTypeEnum.POSE_ESTIMATION: "估计人体或对象的姿态和关键点",
        AnalysisTypeEnum.FACE_RECOGNITION: "识别和验证人脸身份",
    }
    
    for type_number, analysis_type in ANALYSIS_TYPE_MAPPING.items():
        info_list.append({
            'id': type_number,
            'name': analysis_type.value,
            'display_name': _get_display_name(analysis_type),
            'description': type_descriptions.get(analysis_type, ""),
            'enum_value': analysis_type
        })
    
    return info_list


def _get_display_name(analysis_type: AnalysisTypeEnum) -> str:
    """获取分析类型的显示名称（中文）
    
    Args:
        analysis_type: 分析类型枚举
        
    Returns:
        str: 显示名称
    """
    display_names = {
        AnalysisTypeEnum.DETECTION: "目标检测",
        AnalysisTypeEnum.CLASSIFICATION: "图像分类", 
        AnalysisTypeEnum.SEGMENTATION: "图像分割",
        AnalysisTypeEnum.TRACKING: "目标跟踪",
        AnalysisTypeEnum.POSE_ESTIMATION: "姿态估计",
        AnalysisTypeEnum.FACE_RECOGNITION: "人脸识别",
    }
    return display_names.get(analysis_type, analysis_type.value)


def validate_analysis_type(type_input) -> Optional[AnalysisTypeEnum]:
    """验证并转换分析类型输入
    
    Args:
        type_input: 分析类型输入（可以是数字、字符串或枚举）
        
    Returns:
        Optional[AnalysisTypeEnum]: 验证后的分析类型枚举
    """
    if isinstance(type_input, AnalysisTypeEnum):
        return type_input
    
    if isinstance(type_input, int):
        return get_analysis_type_by_number(type_input)
    
    if isinstance(type_input, str):
        # 尝试作为枚举值解析
        try:
            return AnalysisTypeEnum(type_input)
        except ValueError:
            pass
        
        # 尝试作为数字字符串解析
        try:
            type_number = int(type_input)
            return get_analysis_type_by_number(type_number)
        except ValueError:
            pass
    
    return None


def is_valid_analysis_type(type_input) -> bool:
    """检查分析类型是否有效
    
    Args:
        type_input: 分析类型输入
        
    Returns:
        bool: 是否有效
    """
    return validate_analysis_type(type_input) is not None 