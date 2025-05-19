"""
特征提取器模块
提供目标特征提取功能
"""
import os
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Union, Tuple

from shared.utils.logger import get_normal_logger, get_exception_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, feature_type: int = 0):
        """
        初始化特征提取器
        
        Args:
            feature_type: 特征类型 (0=颜色直方图, 1=HOG, 2=深度特征)
        """
        self.feature_type = feature_type
        self.model = None
        
        # 根据特征类型初始化
        if feature_type == 0:
            # 颜色直方图特征
            pass
        elif feature_type == 1:
            # HOG特征
            self.hog = cv2.HOGDescriptor()
        elif feature_type == 2:
            # 深度特征
            # TODO: 加载深度特征提取模型
            normal_logger.warning("深度特征提取尚未实现")
        else:
            normal_logger.warning(f"不支持的特征类型: {feature_type}，使用默认颜色直方图特征")
            self.feature_type = 0
        
        normal_logger.info(f"初始化特征提取器: 特征类型={self._get_feature_type_name()}")
    
    def _get_feature_type_name(self) -> str:
        """获取特征类型名称"""
        feature_types = {
            0: "颜色直方图",
            1: "HOG",
            2: "深度特征"
        }
        return feature_types.get(self.feature_type, f"未知特征类型({self.feature_type})")
    
    def extract_feature(self, image: np.ndarray, bbox: Dict[str, float]) -> np.ndarray:
        """
        提取目标特征
        
        Args:
            image: 输入图像
            bbox: 目标边界框
            
        Returns:
            np.ndarray: 特征向量
        """
        try:
            # 提取目标区域
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            roi = image[y1:y2, x1:x2]
            
            # 检查ROI是否有效
            if roi.size == 0:
                normal_logger.warning(f"无效的ROI: {bbox}")
                return np.zeros(128, dtype=np.float32)
            
            # 根据特征类型提取特征
            if self.feature_type == 0:
                # 颜色直方图特征
                return self._extract_color_histogram(roi)
            elif self.feature_type == 1:
                # HOG特征
                return self._extract_hog_feature(roi)
            elif self.feature_type == 2:
                # 深度特征
                return self._extract_deep_feature(roi)
            else:
                # 默认使用颜色直方图特征
                return self._extract_color_histogram(roi)
                
        except Exception as e:
            exception_logger.exception(f"特征提取失败: {str(e)}")
            return np.zeros(128, dtype=np.float32)
    
    def _extract_color_histogram(self, roi: np.ndarray) -> np.ndarray:
        """
        提取颜色直方图特征
        
        Args:
            roi: 目标区域
            
        Returns:
            np.ndarray: 特征向量
        """
        # 调整大小
        roi = cv2.resize(roi, (64, 128))
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 计算直方图
        h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        
        # 归一化
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
        
        # 合并特征
        feature = np.concatenate([h_hist, s_hist, v_hist]).flatten()
        
        return feature
    
    def _extract_hog_feature(self, roi: np.ndarray) -> np.ndarray:
        """
        提取HOG特征
        
        Args:
            roi: 目标区域
            
        Returns:
            np.ndarray: 特征向量
        """
        # 调整大小
        roi = cv2.resize(roi, (64, 128))
        
        # 转换为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 计算HOG特征
        feature = self.hog.compute(gray)
        
        # 归一化
        feature = feature.flatten()
        if np.linalg.norm(feature) > 0:
            feature = feature / np.linalg.norm(feature)
        
        return feature
    
    def _extract_deep_feature(self, roi: np.ndarray) -> np.ndarray:
        """
        提取深度特征
        
        Args:
            roi: 目标区域
            
        Returns:
            np.ndarray: 特征向量
        """
        # TODO: 实现深度特征提取
        normal_logger.warning("深度特征提取尚未实现，使用占位特征")
        
        # 返回占位特征
        return np.zeros(128, dtype=np.float32)
    
    def compute_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        计算特征相似度
        
        Args:
            feature1: 特征向量1
            feature2: 特征向量2
            
        Returns:
            float: 相似度 (0-1)
        """
        # 检查特征向量是否有效
        if feature1.size == 0 or feature2.size == 0:
            return 0.0
            
        # 计算余弦相似度
        dot_product = np.dot(feature1, feature2)
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        
        # 确保相似度在0-1范围内
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
