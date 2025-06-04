"""
帧缓存器
用于缓存已渲染的帧，减少重复渲染开销
"""
from typing import Dict, Any, Optional, Tuple
import hashlib
import threading
import time
from collections import OrderedDict
import numpy as np

from shared.utils.logger import get_normal_logger

normal_logger = get_normal_logger(__name__)


class FrameCache:
    """帧缓存器 - 缓存已渲染的帧"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: float = 5.0):
        """
        初始化帧缓存器
        
        Args:
            max_size: 最大缓存数量
            ttl_seconds: 缓存生存时间（秒）
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # 使用OrderedDict实现LRU缓存
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.RLock()
        
        # 统计信息
        self.hit_count = 0
        self.miss_count = 0
        
        normal_logger.info(f"帧缓存器初始化完成，最大缓存: {max_size}, TTL: {ttl_seconds}秒")
    
    def _generate_cache_key(self, frame_shape: Tuple[int, ...], 
                          analysis_result: Dict[str, Any]) -> str:
        """
        生成缓存键
        
        Args:
            frame_shape: 帧形状
            analysis_result: 分析结果
            
        Returns:
            str: 缓存键
        """
        # 提取关键信息用于生成缓存键
        detections = analysis_result.get("detections", [])
        
        # 简化检测结果，只保留位置和类别信息
        simplified_detections = []
        for det in detections:
            simplified_det = {
                "bbox": det.get("bbox_pixels") or det.get("bbox", []),
                "class_name": det.get("class_name", ""),
                "confidence": round(float(det.get("confidence", 0)), 2)  # 保留2位小数
            }
            simplified_detections.append(simplified_det)
        
        # 创建缓存键
        cache_data = {
            "frame_shape": frame_shape,
            "detections": simplified_detections,
            "detection_count": len(detections)
        }
        
        # 使用JSON字符串的哈希作为缓存键
        import json
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _cleanup_expired(self):
        """清理过期的缓存项"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self._timestamps.items():
            if current_time - timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self._cache:
                del self._cache[key]
            if key in self._timestamps:
                del self._timestamps[key]
        
        if expired_keys:
            normal_logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")
    
    def get(self, frame: np.ndarray, analysis_result: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        从缓存获取渲染后的帧
        
        Args:
            frame: 原始帧
            analysis_result: 分析结果
            
        Returns:
            Optional[np.ndarray]: 缓存的渲染帧，如果没有则返回None
        """
        if not analysis_result or not analysis_result.get("detections"):
            return None
        
        with self._lock:
            # 清理过期缓存
            self._cleanup_expired()
            
            # 生成缓存键
            cache_key = self._generate_cache_key(frame.shape, analysis_result)
            
            if cache_key in self._cache:
                # 缓存命中
                self.hit_count += 1
                # 更新访问时间
                self._timestamps[cache_key] = time.time()
                # 移到末尾（LRU）
                cached_frame = self._cache.pop(cache_key)
                self._cache[cache_key] = cached_frame
                
                normal_logger.debug(f"帧缓存命中: {cache_key[:8]}")
                return cached_frame.copy()  # 返回副本
            else:
                # 缓存未命中
                self.miss_count += 1
                return None
    
    def put(self, frame: np.ndarray, analysis_result: Dict[str, Any], 
            rendered_frame: np.ndarray):
        """
        将渲染后的帧放入缓存
        
        Args:
            frame: 原始帧
            analysis_result: 分析结果
            rendered_frame: 渲染后的帧
        """
        if not analysis_result or not analysis_result.get("detections"):
            return
        
        with self._lock:
            # 生成缓存键
            cache_key = self._generate_cache_key(frame.shape, analysis_result)
            
            # 如果缓存已满，删除最旧的项
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                if oldest_key in self._timestamps:
                    del self._timestamps[oldest_key]
            
            # 添加到缓存
            self._cache[cache_key] = rendered_frame.copy()
            self._timestamps[cache_key] = time.time()
            
            normal_logger.debug(f"帧缓存添加: {cache_key[:8]}, 当前缓存大小: {len(self._cache)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate_percent": round(hit_rate, 2),
                "ttl_seconds": self.ttl_seconds
            }
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self.hit_count = 0
            self.miss_count = 0
            normal_logger.info("帧缓存已清空")
    
    def set_ttl(self, ttl_seconds: float):
        """
        设置缓存生存时间
        
        Args:
            ttl_seconds: 新的TTL值（秒）
        """
        self.ttl_seconds = ttl_seconds
        normal_logger.info(f"帧缓存TTL设置为: {ttl_seconds}秒")


# 全局帧缓存实例
frame_cache = FrameCache(max_size=50, ttl_seconds=3.0)  # 3秒TTL，适合实时场景 