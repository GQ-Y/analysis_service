#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: performance_monitor.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-11
描述: 内存性能监控和诊断工具

提供内存管理系统的性能监控和诊断功能：
1. 实时性能指标收集
2. 性能瓶颈分析
3. 内存使用报告生成
4. 优化建议

本文件是分析服务项目的一部分。
"""

import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque
import logging

from .enhanced_memory_manager import get_enhanced_memory_manager
from .smart_memory_pool import get_smart_memory_pool
from ..analyzer.smart_image_extractor import get_smart_image_extractor


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: float
    memory_usage_mb: float
    memory_usage_percent: float
    pool_count: int
    active_buffers: int
    total_buffers: int
    allocation_success_rate: float
    avg_allocation_time_ms: float
    cleanup_frequency: float
    gc_count: int
    extraction_strategy_stats: Dict[str, int]


class PerformanceMonitor:
    """
    性能监控器
    
    实时监控内存管理系统的性能指标，提供诊断和优化建议
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化性能监控器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # 获取组件引用
        self.memory_manager = get_enhanced_memory_manager()
        self.smart_pool = get_smart_memory_pool()
        self.image_extractor = get_smart_image_extractor()
        
        # 性能指标历史
        self.metrics_history = deque(maxlen=1000)  # 保留最近1000条记录
        self.current_metrics: Optional[PerformanceMetrics] = None
        
        # 监控控制
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 1.0  # 1秒采样间隔
        
        # 性能阈值
        self.thresholds = {
            "memory_usage_percent": 80,
            "allocation_success_rate": 90,
            "avg_allocation_time_ms": 10,
            "cleanup_frequency": 10  # 每小时
        }
        
        # 警告回调
        self.warning_callbacks: List[Callable[[PerformanceMetrics], None]] = []
        
        # 统计数据
        self.allocation_times = deque(maxlen=100)
        self.last_gc_count = 0
        
        self.logger.info("📊 性能监控器初始化完成")
    
    def start_monitoring(self):
        """启动性能监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("📊 性能监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        self.logger.info("📊 性能监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        import gc
        
        while self.monitoring:
            try:
                # 收集性能指标
                metrics = self._collect_metrics()
                
                # 更新当前指标
                self.current_metrics = metrics
                
                # 保存历史
                self.metrics_history.append(metrics)
                
                # 检查性能阈值
                self._check_thresholds(metrics)
                
                # 记录关键指标
                if len(self.metrics_history) % 60 == 0:  # 每分钟记录一次
                    self._log_summary()
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"❌ 性能监控错误: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        import gc
        
        # 获取内存统计
        memory_stats = self.memory_manager.get_stats()
        pool_stats = self.smart_pool.get_stats()
        extractor_stats = self.image_extractor.get_stats()
        
        # 计算GC次数
        gc_stats = gc.get_stats()
        current_gc_count = sum(stat.get('collections', 0) for stat in gc_stats)
        gc_count_delta = current_gc_count - self.last_gc_count
        self.last_gc_count = current_gc_count
        
        # 计算平均分配时间
        avg_allocation_time = 0
        if self.allocation_times:
            avg_allocation_time = sum(self.allocation_times) / len(self.allocation_times) * 1000  # 转换为毫秒
        
        # 构建指标
        enhanced_stats = memory_stats.get("enhanced_stats", {})
        
        return PerformanceMetrics(
            timestamp=time.time(),
            memory_usage_mb=enhanced_stats.get("used_memory_mb", 0),
            memory_usage_percent=enhanced_stats.get("usage_percent", 0),
            pool_count=pool_stats.get("pool_count", 0),
            active_buffers=pool_stats.get("active_buffers", 0),
            total_buffers=pool_stats.get("total_buffers", 0),
            allocation_success_rate=enhanced_stats.get("allocation_success_rate", 100),
            avg_allocation_time_ms=avg_allocation_time,
            cleanup_frequency=enhanced_stats.get("cleanup_frequency", 0),
            gc_count=gc_count_delta,
            extraction_strategy_stats={
                "zero_copy": extractor_stats.get("zero_copy_count", 0),
                "copy": extractor_stats.get("copy_count", 0),
                "failures": extractor_stats.get("extraction_failures", 0)
            }
        )
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """检查性能阈值"""
        warnings = []
        
        # 检查内存使用率
        if metrics.memory_usage_percent > self.thresholds["memory_usage_percent"]:
            warnings.append(f"内存使用率过高: {metrics.memory_usage_percent:.1f}%")
        
        # 检查分配成功率
        if metrics.allocation_success_rate < self.thresholds["allocation_success_rate"]:
            warnings.append(f"分配成功率过低: {metrics.allocation_success_rate:.1f}%")
        
        # 检查分配时间
        if metrics.avg_allocation_time_ms > self.thresholds["avg_allocation_time_ms"]:
            warnings.append(f"平均分配时间过长: {metrics.avg_allocation_time_ms:.1f}ms")
        
        # 检查清理频率
        if metrics.cleanup_frequency > self.thresholds["cleanup_frequency"]:
            warnings.append(f"清理频率过高: {metrics.cleanup_frequency:.1f}/小时")
        
        # 触发警告回调
        if warnings:
            self.logger.warning(f"⚠️ 性能警告: {', '.join(warnings)}")
            for callback in self.warning_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    self.logger.error(f"❌ 警告回调失败: {e}")
    
    def _log_summary(self):
        """记录性能摘要"""
        if not self.metrics_history:
            return
        
        # 计算最近一分钟的平均值
        recent_metrics = list(self.metrics_history)[-60:]
        
        avg_memory = sum(m.memory_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_success_rate = sum(m.allocation_success_rate for m in recent_metrics) / len(recent_metrics)
        avg_alloc_time = sum(m.avg_allocation_time_ms for m in recent_metrics) / len(recent_metrics)
        total_gc = sum(m.gc_count for m in recent_metrics)
        
        self.logger.info(
            f"📊 性能摘要 [1分钟]: "
            f"内存 {avg_memory:.1f}%, "
            f"成功率 {avg_success_rate:.1f}%, "
            f"分配时间 {avg_alloc_time:.1f}ms, "
            f"GC次数 {total_gc}"
        )
    
    def record_allocation_time(self, duration: float):
        """
        记录分配时间
        
        Args:
            duration: 分配耗时（秒）
        """
        self.allocation_times.append(duration)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        获取性能报告
        
        Returns:
            Dict[str, Any]: 性能报告
        """
        if not self.metrics_history:
            return {"error": "没有可用的性能数据"}
        
        # 计算统计数据
        metrics_list = list(self.metrics_history)
        
        # 内存使用统计
        memory_usage = [m.memory_usage_percent for m in metrics_list]
        memory_stats = {
            "current": memory_usage[-1] if memory_usage else 0,
            "average": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            "peak": max(memory_usage) if memory_usage else 0,
            "trend": "rising" if len(memory_usage) > 10 and memory_usage[-1] > memory_usage[-10] else "stable"
        }
        
        # 分配性能统计
        alloc_times = [m.avg_allocation_time_ms for m in metrics_list]
        allocation_stats = {
            "avg_time_ms": sum(alloc_times) / len(alloc_times) if alloc_times else 0,
            "max_time_ms": max(alloc_times) if alloc_times else 0,
            "success_rate": metrics_list[-1].allocation_success_rate if metrics_list else 100
        }
        
        # 提取策略统计
        total_extractions = sum(
            sum(m.extraction_strategy_stats.values()) 
            for m in metrics_list
        )
        strategy_stats = {
            "zero_copy_rate": 0,
            "copy_rate": 0,
            "failure_rate": 0
        }
        
        if total_extractions > 0:
            zero_copy_total = sum(m.extraction_strategy_stats.get("zero_copy", 0) for m in metrics_list)
            copy_total = sum(m.extraction_strategy_stats.get("copy", 0) for m in metrics_list)
            failure_total = sum(m.extraction_strategy_stats.get("failures", 0) for m in metrics_list)
            
            strategy_stats["zero_copy_rate"] = zero_copy_total / total_extractions * 100
            strategy_stats["copy_rate"] = copy_total / total_extractions * 100
            strategy_stats["failure_rate"] = failure_total / total_extractions * 100
        
        # 优化建议
        recommendations = self._generate_recommendations(memory_stats, allocation_stats, strategy_stats)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_duration_seconds": len(metrics_list) * self.monitor_interval,
            "memory_stats": memory_stats,
            "allocation_stats": allocation_stats,
            "strategy_stats": strategy_stats,
            "pool_info": {
                "pool_count": metrics_list[-1].pool_count if metrics_list else 0,
                "active_buffers": metrics_list[-1].active_buffers if metrics_list else 0,
                "total_buffers": metrics_list[-1].total_buffers if metrics_list else 0
            },
            "gc_stats": {
                "total_collections": sum(m.gc_count for m in metrics_list),
                "avg_per_minute": sum(m.gc_count for m in metrics_list[-60:]) if len(metrics_list) >= 60 else 0
            },
            "recommendations": recommendations
        }
    
    def _generate_recommendations(self, memory_stats: Dict, allocation_stats: Dict, 
                                 strategy_stats: Dict) -> List[str]:
        """
        生成优化建议
        
        Args:
            memory_stats: 内存统计
            allocation_stats: 分配统计
            strategy_stats: 策略统计
            
        Returns:
            List[str]: 优化建议列表
        """
        recommendations = []
        
        # 内存使用建议
        if memory_stats["average"] > 80:
            recommendations.append("内存使用率持续偏高，建议增加内存限制或优化算法")
        elif memory_stats["peak"] > 90:
            recommendations.append("内存使用存在峰值过高情况，建议检查是否有内存泄漏")
        
        if memory_stats["trend"] == "rising":
            recommendations.append("内存使用呈上升趋势，建议监控是否存在内存泄漏")
        
        # 分配性能建议
        if allocation_stats["avg_time_ms"] > 10:
            recommendations.append("平均分配时间过长，建议增加内存池基础大小")
        
        if allocation_stats["success_rate"] < 95:
            recommendations.append("分配成功率偏低，建议增加内存池最大大小或优化清理策略")
        
        # 策略使用建议
        if strategy_stats["zero_copy_rate"] < 30:
            recommendations.append("零拷贝使用率偏低，建议检查是否可以优化数据结构支持零拷贝")
        
        if strategy_stats["failure_rate"] > 5:
            recommendations.append("提取失败率偏高，建议检查数据格式兼容性")
        
        # 如果没有问题
        if not recommendations:
            recommendations.append("系统运行正常，各项指标良好")
        
        return recommendations
    
    def export_metrics(self, filepath: str):
        """
        导出性能指标
        
        Args:
            filepath: 导出文件路径
        """
        try:
            metrics_data = [asdict(m) for m in self.metrics_history]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "export_time": datetime.now().isoformat(),
                    "metrics": metrics_data,
                    "report": self.get_performance_report()
                }, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 性能指标已导出到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ 导出性能指标失败: {e}")
    
    def reset_metrics(self):
        """重置性能指标"""
        self.metrics_history.clear()
        self.allocation_times.clear()
        self.current_metrics = None
        self.last_gc_count = 0
        
        # 重置组件统计
        self.image_extractor.reset_stats()
        
        self.logger.info("📊 性能指标已重置")
    
    def add_warning_callback(self, callback: Callable[[PerformanceMetrics], None]):
        """
        添加警告回调
        
        Args:
            callback: 警告回调函数
        """
        self.warning_callbacks.append(callback)
    
    def set_threshold(self, name: str, value: float):
        """
        设置性能阈值
        
        Args:
            name: 阈值名称
            value: 阈值
        """
        if name in self.thresholds:
            self.thresholds[name] = value
            self.logger.info(f"📊 性能阈值已更新: {name} = {value}")
        else:
            self.logger.warning(f"⚠️ 未知的阈值名称: {name}")


# 全局性能监控器实例
_performance_monitor: Optional[PerformanceMonitor] = None
_monitor_lock = threading.Lock()


def get_performance_monitor() -> PerformanceMonitor:
    """
    获取全局性能监控器实例
    
    Returns:
        PerformanceMonitor: 性能监控器实例
    """
    global _performance_monitor
    
    with _monitor_lock:
        if _performance_monitor is None:
            _performance_monitor = PerformanceMonitor()
            _performance_monitor.start_monitoring()
        return _performance_monitor


def cleanup_performance_monitor():
    """清理全局性能监控器"""
    global _performance_monitor
    
    with _monitor_lock:
        if _performance_monitor:
            _performance_monitor.stop_monitoring()
            _performance_monitor = None