"""
内存管理工具函数
提供系统内存检查、内存对齐、性能测试等工具函数
"""
import os
import sys
import time
import ctypes
import platform
from typing import Dict, Tuple, Any, Optional
import psutil
import numpy as np

# 使用项目现有的日志系统
try:
    from shared.utils.logger import get_normal_logger
    logger = get_normal_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def check_system_memory() -> Dict[str, Any]:
    """
    检查系统内存情况
    
    Returns:
        Dict[str, Any]: 系统内存信息
    """
    try:
        memory_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()
        
        # 获取详细的内存信息
        memory_details = {
            "total_memory_bytes": memory_info.total,
            "available_memory_bytes": memory_info.available,
            "used_memory_bytes": memory_info.used,
            "free_memory_bytes": memory_info.free,
            "memory_usage_percent": memory_info.percent,
            
            "total_memory_gb": memory_info.total / 1024 / 1024 / 1024,
            "available_memory_gb": memory_info.available / 1024 / 1024 / 1024,
            "used_memory_gb": memory_info.used / 1024 / 1024 / 1024,
            "free_memory_gb": memory_info.free / 1024 / 1024 / 1024,
            
            "swap_total_gb": swap_info.total / 1024 / 1024 / 1024,
            "swap_used_gb": swap_info.used / 1024 / 1024 / 1024,
            "swap_usage_percent": swap_info.percent,
            
            "platform": platform.system(),
            "architecture": platform.architecture()[0],
        }
        
        # 添加平台特定的内存信息
        if platform.system() == "Linux":
            memory_details.update(_get_linux_memory_info())
        elif platform.system() == "Darwin":  # macOS
            memory_details.update(_get_macos_memory_info())
        elif platform.system() == "Windows":
            memory_details.update(_get_windows_memory_info())
        
        logger.info(f"系统内存检查完成: "
                   f"总内存{memory_details['total_memory_gb']:.2f}GB, "
                   f"可用{memory_details['available_memory_gb']:.2f}GB, "
                   f"使用率{memory_details['memory_usage_percent']:.1f}%")
        
        return memory_details
        
    except Exception as e:
        logger.error(f"检查系统内存失败: {str(e)}")
        return {}


def _get_linux_memory_info() -> Dict[str, Any]:
    """获取Linux特定的内存信息"""
    info = {}
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        # 解析关键内存信息
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                info['meminfo_total_kb'] = int(line.split()[1])
            elif line.startswith('MemAvailable:'):
                info['meminfo_available_kb'] = int(line.split()[1])
            elif line.startswith('Buffers:'):
                info['buffers_kb'] = int(line.split()[1])
            elif line.startswith('Cached:'):
                info['cached_kb'] = int(line.split()[1])
            elif line.startswith('Hugepagesize:'):
                info['hugepage_size_kb'] = int(line.split()[1])
        
        # 检查是否支持大页内存
        info['hugepages_supported'] = os.path.exists('/sys/kernel/mm/hugepages')
        
    except Exception as e:
        logger.warning(f"获取Linux内存信息失败: {str(e)}")
    
    return info


def _get_macos_memory_info() -> Dict[str, Any]:
    """获取macOS特定的内存信息"""
    info = {}
    try:
        # 使用vm_stat命令获取虚拟内存统计
        import subprocess
        result = subprocess.run(['vm_stat'], capture_output=True, text=True)
        if result.returncode == 0:
            info['vm_stat_available'] = True
            # 可以进一步解析vm_stat输出
        else:
            info['vm_stat_available'] = False
            
    except Exception as e:
        logger.warning(f"获取macOS内存信息失败: {str(e)}")
    
    return info


def _get_windows_memory_info() -> Dict[str, Any]:
    """获取Windows特定的内存信息"""
    info = {}
    try:
        # Windows特定的内存信息
        info['windows_memory'] = True
        
    except Exception as e:
        logger.warning(f"获取Windows内存信息失败: {str(e)}")
    
    return info


def calculate_memory_requirements(resolutions: list, blocks_per_resolution: dict, 
                                channels: int = 3, alignment: int = 64) -> Dict[str, Any]:
    """
    计算内存需求
    
    Args:
        resolutions: 支持的分辨率列表 [(width, height), ...]
        blocks_per_resolution: 每种分辨率的块数量 {"widthxheight": count}
        channels: 图像通道数，默认3（RGB）
        alignment: 内存对齐字节数，默认64
    
    Returns:
        Dict[str, Any]: 内存需求详情
    """
    requirements = {
        "resolutions": {},
        "total_bytes": 0,
        "total_blocks": 0,
        "alignment_bytes": alignment,
        "channels": channels,
    }
    
    for width, height in resolutions:
        res_key = f"{width}x{height}"
        block_count = blocks_per_resolution.get(res_key, 100)
        
        # 计算单个帧的内存需求
        frame_size = width * height * channels
        
        # 应用内存对齐
        aligned_frame_size = align_size(frame_size, alignment)
        
        # 计算该分辨率的总内存需求
        resolution_memory = aligned_frame_size * block_count
        
        requirements["resolutions"][res_key] = {
            "width": width,
            "height": height,
            "block_count": block_count,
            "frame_size_bytes": frame_size,
            "aligned_frame_size_bytes": aligned_frame_size,
            "total_memory_bytes": resolution_memory,
            "total_memory_mb": resolution_memory / 1024 / 1024,
        }
        
        requirements["total_bytes"] += resolution_memory
        requirements["total_blocks"] += block_count
    
    requirements["total_mb"] = requirements["total_bytes"] / 1024 / 1024
    requirements["total_gb"] = requirements["total_bytes"] / 1024 / 1024 / 1024
    
    logger.info(f"内存需求计算完成: 总计{requirements['total_gb']:.2f}GB, "
               f"{requirements['total_blocks']}个内存块")
    
    return requirements


def align_size(size: int, alignment: int) -> int:
    """
    对齐内存大小
    
    Args:
        size: 原始大小
        alignment: 对齐字节数
    
    Returns:
        int: 对齐后的大小
    """
    return ((size + alignment - 1) // alignment) * alignment


def align_memory_address(ptr: int, alignment: int) -> int:
    """
    对齐内存地址
    
    Args:
        ptr: 内存地址
        alignment: 对齐字节数
    
    Returns:
        int: 对齐后的地址
    """
    return ((ptr + alignment - 1) // alignment) * alignment


def check_memory_alignment(ptr: int, alignment: int) -> bool:
    """
    检查内存地址是否对齐
    
    Args:
        ptr: 内存地址
        alignment: 对齐字节数
    
    Returns:
        bool: 是否对齐
    """
    return (ptr % alignment) == 0


def benchmark_memory_operations(test_size_mb: int = 100, iterations: int = 10) -> Dict[str, float]:
    """
    内存操作性能测试
    
    Args:
        test_size_mb: 测试数据大小（MB）
        iterations: 测试迭代次数
    
    Returns:
        Dict[str, float]: 性能测试结果
    """
    logger.info(f"开始内存性能测试: {test_size_mb}MB数据, {iterations}次迭代")
    
    test_size_bytes = test_size_mb * 1024 * 1024
    results = {}
    
    try:
        # 测试内存分配性能
        alloc_times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            data = np.zeros(test_size_bytes, dtype=np.uint8)
            end_time = time.perf_counter()
            alloc_times.append(end_time - start_time)
            del data  # 释放内存
        
        results["allocation_avg_ms"] = (sum(alloc_times) / len(alloc_times)) * 1000
        results["allocation_min_ms"] = min(alloc_times) * 1000
        results["allocation_max_ms"] = max(alloc_times) * 1000
        
        # 测试内存拷贝性能
        source = np.random.randint(0, 255, test_size_bytes, dtype=np.uint8)
        copy_times = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            dest = source.copy()
            end_time = time.perf_counter()
            copy_times.append(end_time - start_time)
            del dest
        
        results["copy_avg_ms"] = (sum(copy_times) / len(copy_times)) * 1000
        results["copy_min_ms"] = min(copy_times) * 1000
        results["copy_max_ms"] = max(copy_times) * 1000
        
        # 计算带宽
        results["copy_bandwidth_gb_s"] = (test_size_mb / 1024) / (results["copy_avg_ms"] / 1000)
        
        # 测试内存访问性能
        access_times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            checksum = np.sum(source)
            end_time = time.perf_counter()
            access_times.append(end_time - start_time)
        
        results["access_avg_ms"] = (sum(access_times) / len(access_times)) * 1000
        results["access_bandwidth_gb_s"] = (test_size_mb / 1024) / (results["access_avg_ms"] / 1000)
        
        del source
        
        logger.info(f"内存性能测试完成:")
        logger.info(f"  分配: {results['allocation_avg_ms']:.2f}ms")
        logger.info(f"  拷贝: {results['copy_avg_ms']:.2f}ms ({results['copy_bandwidth_gb_s']:.2f}GB/s)")
        logger.info(f"  访问: {results['access_avg_ms']:.2f}ms ({results['access_bandwidth_gb_s']:.2f}GB/s)")
        
    except Exception as e:
        logger.error(f"内存性能测试失败: {str(e)}")
        results["error"] = str(e)
    
    return results


def get_optimal_alignment() -> int:
    """
    获取最优的内存对齐字节数
    
    Returns:
        int: 推荐的对齐字节数
    """
    # 根据平台和架构确定最优对齐
    if platform.architecture()[0] == "64bit":
        # 64位系统，使用64字节对齐（缓存行大小）
        return 64
    else:
        # 32位系统，使用32字节对齐
        return 32


def check_numa_topology() -> Dict[str, Any]:
    """
    检查NUMA拓扑结构
    
    Returns:
        Dict[str, Any]: NUMA信息
    """
    numa_info = {
        "numa_available": False,
        "numa_nodes": 0,
        "current_node": -1,
    }
    
    try:
        if platform.system() == "Linux":
            # 检查是否有NUMA支持
            if os.path.exists("/sys/devices/system/node"):
                numa_info["numa_available"] = True
                # 计算NUMA节点数
                nodes = [d for d in os.listdir("/sys/devices/system/node") 
                        if d.startswith("node") and d[4:].isdigit()]
                numa_info["numa_nodes"] = len(nodes)
                
                # 获取当前进程的NUMA节点
                try:
                    with open(f"/proc/{os.getpid()}/numa_maps", "r") as f:
                        # 简单解析，实际实现可能需要更复杂的逻辑
                        numa_info["numa_maps_available"] = True
                except:
                    numa_info["numa_maps_available"] = False
        
        logger.info(f"NUMA检查: 可用={numa_info['numa_available']}, "
                   f"节点数={numa_info['numa_nodes']}")
        
    except Exception as e:
        logger.warning(f"NUMA检查失败: {str(e)}")
    
    return numa_info


def estimate_memory_pressure(threshold: float = 0.8) -> Dict[str, Any]:
    """
    评估当前内存压力
    
    Args:
        threshold: 内存压力阈值
    
    Returns:
        Dict[str, Any]: 内存压力信息
    """
    memory_info = psutil.virtual_memory()
    
    pressure_info = {
        "memory_usage_percent": memory_info.percent,
        "is_under_pressure": memory_info.percent > (threshold * 100),
        "pressure_level": "low",
        "available_gb": memory_info.available / 1024 / 1024 / 1024,
        "threshold_percent": threshold * 100,
    }
    
    # 确定压力级别
    usage_percent = memory_info.percent
    if usage_percent > 95:
        pressure_info["pressure_level"] = "critical"
    elif usage_percent > 90:
        pressure_info["pressure_level"] = "high"
    elif usage_percent > threshold * 100:
        pressure_info["pressure_level"] = "medium"
    else:
        pressure_info["pressure_level"] = "low"
    
    logger.debug(f"内存压力评估: {pressure_info['pressure_level']} "
                f"({usage_percent:.1f}%)")
    
    return pressure_info
