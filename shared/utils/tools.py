"""
系统工具方法
提供各种系统级工具函数
"""
import logging
import uuid
import random
import platform
import socket
import psutil
from typing import Dict, Any, Optional

# 配置日志
logger = logging.getLogger(__name__)

def get_mac_address() -> str:
    """
    获取MAC地址
    
    Returns:
        str: MAC地址
    """
    try:
        mac = uuid.getnode()
        mac_str = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])
        return mac_str
    except Exception as e:
        logger.error(f"获取MAC地址失败: {str(e)}")
        # 使用随机生成的MAC地址
        mac = [random.randint(0x00, 0xff) for _ in range(6)]
        mac_str = ':'.join(['{:02x}'.format(x) for x in mac])
        return mac_str

def get_local_ip() -> str:
    """
    获取本地IP地址
    
    Returns:
        str: 本地IP地址
    """
    try:
        # 创建一个临时socket连接来获取本地IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.error(f"获取本地IP地址失败: {e}")
        return "127.0.0.1"

def get_hostname() -> str:
    """
    获取主机名
    
    Returns:
        str: 主机名
    """
    return platform.node()

def get_system_info() -> Dict[str, Any]:
    """
    获取系统信息
    
    Returns:
        Dict[str, Any]: 系统信息
    """
    try:
        # 获取CPU信息
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # 获取内存信息
        memory = psutil.virtual_memory()
        
        # 获取磁盘信息
        disk = psutil.disk_usage('/')
        
        # 获取网络信息
        net_io = psutil.net_io_counters()
        
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "cpu": {
                "count": cpu_count,
                "frequency": {
                    "current": cpu_freq.current,
                    "min": cpu_freq.min,
                    "max": cpu_freq.max
                }
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            },
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        }
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        return {}

def get_resource_usage() -> Dict[str, Any]:
    """
    获取资源使用情况
    
    Returns:
        Dict[str, Any]: 资源使用情况
    """
    try:
        # 获取CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 获取内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / (1024 * 1024)  # 转换为MB
        
        # 获取磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used = disk.used / (1024 * 1024 * 1024)  # 转换为GB
        
        # 获取网络IO
        net_io = psutil.net_io_counters()
        net_sent = net_io.bytes_sent / (1024 * 1024)  # 转换为MB
        net_recv = net_io.bytes_recv / (1024 * 1024)  # 转换为MB
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_used": memory_used,
            "disk_percent": disk_percent,
            "disk_used": disk_used,
            "net_sent": net_sent,
            "net_recv": net_recv
        }
    except Exception as e:
        logger.error(f"获取资源使用情况失败: {e}")
        return {} 