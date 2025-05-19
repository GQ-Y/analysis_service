#!/usr/bin/env python3
"""
MeekYolo分析服务测试脚本
用于执行测试计划中的API测试
"""

import json
import argparse
import requests
import time
import sys
from typing import Dict, Any, Optional

# API基础URL
BASE_URL = "http://localhost:8002"

# 测试请求示例
TEST_REQUESTS = {
    # 基本检测任务
    "base_detection": {
        "model_code": "yoloe",
        "task_name": "基础检测任务",
        "stream_url": "rtsp://223.85.203.115:554/rtp/34020000001110000038_34020000001320000001",
        "output_url": "",
        "save_result": False,
        "save_images": False,
        "analysis_type": "detection",
        "frame_rate": 10,
        "device": 0,
        "enable_callback": False,
        "callback_url": "",
        "config": {
            "confidence": 0.5,
            "iou_threshold": 0.45,
            "classes": [],
            "roi_type": 0,
            "engine": 0,
            "yolo_version": 2
        }
    },
    
    # 带跟踪的检测任务
    "tracking": {
        "model_code": "yoloe",
        "task_name": "跟踪任务",
        "stream_url": "rtsp://223.85.203.115:554/rtp/34020000001110000038_34020000001320000001",
        "output_url": "",
        "save_result": False,
        "save_images": False,
        "analysis_type": "detection",
        "frame_rate": 10,
        "device": 0,
        "enable_callback": False,
        "callback_url": "",
        "config": {
            "confidence": 0.5,
            "iou_threshold": 0.45,
            "classes": [],
            "roi_type": 0,
            "engine": 0,
            "yolo_version": 2,
            "tracking_type": 1,
            "max_tracks": 50,
            "max_lost_time": 30
        }
    },
    
    # RTMP流测试
    "rtmp": {
        "model_code": "yoloe",
        "task_name": "RTMP流测试",
        "stream_url": "rtmp://223.85.203.115:1935/rtp/34020000001110000038_34020000001320000001",
        "output_url": "",
        "save_result": False,
        "save_images": False,
        "analysis_type": "detection",
        "frame_rate": 10,
        "device": 0,
        "enable_callback": False,
        "callback_url": "",
        "config": {
            "confidence": 0.5,
            "iou_threshold": 0.45,
            "classes": [],
            "roi_type": 0,
            "engine": 0,
            "yolo_version": 2
        }
    },
    
    # HLS流测试
    "hls": {
        "model_code": "yoloe",
        "task_name": "HLS流测试",
        "stream_url": "ws://223.85.203.115:3001/rtp/34020000001110000038_34020000001320000001/hls.m3u8",
        "output_url": "",
        "save_result": False,
        "save_images": False,
        "analysis_type": "detection",
        "frame_rate": 10,
        "device": 0,
        "enable_callback": False,
        "callback_url": "",
        "config": {
            "confidence": 0.5,
            "iou_threshold": 0.45,
            "classes": [],
            "roi_type": 0,
            "engine": 0,
            "yolo_version": 2
        }
    },
    
    # HTTP流测试
    "http": {
        "model_code": "yoloe",
        "task_name": "HTTP流测试",
        "stream_url": "http://223.85.203.115:3001/rtp/34020000001110000038_34020000001320000001.live.flv",
        "output_url": "",
        "save_result": False,
        "save_images": False,
        "analysis_type": "detection",
        "frame_rate": 10,
        "device": 0,
        "enable_callback": False,
        "callback_url": "",
        "config": {
            "confidence": 0.5,
            "iou_threshold": 0.45,
            "classes": [],
            "roi_type": 0,
            "engine": 0,
            "yolo_version": 2
        }
    },
    
    # WebRTC流测试
    "webrtc": {
        "model_code": "yoloe",
        "task_name": "WebRTC流测试",
        "stream_url": "ws://223.85.203.115:3001/rtp/34020000001110000038_34020000001320000001.live.mp4",
        "output_url": "",
        "save_result": False,
        "save_images": False,
        "analysis_type": "detection",
        "frame_rate": 10,
        "device": 0,
        "enable_callback": False,
        "callback_url": "",
        "config": {
            "confidence": 0.5,
            "iou_threshold": 0.45,
            "classes": [],
            "roi_type": 0,
            "engine": 0,
            "yolo_version": 2
        }
    },
    
    # GB28181流测试
    "gb28181": {
        "model_code": "yoloe",
        "task_name": "GB28181测试",
        "stream_url": "gb28181://51010200492000000001:34020000001110000038@223.85.203.115:5060",
        "output_url": "",
        "save_result": False,
        "save_images": False,
        "analysis_type": "detection",
        "frame_rate": 10,
        "device": 0,
        "enable_callback": False,
        "callback_url": "",
        "config": {
            "confidence": 0.5,
            "iou_threshold": 0.45,
            "classes": [],
            "gb28181": {
                "sip_id": "51010200492000000001",
                "sip_domain": "5101020049",
                "sip_password": "zyc666"
            }
        }
    },
    
    # ROI区域分析测试
    "roi": {
        "model_code": "yoloe",
        "task_name": "ROI测试",
        "stream_url": "rtsp://223.85.203.115:554/rtp/34020000001110000038_34020000001320000001",
        "output_url": "",
        "save_result": False,
        "save_images": False,
        "analysis_type": "detection",
        "frame_rate": 10,
        "device": 0,
        "enable_callback": False,
        "callback_url": "",
        "config": {
            "confidence": 0.5,
            "iou_threshold": 0.45,
            "classes": [],
            "roi_type": 1,
            "roi": {
                "x1": 0.1,
                "y1": 0.1,
                "x2": 0.9,
                "y2": 0.9
            },
            "engine": 0,
            "yolo_version": 2
        }
    }
}

def check_health() -> bool:
    """检查API服务健康状态
    
    Returns:
        bool: 服务是否健康
    """
    try:
        # 正确的健康检查API路径
        endpoint = "/api/v1/health"
        
        print(f"尝试访问健康检查端点: {endpoint}")
        response = requests.get(f"{BASE_URL}{endpoint}")
        
        if response.status_code == 200:
            print(f"服务健康状态: 正常")
            result = response.json()
            print(f"健康检查详情: {result.get('data', {}).get('status')} - {result.get('message')}")
            print(f"CPU使用率: {result.get('data', {}).get('cpu')}")
            print(f"内存使用率: {result.get('data', {}).get('memory')}")
            return True
        else:
            print(f"服务健康状态: 异常 (状态码: {response.status_code})")
            return False
    except Exception as e:
        print(f"服务健康检查失败: {str(e)}")
        return False

def create_task(test_type: str) -> Optional[str]:
    """创建分析任务
    
    Args:
        test_type: 测试类型，对应TEST_REQUESTS中的键
        
    Returns:
        Optional[str]: 成功创建的任务ID，失败则返回None
    """
    if test_type not in TEST_REQUESTS:
        print(f"未知测试类型: {test_type}")
        return None
    
    try:
        request_data = TEST_REQUESTS[test_type]
        print(f"创建{test_type}测试任务: {request_data['task_name']}")
        
        # 使用正确的API路径
        response = requests.post(
            f"{BASE_URL}/api/v1/tasks/start", 
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success") and "data" in result and "task_id" in result["data"]:
                task_id = result["data"]["task_id"]
                print(f"任务创建成功，ID: {task_id}")
                return task_id
            else:
                print(f"任务创建失败: {result.get('message', '未知错误')}")
                return None
        else:
            print(f"任务创建失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
    except Exception as e:
        print(f"创建任务时出错: {str(e)}")
        return None

def check_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """检查任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        Optional[Dict[str, Any]]: 任务状态信息，查询失败则返回None
    """
    try:
        response = requests.get(f"{BASE_URL}/api/v1/tasks/status/{task_id}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success") and "data" in result:
                task_data = result["data"]
                print(f"任务状态: {task_data.get('status')}")
                return task_data
            else:
                print(f"获取任务状态失败: {result.get('message', '未知错误')}")
                return None
        else:
            print(f"获取任务状态失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
    except Exception as e:
        print(f"检查任务状态时出错: {str(e)}")
        return None

def stop_task(task_id: str) -> bool:
    """停止任务
    
    Args:
        task_id: 任务ID
        
    Returns:
        bool: 是否成功停止任务
    """
    try:
        response = requests.post(f"{BASE_URL}/api/v1/tasks/stop/{task_id}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"任务 {task_id} 已发送停止指令")
                return True
            else:
                print(f"停止任务失败: {result.get('message', '未知错误')}")
                return False
        else:
            print(f"停止任务失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
    except Exception as e:
        print(f"停止任务时出错: {str(e)}")
        return False

def get_task_details(task_id: str) -> Optional[Dict[str, Any]]:
    """获取任务详细信息
    
    Args:
        task_id: 任务ID
        
    Returns:
        Optional[Dict[str, Any]]: 任务详细信息，查询失败则返回None
    """
    try:
        response = requests.get(f"{BASE_URL}/api/v1/tasks/list")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success") and "data" in result and "tasks" in result["data"]:
                for task in result["data"]["tasks"]:
                    if task.get("task_id") == task_id:
                        return task
                print(f"未找到ID为 {task_id} 的任务")
                return None
            else:
                print(f"获取任务列表失败: {result.get('message', '未知错误')}")
                return None
        else:
            print(f"获取任务列表失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
    except Exception as e:
        print(f"获取任务详细信息时出错: {str(e)}")
        return None

def run_test(test_type: str, duration: int = 60) -> None:
    """运行指定测试
    
    Args:
        test_type: 测试类型
        duration: 测试持续时间(秒)
    """
    # 检查服务健康状态
    if not check_health():
        print("服务不健康，无法执行测试")
        return
    
    # 创建任务
    task_id = create_task(test_type)
    if not task_id:
        print("创建任务失败，测试中止")
        return
    
    # 等待并监控任务
    print(f"测试将运行 {duration} 秒，期间每10秒检查一次任务状态")
    start_time = time.time()
    
    try:
        last_status = None
        
        while time.time() - start_time < duration:
            # 等待10秒
            time.sleep(10)
            
            # 检查任务状态
            status = check_task_status(task_id)
            if not status:
                print("获取任务状态失败")
                continue
            
            current_status = status.get("status")
            
            # 如果状态变化并且是错误状态，获取详细信息
            if last_status != current_status and current_status in [-1, -2, -3]:
                details = get_task_details(task_id)
                if details:
                    print(f"任务错误详情: {details.get('error', '无详细信息')}")
            
            last_status = current_status
                
            # 如果任务已经终止，结束测试
            if current_status in ["completed", "failed", "stopped", 2, -1, -5]:
                print(f"任务已终止，状态: {current_status}")
                # 获取详细信息
                details = get_task_details(task_id)
                if details:
                    print(f"任务错误详情: {details.get('error', '无详细信息')}")
                break
            
            # 打印已经运行的时间
            elapsed = time.time() - start_time
            print(f"已运行: {int(elapsed)}秒，剩余: {int(duration - elapsed)}秒")
    except KeyboardInterrupt:
        print("测试被用户中断")
    finally:
        # 停止任务
        if not stop_task(task_id):
            print("警告: 可能未能成功停止任务")
        
        # 最后一次获取详细信息
        details = get_task_details(task_id)
        if details and details.get("status") in [-1, -2, -3]:
            print(f"任务最终错误详情: {details.get('error', '无详细信息')}")
            
        print(f"{test_type}测试完成")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MeekYolo 分析服务测试脚本")
    parser.add_argument("test_type", type=str, choices=list(TEST_REQUESTS.keys()) + ["all"],
                        help="要执行的测试类型")
    parser.add_argument("-d", "--duration", type=int, default=60,
                        help="每项测试运行时间(秒)，默认60秒")
    parser.add_argument("-u", "--url", type=str, default="http://localhost:8002",
                        help="API服务基础URL")
    
    args = parser.parse_args()
    
    # 设置基础 URL
    global BASE_URL
    BASE_URL = args.url
    
    # 执行所有测试或单个测试
    if args.test_type == "all":
        print("执行所有测试...")
        for test_type in TEST_REQUESTS.keys():
            print(f"\n--- 开始 {test_type} 测试 ---")
            run_test(test_type, args.duration)
            print(f"--- 完成 {test_type} 测试 ---\n")
    else:
        run_test(args.test_type, args.duration)

if __name__ == "__main__":
    main() 