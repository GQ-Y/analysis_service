#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速创建和启动分析任务的脚本
"""

import requests
import json
import sys

# API配置
BASE_URL = "http://127.0.0.1:8002"
API_PREFIX = "/api/v1"

# 测试任务配置
TEST_TASK = {
    "analysis_type": 3,
    "callback_interval": 10,
    "callback_urls": [],
    "config": {
        "enable_tracking": True,
        "max_objects": 100
    },
    "description": "检测商场入口的人流情况，统计进出人数",
    "model_configs": [
        {
            "analysis_fps": 10,
            "confidence_threshold": 0.1,
            "iou_threshold": 0.45,
            "model_code": "yolo11n"
        }
    ],
    "name": "商场人流检测任务",
    "roi_config": {
        "enabled": False,
        "regions": [
            {
                "name": "entrance",
                "points": [
                    [100, 100],
                    [500, 100],
                    [500, 400],
                    [100, 400]
                ]
            }
        ]
    },
    "save_images": True,
    "save_result": False,
    "playback_duration": 10,
    "stream_urls": [
        "rtsp://admin:zyckj2021@192.168.1.201:554/1/1"
    ],
    "target_classes": []
}


def create_and_start_task():
    """创建并启动任务"""
    
    # 1. 创建任务
    print("=== 创建分析任务 ===")
    try:
        response = requests.post(
            f"{BASE_URL}{API_PREFIX}/tasks/create",
            json=TEST_TASK
        )
        
        if response.status_code == 200:
            task_data = response.json()
            task_id = task_data["data"]["task_id"]
            print(f"✅ 任务创建成功")
            print(f"   任务ID: {task_id}")
            print(f"   任务名称: {task_data['data']['name']}")
        else:
            print(f"❌ 任务创建失败")
            print(f"   状态码: {response.status_code}")
            print(f"   响应: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 创建任务异常: {e}")
        return None
    
    # 2. 启动任务
    print(f"\n=== 启动任务 {task_id} ===")
    try:
        response = requests.post(
            f"{BASE_URL}{API_PREFIX}/tasks/{task_id}/start"
        )
        
        if response.status_code == 200:
            print("✅ 任务启动成功")
            result = response.json()
            print(f"   消息: {result.get('message', '')}")
        else:
            print(f"❌ 任务启动失败")
            print(f"   状态码: {response.status_code}")
            print(f"   响应: {response.text}")
            
    except Exception as e:
        print(f"❌ 启动任务异常: {e}")
    
    return task_id


def query_task_status(task_id):
    """查询任务状态"""
    try:
        response = requests.get(f"{BASE_URL}{API_PREFIX}/tasks/{task_id}")
        if response.status_code == 200:
            data = response.json()["data"]
            print(f"\n任务状态: {data['status']}")
            print(f"创建时间: {data['created_at']}")
            if data.get('start_time'):
                print(f"启动时间: {data['start_time']}")
        else:
            print(f"查询失败: {response.status_code}")
    except Exception as e:
        print(f"查询异常: {e}")


def stop_task(task_id):
    """停止任务"""
    print(f"\n=== 停止任务 {task_id} ===")
    try:
        response = requests.post(f"{BASE_URL}{API_PREFIX}/tasks/{task_id}/stop")
        if response.status_code == 200:
            print("✅ 任务停止成功")
        else:
            print(f"❌ 停止失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 停止异常: {e}")


if __name__ == "__main__":
    # 支持命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "stop" and len(sys.argv) > 2:
            stop_task(sys.argv[2])
        elif sys.argv[1] == "status" and len(sys.argv) > 2:
            query_task_status(sys.argv[2])
        else:
            print("用法:")
            print("  python quick_task.py          # 创建并启动新任务")
            print("  python quick_task.py stop <task_id>    # 停止任务")
            print("  python quick_task.py status <task_id>  # 查询任务状态")
    else:
        # 默认创建并启动任务
        task_id = create_and_start_task()
        if task_id:
            print(f"\n提示: 可以使用以下命令管理任务:")
            print(f"  python quick_task.py status {task_id}")
            print(f"  python quick_task.py stop {task_id}")