import requests
import json

def test_task_api():
    """测试任务启动API"""
    url = "http://localhost:8002/api/v1/tasks/start"
    
    data = {
        "model_code": "yolov8n",
        "stream_url": "test.mp4",
        "task_name": "测试任务"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("正在测试任务启动API...")
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            print("✅ 任务服务已正常工作！")
        elif response.status_code == 500:
            if "任务服务未初始化" in response.text:
                print("❌ 任务服务仍未初始化")
            else:
                print(f"❌ 服务器内部错误: {response.text}")
        else:
            print(f"❌ 未预期的状态码: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保应用程序正在运行")
    except Exception as e:
        print(f"❌ 测试出错: {str(e)}")

if __name__ == "__main__":
    test_task_api() 