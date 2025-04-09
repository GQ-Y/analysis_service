"""
MQTT客户端服务，支持单例模式，用于与API服务通过MQTT协议通信
"""
import os
import sys
import threading
import time
import json
import uuid
import socket
import platform
import psutil
from typing import Dict, Any, List, Callable, Optional, Union
from datetime import datetime
import paho.mqtt.client as mqtt

# 添加父级目录到sys.path以允许导入core.config
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from shared.utils.logger import setup_logger
from core.config import settings

# 设置日志
logger = setup_logger(__name__)

# 全局MQTT客户端实例
_MQTT_CLIENT_INSTANCE = None
_MQTT_CLIENT_LOCK = threading.Lock()

def get_mqtt_client() -> 'MQTTClient':
    """
    获取MQTT客户端实例（单例模式）
    
    Returns:
        MQTTClient: MQTT客户端实例
    """
    global _MQTT_CLIENT_INSTANCE
    
    if _MQTT_CLIENT_INSTANCE is None:
        with _MQTT_CLIENT_LOCK:
            if _MQTT_CLIENT_INSTANCE is None:
                # 获取MAC地址作为客户端ID
                def get_mac_address():
                    try:
                        mac = uuid.getnode()
                        mac_str = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])
                        # 转换为大写形式
                        mac_str = mac_str.upper()
                        return mac_str
                    except Exception as e:
                        logger.error(f"获取MAC地址失败: {str(e)}")
                        # 使用随机生成的MAC地址，并转为大写
                        import random
                        mac = [random.randint(0x00, 0xff) for _ in range(6)]
                        mac_str = ':'.join(['{:02x}'.format(x) for x in mac]).upper()
                        return mac_str

                mac_address = get_mac_address()
                client_id = mac_address  # 直接使用MAC地址，不添加前缀
                
                # 使用socket.gethostname()作为device_id
                device_id = socket.gethostname()
                broker_host = settings.MQTT.broker_host
                broker_port = settings.MQTT.broker_port
                username = settings.MQTT.username
                password = settings.MQTT.password
                topic_prefix = settings.MQTT.topic_prefix
                
                logger.info(f"创建MQTT客户端实例: {client_id}")
                
                _MQTT_CLIENT_INSTANCE = MQTTClient(
                    device_id=device_id,  # 提供device_id参数
                    client_id=client_id,
                    broker_host=broker_host,
                    broker_port=broker_port,
                    username=username,
                    password=password,
                    topic_prefix=topic_prefix,  # 作为关键字参数传递
                )
    
    return _MQTT_CLIENT_INSTANCE

class MQTTClient:
    """MQTT客户端类，处理与MQTT服务器的通信"""

    def __init__(self, device_id, broker_host, broker_port=1883, username=None, password=None,
                 command_topic=None, response_topic=None, status_topic=None, client_id=None,
                 **kwargs):
        """
        初始化MQTT客户端
        
        Args:
            device_id: 设备ID
            broker_host: MQTT代理服务器地址
            broker_port: MQTT代理服务器端口
            username: MQTT用户名
            password: MQTT密码
            command_topic: 命令主题
            response_topic: 响应主题
            status_topic: 状态主题
            client_id: 客户端ID，如果不提供则使用设备ID
        """
        self.device_id = device_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        
        # 设置客户端ID，使用MAC地址作为基础
        self.mac_address = self._get_mac_address()
        logger.info(f"初始化MQTT客户端: mac_address={self.mac_address}")
        
        # 如果传入了client_id则使用，否则使用MAC地址
        if client_id:
            self.client_id = client_id
        else:
            self.client_id = self.mac_address
        
        # 使用MAC地址作为节点ID
        self.node_id = self.mac_address
        
        # 从kwargs获取topic_prefix或使用默认值
        self.topic_prefix = kwargs.get("topic_prefix", "meek/")
        
        # 创建MQTT客户端
        self.client = mqtt.Client(client_id=self.client_id, protocol=mqtt.MQTTv311, clean_session=True)
        
        # 设置回调函数
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # 设置用户名和密码
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)
            
        # 初始化任务处理器字典
        self.task_handlers = {}
        
        # 连接标志
        self.connected = False

        
        # 状态标志
        self.is_connected = False
        self.is_running = False
        self.reconnect_delay = 1  # 重连延迟（秒）
        self.max_reconnect_delay = 60  # 最大重连延迟（秒）
        
        # 活跃任务
        self.active_tasks = {}
        self.active_tasks_lock = threading.Lock()
        
        # 停止事件
        self.stop_event = threading.Event()
        
    def _get_mac_address(self) -> str:
        """
        获取MAC地址
        
        Returns:
            str: MAC地址
        """
        try:
            mac = uuid.getnode()
            mac_str = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])
            # 转换为大写形式
            mac_str = mac_str.upper()
            return mac_str
        except Exception as e:
            logger.error(f"获取MAC地址失败: {str(e)}")
            # 使用随机生成的MAC地址，并转为大写
            import random
            mac = [random.randint(0x00, 0xff) for _ in range(6)]
            mac_str = ':'.join(['{:02x}'.format(x) for x in mac]).upper()
            return mac_str
            
    def start(self) -> bool:
        """
        启动MQTT客户端
        
        Returns:
            bool: 是否成功启动
        """
        if self.is_running:
            logger.info("MQTT客户端已经在运行")
            return True
            
        logger.info(f"启动MQTT客户端: {self.client_id}")
        
        # 连接MQTT代理服务器
        if not self.connect():
            logger.error("MQTT客户端连接失败，无法启动")
            return False
            
        # 订阅主题
        self._subscribe_topics()
        
        # 启动心跳线程
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        
        # 标记为运行中
        self.is_running = True
        self.stop_event.clear()
        
        logger.info("MQTT客户端已启动")
        return True
        
    def stop(self) -> bool:
        """
        停止MQTT客户端
        
        Returns:
            bool: 是否成功停止
        """
        if not self.is_running:
            logger.info("MQTT客户端未运行")
            return True
            
        logger.info("停止MQTT客户端")
        
        # 发送离线状态
        self._publish_connection_status(False)
        
        # 设置停止事件
        self.stop_event.set()
        
        # 断开连接
        try:
            self.client.disconnect()
            logger.info("MQTT客户端已断开连接")
        except Exception as e:
            logger.error(f"断开MQTT连接失败: {str(e)}")
            
        # 标记为已停止
        self.is_running = False
        self.is_connected = False
        
        logger.info("MQTT客户端已停止")
        return True
        
    def connect(self) -> bool:
        """
        连接到MQTT代理服务器。如果重试次数用尽依然无法连接，则返回False。
        
        Returns:
            bool: 是否成功连接
        """
        # 检查是否已连接
        if self.is_connected:
            logger.info("MQTT客户端已连接")
            return True
            
        logger.info(f"连接到MQTT代理服务器: {self.broker_host}:{self.broker_port}")
        
        # 设置连接回调
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # 设置遗嘱消息
        if self.topic_prefix and self.client_id:
            will_topic = f"{self.topic_prefix}connection"
            will_payload = json.dumps({
                "message_type": "connection",
                "status": "offline",
                "mac_address": self.mac_address,
                "node_id": self.node_id,
                "mqtt_node_id": self.node_id,
                "node_type": "analysis",  # 分析节点类型
                "client_id": self.client_id,  # 添加客户端ID
                "service_type": "analysis",  # 添加服务类型
                "is_active": False,  # 离线状态
                "max_tasks": 100,  # 最大任务数
                "timestamp": int(time.time()),
                "metadata": {
                    "version": settings.VERSION,
                    "hostname": platform.node(),
                    "is_active": False  # 离线状态
                }
            })
            logger.info(f"设置遗嘱消息: {will_topic}")
            self.client.will_set(will_topic, will_payload, qos=1, retain=True)
        
        # 尝试连接
        max_retries = 3
        retry_delay = 1
        
        for retry in range(max_retries):
            try:
                logger.info(f"尝试连接MQTT代理服务器 (重试 {retry+1}/{max_retries})...")
                self.client.connect(self.broker_host, self.broker_port, keepalive=60)
                
                # 启动网络循环
                self.client.loop_start()
                
                # 等待连接确认
                for i in range(10):  # 等待最多5秒
                    if self.is_connected:
                        logger.info(f"MQTT客户端连接成功: {self.broker_host}:{self.broker_port}")
                        
                        # 发布一次上线状态
                        self._publish_connection_status(True)
                        
                        return True
                    time.sleep(0.5)
                    if i % 2 == 0:
                        logger.debug(f"等待MQTT连接确认... ({i/2+0.5}秒)")
                    
                # 连接超时
                logger.warning(f"MQTT连接超时，已等待5秒")
                self.client.loop_stop()
                
            except Exception as e:
                logger.error(f"连接MQTT代理服务器失败: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
            # 等待重试
            if retry < max_retries - 1:
                logger.info(f"等待 {retry_delay} 秒后重试MQTT连接...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
                
        logger.error(f"连接MQTT代理服务器失败，已尝试 {max_retries} 次")
        return False
        
    def _on_connect(self, client, userdata, flags, rc):
        """
        MQTT连接回调
        
        Args:
            client: MQTT客户端
            userdata: 用户数据
            flags: 连接标志
            rc: 返回码
        """
        if rc == 0:
            self.is_connected = True
            logger.info("已连接到MQTT代理服务器")
            
            # 订阅主题
            self._subscribe_topics()
            
            # 发送上线状态
            self._publish_connection_status(True)
            
        else:
            self.is_connected = False
            logger.error(f"连接MQTT代理服务器失败，返回码: {rc}")
            
    def _on_disconnect(self, client, userdata, rc):
        """
        MQTT断开连接回调
        
        Args:
            client: MQTT客户端
            userdata: 用户数据
            rc: 返回码
        """
        self.is_connected = False
        
        if rc == 0:
            logger.info("已与MQTT代理服务器断开连接")
        else:
            logger.warning(f"与MQTT代理服务器断开连接，返回码: {rc}")
            
            # 如果客户端仍在运行，则尝试重新连接
            if self.is_running and not self.stop_event.is_set():
                reconnect_thread = threading.Thread(target=self._reconnect_loop)
                reconnect_thread.daemon = True
                reconnect_thread.start()
                
    def _reconnect_loop(self):
        """MQTT重连循环"""
        delay = self.reconnect_delay
        
        while self.is_running and not self.stop_event.is_set() and not self.is_connected:
            logger.info(f"尝试重新连接MQTT代理服务器，等待 {delay} 秒...")
            time.sleep(delay)
            
            try:
                # 尝试重新连接
                if self.connect():
                    logger.info("重新连接MQTT代理服务器成功")
                    return
                    
                # 增加重连延迟，最大不超过max_reconnect_delay
                delay = min(delay * 2, self.max_reconnect_delay)
                
            except Exception as e:
                logger.error(f"重新连接MQTT代理服务器失败: {str(e)}")
                
    def _subscribe_topics(self):
        """订阅MQTT主题"""
        if not self.is_connected:
            logger.warning("MQTT客户端未连接，无法订阅主题")
            return
            
        # 订阅节点配置和任务分配主题 - 使用自己的MAC地址
        request_setting_topic = f"{self.topic_prefix}{self.node_id}/request_setting"
        result = self.client.subscribe(request_setting_topic, qos=2)
        logger.info(f"已订阅节点配置主题: {request_setting_topic}, 结果: {result}")
        
        # 订阅系统广播主题
        broadcast_topic = f"{self.topic_prefix}system/broadcast"
        result = self.client.subscribe(broadcast_topic, qos=1)
        logger.info(f"已订阅系统广播主题: {broadcast_topic}, 结果: {result}")
        
        # 打印出所有订阅的主题
        logger.info(f"节点ID: {self.node_id}")
        logger.info(f"MAC地址: {self.mac_address}")
        logger.info(f"主题前缀: {self.topic_prefix}")
        logger.info("已订阅的主题列表:")
        logger.info(f" - {request_setting_topic}")
        logger.info(f" - {broadcast_topic}")
        
    def _on_message(self, client, userdata, msg):
        """
        MQTT消息回调
        
        Args:
            client: MQTT客户端
            userdata: 用户数据
            msg: 消息
        """
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            logger.info(f"收到MQTT消息: {topic}")
            logger.info(f"消息内容: {payload}")
            
            # 解析消息
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                logger.error(f"解析MQTT消息失败，非JSON格式: {payload[:100]}...")
                return
            
            # 检查是否为request_setting主题
            if "/request_setting" in topic:
                logger.info("=" * 80)
                logger.info(f"接收到request_setting主题消息: {topic}")
                logger.info(f"完整原始消息内容: {payload}")
                logger.info("JSON格式化消息内容:")
                logger.info(json.dumps(data, ensure_ascii=False, indent=2))
                logger.info("=" * 80)
                
            # 处理不同主题的消息
            # 1. 处理自己的请求主题
            if topic.startswith(f"{self.topic_prefix}{self.node_id}/request_setting"):
                logger.info(f"接收到自己节点的配置请求: {topic}")
                self._handle_request_setting(data)
            
            # 2. 处理任何节点的request_setting主题 - 我们都尝试处理
            elif "/request_setting" in topic:
                logger.info(f"接收到其他节点的配置请求: {topic}")
                # 尝试从主题中提取MAC地址
                parts = topic.split('/')
                if len(parts) > 1:
                    target_node = parts[-2]  # 倒数第二个部分应该是MAC地址
                    logger.info(f"目标节点: {target_node}")
                    
                    # 临时将自己的node_id设为目标节点
                    original_node_id = self.node_id
                    self.node_id = target_node
                    try:
                        self._handle_request_setting(data)
                    finally:
                        # 恢复原始node_id
                        self.node_id = original_node_id
                else:
                    logger.warning(f"无法从主题中提取节点ID: {topic}")
                    # 尝试处理消息
                    self._handle_request_setting(data)
                
            # 3. 处理系统广播消息
            elif topic == f"{self.topic_prefix}system/broadcast":
                # 处理系统广播消息
                self._handle_broadcast(data)
            
            # 4. 其他未处理的主题
            else:
                logger.info(f"未处理的消息主题: {topic}")
                
        except Exception as e:
            logger.error(f"处理MQTT消息失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
    def _handle_request_setting(self, data):
        """
        处理节点配置和任务分配消息
        
        Args:
            data: 消息数据
        """
        try:
            # 提取消息字段
            request_type = data.get("request_type")
            message_id = data.get("message_id")
            message_uuid = data.get("message_uuid")
            
            # 提取确认主题，这是API服务指定的回复目的地
            confirmation_topic = data.get("confirmation_topic")
            
            if not request_type or not message_id or not message_uuid:
                logger.error("请求消息缺少必要字段")
                return
                
            logger.info(f"处理请求: {request_type}, message_id: {message_id}, confirmation_topic: {confirmation_topic}")
            
            # 打印完整的消息数据，用于调试
            logger.info("收到的完整消息数据:")
            logger.info(json.dumps(data, ensure_ascii=False, indent=2))
            
            # 处理不同类型的请求
            if request_type == "node_cmd":
                # 处理节点命令
                self._handle_node_cmd(data, message_id, message_uuid, confirmation_topic)
                
            elif request_type == "task_cmd":
                # 处理任务命令
                self._handle_task_cmd(data, message_id, message_uuid, confirmation_topic)
                
            else:
                logger.warning(f"未知的请求类型: {request_type}")
                
        except Exception as e:
            logger.error(f"处理节点配置和任务分配消息失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _handle_node_cmd(self, data, message_id, message_uuid, confirmation_topic):
        """
        处理节点命令
        
        Args:
            data: 消息数据
            message_id: 消息ID
            message_uuid: 消息UUID
            confirmation_topic: 确认主题
        """
        cmd_data = data.get("data", {})
        cmd_type = cmd_data.get("cmd_type")
        
        if not cmd_type:
            logger.error("节点命令缺少cmd_type字段")
            self._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", "缺少cmd_type字段")
            return
            
        logger.info(f"处理节点命令: {cmd_type}")
        
        # 处理不同类型的命令
        if cmd_type == "sync_time":
            # 时间同步命令
            self._handle_sync_time(cmd_data, message_id, message_uuid, confirmation_topic)
            
        elif cmd_type == "update_config":
            # 更新配置命令
            self._handle_update_config(cmd_data, message_id, message_uuid, confirmation_topic)
            
        else:
            logger.warning(f"未知的节点命令类型: {cmd_type}")
            self._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", f"未知的命令类型: {cmd_type}")
    
    def _handle_task_cmd(self, data, message_id, message_uuid, confirmation_topic):
        """
        处理任务命令
        
        Args:
            data: 消息数据
            message_id: 消息ID
            message_uuid: 消息UUID
            confirmation_topic: 确认主题
        """
        cmd_data = data.get("data", {})
        cmd_type = cmd_data.get("cmd_type")
        
        if not cmd_type:
            logger.error("任务命令缺少cmd_type字段")
            self._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", "缺少cmd_type字段")
            return
            
        logger.info(f"处理任务命令: {cmd_type}")
        
        # 处理不同类型的命令
        if cmd_type == "start_task":
            # 启动任务命令
            self._handle_start_task_cmd(data, message_id, message_uuid, confirmation_topic)
            
        elif cmd_type == "stop_task":
            # 停止任务命令
            self._handle_stop_task_cmd(data, message_id, message_uuid, confirmation_topic)
            
        else:
            logger.warning(f"未知的任务命令类型: {cmd_type}")
            self._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", f"未知的命令类型: {cmd_type}")
    
    def _send_cmd_reply(self, message_id, message_uuid, confirmation_topic, status, message=None, data=None):
        """
        发送命令回复
        
        Args:
            message_id: 消息ID
            message_uuid: 消息UUID
            confirmation_topic: 确认主题，由API服务在请求中指定
            status: 状态（success/error）
            message: 消息内容
            data: 回复数据
        """
        # 如果未提供确认主题，则使用默认主题，但正常情况下API服务应该提供
        if not confirmation_topic:
            logger.warning("未提供确认主题，使用默认主题")
            confirmation_topic = f"{self.topic_prefix}device_config_reply"
            
        logger.info(f"使用确认主题回复: {confirmation_topic}")
            
        reply = {
            "message_id": message_id,
            "message_uuid": message_uuid,
            "response_type": "cmd_reply",
            "status": status,
            "mac_address": self.mac_address,
            "node_id": self.node_id,
            "mqtt_node_id": self.node_id,
            "client_id": self.client_id,
            "service_type": "analysis",
            "is_active": True,
            "timestamp": int(time.time()),
            "data": data or {}
        }
        
        if message and status == "error":
            reply["data"]["message"] = message
            
        self._publish_message(confirmation_topic, reply, qos=1)
        logger.info(f"已发送命令回复到 {confirmation_topic}: {status}, message_id: {message_id}")
        
    def _handle_broadcast(self, data):
        """
        处理系统广播消息
        
        Args:
            data: 消息数据
        """
        logger.info(f"收到系统广播消息: {data}")
        
    def _handle_connection(self, data):
        """
        处理连接状态消息
        
        注意：此方法已废弃。我们不再订阅meek/connection主题，
        因此此方法不会被调用。保留此方法仅为保持代码结构完整性。
        
        Args:
            data: 消息数据
        """
        pass
    
    def _handle_sync_time(self, data, message_id, message_uuid, confirmation_topic):
        """
        处理时间同步命令
        
        Args:
            data: 命令数据
            message_id: 消息ID
            message_uuid: 消息UUID
            confirmation_topic: 确认主题
        """
        logger.info(f"收到时间同步命令: {data}")
        
    def _handle_update_config(self, data, message_id, message_uuid, confirmation_topic):
        """
        处理更新配置命令
        
        Args:
            data: 命令数据
            message_id: 消息ID
            message_uuid: 消息UUID
            confirmation_topic: 确认主题
        """
        logger.info(f"收到更新配置命令: {data}")
        
    def _handle_start_task_cmd(self, data, message_id, message_uuid, confirmation_topic):
        """
        处理开始任务命令
        
        Args:
            data: 命令数据
            message_id: 消息ID
            message_uuid: 消息UUID
            confirmation_topic: 确认主题
        """
        logger.info(f"处理开始任务命令: {data}")
        
        # 提取任务信息 - 修正：从data["data"]中获取数据，而不是直接从data中获取
        cmd_data = data.get("data", {})
        task_id = cmd_data.get("task_id")
        subtask_id = cmd_data.get("subtask_id")
        source = cmd_data.get("source", {})
        source_type = source.get("type")
        config = cmd_data.get("config", {})
        result_config = cmd_data.get("result_config", {})
        
        # 记录详细任务信息
        logger.info(f"任务ID: {task_id}")
        logger.info(f"子任务ID: {subtask_id}")
        logger.info(f"源类型: {source_type}")
        logger.info(f"源配置: {source}")
        logger.info(f"任务配置: {config}")
        logger.info(f"结果配置: {result_config}")
        
        # 记录模型代码和分析类型
        model_code = config.get("model_code", "")
        analysis_type = config.get("analysis_type", "")
        logger.info(f"模型代码: {model_code}")
        logger.info(f"分析类型: {analysis_type}")
        
        # 检查必要的信息是否存在
        if not task_id:
            error_msg = "缺少任务ID"
            logger.error(error_msg)
            if confirmation_topic:
                self._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", 
                                     data={"error_message": error_msg})
            return False
            
        if not subtask_id:
            error_msg = "缺少子任务ID"
            logger.error(error_msg)
            if confirmation_topic:
                self._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", 
                                     data={"error_message": error_msg})
            return False
            
        if not source_type:
            error_msg = "缺少源类型"
            logger.error(error_msg)
            if confirmation_topic:
                self._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", 
                                     data={"error_message": error_msg})
            return False
            
        # 处理不同类型的源
        url = ""
        if source_type in ["image", "video", "stream"]:
            # 获取URL
            url = source.get("url", "")
            if not url and "urls" in source and source["urls"]:
                url = source["urls"][0]
            
        # 记录URL信息
        logger.info(f"解析后的URL: {url}")
        
        # 检查任务类型是否支持
        logger.info(f"当前已注册的任务处理器: {list(self.task_handlers.keys())}")
        
        if source_type in self.task_handlers:
            # 使用源类型作为任务类型
            handler = self.task_handlers[source_type]
            logger.info(f"使用{source_type}处理器处理任务")
        elif analysis_type and analysis_type in self.task_handlers:
            # 如果源类型没有对应的处理器，尝试使用分析类型
            handler = self.task_handlers[analysis_type]
            logger.info(f"使用{analysis_type}处理器处理任务")
        else:
            # 尝试使用默认处理器
            if "detection" in self.task_handlers and source_type == "stream":
                handler = self.task_handlers["detection"]
                logger.info("使用默认detection处理器处理流任务")
            else:
                # 无法找到合适的处理器
                available_handlers = list(self.task_handlers.keys())
                error_msg = f"不支持的任务类型: {source_type}, 分析类型: {analysis_type}. 已注册的任务类型: {available_handlers}"
                logger.error(error_msg)
                if confirmation_topic:
                    self._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", 
                                      data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
                return False
        
        # 执行任务
        try:
            success = handler(task_id, subtask_id, source, config, result_config, 
                             message_id, message_uuid, confirmation_topic)
            return success
        except Exception as e:
            error_msg = f"执行任务时出错: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            if confirmation_topic:
                self._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", 
                                   data={"error_message": error_msg, "task_id": task_id, "subtask_id": subtask_id})
            return False
    
    def _handle_stop_task_cmd(self, data, message_id, message_uuid, confirmation_topic):
        """
        处理停止任务命令
        
        Args:
            data: 命令数据
            message_id: 消息ID
            message_uuid: 消息UUID
            confirmation_topic: 确认主题
        """
        # 提取任务信息 - 修正：从data["data"]中获取数据，而不是直接从data中获取
        cmd_data = data.get("data", {})
        task_id = cmd_data.get("task_id")
        subtask_id = cmd_data.get("subtask_id")
        
        # 验证必要字段
        if not task_id or not subtask_id:
            logger.error("无效的停止任务命令，缺少必要字段")
            error_data = {
                "cmd_type": "stop_task",
                "error_code": "ERR_001",
                "error_type": "INVALID_PARAMS",
                "message": "缺少必要字段: task_id或subtask_id"
            }
            self._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", data=error_data)
            return
            
        logger.info(f"收到停止任务命令: {task_id}/{subtask_id}")
        
        # 检查任务是否存在 - 使用subtask_id作为任务键
        with self.active_tasks_lock:
            if subtask_id not in self.active_tasks:
                logger.warning(f"任务不存在: {task_id}/{subtask_id}")
                error_data = {
                    "cmd_type": "stop_task",
                    "task_id": task_id,
                    "subtask_id": subtask_id,
                    "error_code": "ERR_004",
                    "error_type": "TASK_NOT_FOUND",
                    "message": "任务不存在"
                }
                self._send_cmd_reply(message_id, message_uuid, confirmation_topic, "error", data=error_data)
                return
                
            # 移除任务 - 使用subtask_id作为任务键
            del self.active_tasks[subtask_id]
            
        # 发送任务状态：已停止
        self._send_task_status(task_id, subtask_id, "stopped")
        
        # 无需从TaskManager中删除任务，因为我们采用无状态设计，任务从未被注册到TaskManager中
        
        # 发送接受停止命令的响应
        success_data = {
            "cmd_type": "stop_task",
            "task_id": task_id,
            "subtask_id": subtask_id,
            "message": "任务已成功停止",
            "timestamp": int(time.time())
        }
        self._send_cmd_reply(message_id, message_uuid, confirmation_topic, "success", data=success_data)
        logger.info(f"已停止任务: {task_id}/{subtask_id}")
        
    def _run_task(self, task_id, subtask_id, task_type, source, config, should_stop):
        """
        运行任务
        
        Args:
            task_id: 任务ID
            subtask_id: 子任务ID
            task_type: 任务类型
            source: 数据源
            config: 配置
            should_stop: 停止检查函数
        """
        try:
            logger.info(f"开始执行任务: {task_id}/{subtask_id}, 类型: {task_type}")
            
            # 记录详细的任务信息用于调试
            logger.debug(f"任务详情 - task_id: {task_id}, subtask_id: {subtask_id}")
            logger.debug(f"任务类型: {task_type}")
            logger.debug(f"数据源: {json.dumps(source, ensure_ascii=False)}")
            logger.debug(f"配置: {json.dumps(config, ensure_ascii=False)}")
            
            # 获取任务处理器
            handler = self.task_handlers.get(task_type)
            if not handler:
                error_msg = f"找不到任务处理器: {task_type}"
                logger.error(error_msg)
                self._send_task_status(task_id, subtask_id, "error", error=error_msg)
                self._send_task_result(task_id, subtask_id, "error", {"error": error_msg}, error=error_msg)
                return
                
            # 执行任务
            start_time = time.time()
            logger.info(f"开始执行任务处理器: {task_type}, task_id: {task_id}, subtask_id: {subtask_id}")
            result = handler(task_id, subtask_id, source, config, should_stop)
            end_time = time.time()
            process_time = end_time - start_time
            
            # 检查任务是否被中止
            if should_stop():
                logger.info(f"任务已被中止: {task_id}/{subtask_id}")
                return
                
            # 发送任务结果
            if result and "error" in result:
                # 任务执行出错
                error_msg = result.get("error", "未知错误")
                logger.error(f"任务执行失败: {task_id}/{subtask_id}, 错误: {error_msg}")
                
                # 添加处理时间信息
                result["processing_time"] = process_time
                self._send_task_result(task_id, subtask_id, "error", result, error=error_msg)
            else:
                # 任务执行成功
                logger.info(f"任务执行成功: {task_id}/{subtask_id}, 处理时间: {process_time:.2f}秒")
                
                # 添加处理时间信息
                if isinstance(result, dict):
                    result["processing_time"] = process_time
                else:
                    # 如果result不是字典，创建一个包含基本信息的结果
                    result = {
                        "status": "success",
                        "source_type": task_type,
                        "model_code": config.get("model_code", ""),
                        "timestamp": int(time.time()),
                        "processing_time": process_time,
                        "result_data": result
                    }
                
                self._send_task_result(task_id, subtask_id, "completed", result)
                
                # 从活跃任务中移除
                with self.active_tasks_lock:
                    if subtask_id in self.active_tasks:
                        del self.active_tasks[subtask_id]
                    
        except Exception as e:
            logger.error(f"执行任务失败: {task_id}/{subtask_id}, 错误: {str(e)}")
            import traceback
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            
            # 构建详细的错误信息
            error_result = {
                "error": str(e),
                "timestamp": int(time.time()),
                "task_id": task_id,
                "subtask_id": subtask_id,
                "task_type": task_type,
                "source_info": source.get("type", "unknown"),
                "error_detail": error_trace.splitlines()[-3:] if error_trace else ""
            }
            
            # 发送任务状态：出错
            self._send_task_status(task_id, subtask_id, "error", error=str(e))
            self._send_task_result(task_id, subtask_id, "error", error_result, error=str(e))
            
            # 从活跃任务中移除
            with self.active_tasks_lock:
                if subtask_id in self.active_tasks:
                    del self.active_tasks[subtask_id]
                    
    def _send_task_status(self, task_id, subtask_id, status, error=None):
        """
        发送任务状态 (实际发送到节点状态主题)
        
        Args:
            task_id: 任务ID
            subtask_id: 子任务ID
            status: 状态（processing, completed, error, stopped）
            error: 错误信息（可选）
        """
        payload = {
            "message_id": int(time.time()),
            "message_uuid": str(uuid.uuid4()),
            "task_id": task_id,
            "subtask_id": subtask_id,
            "status": status,
            "progress": 100 if status in ["completed", "error", "stopped"] else 50,  # 进度百分比
            "timestamp": int(time.time()),
            "node_id": self.node_id,
            "mac_address": self.mac_address,
            "mqtt_node_id": self.node_id,
            "client_id": self.client_id,
            "service_type": "analysis",
            "is_active": True
        }
        
        # 添加错误信息
        if error and status == "error":
            payload["error"] = str(error)
            
        # 发布状态消息到标准 /status 主题
        topic = f"{self.topic_prefix}{self.node_id}/status"
        self._publish_message(topic, payload, qos=1)
        logger.info(f"已发送任务状态到 {topic}: {task_id}/{subtask_id}, 状态: {status}")
        
    def _send_task_result(self, task_id, subtask_id, status, result, error=None):
        """
        发送任务结果到标准 /result 主题
        
        Args:
            task_id: 任务ID
            subtask_id: 子任务ID
            status: 状态（completed, error）
            result: 结果数据
            error: 错误信息（可选）
        """
        # 确保结果是字典类型
        if not isinstance(result, dict):
            result = {"data": result}
            
        # 查找任务配置中的回调主题 - 使用subtask_id作为任务键
        result_callback_topic = None
        stream_url = ""
        model_code = ""
        config = {}
        task_name = ""
        base64_image = ""
        
        # 获取任务详情
        with self.active_tasks_lock:
            if subtask_id in self.active_tasks:
                task_data = self.active_tasks.get(subtask_id, {})
                result_config = task_data.get("result_config", {})
                result_callback_topic = result_config.get("callback_topic")
                
                # 获取任务的额外信息用于构建完整回调
                source = task_data.get("source", {})
                if source:
                    stream_url = source.get("url", "")
                    if not stream_url and "urls" in source and source["urls"]:
                        stream_url = source["urls"][0]
                
                config = task_data.get("config", {})
                model_code = config.get("model_code", "")
                task_name = task_data.get("task_name", f"{task_id}/{subtask_id}")
                
        # 提取图像尺寸和检测结果
        frame_width = 1920
        frame_height = 1080
        detections = []
        
        # 尝试从result中提取关键信息
        if "detections" in result:
            detections = result.get("detections", [])
        elif "result_data" in result and "detections" in result["result_data"]:
            detections = result["result_data"]["detections"]
            
        # 提取图像尺寸
        if "frame_size" in result:
            frame_width = result.get("frame_size", {}).get("width", 1920)
            frame_height = result.get("frame_size", {}).get("height", 1080)
        
        # 提取base64图像，如果有的话
        if "src_pic_data" in result:
            base64_image = result.get("src_pic_data", "")
        elif "frame_base64" in result:
            base64_image = result.get("frame_base64", "")
                
        # 强制使用标准 /result 主题，忽略配置中的 callback_topic
        topic = f"{self.topic_prefix}{self.node_id}/result"
        
        # 尝试将task_id转换为整数，用于标准HTTP回调格式
        try:
            numeric_task_id = int(task_id.split('_')[-1], 16) if '_' in task_id and task_id.split('_')[-1].isalnum() else int(task_id) if task_id.isdigit() else 0
        except (ValueError, TypeError):
            numeric_task_id = 0
            
        # 构建与HTTP回调格式一致的数据结构
        http_callback_data = {
            "cameraDeviceType": 1,
            "cameraDeviceStreamUrl": stream_url,
            "cameraDeviceStatus": 1,
            "cameraDeviceGroup": "",
            "cameraDeviceGps": "",
            "cameraDeviceId": 0,
            "cameraDeviceName": task_name,
            "algorithmId": 0,
            "algorithmName": model_code,
            "algorithmNameEn": "",
            "dataID": task_id,
            "parameter": config,
            "picture": "",
            "srcUrl": stream_url,
            "alarmUrl": "",
            "taskId": numeric_task_id,
            "cameraId": 0,
            "cameraUrl": stream_url,
            "cameraName": task_name,
            "timestamp": int(time.time()),
            "imageWidth": frame_width,
            "imageHeight": frame_height,
            "srcPicData": base64_image,
            "srcPicName": "",
            "alarmPicName": "",
            "src": "",
            "alarm": "",
            "alarmPicData": base64_image,
            "other": "",
            "result": "",
            "extraInfo": detections,
            "resultData": {
                "detections": detections,
                "task_id": task_id,
                "subtask_id": subtask_id,
                "timestamp": int(time.time()),
                "frame_size": {
                    "width": frame_width,
                    "height": frame_height
                }
            },
            "degree": 3
        }
        
        # 构建MQTT结果消息 - 保留MQTT特有的元数据，但result字段使用HTTP格式
        payload = {
            "message_id": int(time.time()),
            "message_uuid": str(uuid.uuid4()),
            "task_id": task_id,
            "subtask_id": subtask_id,
            "status": status,
            "progress": 100,  # 结果发送时进度应为100%
            "timestamp": int(time.time()),
            "node_id": self.node_id,
            "mac_address": self.mac_address,
            "mqtt_node_id": self.node_id,
            "client_id": self.client_id,
            "service_type": "analysis",
            "is_active": True,
            "result": http_callback_data  # 使用HTTP回调格式的数据
        }
        
        # 添加错误信息
        if error and status == "error":
            payload["error"] = str(error)
            http_callback_data["error"] = str(error)
            
        # 记录发送的结果消息
        logger.info(f"发送任务结果到主题 {topic}: task_id={task_id}, subtask_id={subtask_id}, 状态={status}")
        if status == "error":
            logger.error(f"任务错误信息: {error}")
            
        # 发布结果消息
        self._publish_message(topic, payload, qos=1)
        logger.info(f"已发送任务结果到 {topic}: {task_id}/{subtask_id}, 状态: {status}")
        
    def _publish_connection_status(self, is_online):
        """
        发布连接状态
        
        Args:
            is_online: 是否在线
        """
        if not self.is_connected and is_online:
            logger.warning("MQTT客户端未连接，无法发布上线状态")
            return
            
        try:
            # 获取系统信息
            import platform
            import psutil
            import os
            
            # 获取CPU使用率
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # 获取内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 获取磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # 获取GPU信息
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                gpu_info = []
                
                if gpus:
                    for i, gpu in enumerate(gpus):
                        gpu_info.append({
                            "id": i,
                            "name": gpu.name,
                            "load": f"{gpu.load * 100:.1f}%",
                            "memory_used": gpu.memoryUsed,
                            "memory_total": gpu.memoryTotal,
                            "temperature": f"{gpu.temperature}°C"
                        })
                else:
                    gpu_info = [{"info": "无可用GPU"}]
                    
            except Exception as e:
                logger.warning(f"获取GPU信息失败: {str(e)}")
                gpu_info = [{"error": str(e)}]
            
            # 获取本地已缓存的模型列表
            models = self._get_available_models()
            logger.info(f"本地可用模型列表: {models}")
            
            # 构建连接状态消息
            payload = {
                "message_type": "connection",
                "status": "online" if is_online else "offline",
                "mac_address": self.mac_address,
                "node_id": self.node_id,
                "mqtt_node_id": self.node_id,
                "node_type": "analysis",  # 分析节点类型
                "client_id": self.client_id,  # 添加客户端ID
                "service_type": "analysis",  # 添加服务类型
                "timestamp": int(time.time()),
                "is_active": True,  # 添加到顶层
                "max_tasks": 100,  # 添加到顶层
                "metadata": {
                    "version": settings.VERSION,
                    "ip": self._get_local_ip(),
                    "port": settings.SERVICES.port,
                    "hostname": platform.node(),
                    "is_active": True,
                    "capabilities": {
                        "models": models,
                        "gpu_available": len(gpu_info) > 0 and "error" not in gpu_info[0],
                        "max_tasks": 100,
                        "cpu_cores": psutil.cpu_count(),
                        "memory": round(psutil.virtual_memory().total / (1024**3))
                    },
                    "resources": {
                        "cpu": cpu_percent,
                        "memory": memory_percent,
                        "memory_used": round(memory.used / (1024**3), 2),
                        "memory_total": round(memory.total / (1024**3), 2),
                        "disk": disk_percent,
                        "disk_free": round(disk.free / (1024**3), 2),
                        "disk_total": round(disk.total / (1024**3), 2),
                        "gpu": gpu_info
                    },
                    "active_tasks": len(self.active_tasks)
                }
            }
            
            # 发布连接状态消息
            topic = f"{self.topic_prefix}connection"
            self._publish_message(topic, payload, qos=1, retain=True)
            logger.debug(f"已发布{'上线' if is_online else '离线'}状态")
            
        except Exception as e:
            logger.error(f"发布连接状态失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _get_available_models(self):
        """获取本地可用的模型列表"""
        try:
            # 使用base_dir和model_dir组合成完整路径
            model_dir = settings.STORAGE.model_dir
            base_dir = settings.STORAGE.base_dir
            complete_model_dir = os.path.join(base_dir, model_dir)
            
            
            # 检查模型目录是否存在
            if not os.path.exists(complete_model_dir):
                logger.warning(f"模型目录不存在: {complete_model_dir}")
                return []  # 返回空列表
                
            # 打印目录中的文件列表
            try:
                dir_content = os.listdir(complete_model_dir)
                logger.info(f"模型目录内容: {dir_content}")
            except Exception as e:
                logger.warning(f"无法列出目录内容: {str(e)}")
            
            # 获取模型目录中的所有子目录作为模型
            models = []
            
            for item in os.listdir(complete_model_dir):
                item_path = os.path.join(complete_model_dir, item)
                # 检查是否是目录
                is_dir = os.path.isdir(item_path)
                logger.info(f"检查模型: {item}, 是目录: {is_dir}")
                
                if is_dir:
                    # 将目录名作为模型代码
                    model_code = item
                    models.append(model_code)
                    logger.info(f"找到模型: {model_code}")
            
            # 如果没有找到模型，返回空列表
            if not models:
                logger.warning(f"模型目录中未找到有效模型目录: {complete_model_dir}")
                return []
                
            logger.info(f"找到的所有模型: {models}")
            return models
            
        except Exception as e:
            logger.error(f"获取可用模型列表失败: {str(e)}")
            return []  # 错误时返回空列表
    
    def _get_local_ip(self):
        """获取本地IP地址"""
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            logger.warning(f"获取本地IP失败: {str(e)}")
            return "127.0.0.1"

    def _heartbeat_loop(self):
        """心跳线程，定期发送连接状态"""
        logger.info("启动MQTT心跳线程")
        last_resource_check = 0
        heartbeat_interval = 30  # 心跳间隔（秒）
        resource_check_interval = 60  # 资源检查间隔（秒）
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # 检查连接状态
                if not self.is_connected:
                    logger.warning("MQTT客户端未连接，尝试重新连接...")
                    if self.connect():
                        logger.info("MQTT客户端重新连接成功")
                    else:
                        logger.error("MQTT客户端重新连接失败")
                        
                # 发送心跳
                if self.is_connected:
                    logger.debug("发送MQTT心跳消息")
                    self._publish_heartbeat()
                    
                    # 每隔resource_check_interval秒发送一次资源状态
                    if current_time - last_resource_check >= resource_check_interval:
                        logger.debug("发送资源状态...")
                        self._publish_connection_status(True)
                        last_resource_check = current_time
                
                # 等待下一次心跳
                for _ in range(heartbeat_interval):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"心跳线程异常: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(5)  # 发生异常时等待5秒
                
        logger.info("MQTT心跳线程已停止")

    def _publish_heartbeat(self):
        """发送心跳消息"""
        if not self.is_connected:
            return
            
        try:
            # 获取CPU和内存使用情况
            import psutil
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            
            # 获取GPU使用情况
            gpu_usage = 0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except:
                pass
            
            # 构建心跳消息
            heartbeat = {
                "timestamp": int(time.time()),
                "node_id": self.node_id,
                "mac_address": self.mac_address,
                "mqtt_node_id": self.node_id,
                "client_id": self.client_id,
                "service_type": "analysis",
                "type": "heartbeat",
                "is_active": True,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "gpu_usage": gpu_usage,
                "task_count": len(self.active_tasks),
                "max_tasks": 100,
                "status": "online"
            }
            
            # 发布心跳消息到节点状态主题
            topic = f"{self.topic_prefix}{self.node_id}/status"
            self._publish_message(topic, heartbeat)
            
        except Exception as e:
            logger.error(f"发送心跳消息失败: {str(e)}")
            
    def _publish_message(self, topic, payload, qos=1, retain=False):
        """
        发布MQTT消息
        
        Args:
            topic: 主题
            payload: 消息负载
            qos: 服务质量
            retain: 是否保留消息
        """
        if not self.is_connected:
            logger.warning(f"MQTT客户端未连接，无法发布消息到: {topic}")
            return False
            
        try:
            # 将负载转换为JSON字符串
            if isinstance(payload, (dict, list)):
                payload = json.dumps(payload)
                
            # 发布消息
            self.client.publish(topic, payload, qos=qos, retain=retain)
            return True
            
        except Exception as e:
            logger.error(f"发布MQTT消息失败: {str(e)}")
            return False
            
    def register_task_handler(self, task_type, handler):
        """
        注册任务处理器
        
        Args:
            task_type: 任务类型
            handler: 处理函数
        """
        self.task_handlers[task_type] = handler
        logger.info(f"已注册任务处理器: {task_type}") 