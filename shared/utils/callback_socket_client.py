#!/usr/bin/env python3
"""
Socket客户端模块
"""
import socket
import json
import asyncio
import time # For retry delay
from typing import Dict, Any, Optional

from shared.utils.logger import get_normal_logger, get_exception_logger
# Import settings to access retry configurations
from core.config import settings

class CallbackSocketClient:
    """回调Socket客户端类"""

    def __init__(self, host: str = "localhost", port: int = 8089):
        """
        初始化Socket客户端

        Args:
            host: Socket服务器地址
            port: Socket服务器端口
        """
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.connected: bool = False
        self.logger = get_normal_logger(f"CallbackSocketClient[{self.host}:{self.port}]")
        self.exception_logger = get_exception_logger(f"CallbackSocketClient[{self.host}:{self.port}]")
        self._lock = asyncio.Lock() # Lock for connect/disconnect operations
        self._connecting = False # Flag to prevent multiple concurrent connect attempts

    async def connect(self) -> bool:
        """
        连接到Socket服务器，带重试逻辑。
        """
        if self.connected:
            self.logger.info("已连接，无需重复连接。")
            return True
        
        if self._connecting:
            self.logger.info("正在尝试连接中，请稍候...")
            # Optionally, wait for the existing connection attempt or return False
            # For now, just return False to prevent re-entry
            return False

        async with self._lock: # Ensure only one connect operation at a time
            if self.connected: # Double check after acquiring lock
                return True
            if self._connecting: # Double check after acquiring lock
                return False
            
            self._connecting = True
            
            attempts = 0
            max_attempts = settings.callback.socket_max_connect_attempts
            retry_delay = settings.callback.socket_connect_retry_delay
            connect_timeout = settings.callback.socket_connect_timeout

            while attempts < max_attempts:
                attempts += 1
                self.logger.info(f"尝试连接到Socket服务器: {self.host}:{self.port} (第 {attempts}/{max_attempts} 次)")
                try:
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    # Set a timeout for the connection attempt itself
                    self.socket.settimeout(connect_timeout) 
                    
                    # socket.connect() is blocking, run in executor for async context
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self.socket.connect, (self.host, self.port))
                    
                    self.socket.settimeout(None) # Reset timeout for normal operations
                    self.connected = True
                    self.logger.info(f"✅ 已连接到Socket服务器: {self.host}:{self.port}")
                    self._connecting = False
                    return True
                except socket.timeout:
                    self.logger.warning(f"连接Socket服务器超时 (第 {attempts}/{max_attempts} 次)。")
                    if self.socket:
                        self.socket.close()
                    self.socket = None
                    if attempts < max_attempts:
                        self.logger.info(f"将在 {retry_delay} 秒后重试...")
                        await asyncio.sleep(retry_delay)
                    else:
                        self.logger.error(f"达到最大连接尝试次数 ({max_attempts})，连接失败。")
                        self._connecting = False
                        return False
                except Exception as e:
                    self.exception_logger.error(f"连接Socket服务器失败 (第 {attempts}/{max_attempts} 次): {e}")
                    if self.socket:
                        self.socket.close()
                    self.socket = None
                    if attempts < max_attempts:
                        self.logger.info(f"将在 {retry_delay} 秒后重试...")
                        await asyncio.sleep(retry_delay)
                    else:
                        self.logger.error(f"达到最大连接尝试次数 ({max_attempts})，连接失败。")
                        self._connecting = False
                        return False
            
            self._connecting = False
            return False # Should be unreachable if logic is correct

    async def send_callback(self, data: Dict[str, Any]) -> bool:
        """
        发送回调数据

        Args:
            data: 要发送的回调数据（字典格式）

        Returns:
            bool: 发送是否成功
        """
        if not self.connected or not self.socket:
            self.logger.warning("❌ 未连接到Socket服务器或socket对象不存在，无法发送数据。")
            self.connected = False # Ensure state is accurate
            return False

        try:
            json_data = json.dumps(data, ensure_ascii=False)
            
            # 添加调试日志
            self.logger.info(f"准备发送JSON数据，长度: {len(json_data)} 字符")
            self.logger.debug(f"JSON数据包含的顶层字段: {list(data.keys())}")
            if 'data' in data:
                self.logger.debug(f"data字段包含的子字段: {list(data['data'].keys())}")
            
            # 添加换行符作为消息分隔符，便于接收端识别消息边界
            message_bytes = (json_data + '\n').encode('utf-8')
            
            self.logger.info(f"编码后的消息长度: {len(message_bytes)} 字节")

            # Use socket.sendall for ensuring all data is sent, run in executor
            loop = asyncio.get_running_loop()
            # Set a timeout for sending data
            self.socket.settimeout(settings.callback.socket_send_timeout)
            await loop.run_in_executor(None, self.socket.sendall, message_bytes)
            self.socket.settimeout(None) # Reset timeout

            # Note: Original code expected a response. 
            # For a one-way push, receiving a response might not be necessary or could be optional.
            # If a response is required, uncomment and adapt the recv logic.
            # response = await loop.run_in_executor(None, self.socket.recv, 4096)
            # response_data = json.loads(response.decode('utf-8'))
            # self.logger.info(f"✅ 回调数据发送成功，服务器响应: {response_data}")

            self.logger.info(f"✅ 回调数据发送成功，已发送 {len(message_bytes)} 字节。")
            return True

        except socket.timeout:
            self.exception_logger.error(f"❌ 发送回调数据超时: {self.host}:{self.port}")
            self.connected = False # Assume connection is lost on timeout
            if self.socket:
                try:
                    self.socket.close()
                except Exception:
                    pass
                self.socket = None
            return False
        except Exception as e:
            self.exception_logger.error(f"❌ 发送回调数据失败: {e}")
            self.connected = False # Assume connection is lost on other errors too
            if self.socket:
                try:
                    self.socket.close()
                except Exception:
                    pass
                self.socket = None
            return False

    async def disconnect(self):
        """断开Socket连接"""
        async with self._lock: # Use lock to prevent race conditions with connect
            if self.socket:
                try:
                    self.socket.close()
                    self.logger.info("🔌 Socket连接已断开。")
                except Exception as e:
                    self.exception_logger.warning(f"关闭socket时发生错误: {e}")
                finally:
                    self.socket = None
                    self.connected = False
            else:
                self.logger.info("Socket未连接或已断开。")
            # Ensure connected status is false even if socket was already None
            self.connected = False
            self._connecting = False # Reset connecting flag

    # Context manager methods can remain if useful for direct client usage,
    # but the manager will handle connect/disconnect mostly.
    async def __aenter__(self):
        """上下文管理器入口 (异步)"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口 (异步)"""
        await self.disconnect()

# Example usage remains commented out as it's not part of the library code.
# async def example_usage():
# ...

# Example of how to use the client (typically managed by a service/manager)
# async def main_example():
#     client = CallbackSocketClient(host="localhost", port=8089)
#     if await client.connect():
#         test_payload = {"message": "Hello from CallbackSocketClient", "timestamp": time.time()}
#         await client.send_data(test_payload)
#         await client.disconnect()
#
# if __name__ == "__main__":
# import asyncio
# import time
# asyncio.run(main_example()) 