#!/usr/bin/env python3
"""
Socket回调管理器
负责管理全局的CallbackSocketClient实例及其生命周期。
"""
import asyncio
from typing import Dict, Any, Optional

from shared.utils.logger import get_normal_logger, get_exception_logger
from shared.utils.callback_socket_client import CallbackSocketClient
from shared.config.settings import settings # 使用全局settings实例

class SocketCallbackManager:
    _instance: Optional['SocketCallbackManager'] = None
    _client: Optional[CallbackSocketClient] = None
    _manager_lock = asyncio.Lock()  # Lock for manager instance creation and client initialization
    _connect_lock = asyncio.Lock()  # Lock specifically for connection attempts
    _reconnect_task: Optional[asyncio.Task] = None
    _shutting_down: bool = False # Flag to indicate shutdown process

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.logger = get_normal_logger("SocketCallbackManager")
        self.exception_logger = get_exception_logger("SocketCallbackManager")
        # Configuration is accessed via settings directly in methods
        self._initialized = True

    @classmethod
    async def get_instance(cls) -> 'SocketCallbackManager':
        if cls._instance is None:
            async with cls._manager_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    # Client initialization moved to establish_connection or a dedicated init method if preferred
        return cls._instance

    async def initialize_client(self):
        """Initializes the socket client if not already initialized."""
        # This method is kept for potential explicit initialization scenarios,
        # but establish_connection will also handle client creation.
        if not self._client and settings.SOCKET_CALLBACK_ENABLED:
            async with self._manager_lock: # Protect client creation
                if not self._client: # Double check
                    self.logger.info(f"初始化Socket客户端，目标: {settings.SOCKET_CALLBACK_HOST}:{settings.SOCKET_CALLBACK_PORT}")
                    self._client = CallbackSocketClient(
                        host=settings.SOCKET_CALLBACK_HOST,
                        port=settings.SOCKET_CALLBACK_PORT
                    )
        elif not settings.SOCKET_CALLBACK_ENABLED:
            self.logger.info("Socket回调功能已禁用，不初始化客户端。")
            if self._client:
                 await self._client.disconnect()
                 self._client = None # Ensure client is cleared if disabled

    async def establish_connection(self, from_startup: bool = False) -> bool:
        """尝试建立或重新建立到Socket服务器的连接。"""
        if self._shutting_down:
            self.logger.info("管理器正在关闭，跳过建立连接。")
            return False
        
        if not settings.SOCKET_CALLBACK_ENABLED:
            self.logger.info("Socket回调未启用，不尝试连接。")
            if self._client and self._client.connected:
                await self._client.disconnect()
            return False

        # Ensure client is initialized
        if not self._client:
            await self.initialize_client()
            if not self._client: # Still no client (e.g., disabled)
                return False
        
        # Use a specific lock for connection attempts to allow other manager operations
        async with self._connect_lock:
            if self._client.connected:
                # self.logger.debug("连接已建立。") # Can be noisy
                if self._reconnect_task and not self._reconnect_task.done():
                    self.logger.info("连接已成功建立，取消后台重连任务。")
                    self._reconnect_task.cancel()
                    self._reconnect_task = None
                return True

            self.logger.info("尝试连接到Socket服务器...")
            connected = await self._client.connect() # connect now has retries

            if connected:
                self.logger.info("成功连接到Socket服务器。")
                if self._reconnect_task and not self._reconnect_task.done():
                    self.logger.info("连接成功，取消现有的后台重连任务。")
                    self._reconnect_task.cancel()
                    self._reconnect_task = None
                return True
            else:
                self.logger.warning("连接Socket服务器失败 (客户端多次尝试后)。")
                if (from_startup or not self._reconnect_task or self._reconnect_task.done()) and not self._shutting_down:
                    self.logger.info("启动后台连接监控任务...")
                    if self._reconnect_task and not self._reconnect_task.done():
                        self._reconnect_task.cancel() # Cancel existing before starting new
                    self._reconnect_task = asyncio.create_task(self._connection_monitor_task())
                return False

    async def _connection_monitor_task(self):
        """后台任务，定期检查并尝试重新连接Socket。"""
        self.logger.info("Socket后台连接监控任务已启动。")
        # Use a longer initial delay if just failed, then regular interval
        await asyncio.sleep(settings.SOCKET_CONNECT_RETRY_DELAY * 2) 

        while not self._shutting_down:
            if not self._client or not self._client.connected:
                self.logger.info("监控任务: 检测到连接断开，尝试重新连接...")
                # Do not pass from_startup=True here, as this is the monitor task
                await self.establish_connection() 
            else:
                # self.logger.debug("监控任务: 连接状态正常。") # Can be very noisy
                pass
            
            # Wait before next check
            # Consider a different, possibly longer, interval for monitoring vs initial retries
            # For now, using SOCKET_CONNECT_RETRY_DELAY for simplicity
            await asyncio.sleep(settings.SOCKET_CONNECT_RETRY_DELAY * 3) # e.g., check every 15s if delay is 5s
        self.logger.info("Socket后台连接监控任务已停止。")

    async def send_socket_callback(self, data: Dict[str, Any]) -> bool:
        """尝试通过Socket发送回调数据"""
        if not settings.SOCKET_CALLBACK_ENABLED:
            return False

        if not self._client:
            self.logger.warning("Socket客户端未初始化，无法发送回调。")
            return False
        
        if not self._client.connected:
            self.logger.warning("Socket客户端未连接，无法发送回调。后台任务将尝试重连。")
            # Trigger a connection attempt if not already happening via monitor,
            # but don't block. The monitor task is the primary reconductor.
            if (not self._reconnect_task or self._reconnect_task.done()) and not self._shutting_down:
                 asyncio.create_task(self.establish_connection())
            return False

        try:
            # self.logger.info(f"准备通过管理器发送数据到 {self._client.host}:{self._client.port}")
            success = await self._client.send_callback(data)
            if success:
                # self.logger.info("通过管理器发送Socket回调数据成功。") # Can be noisy
                pass
            else:
                self.logger.warning("通过管理器发送Socket回调数据失败 (客户端报告)。可能需要后台重连。")
                # Client's send_callback would set its own .connected to False if send failed due to connection issue.
                # The monitor task will pick this up.
                if (not self._reconnect_task or self._reconnect_task.done()) and not self._shutting_down:
                    asyncio.create_task(self.establish_connection()) # Ensure monitor starts if send fails and client disconnects
            return success
        except Exception as e:
            self.exception_logger.error(f"通过管理器发送Socket回调时发生异常: {e}")
            if self._client and self._client.connected: # If exception didn't come from client's disconnect
                await self._client.disconnect() 
            return False

    async def shutdown(self):
        """关闭SocketCallbackManager，断开客户端连接并停止后台任务。"""
        self.logger.info("开始关闭SocketCallbackManager...")
        self._shutting_down = True # Signal monitor task to stop
        async with self._manager_lock:
            if self._reconnect_task and not self._reconnect_task.done():
                self.logger.info("正在取消后台重连任务...")
                self._reconnect_task.cancel()
                try:
                    await self._reconnect_task # Wait for cancellation to complete
                except asyncio.CancelledError:
                    self.logger.info("后台重连任务已成功取消。")
                self._reconnect_task = None
            
            if self._client:
                self.logger.info("正在断开Socket客户端连接...")
                await self._client.disconnect()
                self._client = None # Clear the client instance
            self.logger.info("SocketCallbackManager已成功关闭。")
            # Reset instance for potential re-initialization if app restarts parts, though usually not needed for singletons
            # SocketCallbackManager._instance = None 

# 单例模式的辅助函数
_socket_manager_instance: Optional[SocketCallbackManager] = None
_socket_manager_lock = asyncio.Lock()

async def get_socket_manager() -> SocketCallbackManager:
    global _socket_manager_instance
    if _socket_manager_instance is None:
        async with _socket_manager_lock:
            if _socket_manager_instance is None:
                _socket_manager_instance = await SocketCallbackManager.get_instance()
                # Perform initial client initialization and connection attempt when manager is first obtained
                # This ensures that by the time manager is used, connection setup has been initiated.
                # No, startup_socket_manager will call establish_connection
    return _socket_manager_instance

async def startup_socket_manager():
    """在应用启动时初始化并连接Socket管理器。"""
    manager = await get_socket_manager()
    manager._shutting_down = False # Ensure manager is not in shutting down state if re-starting
    await manager.initialize_client() # Ensure client object exists if enabled
    if settings.SOCKET_CALLBACK_ENABLED and manager._client:
        await manager.establish_connection(from_startup=True)
    else:
        manager.logger.info("Socket回调未启用或客户端未初始化，跳过启动时连接。")

async def shutdown_socket_manager():
    """在应用关闭时关闭Socket管理器。"""
    manager = await get_socket_manager()
    await manager.shutdown() 