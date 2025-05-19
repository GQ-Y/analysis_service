"""
Redis管理模块
负责连接和管理Redis缓存
"""
import os
import json
import asyncio
from typing import Optional, Any, List, Dict, Callable
from shared.utils.logger import get_normal_logger, get_exception_logger
import redis.asyncio as redis

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)

class RedisManager:
    """Redis管理器，单例模式"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return

        self.redis_host = os.environ.get('REDIS_HOST', 'localhost')
        self.redis_port = int(os.environ.get('REDIS_PORT', 6379))
        self.redis_db = int(os.environ.get('REDIS_DB', 0))
        self.redis_password = os.environ.get('REDIS_PASSWORD', None)
        self.redis_pool = None
        self.redis_client = None

        try:
            normal_logger.info(f"初始化Redis连接: {self.redis_host}:{self.redis_port}, 数据库: {self.redis_db}")
            self.redis_pool = redis.ConnectionPool(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True
            )
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            self.initialized = True
            normal_logger.info("Redis连接池初始化成功")
        except Exception as e:
            exception_logger.exception(f"Redis连接失败: {str(e)}")
            self.initialized = False

    async def ping(self) -> bool:
        """
        测试Redis连接

        Returns:
            bool: 连接是否可用
        """
        try:
            if not self.redis_client:
                normal_logger.warning("Redis客户端未初始化")
                return False

            result = await self.redis_client.ping()
            return result
        except Exception as e:
            exception_logger.exception(f"Redis ping失败: {str(e)}")
            return False

    async def get(self, key: str) -> Optional[str]:
        """
        获取键值

        Args:
            key: 键名

        Returns:
            Optional[str]: 键值
        """
        try:
            return await self.redis_client.get(key)
        except Exception as e:
            exception_logger.exception(f"Redis get失败: {str(e)}")
            return None

    async def set(self, key: str, value: str, expire: int = None) -> bool:
        """
        设置键值

        Args:
            key: 键名
            value: 键值
            expire: 过期时间（秒）

        Returns:
            bool: 是否设置成功
        """
        try:
            await self.redis_client.set(key, value, ex=expire)
            return True
        except Exception as e:
            exception_logger.exception(f"Redis set失败: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """
        删除键

        Args:
            key: 键名

        Returns:
            bool: 是否删除成功
        """
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            exception_logger.exception(f"Redis delete失败: {str(e)}")
            return False

    async def hset(self, name: str, key: str, value: str) -> bool:
        """
        设置哈希表字段值

        Args:
            name: 哈希表名
            key: 字段名
            value: 字段值

        Returns:
            bool: 是否设置成功
        """
        try:
            await self.redis_client.hset(name, key, value)
            return True
        except Exception as e:
            exception_logger.exception(f"Redis hset失败: {str(e)}")
            return False

    async def hget(self, name: str, key: str) -> Optional[str]:
        """
        获取哈希表字段值

        Args:
            name: 哈希表名
            key: 字段名

        Returns:
            Optional[str]: 字段值
        """
        try:
            return await self.redis_client.hget(name, key)
        except Exception as e:
            exception_logger.exception(f"Redis hget失败: {str(e)}")
            return None

    async def hgetall(self, name: str) -> dict:
        """
        获取哈希表所有字段值

        Args:
            name: 哈希表名

        Returns:
            dict: 哈希表所有字段
        """
        try:
            return await self.redis_client.hgetall(name)
        except Exception as e:
            exception_logger.exception(f"Redis hgetall失败: {str(e)}")
            return {}

    async def hdel(self, name: str, key: str) -> bool:
        """
        删除哈希表字段

        Args:
            name: 哈希表名
            key: 字段名

        Returns:
            bool: 是否删除成功
        """
        try:
            await self.redis_client.hdel(name, key)
            return True
        except Exception as e:
            exception_logger.exception(f"Redis hdel失败: {str(e)}")
            return False

    async def exists(self, key: str) -> bool:
        """
        检查键是否存在

        Args:
            key: 键名

        Returns:
            bool: 键是否存在
        """
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            exception_logger.exception(f"Redis exists失败: {str(e)}")
            return False

    async def close(self):
        """关闭Redis连接"""
        if self.redis_pool:
            await self.redis_pool.disconnect()
            normal_logger.info("Redis连接池已关闭")
            
    async def get_value(self, key: str, as_json: bool = False) -> Optional[Any]:
        """获取键值"""
        try:
            normal_logger.info(f"Redis.get_value - 获取键: {key}")
            value = await self.redis_client.get(key)
            
            if value:
                normal_logger.info(f"Redis.get_value - 成功获取键 {key} 的值")
                if as_json:
                    try:
                        parsed_value = json.loads(value)
                        return parsed_value
                    except json.JSONDecodeError as e:
                        exception_logger.error(f"Redis.get_value - JSON解析失败 - {key}: {str(e)}")
                        return None
                return value
            else:
                normal_logger.warning(f"Redis.get_value - 键 {key} 不存在")
                return None
                
        except Exception as e:
            exception_logger.error(f"Redis.get_value - 获取键值失败 - {key}: {str(e)}")
            return None
            
    async def set_value(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """设置键值"""
        try:
            normal_logger.info(f"Redis.set_value - 设置键: {key}")
            
            if isinstance(value, (dict, list)):
                try:
                    value = json.dumps(value)
                    normal_logger.info(f"Redis.set_value - 字典/列表转为JSON字符串: {key}")
                except Exception as e:
                    exception_logger.error(f"Redis.set_value - JSON序列化失败 - {key}: {str(e)}")
                    raise
            
            result = await self.redis_client.set(key, value, ex=ex)
            
            if result:
                normal_logger.info(f"Redis.set_value - 成功设置键 {key} 的值")
                if ex:
                    normal_logger.info(f"Redis.set_value - 键 {key} 设置过期时间: {ex}秒")
            else:
                normal_logger.warning(f"Redis.set_value - 设置键 {key} 返回结果: {result}")
                
            # 验证键是否成功设置
            try:
                exists = await self.exists_key(key)
                if exists:
                    normal_logger.info(f"Redis.set_value - 验证键 {key} 已成功设置")
                else:
                    normal_logger.warning(f"Redis.set_value - 验证失败，键 {key} 不存在")
            except Exception as e:
                exception_logger.error(f"Redis.set_value - 验证键设置失败 - {key}: {str(e)}")
            
            return bool(result)
            
        except Exception as e:
            exception_logger.error(f"Redis.set_value - 设置键值失败 - {key}: {str(e)}")
            return False
            
    async def delete_key(self, key: str) -> bool:
        """删除键"""
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            exception_logger.error(f"删除Redis键失败 - {key}: {str(e)}")
            return False
            
    async def exists_key(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            return await self.redis_client.exists(key)
        except Exception as e:
            exception_logger.error(f"检查Redis键是否存在失败 - {key}: {str(e)}")
            return False
            
    async def zadd_task(self, key: str, member: str, score: float) -> bool:
        """添加任务到有序集合"""
        try:
            await self.redis_client.zadd(key, {member: score})
            return True
        except Exception as e:
            exception_logger.error(f"添加任务到有序集合失败 - {key}: {str(e)}")
            return False
            
    async def zget_tasks(self, key: str, start: int, end: int) -> List[str]:
        """获取有序集合中的任务"""
        try:
            return await self.redis_client.zrange(key, start, end)
        except Exception as e:
            exception_logger.error(f"获取有序集合任务失败 - {key}: {str(e)}")
            return []
            
    async def zrem_task(self, key: str, member: str) -> bool:
        """从有序集合中移除任务"""
        try:
            await self.redis_client.zrem(key, member)
            return True
        except Exception as e:
            exception_logger.error(f"从有序集合移除任务失败 - {key}: {str(e)}")
            return False

    # 哈希操作
    async def hset_dict(self, name: str, mapping: Dict[str, Any]):
        """设置哈希表"""
        try:
            processed_mapping = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in mapping.items()
            }
            await self.redis_client.hset(name, mapping=processed_mapping)
        except Exception as e:
            exception_logger.error(f"设置哈希表失败: {str(e)}")
            raise

    async def hget_dict(self, name: str, key: str, as_json: bool = False) -> Optional[Any]:
        """获取哈希表字段"""
        try:
            value = await self.redis_client.hget(name, key)
            if value and as_json:
                return json.loads(value)
            return value
        except Exception as e:
            exception_logger.error(f"获取哈希表字段失败: {str(e)}")
            return None

    # 列表操作
    async def list_push(self, name: str, value: Any):
        """推入列表"""
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            await self.redis_client.rpush(name, value)
        except Exception as e:
            exception_logger.error(f"推入列表失败: {str(e)}")
            raise

    async def list_pop(self, name: str, as_json: bool = False) -> Optional[Any]:
        """弹出列表"""
        try:
            value = await self.redis_client.lpop(name)
            if value and as_json:
                return json.loads(value)
            return value
        except Exception as e:
            exception_logger.error(f"弹出列表失败: {str(e)}")
            return None

    async def set_expiry(self, name: str, seconds: int):
        """设置键的过期时间"""
        try:
            await self.redis_client.expire(name, seconds)
        except Exception as e:
            exception_logger.error(f"设置过期时间失败: {str(e)}")
            raise

    async def delete_pattern(self, pattern: str):
        """删除匹配模式的键"""
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
        except Exception as e:
            exception_logger.error(f"删除匹配模式的键失败: {str(e)}")
            raise
            
    # 发布订阅
    async def publish(self, channel: str, message: str) -> int:
        """发布消息到频道
        
        Args:
            channel: 频道名称
            message: 消息内容
            
        Returns:
            int: 接收到消息的客户端数量
        """
        try:
            normal_logger.debug(f"发布消息到频道 {channel}")
            return await self.redis_client.publish(channel, message)
        except Exception as e:
            exception_logger.error(f"发布消息到频道 {channel} 失败: {str(e)}")
            return 0
            
    async def subscribe(self, channel: str, callback: Callable):
        """订阅频道
        
        Args:
            channel: 频道名称
            callback: 回调函数，接收频道名和消息内容 async def callback(channel, message)
        """
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(channel)
            normal_logger.info(f"已订阅频道: {channel}")
            
            try:
                while True:
                    message = await pubsub.get_message(ignore_subscribe_messages=True)
                    if message:
                        normal_logger.debug(f"从频道 {channel} 收到消息")
                        await callback(channel, message["data"])
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                normal_logger.info(f"取消订阅频道: {channel}")
                raise
            finally:
                await pubsub.unsubscribe(channel)
                
        except Exception as e:
            exception_logger.error(f"订阅频道 {channel} 失败: {str(e)}")
            raise 