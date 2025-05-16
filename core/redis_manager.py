"""Redis管理器模块"""
import json
from redis import asyncio as aioredis
from typing import Optional, Any, Dict, List, Callable
import asyncio
from loguru import logger
from shared.utils.logger import setup_logger
from core.config import settings

logger = setup_logger(__name__)

class RedisManager:
    """Redis连接管理器"""
    
    def __init__(self):
        """初始化Redis连接"""
        self.redis = None
        self.pool = None
        self._init_connection()
        
    def _init_connection(self):
        """初始化Redis连接池"""
        try:
            self.pool = aioredis.ConnectionPool(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                decode_responses=True
            )
            self.redis = aioredis.Redis(connection_pool=self.pool)
            logger.info(f"Redis连接池初始化成功: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except Exception as e:
            logger.error(f"Redis连接池初始化失败: {str(e)}")
            raise
            
    async def close(self):
        """关闭Redis连接"""
        if self.pool:
            await self.pool.disconnect()
            logger.info("Redis连接池已关闭")
            
    async def get_value(self, key: str, as_json: bool = False) -> Any:
        """获取键值"""
        try:
            logger.info(f"Redis.get_value - 获取键: {key}")
            value = await self.redis.get(key)
            
            if value:
                logger.info(f"Redis.get_value - 成功获取键 {key} 的值")
                if as_json:
                    try:
                        parsed_value = json.loads(value)
                        return parsed_value
                    except json.JSONDecodeError as e:
                        logger.error(f"Redis.get_value - JSON解析失败 - {key}: {str(e)}")
                        return None
                return value
            else:
                logger.warning(f"Redis.get_value - 键 {key} 不存在")
                return None
                
        except Exception as e:
            logger.error(f"Redis.get_value - 获取键值失败 - {key}: {str(e)}")
            return None
            
    async def set_value(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """设置键值"""
        try:
            logger.info(f"Redis.set_value - 设置键: {key}")
            
            if isinstance(value, (dict, list)):
                try:
                    value = json.dumps(value)
                    logger.info(f"Redis.set_value - 字典/列表转为JSON字符串: {key}")
                except Exception as e:
                    logger.error(f"Redis.set_value - JSON序列化失败 - {key}: {str(e)}")
                    raise
            
            result = await self.redis.set(key, value, ex=ex)
            
            if result:
                logger.info(f"Redis.set_value - 成功设置键 {key} 的值")
                if ex:
                    logger.info(f"Redis.set_value - 键 {key} 设置过期时间: {ex}秒")
            else:
                logger.warning(f"Redis.set_value - 设置键 {key} 返回结果: {result}")
                
            # 验证键是否成功设置
            try:
                exists = await self.exists_key(key)
                if exists:
                    logger.info(f"Redis.set_value - 验证键 {key} 已成功设置")
                else:
                    logger.warning(f"Redis.set_value - 验证失败，键 {key} 不存在")
            except Exception as e:
                logger.error(f"Redis.set_value - 验证键设置失败 - {key}: {str(e)}")
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis.set_value - 设置键值失败 - {key}: {str(e)}")
            return False
            
    async def delete_key(self, key: str) -> bool:
        """删除键"""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"删除Redis键失败 - {key}: {str(e)}")
            return False
            
    async def exists_key(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            return await self.redis.exists(key)
        except Exception as e:
            logger.error(f"检查Redis键是否存在失败 - {key}: {str(e)}")
            return False
            
    async def zadd_task(self, key: str, member: str, score: float) -> bool:
        """添加任务到有序集合"""
        try:
            await self.redis.zadd(key, {member: score})
            return True
        except Exception as e:
            logger.error(f"添加任务到有序集合失败 - {key}: {str(e)}")
            return False
            
    async def zget_tasks(self, key: str, start: int, end: int) -> List[str]:
        """获取有序集合中的任务"""
        try:
            return await self.redis.zrange(key, start, end)
        except Exception as e:
            logger.error(f"获取有序集合任务失败 - {key}: {str(e)}")
            return []
            
    async def zrem_task(self, key: str, member: str) -> bool:
        """从有序集合中移除任务"""
        try:
            await self.redis.zrem(key, member)
            return True
        except Exception as e:
            logger.error(f"从有序集合移除任务失败 - {key}: {str(e)}")
            return False

    async def ping(self) -> bool:
        """测试连接"""
        try:
            return await self.redis.ping()
        except Exception as e:
            logger.error(f"Redis连接测试失败: {str(e)}")
            return False

    # 哈希操作
    async def hset_dict(self, name: str, mapping: Dict[str, Any]):
        """设置哈希表"""
        try:
            processed_mapping = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in mapping.items()
            }
            await self.redis.hset(name, mapping=processed_mapping)
        except Exception as e:
            logger.error(f"设置哈希表失败: {str(e)}")
            raise

    async def hget_dict(self, name: str, key: str, as_json: bool = False) -> Any:
        """获取哈希表字段"""
        try:
            value = await self.redis.hget(name, key)
            if value and as_json:
                return json.loads(value)
            return value
        except Exception as e:
            logger.error(f"获取哈希表字段失败: {str(e)}")
            return None

    # 列表操作
    async def list_push(self, name: str, value: Any):
        """推入列表"""
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            await self.redis.rpush(name, value)
        except Exception as e:
            logger.error(f"推入列表失败: {str(e)}")
            raise

    async def list_pop(self, name: str, as_json: bool = False) -> Any:
        """弹出列表"""
        try:
            value = await self.redis.lpop(name)
            if value and as_json:
                return json.loads(value)
            return value
        except Exception as e:
            logger.error(f"弹出列表失败: {str(e)}")
            return None

    async def set_expiry(self, name: str, seconds: int):
        """设置键的过期时间"""
        try:
            await self.redis.expire(name, seconds)
        except Exception as e:
            logger.error(f"设置过期时间失败: {str(e)}")
            raise

    async def delete_pattern(self, pattern: str):
        """删除匹配模式的键"""
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
        except Exception as e:
            logger.error(f"删除匹配模式的键失败: {str(e)}")
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
            logger.debug(f"发布消息到频道 {channel}")
            return await self.redis.publish(channel, message)
        except Exception as e:
            logger.error(f"发布消息到频道 {channel} 失败: {str(e)}")
            return 0
            
    async def subscribe(self, channel: str, callback: Callable):
        """订阅频道
        
        Args:
            channel: 频道名称
            callback: 回调函数，接收频道名和消息内容 async def callback(channel, message)
        """
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(channel)
            logger.info(f"已订阅频道: {channel}")
            
            try:
                while True:
                    message = await pubsub.get_message(ignore_subscribe_messages=True)
                    if message:
                        logger.debug(f"从频道 {channel} 收到消息")
                        await callback(channel, message["data"])
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                logger.info(f"取消订阅频道: {channel}")
                raise
            finally:
                await pubsub.unsubscribe(channel)
                
        except Exception as e:
            logger.error(f"订阅频道 {channel} 失败: {str(e)}")
            raise 