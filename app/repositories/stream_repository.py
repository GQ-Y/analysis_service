#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: stream_repository.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 流仓储

处理视频流相关的数据访问操作，包括流的增删改查、状态管理等。

本文件是分析服务项目的一部分。
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .base_repository import InMemoryRepository
from app.models.stream_model import StreamModel, StreamStatistics, StreamHealth
from app.models.base_model import StreamStatusEnum, StreamTypeEnum
from app.exceptions.business_exception import BusinessException


class StreamRepository(InMemoryRepository[StreamModel]):
    """流仓储"""
    
    def __init__(self):
        """初始化流仓储"""
        super().__init__(StreamModel)
    
    async def find_by_status(self, status: StreamStatusEnum) -> List[StreamModel]:
        """根据状态查找流
        
        Args:
            status: 流状态
            
        Returns:
            List[StreamModel]: 流列表
        """
        return await self.find_by_field('status', status)
    
    async def find_by_type(self, stream_type: StreamTypeEnum) -> List[StreamModel]:
        """根据类型查找流
        
        Args:
            stream_type: 流类型
            
        Returns:
            List[StreamModel]: 流列表
        """
        return await self.find_by_field('stream_type', stream_type)
    
    async def find_by_user(self, user_id: str) -> List[StreamModel]:
        """根据用户ID查找流
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[StreamModel]: 流列表
        """
        return await self.find_by_field('user_id', user_id)
    
    async def find_online_streams(self) -> List[StreamModel]:
        """查找在线流
        
        Returns:
            List[StreamModel]: 在线流列表
        """
        return await self.find_by_status(StreamStatusEnum.ONLINE)
    
    async def find_offline_streams(self) -> List[StreamModel]:
        """查找离线流
        
        Returns:
            List[StreamModel]: 离线流列表
        """
        return await self.find_by_status(StreamStatusEnum.OFFLINE)
    
    async def find_by_url(self, stream_url: str) -> Optional[StreamModel]:
        """根据URL查找流
        
        Args:
            stream_url: 流URL
            
        Returns:
            Optional[StreamModel]: 流对象
        """
        return await self.find_one_by_field('stream_url', stream_url)
    
    async def find_by_location(self, location: str) -> List[StreamModel]:
        """根据位置查找流
        
        Args:
            location: 位置信息
            
        Returns:
            List[StreamModel]: 流列表
        """
        return await self.find_by_field('location', location)
    
    async def find_by_tags(self, tags: List[str]) -> List[StreamModel]:
        """根据标签查找流
        
        Args:
            tags: 标签列表
            
        Returns:
            List[StreamModel]: 流列表
        """
        streams = []
        for stream in self._data.values():
            if stream.tags and any(tag in stream.tags for tag in tags):
                streams.append(stream)
        return streams
    
    async def update_status(self, stream_id: str, status: StreamStatusEnum) -> bool:
        """更新流状态
        
        Args:
            stream_id: 流ID
            status: 新状态
            
        Returns:
            bool: 是否更新成功
        """
        stream = await self.get_by_id(stream_id)
        if not stream:
            return False
        
        old_status = stream.status
        stream.status = status
        stream.updated_at = datetime.now()
        
        # 根据状态更新时间字段
        if status == StreamStatusEnum.ONLINE and old_status != StreamStatusEnum.ONLINE:
            stream.last_connected_at = datetime.now()
            stream.connection_count += 1
        elif status in [StreamStatusEnum.OFFLINE, StreamStatusEnum.ERROR] and old_status == StreamStatusEnum.ONLINE:
            stream.last_disconnected_at = datetime.now()
        
        await self.update(stream)
        return True
    
    async def update_connection_info(self, stream_id: str, connected: bool, error_message: str = None) -> bool:
        """更新连接信息
        
        Args:
            stream_id: 流ID
            connected: 是否连接成功
            error_message: 错误消息
            
        Returns:
            bool: 是否更新成功
        """
        stream = await self.get_by_id(stream_id)
        if not stream:
            return False
        
        if connected:
            stream.status = StreamStatusEnum.ONLINE
            stream.last_connected_at = datetime.now()
            stream.connection_count += 1
            stream.last_error = None
        else:
            stream.status = StreamStatusEnum.ERROR if error_message else StreamStatusEnum.OFFLINE
            stream.last_disconnected_at = datetime.now()
            if error_message:
                stream.last_error = error_message
                stream.error_count += 1
        
        stream.updated_at = datetime.now()
        await self.update(stream)
        return True
    
    async def get_user_stream_count(self, user_id: str, status: StreamStatusEnum = None) -> int:
        """获取用户流数量
        
        Args:
            user_id: 用户ID
            status: 流状态（可选）
            
        Returns:
            int: 流数量
        """
        filters = {'user_id': user_id}
        if status:
            filters['status'] = status
        
        return await self.count(filters)
    
    async def get_statistics(self, user_id: str = None) -> Dict[str, Any]:
        """获取流统计信息
        
        Args:
            user_id: 用户ID（可选）
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        # 构建过滤条件
        filters = {}
        if user_id:
            filters['user_id'] = user_id
        
        # 获取所有流
        all_streams = await self.find_all(filters)
        
        # 统计各状态流数量
        status_counts = {}
        for status in StreamStatusEnum:
            status_counts[status.value] = len([stream for stream in all_streams if stream.status == status])
        
        # 统计类型分布
        type_counts = {}
        for stream_type in StreamTypeEnum:
            type_counts[stream_type.value] = len([stream for stream in all_streams if stream.stream_type == stream_type])
        
        # 计算连接成功率
        total_connections = sum(stream.connection_count for stream in all_streams)
        total_errors = sum(stream.error_count for stream in all_streams)
        success_rate = ((total_connections - total_errors) / total_connections) if total_connections > 0 else 0
        
        # 计算平均在线时间
        online_streams = [stream for stream in all_streams if stream.status == StreamStatusEnum.ONLINE]
        avg_uptime = 0
        if online_streams:
            total_uptime = 0
            for stream in online_streams:
                if stream.last_connected_at:
                    uptime = (datetime.now() - stream.last_connected_at).total_seconds()
                    total_uptime += uptime
            avg_uptime = total_uptime / len(online_streams)
        
        return {
            'total_streams': len(all_streams),
            'status_distribution': status_counts,
            'type_distribution': type_counts,
            'connection_success_rate': success_rate,
            'average_uptime_seconds': avg_uptime,
            'total_connections': total_connections,
            'total_errors': total_errors,
            'user_id': user_id
        }
    
    async def find_streams_by_health(self, is_healthy: bool) -> List[StreamModel]:
        """根据健康状态查找流
        
        Args:
            is_healthy: 是否健康
            
        Returns:
            List[StreamModel]: 流列表
        """
        if is_healthy:
            return await self.find_by_status(StreamStatusEnum.ONLINE)
        else:
            unhealthy_statuses = [StreamStatusEnum.OFFLINE, StreamStatusEnum.ERROR]
            streams = []
            for status in unhealthy_statuses:
                streams.extend(await self.find_by_status(status))
            return streams
    
    async def cleanup_old_streams(self, days: int = 180) -> int:
        """清理旧流
        
        Args:
            days: 保留天数
            
        Returns:
            int: 清理的流数量
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        old_streams = []
        for stream in self._data.values():
            # 只清理长时间离线且无活动的流
            if (stream.created_at < cutoff_date and 
                stream.status in [StreamStatusEnum.OFFLINE, StreamStatusEnum.ERROR] and
                (not stream.last_connected_at or stream.last_connected_at < cutoff_date)):
                old_streams.append(stream.id)
        
        # 删除旧流
        for stream_id in old_streams:
            await self.delete(stream_id)
        
        self.logger.info(f"清理了 {len(old_streams)} 个旧流")
        return len(old_streams)


class StreamStatisticsRepository(InMemoryRepository[StreamStatistics]):
    """流统计仓储"""
    
    def __init__(self):
        """初始化流统计仓储"""
        super().__init__(StreamStatistics)
    
    async def find_by_stream_id(self, stream_id: str) -> Optional[StreamStatistics]:
        """根据流ID查找统计
        
        Args:
            stream_id: 流ID
            
        Returns:
            Optional[StreamStatistics]: 统计信息
        """
        return await self.find_one_by_field('stream_id', stream_id)
    
    async def update_connection_stats(self, stream_id: str, success: bool, duration: float = None) -> bool:
        """更新连接统计
        
        Args:
            stream_id: 流ID
            success: 是否成功
            duration: 连接时长
            
        Returns:
            bool: 是否更新成功
        """
        stats = await self.find_by_stream_id(stream_id)
        
        if not stats:
            # 创建新的统计记录
            stats = StreamStatistics(
                stream_id=stream_id,
                start_time=datetime.now()
            )
            await self.create(stats)
        
        # 更新连接统计
        stats.update_connection_stats(success, duration)
        await self.update(stats)
        
        return True
    
    async def update_frame_stats(self, stream_id: str, frame_count: int, bytes_count: int, duration: float) -> bool:
        """更新帧统计
        
        Args:
            stream_id: 流ID
            frame_count: 帧数
            bytes_count: 字节数
            duration: 时长
            
        Returns:
            bool: 是否更新成功
        """
        stats = await self.find_by_stream_id(stream_id)
        
        if not stats:
            # 创建新的统计记录
            stats = StreamStatistics(
                stream_id=stream_id,
                start_time=datetime.now()
            )
            await self.create(stats)
        
        # 更新帧统计
        stats.update_frame_stats(frame_count, bytes_count, duration)
        await self.update(stats)
        
        return True
    
    async def get_stream_performance(self, stream_id: str) -> Dict[str, Any]:
        """获取流性能信息
        
        Args:
            stream_id: 流ID
            
        Returns:
            Dict[str, Any]: 性能信息
        """
        stats = await self.find_by_stream_id(stream_id)
        
        if not stats:
            return {
                'stream_id': stream_id,
                'performance_score': 0,
                'message': '无统计数据'
            }
        
        # 计算性能分数
        performance_score = 100
        
        # 连接成功率影响
        success_rate = stats.calculate_connection_success_rate()
        if success_rate < 0.9:
            performance_score -= (0.9 - success_rate) * 50
        
        # 丢帧率影响
        if stats.frame_drop_rate and stats.frame_drop_rate > 0.05:
            performance_score -= stats.frame_drop_rate * 30
        
        # 帧率影响
        if stats.avg_frame_rate and stats.avg_frame_rate < 15:
            performance_score -= (15 - stats.avg_frame_rate) * 2
        
        performance_score = max(0, min(100, performance_score))
        
        return {
            'stream_id': stream_id,
            'performance_score': performance_score,
            'connection_success_rate': success_rate,
            'frame_drop_rate': stats.frame_drop_rate or 0,
            'average_frame_rate': stats.avg_frame_rate or 0,
            'average_bitrate': stats.avg_bitrate or 0,
            'total_uptime': stats.total_uptime,
            'total_frames': stats.total_frames,
            'total_bytes': stats.total_bytes
        }


class StreamHealthRepository(InMemoryRepository[StreamHealth]):
    """流健康仓储"""
    
    def __init__(self):
        """初始化流健康仓储"""
        super().__init__(StreamHealth)
    
    async def find_by_stream_id(self, stream_id: str) -> Optional[StreamHealth]:
        """根据流ID查找健康状态
        
        Args:
            stream_id: 流ID
            
        Returns:
            Optional[StreamHealth]: 健康状态
        """
        return await self.find_one_by_field('stream_id', stream_id)
    
    async def update_health(self, stream_id: str, is_healthy: bool, 
                           response_time: float = None, error_message: str = None) -> bool:
        """更新健康状态
        
        Args:
            stream_id: 流ID
            is_healthy: 是否健康
            response_time: 响应时间
            error_message: 错误消息
            
        Returns:
            bool: 是否更新成功
        """
        health = await self.find_by_stream_id(stream_id)
        
        if not health:
            # 创建新的健康记录
            health = StreamHealth(
                stream_id=stream_id,
                is_healthy=is_healthy,
                health_score=100.0 if is_healthy else 0.0,
                last_check_time=datetime.now()
            )
            await self.create(health)
        else:
            # 更新健康状态
            health.update_health(is_healthy, response_time, error_message)
            await self.update(health)
        
        return True
    
    async def find_unhealthy_streams(self) -> List[StreamHealth]:
        """查找不健康的流
        
        Returns:
            List[StreamHealth]: 不健康的流列表
        """
        return await self.find_by_field('is_healthy', False)
    
    async def find_by_health_score_range(self, min_score: float, max_score: float) -> List[StreamHealth]:
        """根据健康分数范围查找流
        
        Args:
            min_score: 最小分数
            max_score: 最大分数
            
        Returns:
            List[StreamHealth]: 流健康状态列表
        """
        health_list = []
        for health in self._data.values():
            if min_score <= health.health_score <= max_score:
                health_list.append(health)
        return health_list
    
    async def cleanup_old_health_records(self, days: int = 30) -> int:
        """清理旧健康记录
        
        Args:
            days: 保留天数
            
        Returns:
            int: 清理的记录数量
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        old_records = []
        for health in self._data.values():
            if health.last_check_time < cutoff_date:
                old_records.append(health.id)
        
        # 删除旧记录
        for record_id in old_records:
            await self.delete(record_id)
        
        self.logger.info(f"清理了 {len(old_records)} 个旧健康记录")
        return len(old_records)
