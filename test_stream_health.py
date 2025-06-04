#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流健康检查测试脚本
用于验证改进的流管理逻辑
"""

import asyncio
import logging
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.media_kit.zlm_manager import ZLMManager
from core.media_kit.zlm_stream import ZLMVideoStream
from core.media_kit.zlm_config import ZLMConfig
from utils.logger import normal_logger, exception_logger

class StreamHealthTester:
    """流健康检查测试器"""
    
    def __init__(self):
        """初始化测试器"""
        # 配置ZLM
        self.zlm_config = ZLMConfig(
            server_address="127.0.0.1",
            http_port=80,
            rtsp_port=554,
            secret="035c73f7-bb6b-4889-a715-d9eb2d1925cc"
        )
        
        self.zlm_manager = ZLMManager(self.zlm_config)
        
    async def test_stream_health_check(self):
        """测试流健康检查功能"""
        normal_logger.info("开始测试流健康检查功能")
        
        try:
            # 1. 测试检查不存在的流
            normal_logger.info("=== 测试1: 检查不存在的流 ===")
            health_result = await self.zlm_manager.check_stream_health(
                vhost="__defaultVhost__",
                app="live", 
                stream_name="non_existent_stream"
            )
            
            normal_logger.info(f"不存在流的健康检查结果: {health_result}")
            assert not health_result["healthy"], "不存在的流应该标记为不健康"
            
            # 2. 创建一个测试流
            normal_logger.info("=== 测试2: 创建测试流并检查健康状态 ===")
            test_rtsp_url = "rtsp://admin:password@192.168.1.100:554/stream1"
            
            stream = ZLMVideoStream(
                stream_id="test_stream_001",
                rtsp_url=test_rtsp_url,
                zlm_manager=self.zlm_manager
            )
            
            # 尝试启动流
            normal_logger.info("启动测试流...")
            await stream.start()
            
            # 等待流稳定
            await asyncio.sleep(3)
            
            # 检查流健康状态
            normal_logger.info("检查新创建流的健康状态...")
            if hasattr(stream, '_proxy_key') and stream._proxy_key:
                # 获取流信息进行健康检查
                stream_info = await self.zlm_manager.get_stream_info(stream.stream_id)
                if stream_info:
                    health_result = await self.zlm_manager.check_stream_health(
                        vhost=stream_info.get("vhost", "__defaultVhost__"),
                        app=stream_info.get("app", "live"),
                        stream_name=stream_info.get("stream", ""),
                        original_url=test_rtsp_url
                    )
                    
                    normal_logger.info(f"新创建流的健康检查结果: {health_result}")
                else:
                    normal_logger.warning("无法获取流信息")
            
            # 3. 测试强制删除流
            normal_logger.info("=== 测试3: 测试强制删除流 ===")
            if hasattr(stream, '_proxy_key') and stream._proxy_key:
                stream_info = await self.zlm_manager.get_stream_info(stream.stream_id)
                if stream_info:
                    success = await self.zlm_manager.force_delete_stream(
                        vhost=stream_info.get("vhost", "__defaultVhost__"),
                        app=stream_info.get("app", "live"),
                        stream_name=stream_info.get("stream", "")
                    )
                    normal_logger.info(f"强制删除流结果: {success}")
            
            # 停止流
            await stream.stop()
            
        except Exception as e:
            exception_logger.exception(f"测试过程中出错: {str(e)}")
        
        normal_logger.info("流健康检查测试完成")
    
    async def test_existing_stream_handling(self):
        """测试现有流处理逻辑"""
        normal_logger.info("开始测试现有流处理逻辑")
        
        try:
            test_rtsp_url = "rtsp://admin:password@192.168.1.100:554/stream2"
            
            # 1. 第一次创建流
            normal_logger.info("=== 测试1: 第一次创建流 ===")
            stream1 = ZLMVideoStream(
                stream_id="test_stream_002",
                rtsp_url=test_rtsp_url,
                zlm_manager=self.zlm_manager
            )
            
            await stream1.start()
            await asyncio.sleep(2)
            
            # 2. 第二次创建相同的流（应该触发现有流检查逻辑）
            normal_logger.info("=== 测试2: 第二次创建相同流（测试现有流逻辑） ===")
            stream2 = ZLMVideoStream(
                stream_id="test_stream_003",
                rtsp_url=test_rtsp_url,
                zlm_manager=self.zlm_manager
            )
            
            await stream2.start()  # 这应该触发现有流检查和处理逻辑
            await asyncio.sleep(2)
            
            # 3. 检查两个流的状态
            normal_logger.info("=== 测试3: 检查流状态 ===")
            stream1_info = await self.zlm_manager.get_stream_info(stream1.stream_id)
            stream2_info = await self.zlm_manager.get_stream_info(stream2.stream_id)
            
            normal_logger.info(f"Stream1状态: {stream1_info}")
            normal_logger.info(f"Stream2状态: {stream2_info}")
            
            # 停止流
            await stream1.stop()
            await stream2.stop()
            
        except Exception as e:
            exception_logger.exception(f"测试现有流处理时出错: {str(e)}")
        
        normal_logger.info("现有流处理测试完成")
    
    async def test_comprehensive_scenarios(self):
        """综合场景测试"""
        normal_logger.info("开始综合场景测试")
        
        try:
            # 1. 批量创建流
            normal_logger.info("=== 场景1: 批量创建流 ===")
            streams = []
            for i in range(3):
                stream = ZLMVideoStream(
                    stream_id=f"batch_test_stream_{i:03d}",
                    rtsp_url=f"rtsp://admin:password@192.168.1.{100+i}:554/stream{i}",
                    zlm_manager=self.zlm_manager
                )
                streams.append(stream)
            
            # 并行启动所有流
            start_tasks = [stream.start() for stream in streams]
            await asyncio.gather(*start_tasks, return_exceptions=True)
            
            await asyncio.sleep(3)
            
            # 2. 检查所有流的健康状态
            normal_logger.info("=== 场景2: 批量健康检查 ===")
            for stream in streams:
                try:
                    stream_info = await self.zlm_manager.get_stream_info(stream.stream_id)
                    if stream_info:
                        health = await self.zlm_manager.check_stream_health(
                            vhost=stream_info.get("vhost", "__defaultVhost__"),
                            app=stream_info.get("app", "live"),
                            stream_name=stream_info.get("stream", ""),
                            original_url=stream.rtsp_url
                        )
                        normal_logger.info(f"流 {stream.stream_id} 健康状态: {health['healthy']}")
                except Exception as e:
                    normal_logger.error(f"检查流 {stream.stream_id} 健康状态失败: {str(e)}")
            
            # 3. 获取所有流列表
            normal_logger.info("=== 场景3: 获取所有流列表 ===")
            all_streams = await self.zlm_manager.get_all_streams()
            normal_logger.info(f"当前总共有 {len(all_streams)} 个流")
            
            # 4. 批量停止流
            normal_logger.info("=== 场景4: 批量停止流 ===")
            stop_tasks = [stream.stop() for stream in streams]
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            
        except Exception as e:
            exception_logger.exception(f"综合场景测试出错: {str(e)}")
        
        normal_logger.info("综合场景测试完成")

async def main():
    """主测试函数"""
    # 配置日志级别
    logging.getLogger().setLevel(logging.INFO)
    
    normal_logger.info("开始流健康检查和管理逻辑测试")
    
    tester = StreamHealthTester()
    
    try:
        # 运行所有测试
        await tester.test_stream_health_check()
        await asyncio.sleep(2)
        
        await tester.test_existing_stream_handling()
        await asyncio.sleep(2)
        
        await tester.test_comprehensive_scenarios()
        
    except Exception as e:
        exception_logger.exception(f"测试过程中出现异常: {str(e)}")
    
    normal_logger.info("所有测试完成")

if __name__ == "__main__":
    asyncio.run(main()) 