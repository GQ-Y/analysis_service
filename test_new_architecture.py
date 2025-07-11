#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试新的视频缓存和回放架构
"""

import asyncio
import time
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

async def test_video_cache_service():
    """测试视频缓存服务"""
    logger.info("🧪 开始测试视频缓存服务...")
    
    try:
        from app.services.video_cache_service import VideoCacheService
        
        # 使用示例RTSP流或本地视频文件
        test_url = "rtsp://admin:admin123@192.168.1.100:554/stream1"  # 替换为您的实际RTSP地址
        # 或者使用本地视频文件进行测试
        # test_url = "./test_video.mp4"
        
        cache_service = VideoCacheService(
            stream_url=test_url,
            cache_dir="./storage/test_cache",
            cache_duration=60,  # 1分钟缓存
            target_fps=25.0,
            logger=logger
        )
        
        # 启动缓存服务
        if cache_service.start():
            logger.info("✅ 视频缓存服务启动成功")
            
            # 等待一段时间让服务缓存一些帧
            logger.info("⏳ 等待30秒让服务缓存帧数据...")
            await asyncio.sleep(30)
            
            # 检查缓存状态
            stats = cache_service.get_cache_stats()
            logger.info(f"📊 缓存统计: {stats}")
            
            # 测试查询功能
            current_time = time.time()
            test_range_start = current_time - 10
            test_range_end = current_time
            
            frames = cache_service.get_frames_in_range(test_range_start, test_range_end)
            logger.info(f"🔍 查询测试: 获取到{len(frames)}帧 (时间范围: {test_range_start:.2f}-{test_range_end:.2f})")
            
            # 停止服务
            cache_service.stop()
            logger.info("🛑 视频缓存服务已停止")
        else:
            logger.error("❌ 视频缓存服务启动失败")
            
    except Exception as e:
        logger.error(f"❌ 测试视频缓存服务失败: {e}")
        import traceback
        traceback.print_exc()

async def test_video_playback_service():
    """测试视频回放服务"""
    logger.info("🧪 开始测试视频回放服务...")
    
    try:
        from app.services.video_cache_service import VideoCacheService
        from app.services.video_playback_service import VideoPlaybackService
        
        # 创建视频缓存服务
        test_url = "rtsp://admin:admin123@192.168.1.100:554/stream1"  # 替换为您的实际RTSP地址
        
        cache_service = VideoCacheService(
            stream_url=test_url,
            cache_dir="./storage/test_cache",
            cache_duration=60,
            target_fps=25.0,
            logger=logger
        )
        
        # 创建视频回放服务
        playback_service = VideoPlaybackService(
            video_cache_service=cache_service,
            output_dir="./storage/test_playback",
            playback_duration=10.0,
            fps=25.0,
            logger=logger
        )
        
        # 启动服务
        if cache_service.start() and playback_service.start():
            logger.info("✅ 视频缓存和回放服务启动成功")
            
            # 等待缓存一些帧
            logger.info("⏳ 等待20秒让服务缓存帧数据...")
            await asyncio.sleep(20)
            
            # 模拟分析结果
            analysis_result = {
                "frame_id": 12345,
                "timestamp": time.time(),
                "detections": [
                    {
                        "class_name": "person",
                        "confidence": 0.85,
                        "bbox": [100, 100, 200, 300]
                    },
                    {
                        "class_name": "car",
                        "confidence": 0.92,
                        "bbox": [300, 150, 500, 400]
                    }
                ],
                "model_name": "yolo11n",
                "confidence": 0.85,
                "stream_id": "test_stream"
            }
            
            # 添加分析结果到回放队列
            playback_service.add_analysis_result(analysis_result)
            logger.info("📋 已添加测试分析结果到回放队列")
            
            # 等待视频生成
            logger.info("⏳ 等待15秒让服务生成回放视频...")
            await asyncio.sleep(15)
            
            # 检查输出目录
            output_dir = Path("./storage/test_playback")
            if output_dir.exists():
                video_files = list(output_dir.glob("*.mp4"))
                logger.info(f"🎬 生成的视频文件: {len(video_files)}个")
                for video_file in video_files:
                    file_size = video_file.stat().st_size / (1024 * 1024)  # MB
                    logger.info(f"   - {video_file.name}: {file_size:.2f}MB")
            else:
                logger.warning("⚠️ 输出目录不存在")
            
            # 停止服务
            playback_service.stop()
            cache_service.stop()
            logger.info("🛑 视频缓存和回放服务已停止")
        else:
            logger.error("❌ 服务启动失败")
            
    except Exception as e:
        logger.error(f"❌ 测试视频回放服务失败: {e}")
        import traceback
        traceback.print_exc()

async def test_integration():
    """集成测试"""
    logger.info("🧪 开始集成测试...")
    
    # 创建测试目录
    Path("./storage/test_cache").mkdir(parents=True, exist_ok=True)
    Path("./storage/test_playback").mkdir(parents=True, exist_ok=True)
    
    try:
        # 先测试视频缓存服务
        await test_video_cache_service()
        
        logger.info("\n" + "="*50 + "\n")
        
        # 再测试视频回放服务
        await test_video_playback_service()
        
        logger.info("\n🎉 集成测试完成")
        
    except Exception as e:
        logger.error(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("�� 开始测试新的视频缓存和回放架构")
    logger.info("请确保：")
    logger.info("1. 已配置正确的RTSP流地址")
    logger.info("2. 网络连接正常")
    logger.info("3. 有足够的磁盘空间")
    
    # 运行测试
    asyncio.run(test_integration()) 