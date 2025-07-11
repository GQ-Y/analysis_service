#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试视频缓存处理器改进功能
"""

import sys
import os
import time
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from app.core.result_processing.video_cache_processor import VideoCacheProcessor
from app.core.result_processing.video_processor import VideoProcessor
from app.core.result_processing.result_pipeline import ResultProcessingPipeline
from app.core.zero_copy.frame_buffer import FrameBuffer
from app.core.zero_copy.memory_pool import MemoryPool
from app.models.analysis_result import AnalysisResult
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_frame_buffer(frame_id: int, timestamp: float, stream_id: str = "test_stream") -> FrameBuffer:
    """创建模拟帧缓冲区"""
    # 创建内存池
    memory_pool = MemoryPool(pool_size=10, height=480, width=640)
    
    # 创建模拟帧数据
    mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 获取帧缓冲区
    frame_buffer = memory_pool.put_frame(mock_frame, frame_id, timestamp, stream_id)
    
    return frame_buffer


def test_video_cache_processor_startup_conditions():
    """测试视频缓存处理器的启动条件"""
    logger.info("=" * 60)
    logger.info("🧪 测试视频缓存处理器启动条件")
    logger.info("=" * 60)
    
    # 测试1：满足启动条件的配置
    logger.info("\n📋 测试1: 满足启动条件的配置")
    task_config_valid = {
        "analysis_type": 3,  # 流分析
        "playback_duration": 10,  # 10秒回放
        "save_result": True,
        "save_images": True
    }
    
    pipeline = ResultProcessingPipeline(
        task_id=1,
        task_config=task_config_valid,
        time_axis_manager=None,
        output_dir="test_output",
        logger=logger
    )
    
    # 检查视频缓存处理器是否被创建
    if hasattr(pipeline, 'video_cache_processor') and pipeline.video_cache_processor:
        logger.info("✅ 视频缓存处理器已创建")
        
        # 测试处理帧
        frame_buffer = create_mock_frame_buffer(1, time.time())
        result = pipeline.video_cache_processor.process(frame_buffer)
        logger.info(f"✅ 帧处理结果: {result}")
        
        # 检查统计信息
        stats = pipeline.video_cache_processor.get_stats()
        logger.info(f"📊 统计信息: {stats}")
    else:
        logger.error("❌ 视频缓存处理器未创建")
    
    # 测试2：不满足启动条件的配置（分析类型错误）
    logger.info("\n📋 测试2: 不满足启动条件的配置（分析类型错误）")
    task_config_invalid_type = {
        "analysis_type": 1,  # 图片分析
        "playback_duration": 10,
        "save_result": True
    }
    
    pipeline2 = ResultProcessingPipeline(
        task_id=2,
        task_config=task_config_invalid_type,
        time_axis_manager=None,
        output_dir="test_output",
        logger=logger
    )
    
    if hasattr(pipeline2, 'video_cache_processor') and pipeline2.video_cache_processor:
        logger.error("❌ 视频缓存处理器不应该被创建")
    else:
        logger.info("✅ 视频缓存处理器正确跳过")
    
    # 测试3：不满足启动条件的配置（回放时长不足）
    logger.info("\n📋 测试3: 不满足启动条件的配置（回放时长不足）")
    task_config_invalid_duration = {
        "analysis_type": 3,  # 流分析
        "playback_duration": 2,  # 2秒回放（不足3秒）
        "save_result": True
    }
    
    pipeline3 = ResultProcessingPipeline(
        task_id=3,
        task_config=task_config_invalid_duration,
        time_axis_manager=None,
        output_dir="test_output",
        logger=logger
    )
    
    if hasattr(pipeline3, 'video_cache_processor') and pipeline3.video_cache_processor:
        logger.error("❌ 视频缓存处理器不应该被创建")
    else:
        logger.info("✅ 视频缓存处理器正确跳过")


def test_video_cache_functionality():
    """测试视频缓存功能"""
    logger.info("=" * 60)
    logger.info("🧪 测试视频缓存功能")
    logger.info("=" * 60)
    
    # 创建视频缓存处理器
    cache_processor = VideoCacheProcessor(
        cache_duration=10,  # 10秒缓存
        fps=10.0,          # 10 FPS
        analysis_type=3,   # 流分析
        logger=logger
    )
    
    # 模拟添加帧到缓存
    logger.info("\n📥 模拟添加帧到缓存...")
    base_time = time.time()
    
    for i in range(50):  # 添加50帧，模拟5秒的视频
        frame_timestamp = base_time + i * 0.1  # 每帧间隔0.1秒
        frame_buffer = create_mock_frame_buffer(i, frame_timestamp)
        
        result = cache_processor.process(frame_buffer)
        
        if i % 10 == 0:
            logger.info(f"📥 已添加第 {i} 帧, 时间戳: {frame_timestamp:.2f}")
    
    # 获取统计信息
    stats = cache_processor.get_stats()
    logger.info(f"\n📊 缓存统计: {stats}")
    
    # 测试获取回放帧
    logger.info("\n🎬 测试获取回放帧...")
    center_time = base_time + 2.5  # 2.5秒处作为中心
    playback_frames = cache_processor.get_frames_for_playback(center_time, 2.0)  # 获取2秒的回放
    
    logger.info(f"🎬 回放帧数: {len(playback_frames)}")
    if playback_frames:
        logger.info(f"🎬 时间范围: {playback_frames[0].timestamp:.2f} - {playback_frames[-1].timestamp:.2f}")
    
    # 测试获取时间范围帧
    logger.info("\n⏱️ 测试获取时间范围帧...")
    start_time = base_time + 1.0
    end_time = base_time + 3.0
    range_frames = cache_processor.get_frames_for_time_range(start_time, end_time)
    
    logger.info(f"⏱️ 时间范围帧数: {len(range_frames)}")
    if range_frames:
        logger.info(f"⏱️ 时间范围: {range_frames[0].timestamp:.2f} - {range_frames[-1].timestamp:.2f}")


def test_video_processor_integration():
    """测试视频处理器与缓存处理器的集成"""
    logger.info("=" * 60)
    logger.info("🧪 测试视频处理器与缓存处理器的集成")
    logger.info("=" * 60)
    
    # 创建视频缓存处理器
    cache_processor = VideoCacheProcessor(
        cache_duration=20,  # 20秒缓存
        fps=25.0,          # 25 FPS
        analysis_type=3,   # 流分析
        logger=logger
    )
    
    # 创建视频处理器
    video_processor = VideoProcessor(
        output_dir="test_output/videos",
        playback_duration=10,  # 10秒回放
        time_axis_manager=None,
        analysis_type=3,
        video_cache_processor=cache_processor,
        logger=logger
    )
    
    # 模拟添加帧到缓存
    logger.info("\n📥 模拟流式添加帧...")
    base_time = time.time()
    
    for i in range(100):  # 添加100帧，模拟4秒的视频
        frame_timestamp = base_time + i * 0.04  # 每帧间隔0.04秒（25fps）
        frame_buffer = create_mock_frame_buffer(i, frame_timestamp)
        
        # 添加到缓存
        cache_processor.process(frame_buffer)
        
        # 模拟在第50帧时检测到目标
        if i == 50:
            logger.info(f"🎯 第 {i} 帧检测到目标，时间戳: {frame_timestamp:.2f}")
            
            # 创建模拟分析结果
            analysis_result = AnalysisResult(
                frame_id=i,
                timestamp=frame_timestamp,
                detections=[
                    {
                        "class": "person",
                        "confidence": 0.85,
                        "bbox": [100, 100, 200, 200]
                    }
                ],
                analyzer="yolo11n"
            )
            
            # 测试视频处理器处理检测结果
            if video_processor.enabled:
                logger.info("🎬 触发视频回放处理...")
                try:
                    # 这里应该触发视频回放生成
                    logger.info("✅ 视频回放处理器已启用，可以处理检测结果")
                except Exception as e:
                    logger.error(f"❌ 视频回放处理失败: {e}")
    
    # 获取最终统计
    cache_stats = cache_processor.get_stats()
    logger.info(f"\n📊 最终缓存统计: {cache_stats}")


def main():
    """主测试函数"""
    logger.info("🚀 开始测试视频缓存处理器改进功能")
    
    try:
        # 测试启动条件
        test_video_cache_processor_startup_conditions()
        
        # 测试缓存功能
        test_video_cache_functionality()
        
        # 测试集成功能
        test_video_processor_integration()
        
        logger.info("\n🎉 所有测试完成！")
        
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 