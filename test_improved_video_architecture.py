#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试改进后的视频缓存和回放架构
"""

import sys
import os
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from app.core.result_processing.video_cache_processor import VideoCacheProcessor
from app.core.result_processing.video_processor import VideoProcessor
from app.core.result_processing.result_pipeline import ResultProcessingPipeline
from app.core.zero_copy.frame_buffer import FrameBuffer
from app.core.zero_copy.memory_pool import MemoryPool
from app.core.memory.time_axis_manager import TimeAxisManager
from app.core.memory.time_axis import TimeAxis
from app.models.analysis_result import AnalysisResult, Detection
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_frame_buffer(frame_id: int, timestamp: float, stream_id: str = "test_stream") -> FrameBuffer:
    """创建模拟的帧缓冲区"""
    # 创建内存池
    memory_pool = MemoryPool(pool_size=10, height=720, width=1280)
    
    # 创建模拟图像数据
    image_data = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # 获取帧缓冲区
    frame_buffer = memory_pool.put_frame(image_data, frame_id, timestamp, stream_id)
    
    return frame_buffer


def create_mock_analysis_result(frame_id: int, timestamp: float, has_detection: bool = True) -> AnalysisResult:
    """创建模拟的分析结果"""
    detections = []
    
    if has_detection:
        # 创建一个模拟检测结果
        detection = Detection(
            class_id=1,
            class_name="person",
            confidence=0.85,
            bbox=[100, 100, 200, 200],
            area=10000
        )
        detections.append(detection)
    
    return AnalysisResult(
        frame_id=frame_id,
        timestamp=timestamp,
        model_name="test_model",
        image_shape=(720, 1280, 3),
        detections=detections,
        inference_time=0.1
    )


async def test_video_cache_processor():
    """测试视频缓存处理器"""
    logger.info("🧪 测试视频缓存处理器...")
    
    # 创建视频缓存处理器
    video_cache_processor = VideoCacheProcessor(
        cache_duration=10,  # 10秒缓存
        fps=5,              # 5fps
        analysis_type=3,    # 流分析
        logger=logger
    )
    
    # 测试启动前的状态
    assert not video_cache_processor.enabled
    assert not video_cache_processor.is_started
    
    # 启动缓存
    video_cache_processor.start_caching()
    
    # 测试启动后的状态
    assert video_cache_processor.enabled
    assert video_cache_processor.is_started
    
    # 模拟添加帧到缓存
    logger.info("📹 模拟添加帧到缓存...")
    base_time = time.time()
    
    for i in range(20):
        frame_buffer = create_mock_frame_buffer(i, base_time + i * 0.2)
        result = video_cache_processor.process(frame_buffer)
        assert result is True
    
    # 测试获取回放帧
    logger.info("🎬 测试获取回放帧...")
    center_time = base_time + 2.0  # 中心时间
    duration = 2.0  # 2秒回放
    
    frames = video_cache_processor.get_frames_for_playback(
        center_timestamp=center_time,
        duration=duration
    )
    
    logger.info(f"✅ 获取到 {len(frames)} 帧用于回放")
    assert len(frames) > 0
    
    # 测试统计信息
    stats = video_cache_processor.get_stats()
    logger.info(f"📊 缓存统计: {stats}")
    
    logger.info("✅ 视频缓存处理器测试通过")


async def test_video_processor_with_cache():
    """测试视频回放处理器与缓存处理器的集成"""
    logger.info("🧪 测试视频回放处理器与缓存处理器集成...")
    
    # 创建时间轴管理器
    time_axis = TimeAxis(timeout_seconds=1.0, max_frames=100, logger=logger)
    time_axis_manager = TimeAxisManager(time_axis, logger)
    
    # 创建视频缓存处理器
    video_cache_processor = VideoCacheProcessor(
        cache_duration=10,
        fps=5,
        analysis_type=3,
        logger=logger
    )
    video_cache_processor.start_caching()
    
    # 创建视频回放处理器
    video_processor = VideoProcessor(
        output_dir="test_output",
        playback_duration=4,  # 4秒回放
        time_axis_manager=time_axis_manager,
        analysis_type=3,
        video_cache_processor=video_cache_processor,
        logger=logger
    )
    
    # 模拟添加帧到缓存
    logger.info("📹 模拟添加帧到缓存...")
    base_time = time.time()
    
    for i in range(30):
        frame_buffer = create_mock_frame_buffer(i, base_time + i * 0.2)
        video_cache_processor.process(frame_buffer)
    
    # 创建带检测结果的帧
    detection_time = base_time + 3.0
    frame_buffer = create_mock_frame_buffer(15, detection_time)
    analysis_result = create_mock_analysis_result(15, detection_time, has_detection=True)
    
    # 将分析结果附加到帧缓冲区
    frame_buffer.analysis_result = analysis_result
    
    # 测试视频回放处理器
    logger.info("🎬 测试视频回放处理器...")
    try:
        await video_processor.process_result(analysis_result)
        logger.info("✅ 视频回放处理器测试通过")
    except Exception as e:
        logger.error(f"❌ 视频回放处理器测试失败: {e}")


async def test_result_pipeline_integration():
    """测试结果处理管道集成"""
    logger.info("🧪 测试结果处理管道集成...")
    
    # 创建时间轴管理器
    time_axis = TimeAxis(timeout_seconds=1.0, max_frames=100, logger=logger)
    time_axis_manager = TimeAxisManager(time_axis, logger)
    
    # 创建视频缓存处理器
    video_cache_processor = VideoCacheProcessor(
        cache_duration=10,
        fps=5,
        analysis_type=3,
        logger=logger
    )
    
    # 创建任务配置
    task_config = {
        "task_id": 1,
        "analysis_type": 3,  # 流分析
        "playback_duration": 6,  # 6秒回放
        "callback_url": None,
        "save_result": True
    }
    
    # 创建结果处理管道
    result_pipeline = ResultProcessingPipeline(
        task_id=1,
        task_config=task_config,
        time_axis_manager=time_axis_manager,
        output_dir="test_output",
        video_cache_processor=video_cache_processor,
        logger=logger
    )
    
    # 启动管道
    await result_pipeline.start()
    
    # 模拟拉流成功，启动视频缓存
    logger.info("📡 模拟拉流成功，启动视频缓存...")
    video_cache_processor.start_caching()
    
    # 模拟帧处理
    logger.info("📹 模拟帧处理...")
    base_time = time.time()
    
    for i in range(40):
        frame_buffer = create_mock_frame_buffer(i, base_time + i * 0.2)
        
        # 每10帧添加一个检测结果
        if i % 10 == 5:
            analysis_result = create_mock_analysis_result(i, base_time + i * 0.2, has_detection=True)
            frame_buffer.analysis_result = analysis_result
            logger.info(f"🎯 帧 {i} 包含检测结果")
        
        # 处理帧
        result_pipeline.process_result(frame_buffer, task_id=1)
    
    # 停止管道
    await result_pipeline.stop()
    
    logger.info("✅ 结果处理管道集成测试通过")


async def test_time_calculation_fix():
    """测试时间计算修复"""
    logger.info("🧪 测试时间计算修复...")
    
    # 创建视频缓存处理器
    video_cache_processor = VideoCacheProcessor(
        cache_duration=20,
        fps=10,
        analysis_type=3,
        logger=logger
    )
    video_cache_processor.start_caching()
    
    # 模拟连续帧
    base_time = time.time()
    logger.info(f"📅 基准时间: {base_time}")
    
    for i in range(100):
        frame_time = base_time + i * 0.1  # 每帧间隔0.1秒
        frame_buffer = create_mock_frame_buffer(i, frame_time)
        video_cache_processor.process(frame_buffer)
    
    # 测试以中心时间获取回放帧
    center_time = base_time + 5.0  # 第50帧的时间
    duration = 2.0  # 2秒回放
    
    logger.info(f"🎯 测试中心时间: {center_time}")
    logger.info(f"🎬 回放时长: {duration}秒")
    logger.info(f"📊 预期时间窗口: [{center_time - duration/2:.3f}, {center_time + duration/2:.3f}]")
    
    frames = video_cache_processor.get_frames_for_playback(
        center_timestamp=center_time,
        duration=duration
    )
    
    if frames:
        first_frame_time = frames[0]["timestamp"]
        last_frame_time = frames[-1]["timestamp"]
        actual_duration = last_frame_time - first_frame_time
        
        logger.info(f"✅ 获取到 {len(frames)} 帧")
        logger.info(f"📊 实际时间窗口: [{first_frame_time:.3f}, {last_frame_time:.3f}]")
        logger.info(f"📊 实际时长: {actual_duration:.3f}秒")
        
        # 验证时间窗口是否正确
        expected_start = center_time - duration / 2
        expected_end = center_time + duration / 2
        
        tolerance = 0.2  # 允许0.2秒的误差
        assert abs(first_frame_time - expected_start) < tolerance, f"开始时间偏差过大: {first_frame_time} vs {expected_start}"
        assert abs(last_frame_time - expected_end) < tolerance, f"结束时间偏差过大: {last_frame_time} vs {expected_end}"
        
        logger.info("✅ 时间计算修复验证通过")
    else:
        logger.error("❌ 未获取到回放帧")


async def main():
    """主测试函数"""
    logger.info("🚀 开始测试改进后的视频缓存和回放架构...")
    
    try:
        # 测试1: 视频缓存处理器
        await test_video_cache_processor()
        
        # 测试2: 视频回放处理器与缓存集成
        await test_video_processor_with_cache()
        
        # 测试3: 结果处理管道集成
        await test_result_pipeline_integration()
        
        # 测试4: 时间计算修复
        await test_time_calculation_fix()
        
        logger.info("🎉 所有测试通过！")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 