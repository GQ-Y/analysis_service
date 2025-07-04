#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: test_real_stream.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 真实RTSP流测试

使用真实的RTSP流地址测试零拷贝AI分析引擎的实际性能。

本文件是分析服务项目的一部分。
"""

import sys
import asyncio
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.zero_copy import (
    MemoryPool, TimeAxis, StreamCapture,
    MockAnalyzer, AnalysisWorker, AnalysisEngine
)


async def test_real_rtsp_stream():
    """测试真实RTSP流"""
    
    print("🎥 真实RTSP流零拷贝AI分析测试")
    print("=" * 60)
    print()
    
    # RTSP流地址
    rtsp_url = "rtsp://admin:zyckj2021@192.168.1.201:554/1/1"
    print(f"📡 RTSP流地址: {rtsp_url}")
    print()
    
    # 1. 创建内存池
    print("🔧 创建内存池...")
    memory_pool = MemoryPool(
        pool_size=30,  # 增加到30个缓冲区
        height=1080,
        width=1920,
        channels=3
    )
    print(f"✅ 内存池: {memory_pool.pool_size} 个缓冲区, {memory_pool.get_total_memory_mb():.1f}MB")
    print()
    
    # 2. 创建时间轴
    print("⏰ 创建时间轴...")
    time_axis = TimeAxis(
        timeout_seconds=3.0,  # 3秒超时
        max_frames=100,       # 最大100帧
    )
    print("✅ 时间轴创建完成")
    print()
    
    # 3. 创建流捕获器
    print("📹 创建RTSP流捕获器...")
    stream_capture = StreamCapture(
        stream_url=rtsp_url,
        memory_pool=memory_pool,
        time_axis=time_axis,
        stream_id="rtsp_camera",
        reconnect_delay=3.0
    )
    print("✅ 流捕获器创建完成")
    print()
    
    # 4. 创建AI分析引擎
    print("🤖 创建AI分析引擎...")
    analysis_engine = AnalysisEngine()
    
    # 添加多个分析器模拟真实场景
    analyzers = [
        ("person_detector", MockAnalyzer("person_detector", 0.03)),  # 30ms
        ("vehicle_detector", MockAnalyzer("vehicle_detector", 0.04)), # 40ms
        ("face_detector", MockAnalyzer("face_detector", 0.02)),      # 20ms
    ]
    
    for name, analyzer in analyzers:
        worker = analysis_engine.add_worker(
            name=f"worker_{name}",
            analyzer=analyzer,
            time_axis=time_axis,
            batch_size=3,  # 批处理3帧
            timeout=0.2
        )
        
        # 添加结果回调
        worker.add_result_callback(
            lambda frames, results, analyzer_name=name: log_analysis_result(analyzer_name, frames, results)
        )
        
        print(f"   ✅ 添加分析器: {name}")
    
    print()
    
    # 5. 启动系统
    print("🚀 启动零拷贝AI分析系统...")
    
    try:
        # 启动分析引擎
        analysis_engine.start_all()
        print("   ✅ 分析引擎已启动")
        
        # 启动流捕获
        stream_capture.start()
        print("   ✅ 流捕获已启动")
        print()
        
        # 6. 监控系统运行
        print("📊 系统监控 (运行30秒)...")
        print("-" * 50)
        
        start_time = time.time()
        last_log_time = start_time
        
        while time.time() - start_time < 30:  # 运行30秒
            current_time = time.time()
            
            # 每5秒输出一次详细统计
            if current_time - last_log_time >= 5.0:
                print_detailed_stats(memory_pool, time_axis, analysis_engine, stream_capture)
                last_log_time = current_time
            
            await asyncio.sleep(1)
        
        print()
        print("⏹️ 测试时间结束，正在停止系统...")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断，正在停止系统...")
    
    finally:
        # 7. 清理资源
        print("🧹 清理系统资源...")
        
        # 停止流捕获
        stream_capture.stop()
        print("   ✅ 流捕获已停止")
        
        # 停止分析引擎
        analysis_engine.stop_all()
        print("   ✅ 分析引擎已停止")
        
        # 清理内存池
        memory_pool.cleanup()
        print("   ✅ 内存池已清理")
        
        print()
        print("📊 最终测试报告:")
        print_final_report(memory_pool, time_axis, analysis_engine, stream_capture)
        
        print()
        print("🎉 真实RTSP流测试完成！")


def log_analysis_result(analyzer_name: str, frame_buffers: list, results: list):
    """记录分析结果"""
    if frame_buffers and results:
        frame_ids = [fb.frame_id for fb in frame_buffers]
        detection_counts = [len(result.get("detections", [])) for result in results]
        print(f"🔍 {analyzer_name}: 帧{frame_ids} -> 检测{detection_counts}")


def print_detailed_stats(memory_pool, time_axis, analysis_engine, stream_capture):
    """打印详细统计信息"""
    print(f"\n📊 系统状态 [{time.strftime('%H:%M:%S')}]:")
    
    # 内存池统计
    memory_stats = memory_pool.get_stats()
    print(f"💾 内存池: 使用{memory_stats['current_usage']}/{memory_stats['pool_size']}, "
          f"成功率{memory_stats['success_rate']:.1f}%, "
          f"总请求{memory_stats['total_requests']}")
    
    # 时间轴统计
    time_axis_stats = time_axis.get_stats()
    print(f"⏰ 时间轴: 待处理{time_axis_stats['current_size']}帧, "
          f"已添加{time_axis_stats['total_added']}, "
          f"已获取{time_axis_stats['total_retrieved']}")
    
    # 流捕获统计
    capture_stats = stream_capture.get_stats()
    print(f"📹 流捕获: FPS{capture_stats['fps']:.1f}, "
          f"总帧{capture_stats['total_frames']}, "
          f"成功率{capture_stats['success_rate']:.1f}%, "
          f"重连{capture_stats['reconnect_count']}次")
    
    # 分析引擎统计
    analysis_stats = analysis_engine.get_all_stats()
    total_processed = 0
    for worker_name, stats in analysis_stats.items():
        processed = stats.get("total_processed", 0)
        fps = stats.get("fps", 0)
        avg_time = stats.get("avg_time_per_frame", 0) * 1000
        total_processed += processed
        print(f"🤖 {worker_name}: {processed}帧, FPS{fps:.1f}, 平均{avg_time:.1f}ms/帧")
    
    print(f"📈 总处理: {total_processed}帧")
    print("-" * 50)


def print_final_report(memory_pool, time_axis, analysis_engine, stream_capture):
    """打印最终测试报告"""
    print("=" * 60)
    
    # 内存池报告
    memory_stats = memory_pool.get_stats()
    print(f"💾 内存池性能:")
    print(f"   - 总请求: {memory_stats['total_requests']}")
    print(f"   - 成功率: {memory_stats['success_rate']:.2f}%")
    print(f"   - 峰值使用: {memory_stats['peak_usage']}/{memory_stats['pool_size']}")
    print(f"   - 总回收: {memory_stats['total_recycled']}")
    
    # 时间轴报告
    time_axis_stats = time_axis.get_stats()
    print(f"⏰ 时间轴性能:")
    print(f"   - 总添加: {time_axis_stats['total_added']}")
    print(f"   - 总获取: {time_axis_stats['total_retrieved']}")
    print(f"   - 超时帧: {time_axis_stats['total_timeout']}")
    print(f"   - 丢弃帧: {time_axis_stats['total_dropped']}")
    
    # 流捕获报告
    capture_stats = stream_capture.get_stats()
    runtime = capture_stats.get('runtime_seconds', 0)
    print(f"📹 流捕获性能:")
    print(f"   - 运行时间: {runtime:.1f}秒")
    print(f"   - 总帧数: {capture_stats['total_frames']}")
    print(f"   - 平均FPS: {capture_stats['avg_fps']:.2f}")
    print(f"   - 实时FPS: {capture_stats['fps']:.2f}")
    print(f"   - 成功率: {capture_stats['success_rate']:.2f}%")
    print(f"   - 重连次数: {capture_stats['reconnect_count']}")
    
    # 分析引擎报告
    analysis_stats = analysis_engine.get_all_stats()
    print(f"🤖 AI分析性能:")
    total_processed = 0
    total_time = 0
    for worker_name, stats in analysis_stats.items():
        processed = stats.get("total_processed", 0)
        batches = stats.get("total_batches", 0)
        avg_time = stats.get("avg_time_per_frame", 0) * 1000
        fps = stats.get("fps", 0)
        total_processed += processed
        total_time += stats.get("total_time", 0)
        
        print(f"   - {worker_name}:")
        print(f"     * 处理帧数: {processed}")
        print(f"     * 批次数: {batches}")
        print(f"     * 平均延迟: {avg_time:.2f}ms/帧")
        print(f"     * 处理FPS: {fps:.2f}")
    
    print(f"📊 整体性能:")
    print(f"   - 总处理帧数: {total_processed}")
    print(f"   - 总处理时间: {total_time:.2f}秒")
    if total_processed > 0:
        print(f"   - 平均处理延迟: {(total_time/total_processed)*1000:.2f}ms/帧")
    
    # 性能评估
    print(f"🎯 性能评估:")
    if memory_stats['success_rate'] >= 95:
        print("   ✅ 内存管理: 优秀")
    elif memory_stats['success_rate'] >= 90:
        print("   ⚠️ 内存管理: 良好")
    else:
        print("   ❌ 内存管理: 需要优化")
    
    if capture_stats['success_rate'] >= 95:
        print("   ✅ 流捕获: 优秀")
    elif capture_stats['success_rate'] >= 90:
        print("   ⚠️ 流捕获: 良好")
    else:
        print("   ❌ 流捕获: 需要优化")
    
    if total_processed > 0 and (total_time/total_processed) < 0.1:
        print("   ✅ AI分析: 优秀")
    elif total_processed > 0 and (total_time/total_processed) < 0.2:
        print("   ⚠️ AI分析: 良好")
    else:
        print("   ❌ AI分析: 需要优化")


async def main():
    """主函数"""
    try:
        await test_real_rtsp_stream()
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
