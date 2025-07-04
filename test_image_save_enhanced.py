#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: test_image_save_enhanced.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 增强的图片保存测试

测试零拷贝AI分析引擎的图片保存功能，确保每个分析结果都保存图片。

本文件是分析服务项目的一部分。
"""

import sys
import asyncio
import time
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.zero_copy import (
    MemoryPool, TimeAxis, MockAnalyzer, AnalysisWorker, 
    AnalysisEngine, ResultProcessor
)


async def test_enhanced_image_save():
    """测试增强的图片保存功能"""
    
    print("🖼️ 增强图片保存功能测试")
    print("=" * 60)
    print()
    
    # 1. 创建组件
    print("🔧 创建系统组件...")
    
    # 内存池
    memory_pool = MemoryPool(pool_size=20, height=480, width=640, channels=3)
    
    # 时间轴
    time_axis = TimeAxis(timeout_seconds=1.0, max_frames=50)
    
    # 结果处理器
    result_processor = ResultProcessor(
        output_dir="results/enhanced_test",
        save_images=True,
        save_metadata=True,
        draw_boxes=True
    )
    result_processor.start()
    
    # 分析引擎
    analysis_engine = AnalysisEngine()
    
    # 添加分析器
    analyzer = MockAnalyzer("enhanced_detector", 0.02)
    worker = analysis_engine.add_worker(
        name="enhanced_worker",
        analyzer=analyzer,
        time_axis=time_axis,
        batch_size=1,  # 单帧处理确保每帧都被处理
        timeout=0.1
    )
    
    # 添加结果回调
    def result_callback(frames, results):
        for frame_buffer in frames:
            result_processor.process_result(frame_buffer, task_id=999)
    
    worker.add_result_callback(result_callback)
    
    print("✅ 系统组件创建完成")
    print()
    
    # 2. 启动系统
    print("🚀 启动分析系统...")
    analysis_engine.start_all()
    print("✅ 分析系统已启动")
    print()
    
    # 3. 生产测试帧
    print("📥 生产测试帧...")
    
    frame_count = 20
    for i in range(frame_count):
        frame_id = i + 1
        timestamp = time.time()
        
        # 创建彩色测试图像
        frame_data = create_test_frame(640, 480, frame_id)
        
        # 存储到内存池
        buffer = memory_pool.put_frame(frame_data, frame_id, timestamp, "test_stream")
        
        if buffer:
            # 添加到时间轴
            if time_axis.add_frame(buffer):
                print(f"   📥 生产帧 #{frame_id}: 时间戳={timestamp:.3f}")
            else:
                buffer.release()
        
        await asyncio.sleep(0.1)  # 100ms间隔
    
    print(f"✅ 已生产 {frame_count} 帧")
    print()
    
    # 4. 等待处理完成
    print("⏳ 等待分析处理完成...")
    
    for i in range(10):  # 等待10秒
        await asyncio.sleep(1)
        
        # 检查处理进度
        analysis_stats = analysis_engine.get_all_stats()
        result_stats = result_processor.get_stats()
        
        processed = analysis_stats.get("enhanced_worker", {}).get("total_processed", 0)
        saved_images = result_stats.get("images_saved", 0)
        
        print(f"   [{i+1:2d}s] 已处理: {processed:2d} 帧 | 已保存: {saved_images:2d} 图片")
        
        if processed >= frame_count:
            break
    
    print()
    
    # 5. 停止系统
    print("⏹️ 停止系统...")
    analysis_engine.stop_all()
    result_processor.stop()
    memory_pool.cleanup()
    print("✅ 系统已停止")
    print()
    
    # 6. 检查结果
    print("📊 检查保存结果...")
    
    output_dir = Path("results/enhanced_test")
    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    
    image_count = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
    metadata_count = len(list(metadata_dir.glob("*.json"))) if metadata_dir.exists() else 0
    
    print(f"   🖼️ 保存图片: {image_count} 张")
    print(f"   📄 元数据文件: {metadata_count} 个")
    print()
    
    # 7. 显示文件列表
    if image_count > 0:
        print("📂 保存的图片文件:")
        image_files = sorted(list(images_dir.glob("*.jpg")))
        for i, img_file in enumerate(image_files[:10], 1):  # 显示前10个
            file_size = img_file.stat().st_size / 1024  # KB
            print(f"   {i:2d}. {img_file.name} ({file_size:.1f} KB)")
        
        if image_count > 10:
            print(f"   ... 还有 {image_count - 10} 个文件")
        print()
    
    # 8. 最终统计
    final_analysis_stats = analysis_engine.get_all_stats()
    final_result_stats = result_processor.get_stats()
    
    print("📊 最终统计:")
    print(f"   📥 生产帧数: {frame_count}")
    print(f"   🔍 分析帧数: {final_analysis_stats.get('enhanced_worker', {}).get('total_processed', 0)}")
    print(f"   🖼️ 保存图片: {final_result_stats.get('images_saved', 0)}")
    print(f"   📄 保存元数据: {final_result_stats.get('metadata_saved', 0)}")
    print(f"   ❌ 处理错误: {final_result_stats.get('processing_errors', 0)}")
    print()
    
    # 9. 验证结果
    success_rate = (image_count / frame_count) * 100 if frame_count > 0 else 0
    
    print("🎯 测试结果:")
    if success_rate >= 90:
        print(f"   ✅ 图片保存成功率: {success_rate:.1f}% (优秀)")
    elif success_rate >= 70:
        print(f"   ⚠️ 图片保存成功率: {success_rate:.1f}% (良好)")
    else:
        print(f"   ❌ 图片保存成功率: {success_rate:.1f}% (需要改进)")
    
    print()
    print("📁 文件位置:")
    print(f"   🖼️ 图片: {images_dir.absolute()}")
    print(f"   📄 元数据: {metadata_dir.absolute()}")
    print()
    print("🎉 增强图片保存测试完成！")


def create_test_frame(width: int, height: int, frame_id: int) -> np.ndarray:
    """创建测试帧"""
    # 创建彩色背景
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 设置渐变背景
    for y in range(height):
        for x in range(width):
            frame[y, x] = [
                int(255 * x / width),           # 红色渐变
                int(255 * y / height),          # 绿色渐变
                int(255 * (frame_id % 10) / 10) # 蓝色基于帧ID
            ]
    
    # 添加一些几何图形作为"检测目标"
    import cv2
    
    # 绘制矩形
    cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), 2)
    
    # 绘制圆形
    cv2.circle(frame, (width//2, height//2), 30, (0, 255, 255), -1)
    
    # 添加文字
    cv2.putText(frame, f"Frame {frame_id}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame


async def main():
    """主函数"""
    try:
        await test_enhanced_image_save()
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
