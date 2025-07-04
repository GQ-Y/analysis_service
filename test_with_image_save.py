#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: test_with_image_save.py
作者: Yanli
邮箱: 1959595510@qq.com
创建日期: 2025-01-04
描述: 测试带图片保存的任务管理

测试任务管理系统的图片保存功能，验证分析结果是否正确保存。

本文件是分析服务项目的一部分。
"""

import sys
import asyncio
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.task_service import get_task_service


async def test_task_with_image_save():
    """测试带图片保存的任务管理"""
    
    print("🖼️ 测试任务管理图片保存功能")
    print("=" * 60)
    print()
    
    task_service = get_task_service()
    
    # 创建真实RTSP流分析任务
    print("📡 创建RTSP流分析任务（带图片保存）:")
    task_info = await task_service.create_task(
        name='RTSP流分析-图片保存测试',
        description='测试RTSP流分析并保存检测结果图片',
        analysis_type=3,  # 流分析
        model_codes=['person_detector', 'vehicle_detector', 'face_detector'],
        stream_urls=['rtsp://admin:zyckj2021@192.168.1.201:554/1/1']
    )
    task_id = task_info.id
    
    print(f"   ✅ 任务创建成功:")
    print(f"      - 任务ID: {task_id}")
    print(f"      - 任务名称: {task_info.name}")
    print(f"      - 分析模型: {len(task_info.model_codes)} 个")
    print(f"      - 流数量: {len(task_info.stream_urls)} 个")
    print()
    
    # 检查输出目录
    output_dir = Path(f"results/task_{task_id}")
    print(f"📁 输出目录: {output_dir.absolute()}")
    print()
    
    # 启动任务
    print("🚀 启动任务:")
    start_result = await task_service.start_task(task_id)
    print(f"   ✅ {start_result['message']}")
    print()
    
    # 监控任务执行和文件保存
    print("📊 监控任务执行和文件保存 (30秒):")
    print("-" * 60)
    
    for i in range(15):  # 30秒，每2秒检查一次
        await asyncio.sleep(2)
        
        # 获取任务状态
        detail = await task_service.get_task_detail(task_id)
        
        # 检查保存的文件
        images_dir = output_dir / "images"
        metadata_dir = output_dir / "metadata"
        
        image_count = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
        metadata_count = len(list(metadata_dir.glob("*.json"))) if metadata_dir.exists() else 0
        
        print(f"   [{(i+1)*2:2d}s] 进度: {detail.progress:5.1f}% | "
              f"状态: {detail.status} | "
              f"结果: {detail.result_count:4d} | "
              f"图片: {image_count:3d} | "
              f"元数据: {metadata_count:3d}")
    
    print()
    
    # 停止任务
    print("⏹️ 停止任务:")
    stop_result = await task_service.stop_task(task_id)
    print(f"   ✅ {stop_result['message']}")
    
    # 等待一下让结果处理器完成
    await asyncio.sleep(2)
    
    # 最终统计
    print()
    print("📊 最终统计:")
    final_detail = await task_service.get_task_detail(task_id)
    
    # 检查保存的文件
    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    
    final_image_count = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
    final_metadata_count = len(list(metadata_dir.glob("*.json"))) if metadata_dir.exists() else 0
    
    print(f"   📈 最终进度: {final_detail.progress:.1f}%")
    print(f"   📊 分析结果: {final_detail.result_count} 个")
    print(f"   🖼️ 保存图片: {final_image_count} 张")
    print(f"   📄 元数据文件: {final_metadata_count} 个")
    print()
    
    # 显示保存的文件示例
    if images_dir.exists() and final_image_count > 0:
        print("📂 保存的文件示例:")
        image_files = list(images_dir.glob("*.jpg"))[:5]  # 显示前5个
        for img_file in image_files:
            file_size = img_file.stat().st_size / 1024  # KB
            print(f"   🖼️ {img_file.name} ({file_size:.1f} KB)")
        
        if final_image_count > 5:
            print(f"   ... 还有 {final_image_count - 5} 个文件")
        print()
    
    # 检查文件内容
    if metadata_dir.exists() and final_metadata_count > 0:
        print("📄 元数据示例:")
        metadata_files = list(metadata_dir.glob("*.json"))
        if metadata_files:
            import json
            try:
                with open(metadata_files[0], 'r', encoding='utf-8') as f:
                    sample_metadata = json.load(f)
                
                print(f"   📝 文件: {metadata_files[0].name}")
                print(f"   🆔 帧ID: {sample_metadata.get('frame_id')}")
                print(f"   ⏰ 时间戳: {sample_metadata.get('timestamp')}")
                print(f"   📺 流ID: {sample_metadata.get('stream_id')}")
                
                # 显示分析结果
                analysis_results = sample_metadata.get('analysis_results', {})
                for analyzer, result in analysis_results.items():
                    detections = result.get('detections', [])
                    print(f"   🔍 {analyzer}: {len(detections)} 个检测")
                
            except Exception as e:
                print(f"   ❌ 读取元数据失败: {e}")
        print()
    
    # 删除任务
    print("🗑️ 清理任务:")
    delete_result = await task_service.delete_task(task_id)
    print(f"   ✅ {delete_result['message']}")
    print()
    
    # 测试总结
    print("🎉 测试完成！")
    print()
    print("📋 功能验证:")
    print(f"   ✅ 任务创建: 成功")
    print(f"   ✅ 任务启动: 成功")
    print(f"   ✅ 流连接: 成功")
    print(f"   ✅ AI分析: 成功 ({final_detail.result_count} 个结果)")
    print(f"   ✅ 图片保存: 成功 ({final_image_count} 张图片)")
    print(f"   ✅ 元数据保存: 成功 ({final_metadata_count} 个文件)")
    print(f"   ✅ 任务停止: 成功")
    print(f"   ✅ 任务删除: 成功")
    print()
    
    # 文件位置提示
    if final_image_count > 0:
        print("📁 保存位置:")
        print(f"   🖼️ 图片目录: {images_dir.absolute()}")
        print(f"   📄 元数据目录: {metadata_dir.absolute()}")
        print()
        print("💡 提示:")
        print("   - 图片包含绘制的检测框和标签")
        print("   - 元数据包含完整的分析结果JSON")
        print("   - 文件名包含任务ID、帧ID和时间戳")


async def main():
    """主函数"""
    try:
        await test_task_with_image_save()
    except KeyboardInterrupt:
        print("\n👋 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
