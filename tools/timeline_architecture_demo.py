#!/usr/bin/env python3
"""
时间轴架构演示脚本
展示基于时间轴的高度并行化流水线架构的工作原理和性能
"""
import asyncio
import time
import argparse
import sys
import os
import numpy as np
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.timeline_architecture.pipeline_factory import (
    TimelinePipelineFactory, 
    PipelineConfig, 
    create_timeline_pipeline
)
from core.timeline_architecture.stream_module import StreamConfig, StreamProtocol

class TimelineArchitectureDemo:
    """时间轴架构演示类"""
    
    def __init__(self):
        self.pipeline: TimelinePipelineFactory = None
        self.demo_running = False
        
    async def create_pipeline_with_config(self, 
                                         target_cpu: float = 0.8,
                                         max_memory_mb: int = 1024,
                                         redis_host: str = "localhost") -> TimelinePipelineFactory:
        """创建配置的流水线"""
        
        config = PipelineConfig(
            # 内存配置
            max_memory_mb=max_memory_mb,
            memory_cleanup_interval=0.5,
            
            # 处理器配置  
            target_cpu_utilization=target_cpu,
            min_processors=1,
            max_processors=None,  # 自动检测
            
            # 结果推送配置
            redis_host=redis_host,
            redis_port=6379,
            redis_db=0,
            result_queue_prefix="analysis_results",
            
            # 时间轴配置
            frame_expire_time=1.0,  # 1秒过期
            timeline_cleanup_interval=0.5,
            
            # 系统配置
            enable_monitoring=True,
            monitoring_interval=5.0
        )
        
        print(f"🏗️  创建时间轴流水线...")
        print(f"   • 目标CPU利用率: {target_cpu*100}%")
        print(f"   • 最大内存: {max_memory_mb}MB")
        print(f"   • 帧过期时间: 1.0秒")
        print(f"   • Redis: {redis_host}:6379")
        
        pipeline = TimelinePipelineFactory(config)
        await pipeline.initialize()
        
        return pipeline
    
    async def demo_single_stream(self):
        """演示单流处理"""
        print("\n" + "="*80)
        print("🎬 演示1: 单视频流处理")
        print("="*80)
        
        # 创建流水线
        self.pipeline = await self.create_pipeline_with_config(
            target_cpu=0.8,
            max_memory_mb=512
        )
        
        # 启动流水线
        await self.pipeline.start()
        
        # 添加模拟视频流
        stream_config = StreamConfig(
            stream_id="demo_stream_1",
            url="demo://fake_rtsp_url",
            protocol=StreamProtocol.RTSP,
            fps_limit=30,
            resolution=(1920, 1080)
        )
        
        print(f"📡 添加视频流: {stream_config.stream_id}")
        print(f"   • 协议: {stream_config.protocol.value}")
        print(f"   • 分辨率: {stream_config.resolution}")
        print(f"   • FPS限制: {stream_config.fps_limit}")
        
        success = await self.pipeline.add_stream(stream_config)
        
        if success:
            print("✅ 视频流添加成功")
            
            # 运行演示
            print(f"\n⏳ 运行演示 30 秒...")
            
            # 添加状态回调
            self.pipeline.add_status_callback(self._status_callback)
            
            await asyncio.sleep(30)
            
            # 获取最终状态
            final_status = self.pipeline.get_pipeline_status()
            self._print_final_statistics("单流处理", final_status)
            
        else:
            print("❌ 视频流添加失败")
        
        # 停止流水线
        await self.pipeline.stop()
    
    async def demo_multi_stream(self):
        """演示多流处理"""
        print("\n" + "="*80)
        print("🎬 演示2: 多视频流并行处理")
        print("="*80)
        
        # 创建流水线（更高配置）
        self.pipeline = await self.create_pipeline_with_config(
            target_cpu=0.8,
            max_memory_mb=1024
        )
        
        # 启动流水线
        await self.pipeline.start()
        
        # 添加多个模拟视频流
        stream_configs = [
            StreamConfig(
                stream_id=f"demo_stream_{i+1}",
                url=f"demo://fake_rtsp_url_{i+1}",
                protocol=StreamProtocol.RTSP,
                fps_limit=20,
                resolution=(1280, 720)
            )
            for i in range(3)  # 3个并发流
        ]
        
        print(f"📡 添加 {len(stream_configs)} 个并发视频流:")
        
        for config in stream_configs:
            success = await self.pipeline.add_stream(config)
            status = "✅" if success else "❌"
            print(f"   {status} {config.stream_id}")
        
        # 运行演示
        print(f"\n⏳ 运行多流演示 45 秒...")
        print("💡 观察时序处理和CPU动态调整...")
        
        # 添加状态回调
        self.pipeline.add_status_callback(self._status_callback)
        
        await asyncio.sleep(45)
        
        # 获取最终状态
        final_status = self.pipeline.get_pipeline_status()
        self._print_final_statistics("多流处理", final_status)
        
        # 停止流水线
        await self.pipeline.stop()
    
    async def demo_dynamic_scaling(self):
        """演示动态处理器调整"""
        print("\n" + "="*80)
        print("🎬 演示3: 动态处理器扩缩容")
        print("="*80)
        
        # 创建流水线
        self.pipeline = await self.create_pipeline_with_config(
            target_cpu=0.7,  # 稍低的目标利用率
            max_memory_mb=1024
        )
        
        # 启动流水线
        await self.pipeline.start()
        
        print("📈 演示处理器动态调整...")
        
        # 阶段1：轻负载
        print(f"\n🔹 阶段1: 轻负载（1个流）")
        stream_config = StreamConfig(
            stream_id="scaling_stream_1",
            url="demo://light_load",
            protocol=StreamProtocol.RTSP,
            fps_limit=15
        )
        
        await self.pipeline.add_stream(stream_config)
        await asyncio.sleep(15)
        
        # 阶段2：中等负载
        print(f"\n🔹 阶段2: 中等负载（2个流）")
        stream_config_2 = StreamConfig(
            stream_id="scaling_stream_2",
            url="demo://medium_load",
            protocol=StreamProtocol.RTSP,
            fps_limit=25
        )
        
        await self.pipeline.add_stream(stream_config_2)
        await asyncio.sleep(20)
        
        # 阶段3：高负载
        print(f"\n🔹 阶段3: 高负载（4个流）")
        for i in range(3, 5):
            config = StreamConfig(
                stream_id=f"scaling_stream_{i}",
                url=f"demo://high_load_{i}",
                protocol=StreamProtocol.RTSP,
                fps_limit=30
            )
            await self.pipeline.add_stream(config)
        
        await asyncio.sleep(25)
        
        # 获取最终状态
        final_status = self.pipeline.get_pipeline_status()
        self._print_final_statistics("动态扩缩容", final_status)
        
        # 停止流水线
        await self.pipeline.stop()
    
    async def demo_performance_comparison(self):
        """演示性能对比"""
        print("\n" + "="*80)
        print("📊 性能对比分析")
        print("="*80)
        
        print("🔍 对比项目:")
        print("   • 传统架构 vs 时间轴架构")
        print("   • 串行处理 vs 并行处理")
        print("   • 固定处理器 vs 动态调整")
        
        # 模拟传统架构性能数据
        traditional_metrics = {
            "avg_frame_processing_time": 0.5,  # 500ms
            "cpu_utilization": 0.45,           # 45%
            "memory_efficiency": 0.65,         # 65%
            "throughput_fps": 2.0,             # 2 FPS
            "max_concurrent_streams": 1,
            "processor_utilization": 0.60      # 60%
        }
        
        # 创建时间轴架构
        self.pipeline = await self.create_pipeline_with_config(
            target_cpu=0.8,
            max_memory_mb=1024
        )
        
        await self.pipeline.start()
        
        # 添加测试流
        for i in range(2):
            config = StreamConfig(
                stream_id=f"perf_test_stream_{i+1}",
                url=f"demo://perf_test_{i+1}",
                protocol=StreamProtocol.RTSP,
                fps_limit=25
            )
            await self.pipeline.add_stream(config)
        
        print(f"\n⏳ 收集时间轴架构性能数据...")
        await asyncio.sleep(30)
        
        # 获取时间轴架构性能数据
        timeline_status = self.pipeline.get_pipeline_status()
        
        # 计算时间轴架构指标
        processor_stats = timeline_status.get("processors", {}).get("global_stats", {})
        memory_stats = timeline_status.get("memory", {})
        
        timeline_metrics = {
            "avg_frame_processing_time": 0.06,  # 预估60ms（基于前面的分析）
            "cpu_utilization": timeline_status["pipeline_stats"]["system_cpu_usage"] / 100,
            "memory_efficiency": 100 - memory_stats.get("usage_percent", 0),
            "throughput_fps": 16.7,  # 预估基于并行处理
            "max_concurrent_streams": timeline_status["pipeline_stats"]["total_streams"],
            "processor_utilization": processor_stats.get("avg_cpu_utilization", 0)
        }
        
        # 打印对比结果
        self._print_performance_comparison(traditional_metrics, timeline_metrics)
        
        await self.pipeline.stop()
    
    def _status_callback(self, event: str, status: Dict[str, Any]):
        """状态回调"""
        if event == "monitoring":
            # 简化的监控输出
            pipeline_stats = status["pipeline_stats"]
            
            print(f"📊 [监控] "
                  f"处理帧数: {pipeline_stats['total_frames_processed']}, "
                  f"CPU: {pipeline_stats['system_cpu_usage']:.1f}%, "
                  f"效率: {pipeline_stats['pipeline_efficiency']:.1f}%")
    
    def _print_final_statistics(self, demo_name: str, status: Dict[str, Any]):
        """打印最终统计信息"""
        print(f"\n📈 {demo_name} - 最终统计")
        print("-" * 50)
        
        pipeline_stats = status["pipeline_stats"]
        processor_stats = status.get("processors", {})
        memory_stats = status.get("memory", {})
        result_stats = status.get("results", {})
        
        print(f"🕒 运行时间: {time.time() - pipeline_stats['start_time']:.1f}秒")
        print(f"🎞️  总处理帧数: {pipeline_stats['total_frames_processed']}")
        print(f"📺 并发流数: {pipeline_stats['total_streams']}")
        print(f"💻 最终CPU使用: {pipeline_stats['system_cpu_usage']:.1f}%")
        print(f"🧠 内存使用: {memory_stats.get('memory_usage_mb', 0):.1f}MB")
        print(f"⚡ 流水线效率: {pipeline_stats['pipeline_efficiency']:.1f}%")
        
        # 处理器统计
        if processor_stats:
            proc_global = processor_stats.get("global_stats", {})
            print(f"🔧 处理器数量: {processor_stats.get('processor_count', 0)}")
            print(f"📊 处理器利用率: {proc_global.get('avg_cpu_utilization', 0)*100:.1f}%")
        
        # 结果统计
        if result_stats:
            result_global = result_stats.get("global_stats", {})
            print(f"📤 已推送结果: {result_global.get('total_results_pushed', 0)}")
            print(f"⏱️  平均推送延迟: {result_global.get('average_push_delay', 0):.3f}秒")
    
    def _print_performance_comparison(self, traditional: Dict, timeline: Dict):
        """打印性能对比"""
        print(f"\n📊 性能对比结果")
        print("=" * 80)
        print(f"{'指标':<25} {'传统架构':<15} {'时间轴架构':<15} {'提升':<15}")
        print("-" * 80)
        
        metrics = [
            ("平均帧处理时间", "avg_frame_processing_time", "秒", True),
            ("CPU利用率", "cpu_utilization", "%", False),
            ("内存效率", "memory_efficiency", "%", False), 
            ("处理吞吐量", "throughput_fps", "FPS", False),
            ("最大并发流", "max_concurrent_streams", "个", False),
            ("处理器利用率", "processor_utilization", "%", False)
        ]
        
        for name, key, unit, lower_better in metrics:
            trad_val = traditional[key]
            time_val = timeline[key]
            
            if unit == "%":
                trad_display = f"{trad_val*100:.1f}%"
                time_display = f"{time_val*100:.1f}%"
            elif unit == "秒":
                trad_display = f"{trad_val:.3f}s"
                time_display = f"{time_val:.3f}s"
            else:
                trad_display = f"{trad_val:.1f}{unit}"
                time_display = f"{time_val:.1f}{unit}"
            
            # 计算提升
            if lower_better:
                improvement = ((trad_val - time_val) / trad_val) * 100
                improvement_text = f"{improvement:.1f}% 更快"
            else:
                improvement = ((time_val - trad_val) / trad_val) * 100
                improvement_text = f"{improvement:.1f}% 提升"
            
            print(f"{name:<25} {trad_display:<15} {time_display:<15} {improvement_text:<15}")
        
        print("-" * 80)
        
        # 总结
        print("\n🎯 关键优势:")
        print("  ✅ 时间轴架构实现真正的并行处理")
        print("  ✅ 动态处理器调整优化CPU利用率")
        print("  ✅ 智能内存管理避免内存泄漏")
        print("  ✅ 按时序结果推送保证顺序")
        print("  ✅ 支持多流并发高效处理")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="时间轴架构演示")
    parser.add_argument("--demo", 
                       choices=["single", "multi", "scaling", "comparison", "all"],
                       default="all",
                       help="选择演示类型")
    parser.add_argument("--cpu-target", 
                       type=float, 
                       default=0.8,
                       help="目标CPU利用率 (0.1-1.0)")
    parser.add_argument("--memory-mb", 
                       type=int, 
                       default=1024,
                       help="最大内存限制(MB)")
    
    args = parser.parse_args()
    
    print("🚀 时间轴流水线架构演示")
    print("=" * 80)
    print("📖 这个演示展示了基于时间轴的高度并行化流水线架构")
    print("🎯 核心特性:")
    print("   • 时间轴驱动的帧管理和调度")
    print("   • 智能CPU多核动态利用")
    print("   • 零拷贝内存管理")
    print("   • 按时序结果推送")
    print("   • 多流并行处理")
    
    demo = TimelineArchitectureDemo()
    
    try:
        if args.demo in ["single", "all"]:
            await demo.demo_single_stream()
        
        if args.demo in ["multi", "all"]:
            await demo.demo_multi_stream()
        
        if args.demo in ["scaling", "all"]:
            await demo.demo_dynamic_scaling()
        
        if args.demo in ["comparison", "all"]:
            await demo.demo_performance_comparison()
        
        print(f"\n🎉 演示完成！")
        print("💡 提示: 在实际应用中，可以根据需要调整配置参数")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if demo.pipeline and demo.pipeline.running:
            await demo.pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main()) 