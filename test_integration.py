#!/usr/bin/env python3
"""
快速集成测试脚本
测试ONVIF和GStreamer功能是否正确集成
"""

import sys
import os
import asyncio

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_imports():
    """测试核心模块导入"""
    print("🧪 测试模块导入...")
    
    try:
        # 测试ONVIF模块导入
        from core.media_kit.protocols.onvif.handler import ONVIF_AVAILABLE
        print(f"  ONVIF可用性: {ONVIF_AVAILABLE}")
        
        # 测试GStreamer模块导入
        from core.media_kit.protocols.gstreamer.handler import GSTREAMER_AVAILABLE
        print(f"  GStreamer可用性: {GSTREAMER_AVAILABLE}")
        
        # 测试流工厂导入
        from core.media_kit.factory.stream_factory import StreamFactory, StreamEngine
        print(f"  流工厂导入: ✅")
        
        # 测试GStreamer检测
        gst_detected = StreamFactory.is_gstreamer_available()
        print(f"  GStreamer检测: {gst_detected}")
        
        # 测试Discovery路由器导入
        from routers.discovery import router as discovery_router
        print(f"  Discovery路由器导入: ✅")
        
        print("✅ 所有模块导入成功！")
        return True
        
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_factory():
    """测试流工厂功能"""
    print("\n🔧 测试流工厂...")
    
    try:
        from core.media_kit.factory.stream_factory import StreamFactory, StreamEngine
        
        # 测试配置
        config = {
            "stream_id": "test_stream",
            "url": "rtsp://admin:zyckj2021@192.168.1.200:554/1/1",
            "preferred_engine": "gstreamer"
        }
        
        # 尝试创建GStreamer流
        if StreamFactory.is_gstreamer_available():
            print("  尝试创建GStreamer流...")
            stream = StreamFactory.create_stream(config, StreamEngine.GSTREAMER)
            if stream:
                print("  ✅ GStreamer流创建成功")
            else:
                print("  ❌ GStreamer流创建失败")
        else:
            print("  ⚠️  GStreamer不可用，跳过测试")
        
        # 尝试创建OpenCV流
        print("  尝试创建OpenCV流...")
        opencv_stream = StreamFactory.create_stream(config, StreamEngine.OPENCV)
        if opencv_stream:
            print("  ✅ OpenCV流创建成功")
        else:
            print("  ❌ OpenCV流创建失败")
        
        print("✅ 流工厂测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 流工厂测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_onvif_discovery():
    """测试ONVIF发现功能"""
    print("\n🔍 测试ONVIF发现...")
    
    try:
        # 修复导入路径
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'examples'))
        
        from onvif_discovery_test import ONVIFDiscovery
        
        discovery = ONVIFDiscovery()
        discovery.discovery_timeout = 3  # 短超时用于快速测试
        
        print("  执行设备发现（3秒超时）...")
        devices = discovery.discover_devices()
        
        print(f"  发现 {len(devices)} 个设备")
        for device in devices:
            print(f"    - {device['ip']}: {device.get('name', 'Unknown')}")
        
        print("✅ ONVIF发现测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ ONVIF发现测试失败: {e}")
        return False

async def main():
    """主函数"""
    print("🚀 N-MeekYolo 集成测试")
    print("=" * 60)
    
    # 测试导入
    import_success = await test_imports()
    
    # 测试工厂
    factory_success = await test_factory()
    
    # 测试ONVIF发现
    onvif_success = await test_onvif_discovery()
    
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    print(f"  模块导入: {'✅' if import_success else '❌'}")
    print(f"  流工厂: {'✅' if factory_success else '❌'}")
    print(f"  ONVIF发现: {'✅' if onvif_success else '❌'}")
    
    if all([import_success, factory_success, onvif_success]):
        print("\n🎉 所有测试通过！集成成功！")
        print("\n📝 接下来可以:")
        print("  1. 启动服务: python app.py")
        print("  2. 访问API文档: http://localhost:8002/api/v1/docs")
        print("  3. 测试新的Discovery API端点")
    else:
        print("\n⚠️  部分测试失败，请检查相关模块")

if __name__ == "__main__":
    asyncio.run(main()) 