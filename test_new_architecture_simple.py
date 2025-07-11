#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试新的视频缓存和回放架构
"""

import sys
import os
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_services_import():
    """测试服务导入"""
    logger.info("🧪 测试新架构服务导入...")
    
    try:
        from app.services.video_cache_service import VideoCacheService
        logger.info("✅ VideoCacheService 导入成功")
        
        from app.services.video_playback_service import VideoPlaybackService
        logger.info("✅ VideoPlaybackService 导入成功")
        
        return True
    except Exception as e:
        logger.error(f"❌ 服务导入失败: {e}")
        return False

def test_services_creation():
    """测试服务创建"""
    logger.info("🧪 测试服务创建...")
    
    try:
        from app.services.video_cache_service import VideoCacheService
        from app.services.video_playback_service import VideoPlaybackService
        
        # 创建视频缓存服务
        cache_service = VideoCacheService(
            stream_url="test://test.url",
            cache_dir="./test_cache",
            cache_duration=60
        )
        logger.info("✅ VideoCacheService 创建成功")
        
        # 创建视频回放服务  
        playback_service = VideoPlaybackService(
            video_cache_service=cache_service,
            output_dir="./test_output",
            playback_duration=10.0
        )
        logger.info("✅ VideoPlaybackService 创建成功")
        
        return True
    except Exception as e:
        logger.error(f"❌ 服务创建失败: {e}")
        return False

def test_cleanup():
    """清理测试文件"""
    import shutil
    
    try:
        if os.path.exists("./test_cache"):
            shutil.rmtree("./test_cache")
        if os.path.exists("./test_output"):
            shutil.rmtree("./test_output")
        logger.info("✅ 测试文件清理完成")
    except Exception as e:
        logger.warning(f"⚠️ 清理测试文件失败: {e}")

def main():
    """主测试函数"""
    logger.info("🚀 开始新架构测试...")
    
    tests = [
        ("服务导入测试", test_services_import),
        ("服务创建测试", test_services_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"📋 运行测试: {test_name}")
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} 通过")
        else:
            logger.error(f"❌ {test_name} 失败")
        logger.info("-" * 50)
    
    # 清理
    test_cleanup()
    
    # 总结
    logger.info(f"🎯 测试完成: {passed}/{total} 个测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！新架构准备就绪")
        return True
    else:
        logger.error("❌ 部分测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 