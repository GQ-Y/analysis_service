#!/usr/bin/env python3
"""
临时测试文件 - 验证分析器创建和模型加载
"""

def test_analyzer_creation():
    """测试分析器创建"""
    try:
        print("开始测试分析器创建...")
        
        from core.analyzer.analyzer_factory import analyzer_factory
        
        # 测试配置
        config = {
            'model_code': 'yolo',
            'device': 0,  # 使用CPU而不是CUDA
            'confidence': 0.02,
            'iou_threshold': 0.45
        }
        
        print(f"测试配置: {config}")
        
        # 创建分析器
        analyzer = analyzer_factory.create_analyzer('detection', 'default', config)
        
        print(f"分析器创建: {analyzer is not None}")
        
        if analyzer:
            print(f"分析器类型: {type(analyzer).__name__}")
            print(f"模型代码: {analyzer.model_code}")
            print(f"设备: {analyzer.device}")
            print(f"已加载: {analyzer.loaded}")
            
            # 检查分析器属性
            if hasattr(analyzer, 'detector'):
                print(f"检测器: {type(analyzer.detector).__name__}")
                print(f"检测器模型代码: {analyzer.detector.current_model_code}")
                print(f"检测器模型: {analyzer.detector.model is not None}")
            
            # 获取模型信息
            try:
                model_info = analyzer.model_info
                print(f"模型信息: {model_info}")
            except Exception as e:
                print(f"获取模型信息时出错: {e}")
                # 手动检查一些关键属性
                print(f"分析器loaded状态: {getattr(analyzer, 'loaded', 'N/A')}")
                print(f"分析器model_code: {getattr(analyzer, 'model_code', 'N/A')}")
                print(f"分析器current_model_code: {getattr(analyzer, 'current_model_code', 'N/A')}")
                if hasattr(analyzer, 'detector'):
                    print(f"检测器模型状态: {getattr(analyzer.detector, 'model', 'N/A') is not None}")
                    print(f"检测器current_model_code: {getattr(analyzer.detector, 'current_model_code', 'N/A')}")
            
        else:
            print("分析器创建失败")
            
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analyzer_creation() 