def add_log_to_detection():
    """
    在检测到目标时添加日志
    """
    import re
    
    # 读取文件内容
    with open('core/analyzer/yoloe/yoloe_analyzer.py', 'r') as f:
        content = f.read()
    
    # 查找所有需要修改的位置
    pattern = r'(# 获取边界框\s+boxes = result\.boxes\s+\n\s+)# 处理每个检测结果'
    replacement = r'\1# 只在检测到目标时记录日志\n                log_detections_if_found(result, boxes)\n\n                # 处理每个检测结果'
    
    # 替换所有匹配项
    new_content = re.sub(pattern, replacement, content)
    
    # 写回文件
    with open('core/analyzer/yoloe/yoloe_analyzer.py', 'w') as f:
        f.write(new_content)
    
    print("修改完成")

if __name__ == "__main__":
    add_log_to_detection()
