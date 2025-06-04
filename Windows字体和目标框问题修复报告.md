# Windows系统下目标框不显示问题修复报告

## 问题描述

在Windows系统下，YOLO分析时检测到了目标，但直播流画面中没有显示任何目标框。日志显示检测正常，但视觉效果中缺少目标框和标签。

## 问题分析

通过深入调查，发现了两个主要问题：

### 1. 边界框字段名不匹配

**问题根因：**
- YOLO检测器 (`core/analyzer/detection/yolo_detector.py`) 返回的检测结果使用 `bbox_pixels` 字段存储边界框坐标
- TaskProcessor (`core/task_management/processor.py`) 在处理检测结果时只查找 `bbox` 字段
- 字段名不匹配导致无法获取边界框坐标，因此无法绘制目标框

**日志证据：**
```
"bbox_pixels": [1112.4951171875, 271.96417236328125, 1337.16845703125, 885.6586303710938]
```

### 2. Windows下字体渲染问题

**问题根因：**
- TaskProcessor中使用 `cv2.putText()` 绘制标签
- `cv2.putText()` 在Windows系统下不支持中文字符显示
- 虽然不影响目标框显示，但标签可能显示异常

## 修复方案

### 1. 修复边界框字段匹配

在 `core/task_management/processor.py` 的 `_process_detection_results` 方法中：

```python
# 修复前
bbox = det.get("bbox", [0, 0, 0, 0])

# 修复后  
bbox = det.get("bbox", [])
if not bbox:
    # 尝试其他可能的字段名，特别是YOLO检测器使用的bbox_pixels
    bbox = det.get("bbox_pixels", det.get("box", []))
    if not bbox:
        normal_logger.debug(f"检测结果中未找到有效的边界框字段: {list(det.keys())}")
        continue
```

### 2. 增强文本渲染兼容性

```python
# 修复前
cv2.putText(preview_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 修复后
try:
    # 尝试使用FrameRenderer绘制支持中文的标签
    from services.video.utils.frame_renderer import FrameRenderer
    preview_frame = FrameRenderer._put_chinese_text(
        preview_frame, label, (x1, y1 - 25), 16, (255, 255, 255), (0, 255, 0)
    )
except Exception as text_error:
    # 如果中文渲染失败，回退到OpenCV的英文字体
    normal_logger.debug(f"中文文本渲染失败，使用OpenCV默认字体: {text_error}")
    cv2.putText(preview_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

## 验证测试

### 字体环境检查
- ✅ Windows字体文件存在：`C:/Windows/Fonts/msyh.ttc` (微软雅黑)
- ✅ PIL字体加载正常
- ✅ FrameRenderer类工作正常

### 边界框处理测试
- ✅ 成功处理 `bbox_pixels` 字段
- ✅ 向后兼容 `bbox` 字段  
- ✅ 支持字典格式边界框
- ✅ 正确处理缺失字段情况

### 渲染效果验证
生成的测试图像显示：
- `test_bbox_fix_preview.jpg` - 包含完整的目标框和标签
- `test_bbox_case_*.jpg` - 各种格式兼容性测试结果

## 修复效果

1. **目标框显示** - 现在能正确显示检测到的目标边界框
2. **标签文本** - 支持中英文标签显示，在Windows下有更好的兼容性
3. **向后兼容** - 支持多种边界框字段格式，不影响现有功能
4. **错误处理** - 添加了详细的调试日志，便于问题排查

## 相关文件

**修改的文件：**
- `core/task_management/processor.py` - 主要修复逻辑

**测试文件：**
- `test_font_rendering.py` - 字体渲染测试
- `test_bbox_fix.py` - 边界框修复验证
- `Windows字体和目标框问题修复报告.md` - 本报告

**涉及的组件：**
- `services/video/utils/frame_renderer.py` - 字体渲染器
- `core/analyzer/detection/yolo_detector.py` - YOLO检测器
- `services/video/streaming/live_streamer.py` - 直播推流

## 总结

此次修复解决了Windows系统下YOLO检测结果在直播流中不显示目标框的问题。核心问题是边界框字段名不匹配，通过增强字段名兼容性和改进文本渲染，现在系统能够正确显示检测结果。

修复后的系统具有更好的跨平台兼容性和更强的容错能力。 