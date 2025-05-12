# 分析服务架构设计

## 1. 总体架构

分析服务采用模块化设计，主要包含以下几个核心组件：

1. **分析器工厂 (AnalyzerFactory)**: 负责创建不同类型的分析器实例
2. **基础分析器 (BaseAnalyzer)**: 所有分析器的基类，定义通用接口
3. **模型加载器 (ModelLoader)**: 负责加载不同版本和类型的模型
4. **任务管理器 (TaskManager)**: 管理分析任务的生命周期
5. **任务处理器 (TaskProcessor)**: 处理具体的分析任务

## 2. 分析类型支持

系统支持以下分析类型：

- `1`: 检测 (Detection) - 使用 `DetectionAnalyzer`
- `2`: 跟踪 (Tracking) - 使用 `TrackingAnalyzer`
- `3`: 分割 (Segmentation) - 使用 `SegmentationAnalyzer`
- `4`: 跨摄像头跟踪 (Cross-Camera Tracking) - 使用 `CrossCameraTrackingAnalyzer`
- `5`: 越界检测 (Line Crossing) - 使用 `LineCrossingAnalyzer`

## 3. 推理引擎支持

系统支持以下推理引擎：

- `0`: PyTorch - 使用 `PyTorchEngine`
- `1`: ONNX - 使用 `ONNXEngine`
- `2`: TensorRT - 使用 `TensorRTEngine`
- `3`: OpenVINO - 使用 `OpenVINOEngine`
- `4`: Pytron - 使用 `PytronEngine`

## 4. YOLO版本支持

系统支持以下YOLO版本：

- `0`: v8n（纳米版）- 使用 `YOLOv8nModel`
- `1`: v8s（小型版）- 使用 `YOLOv8sModel`
- `2`: v8l（大型版）- 使用 `YOLOv8lModel`
- `3`: v8x（超大版）- 使用 `YOLOv8xModel`
- `4`: 11s（YOLO11小型版）- 使用 `YOLO11sModel`
- `5`: 11m（YOLO11中型版）- 使用 `YOLO11mModel`
- `6`: 11l（YOLO11大型版）- 使用 `YOLO11lModel`

## 5. YOLOE模型支持

系统特别支持YOLOE模型，该模型具有以下特性：

- 文本提示推理 - 使用 `YOLOETextPromptModel`
- 图片提示推理 - 使用 `YOLOEVisualPromptModel`
- 无提示推理 - 使用 `YOLOEBaseModel`
- 目标检测 - 使用 `YOLOEDetectionModel`
- 目标分割 - 使用 `YOLOESegmentationModel`

## 6. 类图设计

```
                           +----------------+
                           |  BaseAnalyzer  |
                           +-------+--------+
                                   ^
                                   |
           +---------------------+-+------------------+
           |                     |                    |
+----------+---------+ +---------+--------+ +---------+--------+
| DetectionAnalyzer  | | TrackingAnalyzer | |SegmentationAnalyzer|
+--------------------+ +------------------+ +-------------------+
           ^                    ^                    ^
           |                    |                    |
+----------+---------+ +---------+--------+ +---------+--------+
|YOLODetectionAnalyzer| |YOLOTrackingAnalyzer| |YOLOSegmentAnalyzer|
+--------------------+ +------------------+ +-------------------+
           ^                    ^                    ^
           |                    |                    |
+----------+---------+ +---------+--------+ +---------+--------+
|  YOLOEAnalyzer     | |CrossCameraAnalyzer| |LineCrossingAnalyzer|
+--------------------+ +------------------+ +-------------------+
```

## 7. 工厂模式设计

使用工厂模式创建分析器实例：

```python
class AnalyzerFactory:
    @staticmethod
    def create_analyzer(analysis_type, model_code, engine_type, yolo_version, **kwargs):
        if analysis_type == 1:  # 检测
            return DetectionAnalyzer(model_code, engine_type, yolo_version, **kwargs)
        elif analysis_type == 2:  # 跟踪
            return TrackingAnalyzer(model_code, engine_type, yolo_version, **kwargs)
        elif analysis_type == 3:  # 分割
            return SegmentationAnalyzer(model_code, engine_type, yolo_version, **kwargs)
        elif analysis_type == 4:  # 跨摄像头跟踪
            return CrossCameraTrackingAnalyzer(model_code, engine_type, yolo_version, **kwargs)
        elif analysis_type == 5:  # 越界检测
            return LineCrossingAnalyzer(model_code, engine_type, yolo_version, **kwargs)
        else:
            raise ValueError(f"不支持的分析类型: {analysis_type}")
```

## 8. 模型加载器设计

```python
class ModelLoader:
    @staticmethod
    def load_model(model_code, engine_type, yolo_version, **kwargs):
        # 根据引擎类型选择加载器
        if engine_type == 0:  # PyTorch
            return PyTorchModelLoader.load(model_code, yolo_version, **kwargs)
        elif engine_type == 1:  # ONNX
            return ONNXModelLoader.load(model_code, yolo_version, **kwargs)
        # ... 其他引擎类型
        else:
            raise ValueError(f"不支持的引擎类型: {engine_type}")
```

## 9. 任务流程

1. API接收任务请求
2. TaskService创建任务配置
3. TaskManager添加任务
4. TaskProcessor启动任务处理
5. AnalyzerFactory创建适当的分析器
6. ModelLoader加载模型
7. 分析器处理视频流/图像
8. 结果处理和回调

## 10. 配置参数映射

| 参数名 | 类型 | 描述 | 对应组件 |
|-------|-----|------|---------|
| analysis_type | int | 分析类型 | AnalyzerFactory |
| engine | int | 推理引擎类型 | ModelLoader |
| yolo_version | int | YOLO模型版本 | ModelLoader |
| model_code | string | 模型代码 | ModelLoader |
| prompt_type | int | YOLOE提示类型 | YOLOEAnalyzer |
| text_prompt | list | YOLOE文本提示 | YOLOETextPromptModel |
| visual_prompt | dict | YOLOE视觉提示 | YOLOEVisualPromptModel |

## 11. 实现计划

1. 创建基础分析器接口
2. 实现各种分析器类
3. 实现模型加载器
4. 实现分析器工厂
5. 更新任务处理器以使用新架构
6. 添加YOLOE特定支持
7. 实现跨摄像头跟踪和越界检测功能
