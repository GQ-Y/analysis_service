# YOLOE任务配置指南

本文档提供YOLOE视觉分析任务的完整配置指南，包括参数说明、枚举值、帧率控制逻辑和完整示例。

## 帧率控制说明

帧率设置(`frame_rate`)指的是**每秒处理的帧数(FPS)**。系统通过以下方式实现帧率控制：

1. **不是**每隔N帧抽取1帧，而是控制每秒处理的帧数
2. **实际控制逻辑**：
   - 假设摄像头原始帧率为25fps
   - 如果设置`frame_rate=5`，则每秒只取5帧进行分析，即：
   - 系统计算处理间隔 = 1000ms ÷ 5 = 200ms
   - 每200ms从视频流抽取1帧进行分析
3. **多级配置**：
   - 任务级 → 摄像头级 → 算法级，优先级递增
   - 不同算法可设置不同帧率，合理分配计算资源

### 帧率设置建议

- **检测任务**：3-5fps通常足够
- **跟踪任务**：10-15fps获得流畅体验
- **越界检测**：8-12fps平衡准确性和资源
- **高危场景**：可提高至20-25fps，但注意资源消耗

## 完整配置参数说明

### 1. 任务级参数


| 参数名            | 类型    | 说明                               | 默认值 |
| ----------------- | ------- | ---------------------------------- | ------ |
| name              | string  | 任务名称                           | 必填   |
| description       | string  | 任务描述                           | 空     |
| save_result       | boolean | 是否保存分析结果                   | true   |
| save_images       | boolean | 是否保存图像                       | false  |
| analysis_interval | integer | 分析间隔(秒)                       | 1      |
| frame_rate        | integer | 全局帧率设置(fps)                  | 10     |
| specific_node_id  | integer | 指定节点ID                         | null   |
| callback_id       | integer | 全局回调ID                         | null   |
| device            | integer | 推理设备类型：0=CPU, 1=GPU, 2=AUTO | 1      |

### 2. 摄像头级参数


| 参数名      | 类型    | 说明                         | 默认值     |
| ----------- | ------- | ---------------------------- | ---------- |
| id          | string  | 摄像头配置唯一ID(更新时需要) | 自动生成   |
| camera_id   | integer | 数据库中的摄像头ID           | 必填       |
| camera_name | string  | 摄像头名称                   | 空         |
| frame_rate  | integer | 摄像头级帧率设置             | 继承任务级 |

### 3. 算法级参数


| 参数名                 | 类型    | 说明                       | 默认值       |
| ---------------------- | ------- | -------------------------- | ------------ |
| id                     | string  | 算法配置唯一ID(更新时需要) | 自动生成     |
| algorithm_id           | string  | 算法ID                     | 必填         |
| algorithm_name         | string  | 算法名称                   | 空           |
| model_id               | integer | 数据库中的模型ID           | 必填         |
| analysis_type          | integer | 分析类型：见枚举表         | 1            |
| nested_detection       | boolean | 是否启用嵌套检测           | true         |
| enable_callback        | boolean | 是否启用回调               | false        |
| callback_url           | string  | 回调URL                    | 空           |
| enable_alarm_recording | boolean | 是否启用报警录像           | false        |
| alarm_recording_before | integer | 报警前录像时长(秒)         | 5            |
| alarm_recording_after  | integer | 报警后录像时长(秒)         | 5            |
| frame_rate             | integer | 算法级帧率设置             | 继承摄像头级 |

### 4. 算法参数(params)


| 参数名              | 类型    | 说明                                                    | 默认值                 |
| ------------------- | ------- | ------------------------------------------------------- | ---------------------- |
| confidence          | float   | 置信度阈值                                              | 0.5                    |
| iou_threshold       | float   | IOU阈值                                                 | 0.5                    |
| image_size          | object  | 输入图像尺寸                                            | {width:640,height:640} |
| detect_classes      | array   | 检测类别限制                                            | []                     |
| engine              | integer | 推理引擎：见枚举表                                      | 1                      |
| yolo_version        | integer | YOLO版本：见枚举表                                      | 2                      |
| custom_weights_path | string  | 模型权重路径，支持HTTP/HTTPS URL、对象存储URL和本地路径 | 空                     |
| prompt_type         | integer | 提示类型：见枚举表                                      | 1                      |
| text_prompt         | array   | 文本提示(关键词列表)                                    | []                     |
| visual_prompt       | object  | 视觉提示信息                                            | null                   |
| segmentation        | boolean | 是否启用分割                                            | false                  |
| nms_type            | integer | NMS类型：见枚举表                                       | 0                      |
| max_detections      | integer | 最大检测目标数量                                        | 100                    |
| device              | integer | 推理设备                                                | 继承任务级             |
| half_precision      | boolean | 是否使用半精度(FP16)                                    | false                  |
| tracking_type       | integer | 跟踪算法类型：见枚举表                                  | 0                      |
| max_tracks          | integer | 最大跟踪目标数                                          | 50                     |
| max_lost_time       | integer | 最大丢失时间(帧/秒)                                     | 30                     |
| feature_type        | integer | 特征类型：见枚举表                                      | 0                      |
| related_cameras     | array   | 跨摄像头关联ID                                          | []                     |

## 类型枚举值表

### 分析类型 (analysis_type)

- `1`: 检测 (Detection)
- `2`: 跟踪 (Tracking)
- `3`: 分割 (Segmentation)
- `4`: 跨摄像头跟踪 (Cross-Camera Tracking)
- `5`: 越界检测 (Line Crossing)

### 推理引擎 (engine)

- `0`: PyTorch
- `1`: ONNX
- `2`: TensorRT
- `3`: OpenVINO
- `4`: Pytron

### YOLO版本 (yolo_version)

- `0`: v8n（纳米版）
- `1`: v8s（小型版）
- `2`: v8l（大型版）
- `3`: v8x（超大版）
- `4`: 11s（YOLO11小型版）
- `5`: 11m（YOLO11中型版）
- `6`: 11l（YOLO11大型版）

### 提示类型 (prompt_type)

- `1`: 文本提示 (Text Prompt)
- `2`: 视觉提示 (Visual Prompt)
- `3`: 无提示 (Prompt-Free)

### ROI类型/可视化提示类型 (type)

- `1`: 矩形 (Rectangle)
- `2`: 多边形 (Polygon)
- `3`: 线段 (Line)

### 设备类型 (device)

- `0`: CPU
- `1`: GPU
- `2`: AUTO (自动选择)

### 跟踪算法类型 (tracking_type)

- `0`: DeepSORT
- `1`: ByteTrack
- `2`: StrongSORT
- `3`: BoTSORT

### NMS类型 (nms_type)

- `0`: 默认 (Default)
- `1`: 软性 (Soft)
- `2`: 加权 (Weighted)
- `3`: DIoU NMS

### 特征类型 (feature_type)

- `0`: 基础特征 (Basic)
- `1`: ReID特征 (ReID)
- `2`: 深度特征 (Deep Feature)

## 完整配置示例

以下是一个完整的多摄像头YOLOE配置示例：

```json
{
  "name": "多摄像头YOLOE综合任务",
  "description": "测试YOLOE多种能力的综合配置",
  "save_result": true,
  "save_images": true,
  "analysis_interval": 5,
  "frame_rate": 15,
  "specific_node_id": 2,
  "callback_id": 1,
  "device": 1,
  "cameras": [
    {
      "id": "camera-uuid-1",
      "camera_id": 1,
      "camera_name": "工地入口摄像头",
      "frame_rate": 10,
      "algorithms": [
        {
          "id": "algorithm-uuid-1",
          "algorithm_id": "yoloe",
          "algorithm_name": "YOLOE文本提示检测",
          "model_id": 5,
          "analysis_type": 1,
          "nested_detection": true,
          "enable_callback": true,
          "callback_url": "http://example.com/api/callback",
          "enable_alarm_recording": true,
          "alarm_recording_before": 10,
          "alarm_recording_after": 20,
          "frame_rate": 8,
          "params": {
            "confidence": 0.45,
            "iou_threshold": 0.5,
            "image_size": {
              "width": 640,
              "height": 640
            },
            "detect_classes": [],
            "engine": 1,
            "yolo_version": 2,
            "custom_weights_path": "https://models.example.com/weights/yoloe-v8l-seg.pt",
            "prompt_type": 1,
            "text_prompt": [
              "安全帽",
              "工人",
              "反光背心", 
              "车辆"
            ],
            "segmentation": true,
            "nms_type": 0,
            "max_detections": 100,
            "device": 1,
            "half_precision": true
          },
          "roi": {
            "type": 1,
            "x1": 0.1,
            "y1": 0.2,
            "x2": 0.8,
            "y2": 0.9
          }
        }
      ]
    },
    {
      "id": "camera-uuid-2",
      "camera_id": 2,
      "camera_name": "工地场区摄像头",
      "frame_rate": 25,
      "algorithms": [
        {
          "id": "algorithm-uuid-3",
          "algorithm_id": "yoloe",
          "algorithm_name": "YOLOE视觉提示分割",
          "model_id": 5,
          "analysis_type": 3,
          "nested_detection": true,
          "enable_callback": false,
          "frame_rate": 5,
          "params": {
            "confidence": 0.5,
            "iou_threshold": 0.45,
            "image_size": {
              "width": 800,
              "height": 800
            },
            "engine": 2,
            "yolo_version": 2,
            "prompt_type": 2,
            "visual_prompt": {
              "type": 2,
              "points": [
                {"x": 0.2, "y": 0.3},
                {"x": 0.8, "y": 0.3},
                {"x": 0.7, "y": 0.7},
                {"x": 0.3, "y": 0.7}
              ],
              "line_width": 3,
              "use_as_mask": true
            },
            "segmentation": true,
            "max_detections": 50,
            "half_precision": true
          },
          "roi": {
            "type": 2,
            "points": [
              {"x": 0.1, "y": 0.1},
              {"x": 0.9, "y": 0.1},
              {"x": 0.9, "y": 0.9},
              {"x": 0.1, "y": 0.9}
            ]
          }
        }
      ]
    },
    {
      "id": "camera-uuid-6",
      "camera_id": 6,
      "camera_name": "越界检测摄像头",
      "frame_rate": 12,
      "algorithms": [
        {
          "id": "algorithm-uuid-7",
          "algorithm_id": "yoloe",
          "algorithm_name": "YOLOE线段越界检测",
          "model_id": 5,
          "analysis_type": 5,
          "frame_rate": 8,
          "params": {
            "confidence": 0.5,
            "iou_threshold": 0.45,
            "image_size": {
              "width": 640,
              "height": 640
            },
            "engine": 1,
            "yolo_version": 2,
            "prompt_type": 2,
            "visual_prompt": {
              "type": 3,
              "points": [
                {"x": 0.1, "y": 0.5},
                {"x": 0.9, "y": 0.5}
              ],
              "direction": 1,
              "line_width": 5
            },
            "text_prompt": ["person", "car", "truck"],
            "counting_enabled": true,
            "time_threshold": 0.5,
            "speed_estimation": true,
            "object_filter": {
              "min_size": 0.02,
              "max_size": 0.5
            }
          },
          "roi": {
            "type": 3,
            "points": [
              {"x": 0.1, "y": 0.5},
              {"x": 0.9, "y": 0.5}
            ]
          }
        }
      ]
    }
  ]
}
```

## 集成建议

### 1. 适用场景

- **文本提示(Text Prompt)**: 适用于有明确检测目标的场景，如特定物体检测
- **视觉提示(Visual Prompt)**: 适用于需要细粒度控制检测区域的场景，如划定区域监控
- **无提示(Prompt-Free)**: 适用于通用场景检测，无需特定目标

### 2. 性能优化

- **设备选择**: GPU(1)适合复杂模型，CPU(0)适合轻量模型，AUTO(2)自动平衡
- **模型精度**: 开启half_precision可显著提升GPU性能，牺牲少量精度
- **帧率控制**: 根据任务重要性分配不同帧率，关键任务提高帧率，非关键任务降低帧率
- **图像尺寸**: 更大的image_size提高精度但降低速度，需权衡

### 3. 最佳实践

- **逐步调优**: 从基础配置开始，逐步调整参数至最佳
- **分层配置**: 充分利用任务→摄像头→算法的参数继承关系
- **资源分配**: 重要算法分配更多资源(更高帧率、更大图像尺寸)
- **关联配置**: 跨摄像头跟踪中确保related_cameras配置正确


分析时需要实现的方法：


在系统后端实现时，应该：

1. 自动检测路径类型：根据前缀（http://, https://, s3://, huggingface://等）自动判断
2. 模型缓存机制：首次使用在线模型时下载并缓存到本地，避免重复下载
3. 版本检查：定期检查在线模型是否有更新
4. 权限验证：确保系统有权限访问指定的在线资源

#### 在线URL示例

"custom_weights_path": "https://models.example.com/weights/yoloe-v8l-seg.pt"

#### 本地路径示例（确保所有节点都有相同路径的模型文件）

"custom_weights_path": "/models/yoloe-v8l-seg.pt"

#### 对象存储示例

"custom_weights_path": "s3://ai-models/yoloe/v8l-seg.pt"

#### 模型仓库示例

"custom_weights_path": "huggingface://jameslahm/yoloe-v8l-seg"
