#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
存储处理器 - 负责处理分析结果的存储需求
"""

import os
import cv2
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.core.zero_copy.frame_buffer import FrameBuffer
from .base_processor import BaseResultProcessor


class StorageProcessor(BaseResultProcessor):
    """存储处理器 - 保存结果图片和元数据"""
    
    def __init__(self, 
                 save_images: bool = True,
                 save_metadata: bool = True,
                 draw_boxes: bool = True,
                 image_quality: int = 95,
                 logger: Optional[logging.Logger] = None):
        """
        初始化存储处理器
        
        Args:
            save_images: 是否保存图片
            save_metadata: 是否保存元数据
            draw_boxes: 是否在图片上绘制检测框
            image_quality: 图片质量 (1-100)
            logger: 日志记录器
        """
        super().__init__(logger)
        self.save_images = save_images
        self.save_metadata = save_metadata
        self.draw_boxes = draw_boxes
        self.image_quality = image_quality
        
        # 统计信息
        self.total_processed = 0
        self.images_saved = 0
        self.metadata_saved = 0
        self.errors = 0
        
        # 设置文件日志记录器
        self._setup_file_logger()
        
        self.file_logger.info("💾 存储处理器初始化完成")
        self.file_logger.info(f"   配置: save_images={save_images}, save_metadata={save_metadata}")
        self.file_logger.info(f"   参数: draw_boxes={draw_boxes}, image_quality={image_quality}")
    
    def _setup_file_logger(self):
        """设置存储处理器专用的文件日志记录器"""
        try:
            # 使用原有logger作为基础
            if hasattr(self.logger, 'name') and 'result_pipeline_task_' in self.logger.name:
                # 如果传入的是结果管道的logger，创建存储处理器专用的子logger
                task_id = self.logger.name.split('_')[-1] if '_' in self.logger.name else 'unknown'
                self.file_logger = logging.getLogger(f"storage_processor_task_{task_id}")
                
                # 继承父logger的配置
                if self.logger.handlers:
                    parent_handler = self.logger.handlers[0]
                    
                    # 创建新的处理器，使用相同的文件路径但添加存储前缀
                    if hasattr(parent_handler, 'baseFilename'):
                        log_file = Path(parent_handler.baseFilename).parent / f"task_{task_id}_storage_processor.log"
                        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
                        file_handler.setLevel(logging.DEBUG)
                        file_handler.setFormatter(parent_handler.formatter)
                        
                        self.file_logger.handlers.clear()
                        self.file_logger.addHandler(file_handler)
                        self.file_logger.setLevel(logging.DEBUG)
                        self.file_logger.propagate = False
                    else:
                        self.file_logger = self.logger
                else:
                    self.file_logger = self.logger
            else:
                self.file_logger = self.logger
                
        except Exception as e:
            self.logger.error(f"❌ 设置存储处理器文件日志失败: {e}")
            self.file_logger = self.logger
    
    def process(self, frame_buffer: FrameBuffer, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理帧缓冲区，保存图片和元数据
        
        Args:
            frame_buffer: 帧缓冲区
            task_config: 任务配置
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        start_time = time.time()
        self.total_processed += 1
        
        task_id = task_config.get('task_id', 'unknown')
        frame_id = getattr(frame_buffer, 'frame_id', 0)
        
        self.file_logger.info("="*50)
        self.file_logger.info(f"💾 开始存储处理 - 任务{task_id}, 帧{frame_id}")
        self.file_logger.info(f"   配置: save_images={self.save_images}, save_metadata={self.save_metadata}")
        
        result = {
            "success": False,
            "images_saved": 0,
            "metadata_saved": 0,
            "errors": [],
            "paths": {
                "images": [],
                "metadata": []
            }
        }
        
        try:
            # 获取任务根目录
            storage_root = self._get_storage_root()
            task_dir = storage_root / f"task_{task_id}"
            
            self.file_logger.info(f"📁 存储目录: {task_dir}")
            
            # 创建目录结构
            images_dir, metadata_dir = self._create_directories(task_dir)
            
            # 生成文件名
            timestamp = getattr(frame_buffer, 'timestamp', time.time())
            base_filename = self._generate_filename(task_id, frame_id, timestamp)
            
            self.file_logger.info(f"📝 文件名: {base_filename}")
            
            # 处理图片保存
            if self.save_images:
                self.file_logger.info("🖼️ 开始保存图片...")
                image_success = self._save_image(frame_buffer, images_dir, base_filename, result)
                if image_success:
                    self.images_saved += 1
                    result["images_saved"] += 1
            else:
                self.file_logger.info("⏭️ 跳过图片保存 (save_images=False)")
            
            # 处理元数据保存
            if self.save_metadata:
                self.file_logger.info("📋 开始保存元数据...")
                metadata_success = self._save_metadata(frame_buffer, metadata_dir, base_filename, result)
                if metadata_success:
                    self.metadata_saved += 1
                    result["metadata_saved"] += 1
            else:
                self.file_logger.info("⏭️ 跳过元数据保存 (save_metadata=False)")
            
            # 计算处理结果
            result["success"] = (result["images_saved"] > 0 or result["metadata_saved"] > 0) and len(result["errors"]) == 0
            
            processing_time = (time.time() - start_time) * 1000
            
            if result["success"]:
                self.file_logger.info(f"✅ 存储处理成功，耗时: {processing_time:.2f}ms")
                self.file_logger.info(f"   图片: {result['images_saved']}, 元数据: {result['metadata_saved']}")
            else:
                self.file_logger.error(f"❌ 存储处理失败，耗时: {processing_time:.2f}ms")
                self.file_logger.error(f"   错误: {result['errors']}")
                self.errors += 1
            
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_msg = f"❌ 存储处理异常: {e}"
            self.file_logger.error(f"{error_msg}，耗时: {processing_time:.2f}ms")
            self.file_logger.exception("详细异常信息:")
            
            result["success"] = False
            result["errors"].append(str(e))
            self.errors += 1
            
            return result
    
    def _get_storage_root(self) -> Path:
        """获取存储根目录"""
        # 查找项目根目录中的storage目录
        current_dir = Path.cwd()
        
        # 向上查找包含storage目录的目录
        for parent in [current_dir] + list(current_dir.parents):
            storage_dir = parent / "storage"
            if storage_dir.exists():
                results_dir = storage_dir / "results"
                results_dir.mkdir(exist_ok=True)
                self.file_logger.info(f"📁 找到存储根目录: {results_dir}")
                return results_dir
        
        # 如果找不到，在当前目录创建
        fallback_dir = current_dir / "storage" / "results"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        self.file_logger.warning(f"⚠️ 未找到项目storage目录，使用: {fallback_dir}")
        return fallback_dir
    
    def _create_directories(self, task_dir: Path) -> tuple[Path, Path]:
        """创建必要的目录结构"""
        self.file_logger.info(f"📁 创建目录结构: {task_dir}")
        
        try:
            task_dir.mkdir(parents=True, exist_ok=True)
            
            images_dir = task_dir / "images"
            metadata_dir = task_dir / "metadata"
            
            if self.save_images:
                images_dir.mkdir(exist_ok=True)
                self.file_logger.info(f"   ✅ 图片目录: {images_dir}")
            
            if self.save_metadata:
                metadata_dir.mkdir(exist_ok=True)
                self.file_logger.info(f"   ✅ 元数据目录: {metadata_dir}")
            
            return images_dir, metadata_dir
            
        except Exception as e:
            self.file_logger.error(f"❌ 创建目录失败: {e}")
            raise
    
    def _generate_filename(self, task_id: Any, frame_id: int, timestamp: float) -> str:
        """生成文件名"""
        dt = datetime.fromtimestamp(timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
        filename = f"task_{task_id}_frame_{frame_id}_{timestamp_str}"
        self.file_logger.info(f"📝 生成文件名: {filename}")
        return filename
    
    def _save_image(self, frame_buffer: FrameBuffer, images_dir: Path, base_filename: str, result: Dict[str, Any]) -> bool:
        """保存图片"""
        try:
            self.file_logger.info("🖼️ 开始保存图片...")
            
            # 【修复】正确获取图像数据，兼容不同的FrameBuffer实现
            image_data = None
            
            # 方法1: 尝试零拷贝视图（最高效）
            if hasattr(frame_buffer, 'get_frame_view'):
                try:
                    image_data = frame_buffer.get_frame_view()
                    self.file_logger.info("   使用零拷贝视图获取图像数据")
                except Exception as e:
                    self.file_logger.warning(f"   零拷贝视图获取失败: {e}")
            
            # 方法2: 尝试获取副本
            if image_data is None and hasattr(frame_buffer, 'get_frame_copy'):
                try:
                    image_data = frame_buffer.get_frame_copy()
                    self.file_logger.info("   使用副本获取图像数据")
                except Exception as e:
                    self.file_logger.warning(f"   副本获取失败: {e}")
            
            # 方法3: 尝试直接访问frame_data属性
            if image_data is None and hasattr(frame_buffer, 'frame_data'):
                try:
                    frame_data = getattr(frame_buffer, 'frame_data', None)
                    if frame_data is not None:
                        image_data = frame_data
                        self.file_logger.info("   使用frame_data属性获取图像数据")
                except Exception as e:
                    self.file_logger.warning(f"   frame_data属性获取失败: {e}")
            
            # 方法4: 尝试旧的image_data属性（兼容性）
            if image_data is None:
                try:
                    image_data = getattr(frame_buffer, 'image_data', None)
                    if image_data is not None:
                        self.file_logger.info("   使用image_data属性获取图像数据")
                except Exception as e:
                    self.file_logger.warning(f"   image_data属性获取失败: {e}")
            
            # 检查是否成功获取图像数据
            if image_data is None:
                self.file_logger.error("❌ 无法从FrameBuffer获取图像数据")
                self.file_logger.error(f"   FrameBuffer类型: {type(frame_buffer)}")
                self.file_logger.error(f"   FrameBuffer属性: {[attr for attr in dir(frame_buffer) if not attr.startswith('_')]}")
                result["errors"].append("No image data in frame buffer")
                return False
            
            # 验证图像数据
            if not hasattr(image_data, 'shape'):
                self.file_logger.error(f"❌ 获取到的不是有效的图像数据: {type(image_data)}")
                result["errors"].append("Invalid image data type")
                return False
            
            self.file_logger.info(f"✅ 成功获取图像数据，尺寸: {image_data.shape}")
            
            # 处理图像（绘制检测框等）
            processed_image = self._process_image(frame_buffer, image_data)
            
            # 保存图片
            image_path = images_dir / f"{base_filename}.jpg"
            
            # 设置JPEG质量参数
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            
            success = cv2.imwrite(str(image_path), processed_image, encode_params)
            
            if success and image_path.exists():
                file_size = image_path.stat().st_size
                self.file_logger.info(f"✅ 图片保存成功: {image_path} (大小: {file_size} 字节)")
                result["paths"]["images"].append(str(image_path))
                return True
            else:
                error_msg = "图片保存失败，cv2.imwrite返回False或文件不存在"
                self.file_logger.error(f"❌ {error_msg}")
                result["errors"].append(error_msg)
                return False
            
        except Exception as e:
            error_msg = f"保存图片异常: {e}"
            self.file_logger.error(f"❌ {error_msg}")
            self.file_logger.exception("详细异常信息:")
            result["errors"].append(error_msg)
            return False
    
    def _save_metadata(self, frame_buffer: FrameBuffer, metadata_dir: Path, base_filename: str, result: Dict[str, Any]) -> bool:
        """保存元数据"""
        try:
            self.file_logger.info("📋 开始保存元数据...")
            
            # 构建元数据
            metadata = self._build_metadata(frame_buffer)
            
            self.file_logger.info(f"   检测数量: {len(metadata.get('detections', []))}")
            self.file_logger.info(f"   分析结果: {len(metadata.get('analysis_results', []))}")
            
            # 保存元数据
            metadata_path = metadata_dir / f"{base_filename}.json"
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            if metadata_path.exists():
                file_size = metadata_path.stat().st_size
                self.file_logger.info(f"✅ 元数据保存成功: {metadata_path} (大小: {file_size} 字节)")
                result["paths"]["metadata"].append(str(metadata_path))
                return True
            else:
                error_msg = "元数据文件保存后不存在"
                self.file_logger.error(f"❌ {error_msg}")
                result["errors"].append(error_msg)
                return False
            
        except Exception as e:
            error_msg = f"保存元数据异常: {e}"
            self.file_logger.error(f"❌ {error_msg}")
            self.file_logger.exception("详细异常信息:")
            result["errors"].append(error_msg)
            return False
    
    def _process_image(self, frame_buffer: FrameBuffer, image_data) -> any:
        """处理图像（绘制检测框等）"""
        try:
            # 复制图像避免修改原始数据
            processed_image = image_data.copy()
            
            if not self.draw_boxes:
                self.file_logger.info("   跳过绘制检测框 (draw_boxes=False)")
                return processed_image
            
            # 【修复】获取分析结果 - analysis_results是字典格式
            analysis_results_dict = getattr(frame_buffer, 'analysis_results', {})
            if not analysis_results_dict:
                self.file_logger.info("   没有分析结果需要绘制")
                return processed_image
            
            self.file_logger.info(f"   分析结果字典包含 {len(analysis_results_dict)} 个键")
            
            detection_count = 0
            
            # 【修复】遍历字典格式的分析结果
            for key, result in analysis_results_dict.items():
                self.file_logger.info(f"   处理结果键: {key}")
                
                if isinstance(result, dict) and "model_results" in result:
                    # 处理模型结果结构
                    model_results = result["model_results"]
                    self.file_logger.info(f"     找到 {len(model_results)} 个模型结果")
                    
                    for model_code, model_result in model_results.items():
                        detections = model_result.get("detections", [])
                        self.file_logger.info(f"     绘制模型 {model_code} 的 {len(detections)} 个检测框")
                        
                        for detection in detections:
                            self._draw_detection_box(processed_image, detection)
                            detection_count += 1
                
                elif isinstance(result, dict) and "detections" in result:
                    # 处理直接检测结构
                    detections = result["detections"]
                    self.file_logger.info(f"     绘制 {len(detections)} 个检测框")
                    
                    for detection in detections:
                        self._draw_detection_box(processed_image, detection)
                        detection_count += 1
                
                else:
                    self.file_logger.warning(f"     未识别的结果结构: {type(result)}, 键: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            
            self.file_logger.info(f"✅ 已绘制 {detection_count} 个检测框")
            return processed_image
            
        except Exception as e:
            self.file_logger.error(f"❌ 处理图像失败: {e}")
            self.file_logger.exception("详细异常信息:")
            return image_data  # 返回原始图像
    
    def _draw_detection_box(self, image, detection):
        """绘制单个检测框"""
        try:
            bbox = detection.get("bbox", {})
            if isinstance(bbox, dict):
                x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
            else:
                x1, y1, x2, y2 = bbox[:4] if len(bbox) >= 4 else [0, 0, 0, 0]
            
            # 绘制矩形框
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 绘制标签
            class_name = detection.get("class_name", "unknown")
            confidence = detection.get("confidence", 0.0)
            label = f"{class_name} {confidence:.2f}"
            
            # 计算文本位置
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_y = max(int(y1) - 10, label_size[1])
            
            # 绘制标签背景
            cv2.rectangle(image, (int(x1), label_y - label_size[1] - 5), 
                         (int(x1) + label_size[0], label_y + 5), (0, 255, 0), -1)
            
            # 绘制标签文本
            cv2.putText(image, label, (int(x1), label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        except Exception as e:
            self.file_logger.error(f"❌ 绘制检测框失败: {e}")
    
    def _build_metadata(self, frame_buffer: FrameBuffer) -> Dict[str, Any]:
        """构建元数据"""
        # 【修复】正确获取字典格式的analysis_results
        analysis_results_dict = getattr(frame_buffer, 'analysis_results', {})
        
        metadata = {
            "timestamp": getattr(frame_buffer, 'timestamp', time.time()),
            "frame_id": getattr(frame_buffer, 'frame_id', 0),
            "stream_id": getattr(frame_buffer, 'stream_id', 'unknown'),
            "image_shape": getattr(frame_buffer, 'image_shape', None),
            "analysis_results": analysis_results_dict,  # 保存原始字典格式
            "created_at": datetime.now().isoformat(),
            "processor": {
                "name": "StorageProcessor",
                "version": "1.0",
                "settings": {
                    "save_images": self.save_images,
                    "save_metadata": self.save_metadata,
                    "draw_boxes": self.draw_boxes,
                    "image_quality": self.image_quality
                }
            }
        }
        
        # 【修复】提取检测摘要 - 处理字典格式
        detections_summary = []
        
        for key, result in analysis_results_dict.items():
            if isinstance(result, dict) and "model_results" in result:
                for model_code, model_result in result["model_results"].items():
                    for detection in model_result.get("detections", []):
                        detections_summary.append({
                            "model": model_code,
                            "class_name": detection.get("class_name"),
                            "confidence": detection.get("confidence"),
                            "bbox": detection.get("bbox")
                        })
            elif isinstance(result, dict) and "detections" in result:
                for detection in result["detections"]:
                    detections_summary.append({
                        "class_name": detection.get("class_name"),
                        "confidence": detection.get("confidence"),
                        "bbox": detection.get("bbox")
                    })
        
        metadata["detections"] = detections_summary
        metadata["detection_count"] = len(detections_summary)
        
        return metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = {
            "total_processed": self.total_processed,
            "images_saved": self.images_saved,
            "metadata_saved": self.metadata_saved,
            "errors": self.errors,
            "success_rate": (self.total_processed - self.errors) / max(self.total_processed, 1) * 100,
            "config": {
                "save_images": self.save_images,
                "save_metadata": self.save_metadata,
                "draw_boxes": self.draw_boxes,
                "image_quality": self.image_quality
            }
        }
        
        self.file_logger.info(f"📊 存储处理器统计: {stats}")
        return stats
    
    def cleanup(self):
        """清理资源"""
        self.file_logger.info("🧹 存储处理器清理完成")
        self.file_logger.info(f"   最终统计: 处理{self.total_processed}, 图片{self.images_saved}, 元数据{self.metadata_saved}, 错误{self.errors}") 