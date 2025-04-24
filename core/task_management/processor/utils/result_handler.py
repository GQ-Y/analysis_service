"""
结果处理器
处理检测结果和图片保存
"""
import os
import cv2
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class ResultHandler:
    """结果处理器"""
    
    async def process_detection_result(
        self,
        result: Dict[str, Any],
        task_id: str,
        frame: Any,
        frame_count: int,
        task_config: Dict[str, Any],
        tracker: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        处理检测结果
        
        Args:
            result: 检测结果
            task_id: 任务ID
            frame: 视频帧
            frame_count: 帧计数
            task_config: 任务配置
            tracker: 跟踪器实例
            
        Returns:
            Dict[str, Any]: 处理后的结果
        """
        try:
            # 1. 获取检测结果
            detections = result.get("detections", [])
            
            # 2. 应用类别过滤
            if task_config.get("analysis", {}).get("classes"):
                allowed_classes = task_config["analysis"]["classes"]
                detections = [
                    det for det in detections 
                    if det["class_id"] in allowed_classes
                ]
                
            # 3. 应用ROI过滤
            if task_config.get("roi"):
                roi_config = task_config["roi"]["config"]
                roi_type = task_config["roi"].get("type", 1)
                height, width = frame.shape[:2]
                detections = self._filter_by_roi(
                    detections,
                    roi_config,
                    roi_type,
                    height,
                    width
                )
                
            # 4. 执行跟踪
            if tracker:
                tracked_objects = await tracker.update(detections)
                detections = [obj.to_dict() for obj in tracked_objects]
                
            # 5. 保存结果图片
            image_results = {}
            if detections and task_config["result"].get("save_images", False):
                image_results = await self._save_result_image(
                    frame,
                    detections,
                    task_id,
                    frame_count,
                    task_config
                )
                
            # 6. 准备返回结果
            return {
                "task_id": task_id,
                "subtask_id": task_config["subtask"]["id"],
                "status": "0",  # 进行中
                "progress": 1,  # 流分析永远是1
                "timestamp": int(datetime.now().timestamp()),
                "result": {
                    "frame_id": frame_count,
                    "objects": self._format_detections(detections),
                    "frame_info": {
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                        "frame_index": frame_count,
                        "timestamp": int(datetime.now().timestamp())
                    },
                    "image_results": image_results,
                    "analysis_info": {
                        "model_name": task_config["model"]["code"],
                        "model_version": task_config["model"]["version"],
                        "device": task_config["model"]["device"]
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"处理检测结果失败: {str(e)}")
            raise
            
    def _filter_by_roi(
        self,
        detections: List[Dict],
        roi_config: Dict,
        roi_type: int,
        height: int,
        width: int
    ) -> List[Dict]:
        """应用ROI过滤"""
        filtered_detections = []
        for det in detections:
            bbox = det["bbox"]
            # 根据ROI类型和配置进行过滤
            # 这里需要实现具体的ROI过滤逻辑
            filtered_detections.append(det)
        return filtered_detections
        
    def _format_detections(self, detections: List[Dict]) -> List[Dict]:
        """格式化检测结果"""
        formatted = []
        for det in detections:
            obj = {
                "class_id": det["class_id"],
                "class_name": det.get("class_name", ""),
                "confidence": det["confidence"],
                "bbox": det["bbox"]
            }
            if "track_id" in det:
                obj["track_id"] = det["track_id"]
                obj["track_info"] = det.get("track_info", {})
            formatted.append(obj)
        return formatted
        
    async def _save_result_image(
        self,
        frame: Any,
        detections: List[Dict],
        task_id: str,
        frame_count: int,
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """保存结果图片"""
        try:
            result_config = task_config["result"]
            storage_config = result_config.get("storage", {})
            
            # 1. 准备保存路径
            base_save_dir = storage_config.get("save_path", "results")
            file_pattern = storage_config.get(
                "file_pattern",
                "{task_id}/{date}/{time}_{frame_id}.jpg"
            )
            
            # 2. 生成文件名
            timestamp = datetime.now()
            filename = file_pattern.format(
                task_id=task_id,
                date=timestamp.strftime("%Y%m%d"),
                time=timestamp.strftime("%H%M%S"),
                frame_id=frame_count
            )
            
            # 3. 完整保存路径
            save_path = os.path.join(base_save_dir, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 4. 绘制检测结果
            result_image = frame.copy()
            for det in detections:
                bbox = det["bbox"]
                label = f"{det.get('class_name', '')} {det.get('track_id', '')}"
                # 这里需要实现具体的绘制逻辑
                
            # 5. 保存图片
            cv2.imwrite(save_path, result_image)
            
            # 6. 如果需要返回base64
            image_results = {"save_path": save_path}
            if result_config.get("return_base64", False):
                image_format = result_config.get("image_format", {})
                _, buffer = cv2.imencode(
                    f".{image_format.get('format', 'jpg')}",
                    result_image,
                    [cv2.IMWRITE_JPEG_QUALITY, image_format.get("quality", 95)]
                )
                image_results["base64"] = base64.b64encode(buffer).decode('utf-8')
                
            return image_results
            
        except Exception as e:
            logger.error(f"保存结果图片失败: {str(e)}")
            return {} 