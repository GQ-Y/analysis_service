"""
任务处理器
处理分析任务，包括图像分析、视频分析和流分析
"""
import os
import cv2
import time
import uuid
import base64
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable, Union

from core.task_manager import TaskManager
from core.detector import YOLODetector
from shared.utils.logger import setup_logger
from core.config import settings

logger = setup_logger(__name__)

class TaskProcessor:
    """任务处理器，处理分析任务"""
    
    def __init__(self):
        """初始化任务处理器"""
        self.task_manager = TaskManager.get_instance()
        self.detectors = {}  # 存储已加载的检测器
        self.active_tasks = {}  # 存储活动任务的线程或协程
        
        logger.info("任务处理器已初始化")
        
    def get_detector(self, model_code: str) -> YOLODetector:
        """
        获取或创建检测器
        
        Args:
            model_code: 模型代码
            
        Returns:
            YOLODetector: 检测器实例
        """
        if model_code not in self.detectors:
            try:
                logger.info(f"加载检测模型: {model_code}")
                self.detectors[model_code] = YOLODetector(model_code)
            except Exception as e:
                logger.error(f"加载模型失败: {model_code}, 错误: {str(e)}", exc_info=True)
                raise ValueError(f"加载模型失败: {model_code}, 错误: {str(e)}")
                
        return self.detectors[model_code]
        
    def process_image(self, task_id: str) -> Dict[str, Any]:
        """
        处理图像分析任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 处理结果
        """
        try:
            # 获取任务信息
            task = self.task_manager.get_task(task_id)
            if not task:
                return {"success": False, "error": f"任务不存在: {task_id}"}
                
            # 更新任务状态
            self.task_manager.update_task_status(task_id, "processing")
            
            # 获取任务参数
            params = task.get("params", {})
            image_path = params.get("image_path")
            model_code = params.get("model_code")
            conf_threshold = params.get("conf_threshold", 0.25)
            save_result = params.get("save_result", True)
            include_image = params.get("include_image", False)
            
            # 检查参数
            if not image_path or not model_code:
                self.task_manager.update_task_status(task_id, "failed", {"error": "缺少必要参数"})
                return {"success": False, "error": "缺少必要参数"}
                
            # 检查文件是否存在
            if not os.path.exists(image_path):
                self.task_manager.update_task_status(task_id, "failed", {"error": f"图像文件不存在: {image_path}"})
                return {"success": False, "error": f"图像文件不存在: {image_path}"}
                
            # 获取检测器
            detector = self.get_detector(model_code)
            
            # 读取图像
            try:
                image = cv2.imread(image_path)
                if image is None:
                    self.task_manager.update_task_status(task_id, "failed", {"error": f"无法读取图像: {image_path}"})
                    return {"success": False, "error": f"无法读取图像: {image_path}"}
            except Exception as e:
                self.task_manager.update_task_status(task_id, "failed", {"error": f"读取图像错误: {str(e)}"})
                return {"success": False, "error": f"读取图像错误: {str(e)}"}
                
            # 执行检测
            try:
                logger.info(f"执行图像检测: {image_path}, 模型: {model_code}")
                result = detector.detect(image, conf_threshold)
                detections = result.get("detections", [])
                logger.info(f"检测完成，发现 {len(detections)} 个目标")
                
                # 绘制检测结果
                result_image = detector.draw_detections(image, detections)
                
                # 保存结果
                result_data = {
                    "detections": detections,
                    "filename": os.path.basename(image_path),
                    "image_path": image_path,
                    "model_code": model_code,
                    "conf_threshold": conf_threshold
                }
                
                if save_result:
                    # 保存检测结果图像
                    output_dir = f"{settings.OUTPUT.save_dir}/images"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    result_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_result.jpg"
                    result_path = f"{output_dir}/{result_filename}"
                    
                    cv2.imwrite(result_path, result_image)
                    result_data["result_path"] = result_path
                    logger.info(f"已保存检测结果图像: {result_path}")
                
                # 如果需要，添加图像的base64编码
                if include_image:
                    _, buffer = cv2.imencode('.jpg', result_image)
                    result_data["image_base64"] = base64.b64encode(buffer).decode('utf-8')
                
                # 更新任务状态
                self.task_manager.update_task_status(task_id, "completed", result_data)
                
                # 返回结果
                return {
                    "success": True,
                    "task_id": task_id,
                    "detections": detections,
                    "filename": os.path.basename(image_path),
                    "image_base64": result_data.get("image_base64") if include_image else None,
                    "result_path": result_data.get("result_path") if save_result else None
                }
                
            except Exception as e:
                logger.error(f"图像检测失败: {str(e)}", exc_info=True)
                self.task_manager.update_task_status(task_id, "failed", {"error": f"图像检测失败: {str(e)}"})
                return {"success": False, "error": f"图像检测失败: {str(e)}"}
                
        except Exception as e:
            logger.error(f"处理图像分析任务失败: {str(e)}", exc_info=True)
            self.task_manager.update_task_status(task_id, "failed", {"error": f"处理任务失败: {str(e)}"})
            return {"success": False, "error": f"处理任务失败: {str(e)}"}
            
    def start_video_analysis(self, task_id: str) -> Dict[str, Any]:
        """
        启动视频分析任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 启动结果
        """
        try:
            # 获取任务信息
            task = self.task_manager.get_task(task_id)
            if not task:
                return {"success": False, "error": f"任务不存在: {task_id}"}
                
            # 更新任务状态
            self.task_manager.update_task_status(task_id, "pending")
            
            # 获取任务参数
            params = task.get("params", {})
            video_path = params.get("video_path")
            model_code = params.get("model_code")
            
            # 检查参数
            if not video_path or not model_code:
                self.task_manager.update_task_status(task_id, "failed", {"error": "缺少必要参数"})
                return {"success": False, "error": "缺少必要参数"}
                
            # 检查文件是否存在
            if not os.path.exists(video_path):
                self.task_manager.update_task_status(task_id, "failed", {"error": f"视频文件不存在: {video_path}"})
                return {"success": False, "error": f"视频文件不存在: {video_path}"}
                
            # 启动后台线程处理视频
            thread = threading.Thread(
                target=self._process_video,
                args=(task_id,),
                daemon=True
            )
            self.active_tasks[task_id] = thread
            thread.start()
            
            return {
                "success": True,
                "task_id": task_id,
                "message": "已启动视频分析任务"
            }
            
        except Exception as e:
            logger.error(f"启动视频分析任务失败: {str(e)}", exc_info=True)
            self.task_manager.update_task_status(task_id, "failed", {"error": f"启动任务失败: {str(e)}"})
            return {"success": False, "error": f"启动任务失败: {str(e)}"}
            
    def _process_video(self, task_id: str):
        """
        处理视频分析任务
        
        Args:
            task_id: 任务ID
        """
        task = self.task_manager.get_task(task_id)
        if not task:
            logger.error(f"处理视频任务失败: 任务不存在 {task_id}")
            return
            
        params = task.get("params", {})
        video_path = params.get("video_path")
        model_code = params.get("model_code")
        conf_threshold = params.get("conf_threshold", 0.25)
        save_result = params.get("save_result", True)
        
        try:
            # 更新任务状态
            self.task_manager.update_task_status(task_id, "processing")
            
            # 获取检测器
            detector = self.get_detector(model_code)
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.task_manager.update_task_status(task_id, "failed", {"error": f"无法打开视频: {video_path}"})
                return
                
            # 获取视频信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 创建输出目录
            output_dir = f"{settings.OUTPUT.save_dir}/videos"
            os.makedirs(output_dir, exist_ok=True)
            
            # 设置输出视频
            result_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_result.mp4"
            result_path = f"{output_dir}/{result_filename}"
            
            # 输出帧图像的目录
            frames_dir = f"{output_dir}/{os.path.splitext(os.path.basename(video_path))[0]}_frames"
            if save_result:
                os.makedirs(frames_dir, exist_ok=True)
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = None
            if save_result:
                out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
            
            # 处理视频帧
            frame_count = 0
            all_detections = []
            
            logger.info(f"开始处理视频: {video_path}, 总帧数: {total_frames}, 帧率: {fps}")
            
            start_time = time.time()
            
            while True:
                # 检查任务是否被取消
                if self.task_manager.get_task_status(task_id) == "stopping":
                    logger.info(f"视频分析任务被取消: {task_id}")
                    break
                    
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 每10帧更新一次进度
                if frame_count % 10 == 0:
                    progress = frame_count / total_frames * 100 if total_frames > 0 else 0
                    self.task_manager.update_task_progress(task_id, progress)
                    logger.debug(f"视频分析进度: {progress:.2f}%, 帧: {frame_count}/{total_frames}")
                
                # 执行检测
                result = detector.detect(frame, conf_threshold)
                detections = result.get("detections", [])
                
                # 为每个检测添加帧信息
                for det in detections:
                    det["frame"] = frame_count
                    
                all_detections.extend(detections)
                
                # 绘制检测结果
                result_frame = detector.draw_detections(frame, detections)
                
                # 保存结果
                if save_result:
                    # 保存关键帧（每秒一帧或有检测结果的帧）
                    if len(detections) > 0 or frame_count % int(fps) == 0:
                        frame_filename = f"{frames_dir}/frame_{frame_count:06d}.jpg"
                        cv2.imwrite(frame_filename, result_frame)
                    
                    # 写入视频
                    out.write(result_frame)
                
                frame_count += 1
            
            # 处理完成
            elapsed_time = time.time() - start_time
            
            # 释放资源
            cap.release()
            if out:
                out.release()
            
            # 更新任务状态为完成
            if self.task_manager.get_task_status(task_id) != "stopping":
                result_data = {
                    "detections": all_detections,
                    "total_frames": frame_count,
                    "processed_frames": frame_count,
                    "duration": elapsed_time,
                    "fps": frame_count / elapsed_time if elapsed_time > 0 else 0,
                    "video_path": video_path,
                    "model_code": model_code,
                    "conf_threshold": conf_threshold
                }
                
                if save_result:
                    result_data["result_path"] = result_path
                    result_data["frames_dir"] = frames_dir
                
                self.task_manager.update_task_status(task_id, "completed", result_data)
                logger.info(f"视频分析任务完成: {task_id}, 处理帧数: {frame_count}, 耗时: {elapsed_time:.2f}秒")
            else:
                self.task_manager.update_task_status(task_id, "stopped")
                logger.info(f"视频分析任务已停止: {task_id}")
            
        except Exception as e:
            logger.error(f"处理视频失败: {str(e)}", exc_info=True)
            self.task_manager.update_task_status(task_id, "failed", {"error": f"处理视频失败: {str(e)}"})
            
        finally:
            # 从活动任务中移除
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
                
    def start_stream_analysis(self, task_id: str) -> Dict[str, Any]:
        """
        启动流分析任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 启动结果
        """
        try:
            # 获取任务信息
            task = self.task_manager.get_task(task_id)
            if not task:
                return {"success": False, "error": f"任务不存在: {task_id}"}
                
            # 更新任务状态
            self.task_manager.update_task_status(task_id, "pending")
            
            # 获取任务参数
            params = task.get("params", {})
            stream_url = params.get("stream_url")
            model_code = params.get("model_code")
            
            # 检查参数
            if not stream_url or not model_code:
                self.task_manager.update_task_status(task_id, "failed", {"error": "缺少必要参数"})
                return {"success": False, "error": "缺少必要参数"}
                
            # 启动后台线程处理流
            thread = threading.Thread(
                target=self._process_stream,
                args=(task_id,),
                daemon=True
            )
            self.active_tasks[task_id] = thread
            thread.start()
            
            return {
                "success": True,
                "task_id": task_id,
                "message": "已启动流分析任务"
            }
            
        except Exception as e:
            logger.error(f"启动流分析任务失败: {str(e)}", exc_info=True)
            self.task_manager.update_task_status(task_id, "failed", {"error": f"启动任务失败: {str(e)}"})
            return {"success": False, "error": f"启动任务失败: {str(e)}"}
            
    def _process_stream(self, task_id: str):
        """
        处理流分析任务
        
        Args:
            task_id: 任务ID
        """
        task = self.task_manager.get_task(task_id)
        if not task:
            logger.error(f"处理流任务失败: 任务不存在 {task_id}")
            return
            
        params = task.get("params", {})
        stream_url = params.get("stream_url")
        model_code = params.get("model_code")
        conf_threshold = params.get("conf_threshold", 0.25)
        save_interval = params.get("save_interval", 10)  # 保存图像的间隔（秒）
        max_duration = params.get("max_duration", 3600)  # 最大持续时间（秒）
        
        try:
            # 更新任务状态
            self.task_manager.update_task_status(task_id, "processing")
            
            # 获取检测器
            detector = self.get_detector(model_code)
            
            # 打开视频流
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                self.task_manager.update_task_status(task_id, "failed", {"error": f"无法打开流: {stream_url}"})
                return
                
            # 创建输出目录
            output_dir = f"{settings.OUTPUT.save_dir}/streams"
            os.makedirs(output_dir, exist_ok=True)
            
            # 输出帧图像的目录
            stream_id = uuid.uuid4().hex[:8]
            frames_dir = f"{output_dir}/stream_{stream_id}"
            os.makedirs(frames_dir, exist_ok=True)
            
            # 处理视频帧
            frame_count = 0
            latest_detections = []
            latest_frame = None
            latest_frame_time = 0
            last_save_time = time.time()
            start_time = time.time()
            
            logger.info(f"开始处理流: {stream_url}")
            
            while True:
                # 检查任务是否被取消
                if self.task_manager.get_task_status(task_id) == "stopping":
                    logger.info(f"流分析任务被取消: {task_id}")
                    break
                    
                # 检查是否超过最大持续时间
                if time.time() - start_time > max_duration:
                    logger.info(f"流分析任务达到最大持续时间: {task_id}")
                    break
                    
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"无法读取流帧，尝试重新连接: {stream_url}")
                    # 重新连接流
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(stream_url)
                    if not cap.isOpened():
                        logger.error(f"无法重新连接流: {stream_url}")
                        break
                    continue
                
                # 执行检测
                result = detector.detect(frame, conf_threshold)
                detections = result.get("detections", [])
                
                # 更新最新检测结果和帧
                latest_detections = detections
                latest_frame = detector.draw_detections(frame, detections)
                latest_frame_time = time.time()
                
                # 每10帧更新一次任务数据
                if frame_count % 10 == 0:
                    self.task_manager.update_task_data(task_id, {
                        "frame_count": frame_count,
                        "latest_detection_count": len(latest_detections),
                        "running_time": time.time() - start_time
                    })
                
                # 定期保存检测结果图像
                current_time = time.time()
                if current_time - last_save_time >= save_interval:
                    # 保存检测结果图像
                    timestamp = int(current_time)
                    frame_filename = f"{frames_dir}/frame_{timestamp}.jpg"
                    cv2.imwrite(frame_filename, latest_frame)
                    
                    # 更新任务数据
                    self.task_manager.update_task_data(task_id, {
                        "latest_frame": frame_filename,
                        "latest_detections": latest_detections,
                        "latest_frame_time": timestamp
                    })
                    
                    logger.debug(f"已保存流检测结果: {frame_filename}, 检测数量: {len(latest_detections)}")
                    last_save_time = current_time
                
                frame_count += 1
            
            # 处理完成
            elapsed_time = time.time() - start_time
            
            # 保存最后一帧
            if latest_frame is not None:
                final_frame_path = f"{frames_dir}/final_frame.jpg"
                cv2.imwrite(final_frame_path, latest_frame)
            
            # 释放资源
            cap.release()
            
            # 更新任务状态
            if self.task_manager.get_task_status(task_id) != "stopping":
                result_data = {
                    "total_frames": frame_count,
                    "latest_detections": latest_detections,
                    "duration": elapsed_time,
                    "frames_dir": frames_dir,
                    "stream_url": stream_url,
                    "model_code": model_code,
                    "conf_threshold": conf_threshold
                }
                
                if latest_frame is not None:
                    result_data["final_frame"] = final_frame_path
                
                self.task_manager.update_task_status(task_id, "completed", result_data)
                logger.info(f"流分析任务完成: {task_id}, 处理帧数: {frame_count}, 耗时: {elapsed_time:.2f}秒")
            else:
                self.task_manager.update_task_status(task_id, "stopped")
                logger.info(f"流分析任务已停止: {task_id}")
            
        except Exception as e:
            logger.error(f"处理流失败: {str(e)}", exc_info=True)
            self.task_manager.update_task_status(task_id, "failed", {"error": f"处理流失败: {str(e)}"})
            
        finally:
            # 从活动任务中移除
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
                
    def stop_task(self, task_id: str) -> Dict[str, Any]:
        """
        停止任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 停止结果
        """
        try:
            # 获取任务信息
            task = self.task_manager.get_task(task_id)
            if not task:
                return {"success": False, "error": f"任务不存在: {task_id}"}
                
            # 获取任务状态
            status = task.get("status")
            
            # 如果任务已经完成或失败，无需停止
            if status in ["completed", "failed", "stopped"]:
                return {
                    "success": True,
                    "task_id": task_id,
                    "message": f"任务已经处于 {status} 状态，无需停止"
                }
                
            # 更新任务状态为正在停止
            self.task_manager.update_task_status(task_id, "stopping")
            logger.info(f"正在停止任务: {task_id}")
            
            # 等待任务停止（由处理线程自行停止）
            return {
                "success": True,
                "task_id": task_id,
                "message": "正在停止任务"
            }
            
        except Exception as e:
            logger.error(f"停止任务失败: {str(e)}", exc_info=True)
            return {"success": False, "error": f"停止任务失败: {str(e)}"}
            
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 任务状态
        """
        try:
            # 获取任务信息
            task = self.task_manager.get_task(task_id)
            if not task:
                return {"success": False, "error": f"任务不存在: {task_id}"}
                
            return {
                "success": True,
                "task_id": task_id,
                "status": task.get("status"),
                "progress": task.get("progress", 0),
                "created_at": task.get("created_at"),
                "updated_at": task.get("updated_at"),
                "result": task.get("result")
            }
            
        except Exception as e:
            logger.error(f"获取任务状态失败: {str(e)}", exc_info=True)
            return {"success": False, "error": f"获取任务状态失败: {str(e)}"}
            
    def get_tasks(self, protocol: Optional[str] = None) -> Dict[str, Any]:
        """
        获取任务列表
        
        Args:
            protocol: 协议过滤
            
        Returns:
            Dict: 任务列表
        """
        try:
            # 获取所有任务
            tasks = self.task_manager.get_all_tasks()
            
            # 如果指定了协议，过滤任务
            if protocol:
                filtered_tasks = {
                    task_id: task for task_id, task in tasks.items()
                    if task.get("protocol") == protocol
                }
                return {"success": True, "tasks": filtered_tasks}
            
            return {"success": True, "tasks": tasks}
            
        except Exception as e:
            logger.error(f"获取任务列表失败: {str(e)}", exc_info=True)
            return {"success": False, "error": f"获取任务列表失败: {str(e)}"} 