"""
增强版文件编码器
集成帧稳定器和缓冲系统，确保视频流的稳定性
"""
from typing import Dict, Any, Optional, List
import os
import time
import asyncio
import threading
import subprocess
import uuid
import cv2
import numpy as np
from datetime import datetime

from core.task_management.utils.status import TaskStatus
from shared.utils.logger import get_normal_logger, get_exception_logger, get_test_logger
from services.video.utils.frame_dropper import SmartFrameDropper
from services.video.utils.ffmpeg_params import FFmpegParamsGenerator
from services.video.encoders.base_encoder import BaseEncoder
from services.video.utils.frame_stabilizer import FrameStabilizer

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()


class EnhancedFileEncoder(BaseEncoder):
    """增强版文件编码器 - 集成帧稳定器"""
    
    def __init__(self):
        """初始化增强文件编码器"""
        super().__init__()
        self.encoding_tasks = {}  # 存储编码任务信息
        self.frame_stabilizers = {}  # 每个任务的帧稳定器
        normal_logger.info("增强文件编码器初始化完成")
    
    async def start_encoding(self, task_id: str, task_manager, format: str = "mp4",
                           quality: int = 80, width: Optional[int] = None,
                           height: Optional[int] = None, fps: int = 15) -> Dict[str, Any]:
        """
        开启视频编码 - 使用帧稳定器确保输出稳定
        """
        try:
            # 检查任务是否存在
            if not task_manager.has_task(task_id):
                return {
                    "success": False,
                    "message": f"任务不存在: {task_id}",
                    "video_url": None
                }

            # 检查任务是否正在运行
            task_info = task_manager.get_task(task_id)
            if not task_info:
                return {
                    "success": False,
                    "message": f"任务信息不存在: {task_id}",
                    "video_url": None
                }

            if task_info["status"] != TaskStatus.PROCESSING:
                return {
                    "success": False,
                    "message": f"任务未在运行状态: {task_id}, 当前状态: {task_info['status']}",
                    "video_url": None
                }

            # 如果已经有编码任务，先停止
            if task_id in self.encoding_tasks:
                stop_result = await self.stop_encoding(task_id)
                if not stop_result.get("success", False):
                    normal_logger.warning(f"停止旧编码任务失败: {stop_result.get('message')}")
                await asyncio.sleep(1)

            # 验证格式
            if format.lower() not in ["mp4", "flv"]:
                return {
                    "success": False,
                    "message": f"不支持的视频格式: {format}, 仅支持mp4和flv",
                    "video_url": None
                }

            # 创建唯一的编码ID
            encoding_id = str(uuid.uuid4())

            # 创建输出目录
            output_dir = os.path.join(self.output_base_dir, encoding_id)
            os.makedirs(output_dir, exist_ok=True)

            # 确定输出文件路径
            output_file = f"output.{format.lower()}"
            output_path = os.path.join(output_dir, output_file)

            # 创建帧稳定器
            stabilizer = FrameStabilizer(target_fps=fps, buffer_size=60)  # 增大缓冲区
            self.frame_stabilizers[task_id] = stabilizer

            # 存储编码任务信息
            self.encoding_tasks[task_id] = {
                "encoding_id": encoding_id,
                "output_dir": output_dir,
                "output_path": output_path,
                "format": format.lower(),
                "quality": quality,
                "width": width,
                "height": height,
                "fps": fps,
                "start_time": datetime.now().isoformat(),
                "stabilizer": stabilizer
            }

            # 启动增强编码进程
            video_url = await self._start_enhanced_encoding_process(
                task_id=task_id,
                encoding_id=encoding_id,
                output_dir=output_dir,
                output_path=output_path,
                format=format.lower(),
                task_manager=task_manager,
                quality=quality,
                width=width,
                height=height,
                fps=fps
            )

            if not video_url:
                if task_id in self.encoding_tasks:
                    del self.encoding_tasks[task_id]
                if task_id in self.frame_stabilizers:
                    del self.frame_stabilizers[task_id]
                return {
                    "success": False,
                    "message": f"启动视频编码失败: {task_id}",
                    "video_url": None
                }

            return {
                "success": True,
                "message": f"增强视频编码已启动: {task_id}",
                "video_url": video_url
            }

        except Exception as e:
            exception_logger.exception(f"启动增强视频编码失败: {str(e)}")
            return {
                "success": False,
                "message": f"启动增强视频编码失败: {str(e)}",
                "video_url": None
            }
    
    async def stop_encoding(self, task_id: str) -> Dict[str, Any]:
        """停止增强视频编码"""
        try:
            if task_id not in self.encoding_tasks:
                return {
                    "success": False,
                    "message": f"编码任务不存在: {task_id}"
                }

            encoding_info = self.encoding_tasks[task_id]
            encoding_id = encoding_info["encoding_id"]

            # 停止帧稳定器
            if task_id in self.frame_stabilizers:
                self.frame_stabilizers[task_id].stop_stable_output()
                del self.frame_stabilizers[task_id]

            # 停止FFmpeg进程
            if task_id in self.ffmpeg_processes:
                process = self.ffmpeg_processes[task_id]
                try:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                    normal_logger.info(f"FFmpeg进程已停止: {task_id}")
                except Exception as e:
                    exception_logger.exception(f"停止FFmpeg进程时出错: {str(e)}")

                del self.ffmpeg_processes[task_id]

            # 清理任务信息
            del self.encoding_tasks[task_id]

            normal_logger.info(f"增强编码任务已停止: {task_id}, 编码ID: {encoding_id}")

            return {
                "success": True,
                "message": f"增强编码任务已停止: {task_id}"
            }

        except Exception as e:
            exception_logger.exception(f"停止增强编码任务失败: {str(e)}")
            return {
                "success": False,
                "message": f"停止增强编码任务失败: {str(e)}"
            }
    
    async def _start_enhanced_encoding_process(self, task_id: str, encoding_id: str, output_dir: str,
                                             output_path: str, format: str, task_manager, quality: int = 80,
                                             width: Optional[int] = None, height: Optional[int] = None,
                                             fps: int = 15) -> Optional[str]:
        """启动增强编码进程"""
        try:
            # 创建编码线程
            encoding_thread = threading.Thread(
                target=self._enhanced_encoding_thread,
                args=(task_id, encoding_id, output_dir, output_path, format, task_manager, quality, width, height, fps),
                daemon=True
            )
            encoding_thread.start()

            # 存储编码线程
            self.encoding_threads[task_id] = encoding_thread

            # 等待编码进程初始化
            normal_logger.info(f"等待增强编码进程初始化: {task_id}")
            await asyncio.sleep(3)

            # 构建视频URL
            output_file_name = os.path.basename(output_path)
            video_url = f"{self.base_url}/api/v1/tasks/video/{encoding_id}/{output_file_name}"

            return video_url

        except Exception as e:
            normal_logger.exception(f"启动增强编码进程失败: {str(e)}")
            return None
    
    def _enhanced_encoding_thread(self, task_id: str, encoding_id: str, output_dir: str, output_path: str,
                                format: str, task_manager=None, quality: int = 80,
                                width: Optional[int] = None, height: Optional[int] = None, fps: int = 15):
        """
        增强编码线程 - 集成帧稳定器
        """
        normal_logger.info(f"增强编码线程启动: {task_id}, 编码ID: {encoding_id}, 格式: {format}")
        
        stabilizer = self.frame_stabilizers.get(task_id)
        if not stabilizer:
            normal_logger.error(f"帧稳定器不存在: {task_id}")
            return

        try:
            # 获取任务处理器
            task_processor = None
            try:
                if task_manager:
                    task_processor = task_manager.processor
            except Exception as e:
                normal_logger.error(f"获取任务处理器失败: {str(e)}")

            # 获取第一帧以确定分辨率
            frame = None
            retry_count = 0
            max_retries = 100

            if task_processor:
                normal_logger.info(f"开始获取预览帧: {task_id}")
                while frame is None and retry_count < max_retries:
                    try:
                        frame = task_processor.get_preview_frame(task_id)
                        if frame is not None:
                            normal_logger.info(f"成功获取预览帧: {task_id}, 帧形状: {frame.shape}")
                    except Exception as e:
                        normal_logger.error(f"获取预览帧失败: {str(e)}")
                        frame = None

                    if frame is None:
                        time.sleep(0.1)
                        retry_count += 1
                        if retry_count % 10 == 0:
                            normal_logger.info(f"等待预览帧中: {task_id}, 已重试 {retry_count} 次")

                        if task_id not in self.encoding_tasks:
                            normal_logger.info(f"编码任务已停止，编码线程退出: {task_id}")
                            return

            # 如果没有获取到帧，使用默认帧
            if frame is None:
                normal_logger.warning(f"无法获取预览帧，使用默认帧: {task_id}")
                frame = self.create_default_frame(640, 480, "等待视频流...")

            # 确定视频分辨率
            frame_height, frame_width = frame.shape[:2]
            final_width = width if width is not None else frame_width
            final_height = height if height is not None else frame_height

            normal_logger.info(f"增强编码参数 - 任务ID: {task_id}, 分辨率: {final_width}x{final_height}, 帧率: {fps}")

            # 使用优化的比特率计算
            bitrate = FFmpegParamsGenerator.calculate_optimal_bitrate(quality, final_width, final_height, fps)
            
            # 使用优化的FFmpeg参数
            ffmpeg_cmd = FFmpegParamsGenerator.get_realtime_ffmpeg_params(final_width, final_height, fps, bitrate, output_path)

            # 根据格式调整输出参数
            if format == "mp4":
                ffmpeg_cmd = FFmpegParamsGenerator.adjust_mp4_params(ffmpeg_cmd, output_path)
            elif format == "flv":
                ffmpeg_cmd = FFmpegParamsGenerator.adjust_flv_params(ffmpeg_cmd, output_path)

            normal_logger.info(f"增强FFmpeg命令: {' '.join(ffmpeg_cmd)}")

            # 启动FFmpeg进程
            try:
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=20*1024*1024  # 增大缓冲区到20MB
                )

                self.ffmpeg_processes[task_id] = process
                normal_logger.info(f"FFmpeg进程已启动: {task_id}")

                # 启动帧提供循环
                asyncio.run(self._stabilized_frame_loop(
                    task_id, task_processor, process, stabilizer, 
                    final_width, final_height, fps
                ))

            except Exception as e:
                normal_logger.error(f"启动FFmpeg进程失败: {str(e)}")
                return

        except Exception as e:
            normal_logger.error(f"增强编码线程运行时出错: {str(e)}")
            import traceback
            normal_logger.error(traceback.format_exc())

        finally:
            # 清理
            if task_id in self.ffmpeg_processes:
                process = self.ffmpeg_processes[task_id]
                if process.stdin:
                    try:
                        process.stdin.close()
                    except:
                        pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                
                del self.ffmpeg_processes[task_id]

            normal_logger.info(f"增强编码线程结束: {task_id}")
    
    async def _stabilized_frame_loop(self, task_id: str, task_processor, ffmpeg_process, 
                                   stabilizer: FrameStabilizer, width: int, height: int, fps: int):
        """
        稳定化帧循环
        """
        normal_logger.info(f"开始稳定化帧循环: {task_id}")
        
        frame_count = 0
        last_stats_time = time.time()
        
        try:
            while (task_id in self.encoding_tasks and 
                   ffmpeg_process.poll() is None):
                
                # 获取源帧
                source_frame = None
                analysis_result = None
                
                if task_processor:
                    try:
                        source_frame = task_processor.get_preview_frame(task_id)
                        # 这里可以添加获取分析结果的逻辑
                        # analysis_result = task_processor.get_analysis_result(task_id)
                    except Exception as e:
                        normal_logger.debug(f"获取源帧失败: {str(e)}")
                
                # 添加源帧到稳定器
                if source_frame is not None:
                    # 调整帧大小
                    if source_frame.shape[1] != width or source_frame.shape[0] != height:
                        source_frame = cv2.resize(source_frame, (width, height))
                    
                    stabilizer.add_source_frame(source_frame, analysis_result)
                
                # 获取稳定帧
                stable_frame, source_type, metadata = stabilizer.get_stable_frame()
                
                if stable_frame is not None:
                    try:
                        # 写入FFmpeg
                        ffmpeg_process.stdin.write(stable_frame.tobytes())
                        frame_count += 1
                        
                        # 记录使用缓存帧的情况
                        if source_type in ["cached", "repeated", "interpolated"]:
                            normal_logger.debug(f"使用{source_type}帧: {task_id}, 元数据: {metadata}")
                        
                    except BrokenPipeError:
                        normal_logger.error(f"FFmpeg管道已断开: {task_id}")
                        break
                    except Exception as e:
                        normal_logger.error(f"写入FFmpeg失败: {str(e)}")
                        break
                
                # 每100帧输出统计信息
                current_time = time.time()
                if current_time - last_stats_time >= 10.0:  # 每10秒
                    stats = stabilizer.get_stabilizer_stats()
                    normal_logger.info(f"稳定器统计 [{task_id}] - 总帧数: {frame_count}, "
                                     f"重复帧: {stats['stabilizer']['repeated_frames']}, "
                                     f"插值帧: {stats['stabilizer']['interpolated_frames']}, "
                                     f"平均FPS: {stats['stabilizer']['average_output_fps']:.1f}")
                    last_stats_time = current_time
                
                # 控制循环频率
                await asyncio.sleep(1.0 / (fps * 1.5))  # 稍微快一点确保不缺帧
                
        except Exception as e:
            normal_logger.error(f"稳定化帧循环异常: {str(e)}")
        finally:
            stabilizer.stop_stable_output()
            normal_logger.info(f"稳定化帧循环结束: {task_id}")
    
    def get_encoding_stats(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取编码统计信息"""
        if task_id not in self.frame_stabilizers:
            return None
        
        stabilizer = self.frame_stabilizers[task_id]
        return stabilizer.get_stabilizer_stats()


# 全局增强文件编码器实例
enhanced_file_encoder = EnhancedFileEncoder() 