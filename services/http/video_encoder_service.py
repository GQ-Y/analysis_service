"""
视频编码服务
提供实时分析视频的MP4/FLV编码功能
"""
from typing import Dict, Any, Optional, List, Union
import os
import time
import asyncio
import threading
import subprocess
import uuid
import cv2
import numpy as np
from datetime import datetime

from core.config import settings
from core.task_management.utils.status import TaskStatus
from shared.utils.logger import get_normal_logger, get_exception_logger, get_test_logger

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()

class VideoEncoderService:
    """视频编码服务 - 支持MP4和FLV格式"""

    def __init__(self):
        """初始化视频编码服务"""
        self.encoding_tasks = {}  # 存储编码任务信息

        # 创建输出目录结构
        self.output_base_dir = os.path.join(os.getcwd(), "temp", "videos")
        os.makedirs(self.output_base_dir, exist_ok=True)
        normal_logger.info(f"视频输出基础目录: {self.output_base_dir}")

        # 存储FFmpeg进程
        self.ffmpeg_processes = {}

        # 存储编码线程
        self.encoding_threads = {}

        # 服务基础URL
        host = "localhost" if settings.SERVICES_HOST == "0.0.0.0" else settings.SERVICES_HOST
        self.base_url = f"http://{host}:{settings.SERVICES_PORT}"

        # 检查FFmpeg是否可用
        self._check_ffmpeg()

        normal_logger.info(f"视频编码服务初始化完成，输出目录: {self.output_base_dir}")

    def _check_ffmpeg(self):
        """检查FFmpeg是否可用"""
        try:
            # 获取FFmpeg版本信息
            result = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if result.returncode == 0:
                ffmpeg_version = result.stdout.split('\n')[0]
                normal_logger.info(f"FFmpeg可用: {ffmpeg_version}")
                return True
            else:
                exception_logger.error(f"FFmpeg不可用，返回码: {result.returncode}")
                exception_logger.error(f"错误信息: {result.stderr}")
                return False
        except Exception as e:
            exception_logger.exception(f"FFmpeg检查失败: {str(e)}")
            return False

    async def start_encoding(self, task_id: str, task_manager, format: str = "mp4",
                           quality: int = 80, width: Optional[int] = None,
                           height: Optional[int] = None, fps: int = 15) -> Dict[str, Any]:
        """
        开启视频编码 - 将分析结果转为MP4或FLV格式

        Args:
            task_id: 任务ID
            task_manager: 任务管理器实例
            format: 视频格式，支持"mp4"或"flv"
            quality: 视频质量(1-100)
            width: 视频宽度，为空则使用原始宽度
            height: 视频高度，为空则使用原始高度
            fps: 视频帧率

        Returns:
            Dict[str, Any]: 编码结果，包含视频URL
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
                # 等待一小段时间，确保旧的编码任务完全停止
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
                "start_time": datetime.now().isoformat()
            }

            # 启动编码进程
            video_url = await self._start_encoding_process(
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
                # 如果启动失败，删除编码任务信息
                if task_id in self.encoding_tasks:
                    del self.encoding_tasks[task_id]
                return {
                    "success": False,
                    "message": f"启动视频编码失败: {task_id}",
                    "video_url": None
                }
                
            # 添加测试日志
            test_logger.info("TEST_LOG_MARKER: VIDEO_ENCODE_START_SUCCESS")

            return {
                "success": True,
                "message": f"视频编码已启动: {task_id}",
                "video_url": video_url
            }

        except Exception as e:
            exception_logger.exception(f"启动视频编码失败: {str(e)}")
            return {
                "success": False,
                "message": f"启动视频编码失败: {str(e)}",
                "video_url": None
            }

    async def stop_encoding(self, task_id: str) -> Dict[str, Any]:
        """
        停止视频编码

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 停止结果
        """
        try:
            # 检查任务是否存在
            if task_id not in self.encoding_tasks:
                return {
                    "success": False,
                    "message": f"编码任务不存在: {task_id}"
                }

            # 获取编码任务信息
            encoding_info = self.encoding_tasks[task_id]
            encoding_id = encoding_info["encoding_id"]

            # 停止FFmpeg进程
            if task_id in self.ffmpeg_processes:
                process = self.ffmpeg_processes[task_id]
                try:
                    # 尝试正常终止进程
                    if process.poll() is None:  # 如果进程仍在运行
                        process.terminate()
                        # 等待进程终止
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            # 如果超时，强制终止
                            process.kill()

                    normal_logger.info(f"FFmpeg进程已停止: {task_id}")
                except Exception as e:
                    exception_logger.exception(f"停止FFmpeg进程时出错: {str(e)}")

                # 从字典中移除进程
                del self.ffmpeg_processes[task_id]

            # 从字典中移除编码任务信息
            del self.encoding_tasks[task_id]

            # 记录停止信息
            normal_logger.info(f"编码任务已停止: {task_id}, 编码ID: {encoding_id}")
            
            # 添加测试日志
            test_logger.info("TEST_LOG_MARKER: VIDEO_ENCODE_STOP_SUCCESS")

            return {
                "success": True,
                "message": f"编码任务已停止: {task_id}"
            }

        except Exception as e:
            exception_logger.exception(f"停止编码任务失败: {str(e)}")
            return {
                "success": False,
                "message": f"停止编码任务失败: {str(e)}"
            }

    async def _start_encoding_process(self, task_id: str, encoding_id: str, output_dir: str,
                                    output_path: str, format: str, task_manager, quality: int = 80,
                                    width: Optional[int] = None, height: Optional[int] = None,
                                    fps: int = 15) -> Optional[str]:
        """
        启动编码进程

        Args:
            task_id: 任务ID
            encoding_id: 编码ID
            output_dir: 输出目录
            output_path: 输出文件路径
            format: 视频格式
            task_manager: 任务管理器
            quality: 视频质量
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率

        Returns:
            Optional[str]: 视频URL，如果失败则返回None
        """
        try:
            # 创建编码线程
            encoding_thread = threading.Thread(
                target=self._encoding_thread,
                args=(task_id, encoding_id, output_dir, output_path, format, task_manager, quality, width, height, fps),
                daemon=True
            )
            encoding_thread.start()

            # 存储编码线程
            self.encoding_threads[task_id] = encoding_thread

            # 等待一段时间，确保编码进程启动
            normal_logger.info(f"等待编码进程初始化: {task_id}")
            await asyncio.sleep(5)

            # 检查进程是否仍在运行
            if task_id in self.ffmpeg_processes:
                process = self.ffmpeg_processes[task_id]
                if process.poll() is not None:
                    # 进程已退出
                    normal_logger.error(f"FFmpeg进程已退出，返回码: {process.poll()}")

                    # 尝试获取错误输出
                    if hasattr(process, 'stderr') and process.stderr:
                        try:
                            stderr_output = process.stderr.read()
                            if stderr_output:
                                normal_logger.error(f"FFmpeg错误输出: {stderr_output.decode('utf-8', errors='ignore')}")
                        except Exception as e:
                            normal_logger.error(f"读取FFmpeg错误输出失败: {str(e)}")

                    # 返回错误
                    return None

            # 从输出路径中提取文件名
            output_file_name = os.path.basename(output_path)

            # 构建视频URL
            video_url = f"{self.base_url}/api/v1/tasks/video/{encoding_id}/{output_file_name}"

            return video_url

        except Exception as e:
            normal_logger.exception(f"启动编码进程失败: {str(e)}")
            return None

    def _encoding_thread(self, task_id: str, encoding_id: str, output_dir: str, output_path: str,
                       format: str, task_manager=None, quality: int = 80,
                       width: Optional[int] = None, height: Optional[int] = None, fps: int = 15):
        """
        编码线程，从任务处理器获取预览帧并编码为MP4或FLV

        Args:
            task_id: 任务ID
            encoding_id: 编码ID
            output_dir: 输出目录
            output_path: 输出文件路径
            format: 视频格式
            task_manager: 任务管理器
            quality: 视频质量
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率
        """
        # 记录线程启动
        normal_logger.info(f"编码线程启动: {task_id}, 编码ID: {encoding_id}, 格式: {format}")
        try:
            # 获取任务处理器
            task_processor = None
            try:
                if task_manager:
                    task_processor = task_manager.processor
            except Exception as e:
                normal_logger.error(f"获取任务处理器失败: {str(e)}")

            if not task_processor:
                normal_logger.warning(f"无法获取任务处理器，将使用默认帧: {task_id}")
                # 继续执行，使用默认帧

            # 获取第一帧以确定分辨率
            frame = None
            retry_count = 0
            max_retries = 100  # 增加最大重试次数，最多等待10秒

            # 如果有任务处理器，尝试获取预览帧
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
                        # 检查任务是否已停止
                        if task_id not in self.encoding_tasks:
                            normal_logger.info(f"编码任务已停止，编码线程退出: {task_id}")
                            return

            # 如果仍然没有获取到帧，使用默认帧
            if frame is None:
                normal_logger.warning(f"无法获取预览帧，使用默认帧: {task_id}")
                # 创建一个黑色的默认帧 - 使用默认尺寸
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # 在帧上绘制文本
                cv2.putText(
                    frame,
                    "等待视频流...",
                    (int(frame.shape[1]/2) - 100, int(frame.shape[0]/2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

            # 确定视频分辨率
            frame_height, frame_width = frame.shape[:2]
            final_width = width if width is not None else frame_width
            final_height = height if height is not None else frame_height

            normal_logger.info(f"视频编码参数 - 任务ID: {task_id}, 格式: {format}, 分辨率: {final_width}x{final_height}, 帧率: {fps}, 质量: {quality}")

            # 计算比特率 - 基于质量参数
            # 质量范围1-100，映射到比特率范围500k-8000k
            bitrate = int(500 + (quality / 100.0) * 7500)

            # 构建FFmpeg命令 - 优化参数
            ffmpeg_cmd = [
                "ffmpeg",
                "-f", "rawvideo",           # 输入格式为原始视频
                "-pix_fmt", "bgr24",        # 像素格式
                "-s", f"{final_width}x{final_height}",  # 分辨率
                "-r", str(fps),             # 帧率
                "-i", "pipe:0",             # 从标准输入读取
                "-c:v", "libx264",          # 视频编码器
                "-preset", "veryfast",      # 编码速度 (改为veryfast，平衡速度和质量)
                "-tune", "zerolatency",     # 优化低延迟
                "-pix_fmt", "yuv420p",      # 输出像素格式 - 使用更兼容的YUV420P格式（4:2:0）
                "-profile:v", "main",       # 使用main配置文件，它支持yuv420p
                "-level", "4.1",            # 设置H.264级别 (提高到4.1以支持更高分辨率)
                "-b:v", f"{bitrate}k",      # 视频比特率
                "-maxrate", f"{bitrate*1.5}k",  # 最大比特率
                "-bufsize", f"{bitrate*2}k",    # 缓冲区大小
                "-g", "60",                 # GOP大小 (增加到60，提高稳定性)
                "-keyint_min", "30",        # 最小关键帧间隔
                "-sc_threshold", "40",      # 场景切换阈值，降低以增加关键帧
                "-refs", "4",               # 参考帧数量
                "-qmin", "10",              # 最小量化参数
                "-qmax", "51",              # 最大量化参数
                "-qdiff", "4",              # 量化参数差异
                "-threads", "auto",         # 自动线程数
            ]

            # 根据格式添加特定参数
            if format == "mp4":
                ffmpeg_cmd.extend([
                    "-movflags", "+faststart+frag_keyframe+empty_moov+default_base_moof",  # 优化Web播放和流式传输
                    "-frag_duration", "1000",  # 片段持续时间(毫秒)
                    "-f", "mp4",               # 输出格式为MP4
                ])
            elif format == "flv":
                ffmpeg_cmd.extend([
                    "-flvflags", "no_duration_filesize",  # FLV特定标志
                    "-f", "flv",                # 输出格式为FLV
                ])

            # 添加输出文件路径
            ffmpeg_cmd.append(output_path)

            normal_logger.info(f"FFmpeg命令: {' '.join(ffmpeg_cmd)}")

            # 启动FFmpeg进程
            try:
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10*1024*1024  # 使用更大的缓冲区
                )

                # 存储进程
                self.ffmpeg_processes[task_id] = process

                # 循环获取预览帧并写入FFmpeg
                frame_interval = 1.0 / fps
                last_frame_time = time.time()

                # 主循环 - 持续向FFmpeg提供帧数据
                frame_count = 0
                error_count = 0
                max_errors = 5  # 最大连续错误次数
                ffmpeg_restart_count = 0
                max_ffmpeg_restarts = 3  # 最大FFmpeg重启次数
                last_ffmpeg_restart_time = time.time()

                while task_id in self.encoding_tasks:
                    # 检查FFmpeg进程是否仍在运行
                    if process.poll() is not None:
                        # FFmpeg进程已退出
                        ffmpeg_exit_code = process.poll()
                        normal_logger.error(f"FFmpeg进程意外退出，返回码: {ffmpeg_exit_code}, 任务ID: {task_id}")

                        # 尝试获取错误输出
                        if hasattr(process, 'stderr') and process.stderr:
                            try:
                                stderr_output = process.stderr.read()
                                if stderr_output:
                                    normal_logger.error(f"FFmpeg错误输出: {stderr_output.decode('utf-8', errors='ignore')}")
                            except Exception as e:
                                normal_logger.error(f"读取FFmpeg错误输出失败: {str(e)}")

                        # 检查是否可以重启FFmpeg
                        current_time = time.time()
                        if (ffmpeg_restart_count < max_ffmpeg_restarts and
                            current_time - last_ffmpeg_restart_time > 10):  # 至少间隔10秒重启

                            normal_logger.info(f"尝试重启FFmpeg进程，重启次数: {ffmpeg_restart_count+1}/{max_ffmpeg_restarts}")

                            try:
                                # 重新构建FFmpeg命令 - 追加模式
                                restart_ffmpeg_cmd = ffmpeg_cmd.copy()

                                # 如果文件已存在，使用追加模式
                                if os.path.exists(output_path):
                                    # 修改输出选项为追加模式
                                    if format == "mp4":
                                        # 对于MP4，我们需要创建一个新文件，然后合并
                                        temp_output = f"{output_path}.temp"
                                        restart_ffmpeg_cmd[-1] = temp_output
                                    elif format == "flv":
                                        # FLV支持直接追加
                                        restart_ffmpeg_cmd.insert(-1, "-append")
                                        restart_ffmpeg_cmd.insert(-1, "1")

                                # 启动新的FFmpeg进程
                                process = subprocess.Popen(
                                    restart_ffmpeg_cmd,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    bufsize=10*1024*1024
                                )

                                # 更新进程引用
                                self.ffmpeg_processes[task_id] = process

                                # 更新重启计数和时间
                                ffmpeg_restart_count += 1
                                last_ffmpeg_restart_time = time.time()

                                normal_logger.info(f"FFmpeg进程重启成功，任务ID: {task_id}")

                                # 如果是MP4并且使用了临时文件，需要在最后合并
                                if format == "mp4" and os.path.exists(output_path):
                                    # 记录需要在结束时合并文件
                                    self.encoding_tasks[task_id]["needs_merge"] = True
                                    self.encoding_tasks[task_id]["temp_output"] = temp_output

                                # 继续循环
                                continue

                            except Exception as e:
                                normal_logger.error(f"重启FFmpeg进程失败: {str(e)}")
                                # 如果重启失败，跳出循环
                                break
                        else:
                            # 超过最大重启次数或重启间隔太短，退出循环
                            normal_logger.error(f"FFmpeg进程无法重启，已达到最大重启次数或重启间隔太短，任务ID: {task_id}")
                            break
                    # 确保FFmpeg进程正在运行
                    if process.poll() is not None:
                        # 进程已退出，跳过此次循环，让上面的重启逻辑处理
                        continue

                    # 控制帧率
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)

                    last_frame_time = time.time()

                    # 获取当前帧
                    current_frame = None

                    # 如果有任务处理器，尝试获取预览帧
                    if task_processor:
                        try:
                            current_frame = task_processor.get_preview_frame(task_id)
                            if current_frame is not None:
                                # 检查帧是否有效
                                if not isinstance(current_frame, np.ndarray):
                                    normal_logger.warning(f"获取到的预览帧类型无效: {type(current_frame)}, 使用上一帧")
                                    current_frame = None
                                elif current_frame.size == 0 or current_frame.shape[0] == 0 or current_frame.shape[1] == 0:
                                    normal_logger.warning(f"获取到的预览帧尺寸无效: {current_frame.shape}, 使用上一帧")
                                    current_frame = None
                        except Exception as e:
                            normal_logger.error(f"获取预览帧失败: {str(e)}")
                            current_frame = None

                    # 每100帧记录一次日志
                    frame_count += 1
                    if frame_count % 100 == 0:
                        normal_logger.info(f"编码线程已处理 {frame_count} 帧: {task_id}")

                    # 如果没有获取到有效帧，使用上一帧或默认帧
                    if current_frame is None:
                        error_count += 1
                        if error_count > max_errors:
                            normal_logger.warning(f"连续 {max_errors} 次未获取到有效帧，使用默认帧")
                            # 创建一个黑色的默认帧
                            current_frame = np.zeros((final_height, final_width, 3), dtype=np.uint8)
                            # 在帧上绘制文本
                            cv2.putText(
                                current_frame,
                                "视频流中断...",
                                (int(current_frame.shape[1]/2) - 100, int(current_frame.shape[0]/2)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2
                            )
                            error_count = 0  # 重置错误计数
                        else:
                            # 跳过这一帧，等待下一帧
                            continue
                    else:
                        error_count = 0  # 重置错误计数

                    # 调整帧大小
                    if current_frame.shape[1] != final_width or current_frame.shape[0] != final_height:
                        current_frame = cv2.resize(current_frame, (final_width, final_height))

                    # 写入帧数据到FFmpeg - 增加重试机制
                    max_write_retries = 3
                    write_retry_count = 0
                    write_success = False

                    while not write_success and write_retry_count < max_write_retries:
                        try:
                            # 再次检查进程是否仍在运行
                            if process.poll() is not None:
                                normal_logger.error(f"写入前发现FFmpeg进程已退出: {task_id}")
                                break

                            process.stdin.write(current_frame.tobytes())
                            write_success = True
                        except BrokenPipeError:
                            normal_logger.error(f"写入帧数据时管道已断开: {task_id}, 重试次数: {write_retry_count+1}/{max_write_retries}")
                            write_retry_count += 1
                            if write_retry_count >= max_write_retries:
                                break
                            time.sleep(0.1)  # 短暂等待后重试
                        except Exception as e:
                            normal_logger.error(f"写入帧数据时出错: {str(e)}, 重试次数: {write_retry_count+1}/{max_write_retries}")
                            write_retry_count += 1
                            if write_retry_count >= max_write_retries:
                                break
                            time.sleep(0.1)  # 短暂等待后重试

                    # 如果写入失败，跳出循环
                    if not write_success:
                        normal_logger.error(f"写入帧数据失败，达到最大重试次数: {task_id}")
                        break

                # 关闭FFmpeg输入流
                if process.stdin:
                    try:
                        process.stdin.close()
                    except Exception as e:
                        normal_logger.error(f"关闭FFmpeg输入流时出错: {str(e)}")

                # 等待FFmpeg进程完成
                try:
                    process.wait(timeout=30)  # 增加超时时间到30秒
                    normal_logger.info(f"FFmpeg进程已完成: {task_id}, 返回码: {process.returncode}")
                except subprocess.TimeoutExpired:
                    normal_logger.warning(f"等待FFmpeg进程完成超时: {task_id}")
                    # 尝试终止进程
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # 如果仍然无法终止，强制杀死
                        process.kill()
                        normal_logger.error(f"FFmpeg进程无法正常终止，已强制杀死: {task_id}")

                # 检查是否需要合并文件（MP4分段录制的情况）
                if task_id in self.encoding_tasks and self.encoding_tasks[task_id].get("needs_merge", False):
                    temp_output = self.encoding_tasks[task_id].get("temp_output")
                    if temp_output and os.path.exists(temp_output):
                        try:
                            normal_logger.info(f"开始合并MP4文件: {task_id}")

                            # 使用FFmpeg合并文件
                            merge_cmd = [
                                "ffmpeg",
                                "-f", "concat",
                                "-safe", "0",
                                "-i", "-",
                                "-c", "copy",
                                "-y",
                                f"{output_path}.merged"
                            ]

                            # 创建一个临时的concat文件内容
                            concat_content = f"file '{output_path}'\nfile '{temp_output}'"

                            # 启动合并进程
                            merge_process = subprocess.Popen(
                                merge_cmd,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )

                            # 写入concat内容
                            stdout, stderr = merge_process.communicate(input=concat_content)

                            if merge_process.returncode == 0:
                                # 合并成功，替换原文件
                                os.replace(f"{output_path}.merged", output_path)
                                # 删除临时文件
                                if os.path.exists(temp_output):
                                    os.remove(temp_output)
                                normal_logger.info(f"MP4文件合并成功: {task_id}")
                            else:
                                normal_logger.error(f"MP4文件合并失败: {task_id}, 错误: {stderr}")
                        except Exception as e:
                            normal_logger.error(f"合并MP4文件时出错: {str(e)}")

                # 检查输出文件是否存在
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    normal_logger.info(f"编码完成，输出文件: {output_path}, 大小: {file_size} 字节")

                    # 如果文件太小（可能是空文件或损坏的文件），记录警告
                    if file_size < 1024:  # 小于1KB
                        normal_logger.warning(f"输出文件可能损坏，文件大小过小: {file_size} 字节")
                else:
                    normal_logger.error(f"编码失败，输出文件不存在: {output_path}")

            except Exception as e:
                normal_logger.error(f"FFmpeg进程运行时出错: {str(e)}")
                import traceback
                normal_logger.error(traceback.format_exc())

            normal_logger.info(f"编码线程正常退出: {task_id}")

        except Exception as e:
            normal_logger.error(f"编码线程发生错误: {str(e)}")
            import traceback
            normal_logger.error(traceback.format_exc())