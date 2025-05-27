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

    async def start_live_stream(self, task_id: str, task_manager, format: str = "rtmp",
                              quality: int = 80, width: Optional[int] = None,
                              height: Optional[int] = None, fps: int = 15) -> Dict[str, Any]:
        """
        开启直播流编码 - 将分析结果推送到ZLMediaKit直播流

        Args:
            task_id: 任务ID
            task_manager: 任务管理器实例
            format: 流格式，支持"rtmp"、"hls"、"flv"
            quality: 视频质量(1-100)
            width: 视频宽度，为空则使用原始宽度
            height: 视频高度，为空则使用原始高度
            fps: 视频帧率

        Returns:
            Dict[str, Any]: 编码结果，包含流信息和播放地址
        """
        try:
            # 检查任务是否存在
            if not task_manager.has_task(task_id):
                return {
                    "success": False,
                    "message": f"任务不存在: {task_id}",
                    "stream_info": None,
                    "play_urls": None
                }

            # 检查任务是否正在运行或等待中
            task_info = task_manager.get_task(task_id)
            if not task_info:
                return {
                    "success": False,
                    "message": f"任务信息不存在: {task_id}",
                    "stream_info": None,
                    "play_urls": None
                }

            # 支持 WAITING 和 PROCESSING 状态的任务进行视频编码
            allowed_statuses = [TaskStatus.WAITING, TaskStatus.PROCESSING]
            if task_info["status"] not in allowed_statuses:
                status_names = {
                    TaskStatus.WAITING: "等待中",
                    TaskStatus.PROCESSING: "处理中",
                    TaskStatus.COMPLETED: "已完成",
                    TaskStatus.FAILED: "失败",
                    TaskStatus.STOPPED: "已停止"
                }
                current_status_name = status_names.get(task_info["status"], f"未知状态({task_info['status']})")
                return {
                    "success": False,
                    "message": f"任务状态不支持视频编码: {task_id}, 当前状态: {current_status_name}, 支持的状态: 等待中、处理中",
                    "stream_info": None,
                    "play_urls": None
                }

            # 如果已经有编码任务，先停止
            if task_id in self.encoding_tasks:
                stop_result = await self.stop_live_stream(task_id)
                if not stop_result.get("success", False):
                    normal_logger.warning(f"停止旧直播流任务失败: {stop_result.get('message')}")
                # 等待一小段时间，确保旧的编码任务完全停止
                await asyncio.sleep(1)

            # 验证格式
            if format.lower() not in ["rtmp", "hls", "flv"]:
                return {
                    "success": False,
                    "message": f"不支持的流格式: {format}, 仅支持rtmp、hls和flv",
                    "stream_info": None,
                    "play_urls": None
                }

            # 创建唯一的流ID
            stream_id = f"task_{task_id}_{int(time.time())}"
            app_name = "live"
            
            # 导入ZLM管理器
            from core.media_kit.zlm_manager import zlm_manager
            
            # 构建推流地址
            push_url = f"rtmp://127.0.0.1:1935/{app_name}/{stream_id}"
            
            # 存储编码任务信息
            self.encoding_tasks[task_id] = {
                "stream_id": stream_id,
                "app_name": app_name,
                "push_url": push_url,
                "format": format.lower(),
                "quality": quality,
                "width": width,
                "height": height,
                "fps": fps,
                "start_time": datetime.now().isoformat(),
                "is_live_stream": True
            }

            # 启动推流进程
            stream_info = await self._start_live_stream_process(
                task_id=task_id,
                stream_id=stream_id,
                app_name=app_name,
                push_url=push_url,
                format=format.lower(),
                task_manager=task_manager,
                quality=quality,
                width=width,
                height=height,
                fps=fps
            )

            if not stream_info:
                # 如果启动失败，删除编码任务信息
                if task_id in self.encoding_tasks:
                    del self.encoding_tasks[task_id]
                return {
                    "success": False,
                    "message": f"启动视频推流失败: {task_id}",
                    "stream_info": None,
                    "play_urls": None
                }
            
            # 构建播放地址
            play_urls = {
                "rtmp": f"rtmp://127.0.0.1:1935/{app_name}/{stream_id}",
                "hls": f"http://127.0.0.1:8080/{app_name}/{stream_id}.m3u8",
                "flv": f"http://127.0.0.1:8080/{app_name}/{stream_id}.flv",
                "webrtc": f"http://127.0.0.1:8080/index.html?app={app_name}&stream={stream_id}&type=webrtc"
            }
                
            # 添加测试日志
            test_logger.info("TEST_LOG_MARKER: VIDEO_LIVE_STREAM_START_SUCCESS")

            return {
                "success": True,
                "message": f"视频直播流已启动: {task_id}",
                "stream_info": stream_info,
                "play_urls": play_urls
            }

        except Exception as e:
            exception_logger.exception(f"启动视频直播流失败: {str(e)}")
            return {
                "success": False,
                "message": f"启动视频直播流失败: {str(e)}",
                "stream_info": None,
                "play_urls": None
            }

    async def stop_live_stream(self, task_id: str) -> Dict[str, Any]:
        """
        停止视频直播流

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
                    "message": f"直播流任务不存在: {task_id}"
                }

            # 获取编码任务信息
            encoding_info = self.encoding_tasks[task_id]
            stream_id = encoding_info.get("stream_id", "")

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

                    normal_logger.info(f"FFmpeg推流进程已停止: {task_id}")
                except Exception as e:
                    exception_logger.exception(f"停止FFmpeg推流进程时出错: {str(e)}")

                # 从字典中移除进程
                del self.ffmpeg_processes[task_id]

            # 从字典中移除编码任务信息
            del self.encoding_tasks[task_id]

            # 记录停止信息
            normal_logger.info(f"直播流任务已停止: {task_id}, 流ID: {stream_id}")
            
            # 添加测试日志
            test_logger.info("TEST_LOG_MARKER: VIDEO_LIVE_STREAM_STOP_SUCCESS")

            return {
                "success": True,
                "message": f"直播流任务已停止: {task_id}"
            }

        except Exception as e:
            exception_logger.exception(f"停止直播流任务失败: {str(e)}")
            return {
                "success": False,
                "message": f"停止直播流任务失败: {str(e)}"
            }

    async def _start_live_stream_process(self, task_id: str, stream_id: str, app_name: str,
                                        push_url: str, format: str, task_manager, quality: int = 80,
                                        width: Optional[int] = None, height: Optional[int] = None,
                                        fps: int = 15) -> Optional[Dict[str, Any]]:
        """
        启动直播推流进程

        Args:
            task_id: 任务ID
            stream_id: 流ID
            app_name: 应用名称
            push_url: 推流地址
            format: 流格式
            task_manager: 任务管理器
            quality: 视频质量
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率

        Returns:
            Optional[Dict[str, Any]]: 流信息，如果失败则返回None
        """
        try:
            # 创建推流线程
            streaming_thread = threading.Thread(
                target=self._live_streaming_thread,
                args=(task_id, stream_id, app_name, push_url, format, task_manager, quality, width, height, fps),
                daemon=True
            )
            streaming_thread.start()

            # 存储推流线程
            self.encoding_threads[task_id] = streaming_thread

            # 等待一段时间，确保推流进程启动
            normal_logger.info(f"等待推流进程初始化: {task_id}")
            await asyncio.sleep(5)

            # 检查进程是否仍在运行
            if task_id in self.ffmpeg_processes:
                process = self.ffmpeg_processes[task_id]
                if process.poll() is not None:
                    # 进程已退出
                    normal_logger.error(f"FFmpeg推流进程已退出，返回码: {process.poll()}")

                    # 尝试获取错误输出
                    if hasattr(process, 'stderr') and process.stderr:
                        try:
                            stderr_output = process.stderr.read()
                            if stderr_output:
                                normal_logger.error(f"FFmpeg推流错误输出: {stderr_output.decode('utf-8', errors='ignore')}")
                        except Exception as e:
                            normal_logger.error(f"读取FFmpeg推流错误输出失败: {str(e)}")

                    # 返回错误
                    return None

            # 构建流信息
            stream_info = {
                "stream_id": stream_id,
                "app_name": app_name,
                "push_url": push_url,
                "format": format,
                "quality": quality,
                "width": width,
                "height": height,
                "fps": fps,
                "status": "streaming"
            }

            return stream_info

        except Exception as e:
            normal_logger.exception(f"启动推流进程失败: {str(e)}")
            return None

    def _live_streaming_thread(self, task_id: str, stream_id: str, app_name: str,
                              push_url: str, format: str, task_manager=None, quality: int = 80,
                              width: Optional[int] = None, height: Optional[int] = None, fps: int = 15):
        """
        直播推流线程，实时获取任务原始视频流帧并渲染分析目标框后推送到ZLMediaKit

        Args:
            task_id: 任务ID
            stream_id: 流ID
            app_name: 应用名称
            push_url: 推流地址
            format: 流格式
            task_manager: 任务管理器
            quality: 视频质量
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率
        """
        # 记录线程启动
        normal_logger.info(f"直播推流线程启动: {task_id}, 流ID: {stream_id}, 格式: {format}")
        
        # 在线程中创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 运行异步函数
            loop.run_until_complete(self._async_live_streaming_worker(
                task_id, stream_id, app_name, push_url, format, task_manager, quality, width, height, fps
            ))
        except Exception as e:
            normal_logger.error(f"直播推流线程发生错误: {str(e)}")
            import traceback
            normal_logger.error(traceback.format_exc())
        finally:
            loop.close()

    async def _async_live_streaming_worker(self, task_id: str, stream_id: str, app_name: str,
                                         push_url: str, format: str, task_manager=None, quality: int = 80,
                                         width: Optional[int] = None, height: Optional[int] = None, fps: int = 15):
        """
        异步直播推流工作函数

        Args:
            task_id: 任务ID
            stream_id: 流ID
            app_name: 应用名称
            push_url: 推流地址
            format: 流格式
            task_manager: 任务管理器
            quality: 视频质量
            width: 视频宽度
            height: 视频高度
            fps: 视频帧率
        """
        try:
            # 获取任务处理器以访问最新的分析结果
            task_processor = None
            try:
                if task_manager:
                    task_processor = task_manager.processor
            except Exception as e:
                normal_logger.error(f"获取任务处理器失败: {str(e)}")

            # 获取任务的原始视频流
            # 从任务管理器获取任务配置
            task_info = task_manager.get_task(task_id) if task_manager else None
            if not task_info:
                normal_logger.error(f"无法获取任务信息: {task_id}")
                return

            # 获取流配置信息
            task_data = task_info.get("data", {})
            task_params = task_data.get("params", {}) if isinstance(task_data, dict) else {}
            
            # 获取原始视频流ID和URL
            original_stream_id = task_params.get("video_id") or task_params.get("stream_id") or f"stream_{task_id}"
            stream_url = task_params.get("stream_url", "")
            
            normal_logger.info(f"任务 {task_id}: 准备订阅原始视频流 {original_stream_id}, URL: {stream_url}")

            # 直接订阅原始视频流以获取未处理的帧
            from core.task_management.stream import stream_manager
            frame_queue = None
            try:
                # 构建流配置
                stream_config = {
                    "url": stream_url,
                    "rtsp_transport": task_params.get("rtsp_transport", "tcp"),
                    "reconnect_attempts": task_params.get("reconnect_attempts", 3),
                    "reconnect_delay": task_params.get("reconnect_delay", 5),
                    "frame_buffer_size": task_params.get("frame_buffer_size", 100),
                    "task_id": f"{task_id}_encoder",  # 使用不同的任务ID避免冲突
                    "video_id": original_stream_id
                }
                
                success, frame_queue = await stream_manager.subscribe_stream(
                    original_stream_id, 
                    f"{task_id}_encoder", 
                    stream_config
                )
                
                if not success or frame_queue is None:
                    normal_logger.error(f"订阅原始视频流失败: {original_stream_id}")
                    return
                    
                normal_logger.info(f"成功订阅原始视频流: {original_stream_id}")
            except Exception as e:
                normal_logger.error(f"订阅原始视频流时发生异常: {str(e)}")
                return

            # 获取第一帧以确定分辨率
            frame = None
            retry_count = 0
            max_retries = 100

            # 从原始视频流获取第一帧
            normal_logger.info(f"开始从原始流获取帧: {task_id}")
            while frame is None and retry_count < max_retries:
                try:
                    frame_data = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                    if frame_data and isinstance(frame_data, tuple) and len(frame_data) >= 2:
                        frame, timestamp = frame_data[:2]
                        if frame is not None:
                            normal_logger.info(f"成功获取原始流帧: {task_id}, 帧形状: {frame.shape}")
                            break
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    normal_logger.error(f"获取原始流帧失败: {str(e)}")

                retry_count += 1
                if retry_count % 10 == 0:
                    normal_logger.info(f"等待原始流帧中: {task_id}, 已重试 {retry_count} 次")
                
                # 检查任务是否已停止
                if task_id not in self.encoding_tasks:
                    normal_logger.info(f"编码任务已停止，直播推流线程退出: {task_id}")
                    return

            # 如果仍然没有获取到帧，使用默认帧
            if frame is None:
                normal_logger.warning(f"无法获取原始流帧，使用默认帧: {task_id}")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
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

            normal_logger.info(f"视频推流参数 - 任务ID: {task_id}, 格式: {format}, 分辨率: {final_width}x{final_height}, 帧率: {fps}, 质量: {quality}")

            # 计算比特率 - 基于质量参数
            bitrate = int(500 + (quality / 100.0) * 7500)

            # 构建FFmpeg命令
            ffmpeg_cmd = [
                "ffmpeg",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{final_width}x{final_height}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-tune", "zerolatency",
                "-pix_fmt", "yuv420p",
                "-profile:v", "main",
                "-level", "4.1",
                "-b:v", f"{bitrate}k",
                "-maxrate", f"{bitrate*1.5}k",
                "-bufsize", f"{bitrate*2}k",
                "-g", "60",
                "-keyint_min", "30",
                "-sc_threshold", "40",
                "-refs", "4",
                "-qmin", "10",
                "-qmax", "51",
                "-qdiff", "4",
                "-threads", "auto",
                "-f", "flv",
                push_url
            ]

            normal_logger.info(f"FFmpeg命令: {' '.join(ffmpeg_cmd)}")

            # 启动FFmpeg进程
            try:
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10*1024*1024
                )

                # 存储进程
                self.ffmpeg_processes[task_id] = process

                # 主循环 - 实时处理原始视频流帧
                frame_interval = 1.0 / fps
                last_frame_time = time.time()
                frame_count = 0
                error_count = 0
                max_errors = 5
                
                # 存储最新的分析结果，用于渲染
                latest_analysis_result = None
                
                # 记录任务状态并添加到帧上
                task_status = task_info.get("status", "未知")
                task_start_time = datetime.now()
                
                # 添加状态信息到帧上的函数
                def add_status_info(frame, status, elapsed_seconds):
                    h, w = frame.shape[:2]
                    
                    # 状态映射
                    status_map = {
                        TaskStatus.WAITING: "等待中",
                        TaskStatus.PROCESSING: "处理中",
                        TaskStatus.COMPLETED: "已完成",
                        TaskStatus.FAILED: "失败",
                        TaskStatus.STOPPED: "已停止"
                    }
                    
                    status_text = status_map.get(status, f"未知状态({status})")
                    
                    # 根据状态设置颜色
                    if status == TaskStatus.PROCESSING:
                        color = (0, 255, 0)  # 绿色
                    elif status == TaskStatus.WAITING:
                        color = (255, 165, 0)  # 橙色
                    elif status in [TaskStatus.FAILED, TaskStatus.STOPPED]:
                        color = (0, 0, 255)  # 红色
                    else:
                        color = (255, 255, 255)  # 白色
                    
                    # 格式化运行时间
                    minutes, seconds = divmod(elapsed_seconds, 60)
                    hours, minutes = divmod(minutes, 60)
                    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                    
                    # 绘制半透明背景
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (10, h-70), (300, h-10), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
                    # 添加状态文本
                    cv2.putText(frame, f"任务状态: {status_text}", (20, h-50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 添加运行时间
                    cv2.putText(frame, f"运行时间: {time_str}", (20, h-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    return frame

                while task_id in self.encoding_tasks:
                    # 检查FFmpeg进程是否仍在运行
                    if process.poll() is not None:
                        normal_logger.error(f"FFmpeg进程意外退出，返回码: {process.poll()}, 任务ID: {task_id}")
                        break

                    # 控制帧率
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    if elapsed < frame_interval:
                        await asyncio.sleep(frame_interval - elapsed)
                    last_frame_time = time.time()

                    # 获取原始视频流帧
                    current_frame = None
                    try:
                        frame_data = await asyncio.wait_for(frame_queue.get(), timeout=0.1)
                        if frame_data and isinstance(frame_data, tuple) and len(frame_data) >= 2:
                            current_frame, timestamp = frame_data[:2]
                    except asyncio.TimeoutError:
                        # 超时，使用上一帧或默认帧
                        pass
                    except Exception as e:
                        normal_logger.error(f"获取原始流帧失败: {str(e)}")

                    # 如果没有获取到有效帧，使用默认帧
                    if current_frame is None:
                        error_count += 1
                        if error_count > max_errors:
                            current_frame = np.zeros((final_height, final_width, 3), dtype=np.uint8)
                            cv2.putText(
                                current_frame,
                                "视频流中断...",
                                (int(current_frame.shape[1]/2) - 100, int(current_frame.shape[0]/2)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2
                            )
                            error_count = 0
                        else:
                            continue
                    else:
                        error_count = 0

                    # 调整帧大小
                    if current_frame.shape[1] != final_width or current_frame.shape[0] != final_height:
                        current_frame = cv2.resize(current_frame, (final_width, final_height))

                    # 获取最新的分析结果并渲染在帧上
                    if task_processor:
                        try:
                            # 使用辅助方法获取最新的分析结果
                            latest_analysis_result = await self._get_latest_analysis_result(task_processor, task_id)
                            
                            # 检查是否获取到了预览帧（预览帧中包含已渲染的检测框）
                            if latest_analysis_result and "preview_frame" in latest_analysis_result:
                                preview_frame = latest_analysis_result["preview_frame"]
                                if preview_frame is not None:
                                    # 如果有预览帧（已经渲染了检测框的帧），优先使用它
                                    normal_logger.debug(f"使用预览帧作为渲染结果: {task_id}")
                                    # 保留原始帧尺寸，仅使用预览帧中的渲染结果
                                    if preview_frame.shape[:2] != current_frame.shape[:2]:
                                        preview_frame = cv2.resize(preview_frame, (current_frame.shape[1], current_frame.shape[0]))
                                    # 使用预览帧替代当前帧
                                    rendered_frame = preview_frame
                                    # 跳过后续渲染
                                    continue
                        except Exception as e:
                            normal_logger.debug(f"获取分析结果时出错: {str(e)}")
                            # 在出错的情况下，使用默认的渲染逻辑

                    # 渲染分析结果到帧上
                    rendered_frame = self._render_analysis_results(current_frame, latest_analysis_result)
                    
                    # 计算任务运行时间
                    elapsed_time = (datetime.now() - task_start_time).total_seconds()
                    
                    # 每10秒更新一次任务状态
                    if int(elapsed_time) % 10 == 0:
                        try:
                            # 从任务管理器获取最新状态
                            if task_manager and task_manager.has_task(task_id):
                                updated_task_info = task_manager.get_task(task_id)
                                if updated_task_info:
                                    task_status = updated_task_info.get("status", task_status)
                        except Exception as e:
                            normal_logger.debug(f"更新任务状态时出错: {str(e)}")
                    
                    # 添加状态信息到帧上
                    rendered_frame = add_status_info(rendered_frame, task_status, elapsed_time)
                    
                    # 每100帧记录一次日志
                    frame_count += 1
                    if frame_count % 100 == 0:
                        normal_logger.info(f"直播推流线程已处理 {frame_count} 帧: {task_id}")

                    # 写入帧数据到FFmpeg
                    try:
                        if process.poll() is None:  # 确保进程仍在运行
                            process.stdin.write(rendered_frame.tobytes())
                        else:
                            break
                    except BrokenPipeError:
                        normal_logger.error(f"写入帧数据时管道已断开: {task_id}")
                        break
                    except Exception as e:
                        normal_logger.error(f"写入帧数据时出错: {str(e)}")
                        break

                # 关闭FFmpeg输入流
                if process.stdin:
                    try:
                        process.stdin.close()
                    except Exception as e:
                        normal_logger.error(f"关闭FFmpeg输入流时出错: {str(e)}")

                # 等待FFmpeg进程完成
                try:
                    process.wait(timeout=30)
                    normal_logger.info(f"FFmpeg进程已完成: {task_id}, 返回码: {process.returncode}")
                except subprocess.TimeoutExpired:
                    normal_logger.warning(f"等待FFmpeg进程完成超时: {task_id}")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

            except Exception as e:
                normal_logger.error(f"FFmpeg进程运行时出错: {str(e)}")
                import traceback
                normal_logger.error(traceback.format_exc())

            finally:
                # 取消订阅原始视频流
                try:
                    await stream_manager.unsubscribe_stream(original_stream_id, f"{task_id}_encoder")
                    normal_logger.info(f"已取消订阅原始视频流: {original_stream_id}")
                except Exception as e:
                    normal_logger.error(f"取消订阅原始视频流失败: {str(e)}")
                
                # 清理流队列
                if frame_queue:
                    try:
                        # 清空队列中的所有内容
                        while not frame_queue.empty():
                            try:
                                frame_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        normal_logger.debug(f"已清空帧队列: {task_id}")
                    except Exception as e:
                        normal_logger.error(f"清空帧队列失败: {str(e)}")
                
                # 通知ZLMediaKit流已结束（如果需要）
                try:
                    from core.media_kit.zlm_manager import zlm_manager
                    # 某些情况下，我们可能需要显式通知ZLMediaKit流已结束
                    await zlm_manager.close_stream(app_name, stream_id)
                    normal_logger.info(f"已通知ZLMediaKit关闭流: {stream_id}")
                except Exception as e:
                    normal_logger.debug(f"通知ZLMediaKit关闭流失败: {str(e)}")
                
                # 记录线程结束日志
                normal_logger.info(f"直播推流线程清理完成并正常退出: {task_id}")

        except Exception as e:
            normal_logger.error(f"直播推流工作函数发生错误: {str(e)}")
            import traceback
            normal_logger.error(traceback.format_exc())

    async def _get_latest_analysis_result(self, task_processor, task_id: str) -> Optional[Dict[str, Any]]:
        """
        从任务处理器获取最新的分析结果
        
        Args:
            task_processor: 任务处理器实例
            task_id: 任务ID
            
        Returns:
            Optional[Dict[str, Any]]: 最新的分析结果，如果没有则返回None
        """
        if not task_processor:
            return None
            
        try:
            # 首先尝试直接获取最新的预览帧数据
            if hasattr(task_processor, "preview_frames") and task_id in task_processor.preview_frames:
                # 某些实现中，预览帧就是最新的分析结果
                return {"preview_frame": task_processor.preview_frames[task_id]}
                
            # 尝试从任务状态中获取最新结果
            if hasattr(task_processor, "running_tasks") and task_id in task_processor.running_tasks:
                task_data = task_processor.running_tasks[task_id]
                
                # 某些实现可能会直接存储最近的分析结果
                if "latest_result" in task_data:
                    return task_data["latest_result"]
                    
                # 或者可能有专门的分析结果缓存
                if hasattr(task_processor, "_analysis_results") and task_id in task_processor._analysis_results:
                    return task_processor._analysis_results[task_id]
                    
                # 或者可能将结果存在其他地方
                if hasattr(task_processor, "result_cache") and task_id in task_processor.result_cache:
                    return task_processor.result_cache[task_id]
                    
            # 通过处理器的其他接口获取结果
            if hasattr(task_processor, "get_latest_result"):
                result = await task_processor.get_latest_result(task_id)
                if result:
                    return result
                    
            # 如果都没有找到，返回None
            return None
                
        except Exception as e:
            normal_logger.debug(f"获取最新分析结果时出错: {str(e)}")
            return None
            
    def _render_analysis_results(self, frame: np.ndarray, analysis_result: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        在视频帧上渲染分析结果（目标框、跟踪ID等）

        Args:
            frame: 原始视频帧
            analysis_result: 分析结果字典

        Returns:
            np.ndarray: 渲染后的视频帧
        """
        if analysis_result is None:
            return frame
            
        # 如果已经有预处理的预览帧，直接使用它
        if "preview_frame" in analysis_result and analysis_result["preview_frame"] is not None:
            preview = analysis_result["preview_frame"]
            # 确保尺寸一致
            if preview.shape[:2] != frame.shape[:2]:
                preview = cv2.resize(preview, (frame.shape[1], frame.shape[0]))
            return preview

        rendered_frame = frame.copy()
        height, width = frame.shape[:2]

        try:
            # 渲染检测结果
            detections = analysis_result.get("detections", [])
            for det in detections:
                try:
                    # 获取边界框
                    bbox = det.get("bbox", [])
                    if not bbox:
                        # 尝试其他可能的字段名
                        bbox = det.get("bbox_pixels", det.get("box", []))
                        if not bbox:
                            continue

                    # 处理不同格式的边界框
                    if isinstance(bbox, dict):
                        if all(k in bbox for k in ['x1', 'y1', 'x2', 'y2']):
                            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                        elif all(k in bbox for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                            x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                        else:
                            continue
                    elif isinstance(bbox, list) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                    else:
                        continue

                    # 转换为像素坐标
                    try:
                        if isinstance(x1, (int, float)) and isinstance(y1, (int, float)) and isinstance(x2, (int, float)) and isinstance(y2, (int, float)):
                            if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                                # 归一化坐标，转换为像素坐标
                                x1, y1 = int(x1 * width), int(y1 * height)
                                x2, y2 = int(x2 * width), int(y2 * height)
                            else:
                                # 已经是像素坐标
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        else:
                            continue
                    except (ValueError, TypeError):
                        continue

                    # 确保坐标在有效范围内
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width-1, x2), min(height-1, y2)

                    if x1 >= x2 or y1 >= y2:
                        continue

                    # 获取类别和置信度
                    class_name = det.get("class_name", det.get("class", "未知"))
                    confidence = float(det.get("confidence", det.get("score", 0)))

                    # 确定边界框颜色（基于类名）
                    # 使用一个简单的哈希函数来为不同类别生成不同的颜色
                    color_hash = hash(class_name) % 0xFFFFFF
                    r = (color_hash & 0xFF0000) >> 16
                    g = (color_hash & 0x00FF00) >> 8
                    b = color_hash & 0x0000FF
                    color = (b, g, r)  # OpenCV使用BGR顺序

                    # 绘制边界框
                    cv2.rectangle(rendered_frame, (x1, y1), (x2, y2), color, 2)

                    # 绘制标签
                    label = f"{class_name}: {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    
                    # 绘制标签背景
                    cv2.rectangle(rendered_frame, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), color, -1)
                    
                    # 绘制标签文字
                    cv2.putText(rendered_frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                except Exception as e:
                    # 忽略单个检测结果的渲染错误
                    continue

            # 渲染跟踪结果
            tracked_objects = analysis_result.get("tracked_objects", [])
            for track in tracked_objects:
                try:
                    # 获取边界框
                    bbox = track.get("bbox", [])
                    if not bbox:
                        # 尝试其他可能的字段名
                        bbox = track.get("bbox_pixels", track.get("box", []))
                        if not bbox:
                            continue

                    # 处理边界框（与检测结果类似）
                    if isinstance(bbox, dict):
                        if all(k in bbox for k in ['x1', 'y1', 'x2', 'y2']):
                            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                        elif all(k in bbox for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                            x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                        else:
                            continue
                    elif isinstance(bbox, list) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                    else:
                        continue

                    # 转换为像素坐标
                    try:
                        if isinstance(x1, (int, float)) and isinstance(y1, (int, float)) and isinstance(x2, (int, float)) and isinstance(y2, (int, float)):
                            if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                                # 归一化坐标，转换为像素坐标
                                x1, y1 = int(x1 * width), int(y1 * height)
                                x2, y2 = int(x2 * width), int(y2 * height)
                            else:
                                # 已经是像素坐标
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        else:
                            continue
                    except (ValueError, TypeError):
                        continue

                    # 确保坐标在有效范围内
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width-1, x2), min(height-1, y2)

                    if x1 >= x2 or y1 >= y2:
                        continue

                    # 获取跟踪ID
                    track_id = track.get("track_id", track.get("id", "未知"))

                    # 绘制跟踪边界框（使用不同颜色）
                    color = (255, 0, 0)  # 蓝色（BGR顺序）
                    cv2.rectangle(rendered_frame, (x1, y1), (x2, y2), color, 2)

                    # 绘制跟踪ID
                    track_label = f"ID: {track_id}"
                    cv2.putText(rendered_frame, track_label, (x1, y2 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                except Exception as e:
                    # 忽略单个跟踪结果的渲染错误
                    continue

            # 可以添加更多渲染逻辑，如分割掩码、越界检测线等

            # 添加视频信息到帧的右下角
            info_text = f"任务ID: {analysis_result.get('task_id', 'N/A')} | 帧: {analysis_result.get('frame_index', 0)}"
            cv2.putText(rendered_frame, info_text, (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        except Exception as e:
            normal_logger.error(f"渲染分析结果时出错: {str(e)}")

        return rendered_frame

    async def check_stream_status(self, task_id: str) -> Dict[str, Any]:
        """
        检查直播流状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 状态信息
        """
        try:
            # 检查任务是否存在
            if task_id not in self.encoding_tasks:
                return {
                    "success": False,
                    "message": f"直播流任务不存在: {task_id}",
                    "is_streaming": False
                }
                
            encoding_info = self.encoding_tasks[task_id]
            
            # 检查是否是直播流任务
            if not encoding_info.get("is_live_stream", False):
                return {
                    "success": False,
                    "message": f"任务不是直播流任务: {task_id}",
                    "is_streaming": False
                }
                
            # 检查FFmpeg进程是否存在
            if task_id not in self.ffmpeg_processes:
                return {
                    "success": False,
                    "message": f"FFmpeg进程不存在: {task_id}",
                    "is_streaming": False
                }
                
            # 检查FFmpeg进程是否仍在运行
            process = self.ffmpeg_processes[task_id]
            if process.poll() is not None:
                return {
                    "success": False,
                    "message": f"FFmpeg进程已退出，返回码: {process.poll()}",
                    "is_streaming": False
                }
                
            # 检查ZLMediaKit服务器上的流状态
            stream_id = encoding_info.get("stream_id", "")
            app_name = encoding_info.get("app_name", "live")
            
            # 导入ZLM管理器
            from core.media_kit.zlm_manager import zlm_manager
            
            # 查询流状态
            try:
                stream_info = await zlm_manager.get_stream_info(app_name, stream_id)
                if stream_info.get("code") == 0:
                    # 流存在于ZLMediaKit中
                    return {
                        "success": True,
                        "message": f"直播流正常: {task_id}",
                        "is_streaming": True,
                        "stream_info": stream_info.get("data", {})
                    }
                else:
                    return {
                        "success": False,
                        "message": f"直播流在ZLMediaKit中不存在: {task_id}",
                        "is_streaming": False
                    }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"检查ZLMediaKit流状态时出错: {str(e)}",
                    "is_streaming": False
                }
                
        except Exception as e:
            exception_logger.exception(f"检查直播流状态失败: {str(e)}")
            return {
                "success": False,
                "message": f"检查直播流状态失败: {str(e)}",
                "is_streaming": False
            }