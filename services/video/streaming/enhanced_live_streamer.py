"""
增强版直播推流器
集成帧稳定器，确保直播流的稳定性和连续性
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
from services.video.utils.frame_renderer import FrameRenderer
from services.video.encoders.base_encoder import BaseEncoder
from services.video.utils.frame_stabilizer import FrameStabilizer
from shared.utils.app_state import app_state_manager

# 导入优化组件
from services.video.utils.optimized_frame_renderer import optimized_renderer

# 初始化日志记录器
normal_logger = get_normal_logger(__name__)
exception_logger = get_exception_logger(__name__)
test_logger = get_test_logger()

# 创建增强推流专用日志记录器
import logging
from pathlib import Path

def setup_enhanced_stream_logger():
    """设置增强推流专用日志记录器"""
    # 确保logs目录存在
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # 创建专用日志记录器
    logger = logging.getLogger("enhanced_stream")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 创建文件handler
        handler = logging.FileHandler(logs_dir / "enhanced_stream.log", encoding='utf-8')
        handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S.%f'
        )
        handler.setFormatter(formatter)
        
        # 添加handler
        logger.addHandler(handler)
    
    return logger

# 创建增强推流日志记录器
enhanced_stream_logger = setup_enhanced_stream_logger()

class EnhancedLiveStreamer(BaseEncoder):
    """增强版直播推流器 - 集成帧稳定器"""
    
    def __init__(self):
        """初始化增强直播推流器"""
        super().__init__()
        self.streaming_tasks = {}  # 存储直播流任务信息
        self.frame_stabilizers = {}  # 每个任务的帧稳定器
        self.ffmpeg_params = FFmpegParamsGenerator()
        self.frame_renderer = FrameRenderer()
        normal_logger.info("增强直播推流器初始化完成")
    
    async def start_live_stream(self, task_id: str, task_manager, format: str = "rtmp",
                              quality: int = 80, width: Optional[int] = None,
                              height: Optional[int] = None, fps: int = 15, stream_type: str = "ffmpeg") -> Dict[str, Any]:
        """
        开启增强直播流编码 - 使用帧稳定器
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

            # 检查任务状态
            task_info = task_manager.get_task(task_id)
            if not task_info:
                return {
                    "success": False,
                    "message": f"任务信息不存在: {task_id}",
                    "stream_info": None,
                    "play_urls": None
                }

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
                    "message": f"任务状态不支持视频编码: {task_id}, 当前状态: {current_status_name}",
                    "stream_info": None,
                    "play_urls": None
                }

            # 如果已经有编码任务，先停止
            if task_id in self.streaming_tasks:
                stop_result = await self.stop_live_stream(task_id)
                if not stop_result.get("success", False):
                    normal_logger.warning(f"停止旧直播流任务失败: {stop_result.get('message')}")
                await asyncio.sleep(1)

            # 根据推流类型选择处理方式
            if stream_type.lower() == "ffmpeg":
                return await self._start_enhanced_ffmpeg_stream(
                    task_id, task_manager, format, quality, width, height, fps
                )
            elif stream_type.lower() == "zlm":
                return await self._start_enhanced_zlm_stream(
                    task_id, task_manager, format, quality, width, height, fps
                )
            else:
                return {
                    "success": False,
                    "message": f"不支持的推流类型: {stream_type}, 仅支持 'ffmpeg' 和 'zlm'",
                    "stream_info": None,
                    "play_urls": None
                }

        except Exception as e:
            exception_logger.exception(f"启动增强直播流失败: {str(e)}")
            return {
                "success": False,
                "message": f"启动增强直播流失败: {str(e)}",
                "stream_info": None,
                "play_urls": None
            }
    
    async def stop_live_stream(self, task_id: str) -> Dict[str, Any]:
        """停止增强直播流"""
        try:
            if task_id not in self.streaming_tasks:
                return {
                    "success": False,
                    "message": f"直播流任务不存在: {task_id}"
                }

            encoding_info = self.streaming_tasks[task_id]
            stream_id = encoding_info.get("stream_id", "")

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
                    normal_logger.info(f"FFmpeg推流进程已停止: {task_id}")
                except Exception as e:
                    exception_logger.exception(f"停止FFmpeg推流进程时出错: {str(e)}")

                del self.ffmpeg_processes[task_id]

            # 清理任务信息
            del self.streaming_tasks[task_id]
            self.clear_analysis_result_cache(task_id)

            normal_logger.info(f"增强直播流任务已停止: {task_id}, 流ID: {stream_id}")

            return {
                "success": True,
                "message": f"增强直播流任务已停止: {task_id}"
            }

        except Exception as e:
            exception_logger.exception(f"停止增强直播流任务失败: {str(e)}")
            return {
                "success": False,
                "message": f"停止增强直播流任务失败: {str(e)}"
            }
    
    async def _start_enhanced_ffmpeg_stream(self, task_id: str, task_manager, format: str,
                                          quality: int, width: Optional[int], height: Optional[int], fps: int) -> Dict[str, Any]:
        """启动增强FFmpeg直播流"""
        try:
            # 生成流ID和输出目录
            stream_id = f"enhanced_stream_{uuid.uuid4().hex[:8]}"
            
            # 创建输出目录
            output_dir = os.path.join("temp", "enhanced_streams", stream_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # 设置输出文件路径
            output_file = f"stream.flv"
            output_path = os.path.join(output_dir, output_file)
            
            # 获取应用状态管理器中的HTTP端口
            http_server_port = 8002  # 使用当前服务的端口
            try:
                if hasattr(app_state_manager, 'http_server_port'):
                    http_server_port = app_state_manager.http_server_port
            except:
                pass

            # 构建文件访问URL
            stream_url = f"http://localhost:{http_server_port}/api/v1/tasks/video/enhanced/stream/{stream_id}/{output_file}"

            # 创建帧稳定器
            stabilizer = FrameStabilizer(target_fps=fps, buffer_size=60)
            self.frame_stabilizers[task_id] = stabilizer

            # 存储任务信息
            self.streaming_tasks[task_id] = {
                "stream_id": stream_id,
                "output_path": output_path,
                "output_dir": output_dir,
                "stream_url": stream_url,
                "format": format,
                "quality": quality,
                "width": width,
                "height": height,
                "fps": fps,
                "start_time": datetime.now().isoformat(),
                "stabilizer": stabilizer,
                "stream_type": "enhanced_ffmpeg"
            }

            # 启动增强推流进程
            stream_info = await self._start_enhanced_ffmpeg_process(
                task_id, stream_id, output_path, http_server_port,
                task_manager, quality, width, height, fps
            )

            if not stream_info:
                # 启动失败，清理
                if task_id in self.streaming_tasks:
                    del self.streaming_tasks[task_id]
                if task_id in self.frame_stabilizers:
                    del self.frame_stabilizers[task_id]
                return {
                    "success": False,
                    "message": f"启动增强FFmpeg流失败: {task_id}",
                    "stream_info": None,
                    "play_urls": None
                }

            # 构建播放地址
            play_urls = {
                "flv": stream_url,
                "hls": stream_url.replace('.flv', '.m3u8'),  # 如果需要HLS
                "stream_file": stream_url
            }

            return {
                "success": True,
                "message": f"增强FFmpeg直播流已启动: {task_id}",
                "stream_info": stream_info,
                "play_urls": play_urls
            }

        except Exception as e:
            exception_logger.exception(f"启动增强FFmpeg流失败: {str(e)}")
            return {
                "success": False,
                "message": f"启动增强FFmpeg流失败: {str(e)}",
                "stream_info": None,
                "play_urls": None
            }
    
    async def _start_enhanced_zlm_stream(self, task_id: str, task_manager, format: str,
                                       quality: int, width: Optional[int], height: Optional[int], fps: int) -> Dict[str, Any]:
        """启动增强ZLM推流"""
        # 这里可以实现ZLM的增强版本，现在先返回基础实现
        return {
            "success": False,
            "message": "增强ZLM推流功能待实现",
            "stream_info": None,
            "play_urls": None
        }
    
    async def _start_enhanced_ffmpeg_process(self, task_id: str, stream_id: str, output_path: str, http_port: int,
                                           task_manager, quality: int = 80, width: Optional[int] = None, 
                                           height: Optional[int] = None, fps: int = 15) -> Optional[Dict[str, Any]]:
        """启动增强FFmpeg推流进程"""
        try:
            # 创建推流线程
            streaming_thread = threading.Thread(
                target=self._enhanced_ffmpeg_thread,
                args=(task_id, stream_id, output_path, http_port, task_manager, quality, width, height, fps),
                daemon=True
            )
            streaming_thread.start()

            # 存储线程
            self.encoding_threads[task_id] = streaming_thread

            # 等待初始化
            normal_logger.info(f"等待增强FFmpeg流初始化: {task_id}")
            await asyncio.sleep(3)

            return {
                "stream_id": stream_id,
                "output_path": output_path,
                "format": "flv",
                "status": "starting"
            }

        except Exception as e:
            normal_logger.exception(f"启动增强FFmpeg进程失败: {str(e)}")
            return None
    
    def _enhanced_ffmpeg_thread(self, task_id: str, stream_id: str, output_path: str, http_port: int,
                              task_manager=None, quality: int = 80, width: Optional[int] = None, 
                              height: Optional[int] = None, fps: int = 15):
        """增强FFmpeg推流线程"""
        normal_logger.info(f"增强FFmpeg推流线程启动: {task_id}, 流ID: {stream_id}")
        enhanced_stream_logger.info(f"🚀 增强FFmpeg推流线程启动 - 任务: {task_id}, 流ID: {stream_id}")
        
        stabilizer = self.frame_stabilizers.get(task_id)
        if not stabilizer:
            normal_logger.error(f"帧稳定器不存在: {task_id}")
            enhanced_stream_logger.error(f"❌ 帧稳定器不存在: {task_id}")
            return

        try:
            # 获取任务处理器
            task_processor = None
            try:
                if task_manager:
                    task_processor = task_manager.processor
                    enhanced_stream_logger.info(f"✅ 成功获取任务处理器: {type(task_processor)}")
            except Exception as e:
                normal_logger.error(f"获取任务处理器失败: {str(e)}")
                enhanced_stream_logger.error(f"❌ 获取任务处理器失败: {str(e)}")

            # 获取第一帧以确定分辨率
            frame = None
            retry_count = 0
            max_retries = 100

            if task_processor:
                normal_logger.info(f"开始获取预览帧: {task_id}")
                enhanced_stream_logger.info(f"🔍 开始获取预览帧进行分辨率检测...")
                while frame is None and retry_count < max_retries:
                    try:
                        frame = task_processor.get_preview_frame(task_id)
                        if frame is not None:
                            normal_logger.info(f"成功获取预览帧: {task_id}, 帧形状: {frame.shape}")
                            enhanced_stream_logger.info(f"✅ 成功获取预览帧，形状: {frame.shape}")
                    except Exception as e:
                        normal_logger.debug(f"获取预览帧失败: {str(e)}")
                        enhanced_stream_logger.debug(f"获取预览帧失败: {str(e)}")
                        frame = None

                    if frame is None:
                        time.sleep(0.1)
                        retry_count += 1
                        if retry_count % 10 == 0:
                            normal_logger.info(f"等待预览帧中: {task_id}, 已重试 {retry_count} 次")
                            enhanced_stream_logger.info(f"⏳ 等待预览帧中，已重试 {retry_count} 次")

                        if task_id not in self.streaming_tasks:
                            normal_logger.info(f"推流任务已停止，线程退出: {task_id}")
                            enhanced_stream_logger.info(f"🛑 推流任务已停止，线程退出")
                            return

            # 使用默认帧
            if frame is None:
                normal_logger.warning(f"无法获取预览帧，使用默认帧: {task_id}")
                enhanced_stream_logger.warning(f"⚠️ 无法获取预览帧，使用默认帧")
                frame = self.create_default_frame(640, 480, "等待视频流...")

            # 确定分辨率
            frame_height, frame_width = frame.shape[:2]
            final_width = width if width is not None else frame_width
            final_height = height if height is not None else frame_height

            normal_logger.info(f"增强推流参数 - 任务ID: {task_id}, 分辨率: {final_width}x{final_height}, 帧率: {fps}")
            enhanced_stream_logger.info(f"🎬 推流参数 - 分辨率: {final_width}x{final_height}, 帧率: {fps}, 质量: {quality}")

            # 基于稳定LiveStreamer的FFmpeg命令
            ffmpeg_cmd = [
                "ffmpeg",
                # 输入参数 - 与基础版本保持一致
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{final_width}x{final_height}",
                "-r", str(fps),
                "-i", "pipe:0",
                
                # 稳定的编码参数
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-tune", "zerolatency",
                
                # 动态码率计算
                "-b:v", f"{quality * 10}k",  # 基于质量参数计算码率
                "-maxrate", f"{quality * 12}k",
                "-bufsize", f"{quality * 20}k",
                
                # 稳定的GOP设置
                "-g", str(fps * 2),
                "-pix_fmt", "yuv420p",
                
                # 输出格式
                "-f", "flv",
                output_path
            ]

            normal_logger.info(f"增强FFmpeg命令: {' '.join(ffmpeg_cmd)}")
            enhanced_stream_logger.info(f"🔧 FFmpeg命令: {' '.join(ffmpeg_cmd)}")

            # 启动FFmpeg进程 - 稳定版本
            try:
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10*1024*1024  # 10MB缓冲区，与基础LiveStreamer保持一致
                )

                self.ffmpeg_processes[task_id] = process
                normal_logger.info(f"增强FFmpeg推流进程已启动: {task_id}, PID: {process.pid}")
                enhanced_stream_logger.info(f"✅ FFmpeg推流进程已启动, PID: {process.pid}")

                # 启动FFmpeg错误监控线程
                error_monitor_thread = threading.Thread(
                    target=self._monitor_ffmpeg_process,
                    args=(task_id, process),
                    daemon=True
                )
                error_monitor_thread.start()
                
                # 保守的初始化等待 - 确保FFmpeg准备就绪
                time.sleep(1)  # 减少到1秒，更保守
                
                # 检查进程是否还在运行
                if process.poll() is not None:
                    stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                    raise Exception(f"FFmpeg进程启动失败，错误输出: {stderr_output}")

                # 启动稳定化推流循环
                asyncio.run(self._enhanced_streaming_loop(
                    task_id, task_processor, process, stabilizer,
                    final_width, final_height, fps
                ))

            except Exception as e:
                normal_logger.error(f"启动FFmpeg推流进程失败: {str(e)}")
                enhanced_stream_logger.error(f"❌ 启动FFmpeg推流进程失败: {str(e)}")
                return

        except Exception as e:
            normal_logger.error(f"增强FFmpeg推流线程运行时出错: {str(e)}")
            enhanced_stream_logger.error(f"❌ 推流线程运行时出错: {str(e)}")
            import traceback
            normal_logger.error(traceback.format_exc())
            enhanced_stream_logger.error(f"详细错误: {traceback.format_exc()}")

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

            normal_logger.info(f"增强FFmpeg推流线程结束: {task_id}")
            enhanced_stream_logger.info(f"🏁 增强FFmpeg推流线程结束: {task_id}")
    
    async def _enhanced_streaming_loop(self, task_id: str, task_processor, ffmpeg_process, 
                                     stabilizer: FrameStabilizer, width: int, height: int, fps: int):
        """增强推流循环 - 直播优化版本"""
        normal_logger.info(f"开始增强推流循环: {task_id}")
        enhanced_stream_logger.info(f"=== 开始增强推流循环 [{task_id}] ===")
        
        # 强制检查task_processor状态
        normal_logger.info(f"task_processor 状态: {task_processor is not None}, 类型: {type(task_processor)}")
        enhanced_stream_logger.info(f"task_processor 状态: {task_processor is not None}, 类型: {type(task_processor)}")
        
        frame_count = 0
        last_stats_time = time.time()
        frame_received_count = 0  # 实际接收到的帧数计数
        
        # 简化帧间隔控制 - 更宽松的控制策略
        frame_interval = 1.0 / fps
        enhanced_stream_logger.info(f"推流参数: FPS={fps}, 帧间隔={frame_interval:.3f}秒, 目标分辨率={width}x{height}")
        
        try:
            while (task_id in self.streaming_tasks and 
                   ffmpeg_process.poll() is None):
                
                current_time = time.time()
                
                # 检查FFmpeg进程健康状态
                if ffmpeg_process.poll() is not None:
                    normal_logger.error(f"FFmpeg进程已退出: {task_id}")
                    enhanced_stream_logger.error(f"💀 FFmpeg进程已退出")
                    break
                
                # 获取源帧和分析结果 - 不再严格控制获取频率
                source_frame = None
                analysis_result = None
                
                if task_processor:
                    try:
                        # 直接获取最新的预览帧，不受帧间隔限制
                        source_frame = task_processor.get_preview_frame(task_id)
                        if source_frame is not None:
                            frame_received_count += 1
                            
                            if frame_received_count <= 10 or frame_received_count % 50 == 0:  # 前10帧和每50帧记录一次
                                normal_logger.info(f"成功获取源帧: {task_id}, 已接收: {frame_received_count} 帧, 形状: {source_frame.shape}")
                                enhanced_stream_logger.info(f"✅ 成功获取源帧 #{frame_received_count}, 形状: {source_frame.shape}")
                        elif frame_received_count == 0 and frame_count < 20:  # 前20次循环记录
                            if frame_count % 5 == 0:  # 每5次记录一次，减少日志
                                normal_logger.warning(f"获取源帧返回None: {task_id}, 循环次数: {frame_count}")
                                enhanced_stream_logger.warning(f"❌ 获取源帧返回None, 循环次数: {frame_count}")
                        
                        # 尝试从视频服务获取最新分析结果
                        if source_frame is not None:
                            try:
                                from services.video.video_service import video_service
                                analysis_result = video_service.analysis_results_cache.get(task_id)
                                if analysis_result and frame_received_count % 100 == 0:  # 每100帧记录一次
                                    normal_logger.info(f"获取到分析结果: {task_id}, 检测数量: {len(analysis_result.get('detections', []))}")
                                    enhanced_stream_logger.info(f"🎯 获取到分析结果, 检测数量: {len(analysis_result.get('detections', []))}")
                            except Exception as e:
                                if frame_received_count % 200 == 0:  # 每200帧记录一次错误
                                    normal_logger.debug(f"获取分析结果失败: {str(e)}")
                                    enhanced_stream_logger.debug(f"获取分析结果失败: {str(e)}")
                        
                    except Exception as e:
                        if frame_received_count % 100 == 0:  # 减少错误日志频率
                            normal_logger.error(f"获取源帧失败: {str(e)}")
                            enhanced_stream_logger.error(f"获取源帧异常: {str(e)}")
                
                # 渲染分析结果到帧上
                if source_frame is not None:
                    # 调整帧大小
                    if source_frame.shape[1] != width or source_frame.shape[0] != height:
                        source_frame = cv2.resize(source_frame, (width, height))
                    
                    # 使用优化渲染器渲染分析结果
                    if analysis_result:
                        rendered_frame = optimized_renderer.render_analysis_results_optimized(
                            source_frame, analysis_result
                        )
                        if frame_received_count % 100 == 0:  # 每100帧记录一次
                            normal_logger.debug(f"渲染了分析结果到帧: {task_id}")
                            enhanced_stream_logger.debug(f"🎨 渲染了分析结果到帧")
                    else:
                        rendered_frame = source_frame
                        if frame_received_count % 200 == 0:  # 每200帧记录一次
                            normal_logger.debug(f"使用原始帧（无分析结果）: {task_id}")
                            enhanced_stream_logger.debug(f"使用原始帧（无分析结果）")
                    
                    # 添加到稳定器
                    stabilizer.add_source_frame(rendered_frame, analysis_result)
                
                # 始终尝试获取稳定帧并输出 - 确保连续性
                stable_frame, source_type, metadata = stabilizer.get_stable_frame()
                
                if stable_frame is not None:
                    try:
                        # 检查管道状态
                        if ffmpeg_process.stdin.closed:
                            normal_logger.error(f"FFmpeg输入管道已关闭: {task_id}")
                            enhanced_stream_logger.error(f"💔 FFmpeg输入管道已关闭")
                            break
                        
                        # 写入FFmpeg
                        ffmpeg_process.stdin.write(stable_frame.tobytes())
                        frame_count += 1
                        
                        # 记录稳定化信息
                        if source_type in ["cached", "repeated", "interpolated"] and frame_count % 500 == 0:
                            normal_logger.debug(f"推流使用{source_type}帧: {task_id}")
                            enhanced_stream_logger.debug(f"📺 推流使用{source_type}帧")
                        
                    except BrokenPipeError:
                        normal_logger.error(f"FFmpeg推流管道已断开: {task_id}")
                        enhanced_stream_logger.error(f"💔 FFmpeg推流管道已断开")
                        break
                    except Exception as e:
                        normal_logger.error(f"写入FFmpeg推流失败: {str(e)}")
                        enhanced_stream_logger.error(f"写入FFmpeg推流失败: {str(e)}")
                        break
                
                # 定期输出统计信息
                current_time = time.time()
                if current_time - last_stats_time >= 15.0:  # 每15秒
                    stats = stabilizer.get_stabilizer_stats()
                    normal_logger.info(f"推流稳定器统计 [{task_id}] - 输出帧数: {frame_count}, 接收帧数: {frame_received_count}, "
                                     f"重复帧: {stats['stabilizer']['repeated_frames']}, "
                                     f"插值帧: {stats['stabilizer']['interpolated_frames']}, "
                                     f"缓冲状态: {stats['buffer']['is_stalled']}")
                    enhanced_stream_logger.info(f"📊 推流统计 - 输出帧: {frame_count}, 接收帧: {frame_received_count}, "
                                              f"重复帧: {stats['stabilizer']['repeated_frames']}, "
                                              f"插值帧: {stats['stabilizer']['interpolated_frames']}, "
                                              f"卡顿状态: {stats['buffer']['is_stalled']}")
                    last_stats_time = current_time
                
                # 优化的循环控制 - 保持稳定输出但不过度限制
                await asyncio.sleep(max(0.001, frame_interval - 0.01))  # 轻微的控制，确保不会过快
                
        except Exception as e:
            normal_logger.error(f"增强推流循环异常: {str(e)}")
            enhanced_stream_logger.error(f"❌ 增强推流循环异常: {str(e)}")
        finally:
            stabilizer.stop_stable_output()
            normal_logger.info(f"增强推流循环结束: {task_id}, 总输出帧数: {frame_count}, 总接收帧数: {frame_received_count}")
            enhanced_stream_logger.info(f"=== 增强推流循环结束 [{task_id}] === 总输出帧: {frame_count}, 总接收帧: {frame_received_count}")
    
    def get_streaming_stats(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取推流统计信息"""
        if task_id not in self.frame_stabilizers:
            return None
        
        stabilizer = self.frame_stabilizers[task_id]
        return stabilizer.get_stabilizer_stats()
    
    def create_default_frame(self, width: int, height: int, text: str) -> np.ndarray:
        """创建默认帧"""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (255, 255, 255)
        thickness = 2
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return frame
    
    def get_cached_analysis_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取缓存的分析结果"""
        # 这里应该从分析结果缓存中获取
        # 暂时返回None，实际使用时需要实现
        return None
    
    def clear_analysis_result_cache(self, task_id: str):
        """清理分析结果缓存"""
        # 清理逻辑
        pass

    def _monitor_ffmpeg_process(self, task_id: str, process: subprocess.Popen):
        """监控FFmpeg进程的错误输出和健康状态"""
        normal_logger.info(f"开始监控FFmpeg进程: {task_id}")
        enhanced_stream_logger.info(f"🔍 开始监控FFmpeg进程: {task_id}")
        
        try:
            while process.poll() is None and task_id in self.streaming_tasks:
                # 非阻塞读取stderr
                try:
                    import select
                    if select.select([process.stderr], [], [], 0.1)[0]:
                        stderr_line = process.stderr.readline().decode('utf-8', errors='ignore').strip()
                        if stderr_line:
                            # 只记录真正的错误，忽略正常的FFmpeg输出
                            if any(error_keyword in stderr_line.lower() for error_keyword in ['error', 'failed', 'invalid', 'could not']):
                                normal_logger.error(f"FFmpeg错误: {stderr_line}")
                                enhanced_stream_logger.error(f"❌ FFmpeg错误: {stderr_line}")
                            elif 'frame=' in stderr_line.lower():
                                # 这是正常的FFmpeg进度输出，仅调试时记录
                                enhanced_stream_logger.debug(f"FFmpeg进度: {stderr_line}")
                except:
                    pass
                
                time.sleep(0.1)
            
            # 进程结束后检查退出状态
            exit_code = process.poll()
            if exit_code is not None and exit_code != 0:
                normal_logger.error(f"FFmpeg进程异常退出: {task_id}, 退出码: {exit_code}")
                enhanced_stream_logger.error(f"💥 FFmpeg进程异常退出, 退出码: {exit_code}")
            else:
                normal_logger.info(f"FFmpeg进程正常结束: {task_id}")
                enhanced_stream_logger.info(f"🏁 FFmpeg进程正常结束: {task_id}")

        except Exception as e:
            normal_logger.error(f"监控FFmpeg进程失败: {str(e)}")
            enhanced_stream_logger.error(f"❌ 监控FFmpeg进程失败: {str(e)}")


# 全局增强直播推流器实例
enhanced_live_streamer = EnhancedLiveStreamer() 