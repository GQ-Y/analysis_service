"""
FFmpeg参数配置工具
提供各种场景下的FFmpeg命令行参数生成函数
"""
from typing import List


class FFmpegParamsGenerator:
    """FFmpeg参数生成器"""

    @staticmethod
    def calculate_optimal_bitrate(quality: int, width: int, height: int, fps: int) -> int:
        """计算最优比特率"""
        # 基于分辨率的基础比特率
        resolution_bitrates = {
            (640, 480): 800,      # SD
            (1280, 720): 1500,    # HD
            (1920, 1080): 3000,   # FHD
        }

        # 找到最接近的分辨率
        pixel_count = width * height
        base_bitrate = 1000  # 默认值

        for (w, h), bitrate in resolution_bitrates.items():
            if pixel_count <= w * h:
                base_bitrate = bitrate
                break

        # 质量调整 (50%-150%)
        quality_factor = 0.5 + (quality / 100.0)
        adjusted_bitrate = int(base_bitrate * quality_factor)

        # 帧率调整
        if fps > 15:
            fps_factor = fps / 15.0
            adjusted_bitrate = int(adjusted_bitrate * fps_factor)

        # 限制范围
        return max(300, min(adjusted_bitrate, 6000))

    @staticmethod
    def get_realtime_ffmpeg_params(final_width: int, final_height: int, fps: int, bitrate: int, output_path: str) -> List[str]:
        """获取实时流优化的FFmpeg参数"""
        return [
            "ffmpeg",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{final_width}x{final_height}",
            "-r", str(fps),
            "-i", "pipe:0",

            # 实时编码优化
            "-c:v", "libx264",
            "-preset", "ultrafast",        # 最快编码速度
            "-tune", "zerolatency",        # 零延迟优化
            "-profile:v", "main",          # 改为main profile，支持更多格式
            "-level", "3.1",               # 较低级别
            "-pix_fmt", "yuv420p",         # 明确指定输出像素格式

            # 比特率控制 - 固定比特率模式
            "-b:v", f"{bitrate}k",
            "-minrate", f"{bitrate}k",
            "-maxrate", f"{bitrate}k",
            "-bufsize", f"{bitrate//4}k",  # 更小的缓冲区

            # 关键帧优化 - 更频繁的关键帧
            "-g", str(fps),                # GOP大小等于帧率，1秒一个关键帧
            "-keyint_min", str(fps//2),    # 最小关键帧间隔
            "-sc_threshold", "0",          # 禁用场景切换检测

            # 编码优化
            "-threads", "2",               # 限制线程数避免过载
            "-refs", "1",                  # 只使用1个参考帧
            "-me_method", "dia",           # 最快的运动估计
            "-subq", "0",                  # 最快的子像素运动估计
            "-trellis", "0",               # 禁用trellis量化
            "-aq-mode", "0",               # 禁用自适应量化

            # 输出文件
            output_path
        ]

    @staticmethod
    def get_realtime_streaming_ffmpeg_params(final_width: int, final_height: int, fps: int, bitrate: int, push_url: str) -> List[str]:
        """获取实时推流优化的FFmpeg参数"""
        return [
            "ffmpeg",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{final_width}x{final_height}",
            "-r", str(fps),
            "-i", "pipe:0",

            # 实时编码优化
            "-c:v", "libx264",
            "-preset", "ultrafast",        # 最快编码速度
            "-tune", "zerolatency",        # 零延迟优化
            "-profile:v", "main",          # 改为main profile，支持更多格式
            "-level", "3.1",               # 较低级别
            "-pix_fmt", "yuv420p",         # 明确指定输出像素格式

            # 比特率控制 - 固定比特率模式
            "-b:v", f"{bitrate}k",
            "-minrate", f"{bitrate}k",
            "-maxrate", f"{bitrate}k",
            "-bufsize", f"{bitrate//4}k",  # 更小的缓冲区

            # 关键帧优化 - 更频繁的关键帧
            "-g", str(fps),                # GOP大小等于帧率，1秒一个关键帧
            "-keyint_min", str(fps//2),    # 最小关键帧间隔
            "-sc_threshold", "0",          # 禁用场景切换检测

            # 编码优化
            "-threads", "2",               # 限制线程数避免过载
            "-refs", "1",                  # 只使用1个参考帧
            "-me_method", "dia",           # 最快的运动估计
            "-subq", "0",                  # 最快的子像素运动估计
            "-trellis", "0",               # 禁用trellis量化
            "-aq-mode", "0",               # 禁用自适应量化

            # 输出格式
            "-f", "flv",
            "-flvflags", "no_duration_filesize",
            push_url
        ]

    @staticmethod
    def adjust_mp4_params(ffmpeg_cmd: List[str], output_path: str) -> List[str]:
        """调整MP4特定参数"""
        # 移除最后的输出路径
        adjusted_cmd = ffmpeg_cmd[:-1]
        
        # 添加MP4特定参数
        adjusted_cmd.extend([
            "-movflags", "+faststart+frag_keyframe+empty_moov+default_base_moof",  # 优化Web播放和流式传输
            "-frag_duration", "1000",  # 片段持续时间(毫秒)
            "-f", "mp4",               # 输出格式为MP4
            output_path
        ])
        
        return adjusted_cmd

    @staticmethod
    def adjust_flv_params(ffmpeg_cmd: List[str], output_path: str) -> List[str]:
        """调整FLV特定参数"""
        # 移除最后的输出路径
        adjusted_cmd = ffmpeg_cmd[:-1]
        
        # 添加FLV特定参数
        adjusted_cmd.extend([
            "-flvflags", "no_duration_filesize",  # FLV特定标志
            "-f", "flv",                # 输出格式为FLV
            output_path
        ])
        
        return adjusted_cmd 