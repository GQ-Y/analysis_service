"""
帧丢弃器工具
根据目标帧率智能丢弃多余的帧
"""
from typing import Dict, Any
import time


class SmartFrameDropper:
    """智能帧丢弃器"""

    def __init__(self, target_fps: int = 15):
        self.target_fps = target_fps
        self.target_interval = 1.0 / target_fps
        self.last_encode_time = 0
        self.dropped_frames = 0
        self.total_frames = 0

    def should_encode_frame(self) -> bool:
        """判断是否应该编码当前帧"""
        current_time = time.time()
        self.total_frames += 1

        # 如果距离上次编码时间不足目标间隔，丢弃帧
        if current_time - self.last_encode_time < self.target_interval:
            self.dropped_frames += 1
            return False

        self.last_encode_time = current_time
        return True

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.total_frames > 0:
            drop_rate = self.dropped_frames / self.total_frames
        else:
            drop_rate = 0

        return {
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "drop_rate": drop_rate,
            "effective_fps": 1.0 / self.target_interval if self.target_interval > 0 else 0
        } 