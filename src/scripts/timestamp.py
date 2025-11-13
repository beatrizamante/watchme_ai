
import time


def calculate_timestamp(frame_index, video_info):
    """Calculate timestamp based on frame index and video FPS"""
    if video_info and video_info["fps"] > 0:
        seconds = frame_index / video_info["fps"]
        return {
            "seconds": round(seconds, 2),
            "formatted": format_timestamp(seconds),
            "frame": frame_index
        }
    else:
        return {
            "seconds": time.time(),
            "formatted": time.strftime("%H:%M:%S", time.localtime()),
            "frame": frame_index
        }

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
