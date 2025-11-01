
import base64
import logging
from typing import Optional

import cv2
import numpy as np


def decode_base64_frame(frame_data: str) -> Optional[np.ndarray]:
    """Decode base64 encoded frame to numpy array"""
    try:
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logging.error(f"Error decoding frame: {e}")
        return None
