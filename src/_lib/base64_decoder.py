
import base64
import binascii
import logging
from typing import Optional

import cv2
import numpy as np


def decode_base64_frame(frame_data: str) -> Optional[np.ndarray]:
    """
    Decodes a base64-encoded image frame into a NumPy array.

    Args:
        frame_data (str): Base64-encoded string representing the image frame.

    Returns:
        Optional[np.ndarray]: Decoded image as a NumPy array in BGR format if successful, otherwise None.
    """
    try:
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except (binascii.Error, ValueError, cv2.error) as e:
        logging.error("Error decoding frame: %s", e)
        return None
