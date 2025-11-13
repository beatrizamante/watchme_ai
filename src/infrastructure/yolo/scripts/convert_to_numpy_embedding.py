import base64
import logging

import numpy as np

logger = logging.getLogger("watchmeai")

def convert_to_numpy_embedding(chosen_person):
    """Safely convert various embedding formats to numpy array"""
    logger.debug(f"Converting embedding of type: {type(chosen_person)}")

    if isinstance(chosen_person, np.ndarray):
        return chosen_person.astype(np.float32)

    elif isinstance(chosen_person, list):
        return np.array(chosen_person, dtype=np.float32)

    elif isinstance(chosen_person, str):
        try:
            encoding_bytes = base64.b64decode(chosen_person)
            return np.frombuffer(encoding_bytes, dtype=np.float32)
        except:
            pass

        try:
            clean_str = chosen_person.strip()
            if clean_str.startswith('[') and clean_str.endswith(']'):
                data = eval(clean_str)
                return np.array(data, dtype=np.float32)
        except:
            pass

        raise ValueError(f"Cannot convert string embedding: {chosen_person[:100]}...")

    else:
        raise ValueError(f"Unsupported embedding type: {type(chosen_person)}")
