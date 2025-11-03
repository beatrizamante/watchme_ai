from Crypto.Cipher import AES
import numpy as np

from config import KeySetting
key_setting = KeySetting()

def encrypt_embedding(embedding: np.ndarray) -> bytes:
    data = embedding.tobytes()
    cipher = AES.new(key_setting.key_bytes, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return cipher.nonce + tag + ciphertext
