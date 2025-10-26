from Crypto.Cipher import AES
import numpy as np

from config import KeySetting

key_setting = KeySetting()

def decrypt_embedding(encrypted: bytes, shape, dtype) -> np.ndarray:
    nonce = encrypted[:16]
    tag = encrypted[16:32]
    ciphertext = encrypted[32:]
    cipher = AES.new(key_setting.ENCRYPTION_KEY, AES.MODE_EAX, nonce=nonce)
    data = cipher.decrypt_and_verify(ciphertext, tag)
    return np.frombuffer(data, dtype=dtype).reshape(shape)
