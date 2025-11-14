import base64
from Crypto.Cipher import AES
import numpy as np

from config import KeySetting

key_setting = KeySetting()

def decrypt_embedding(encrypted: str, shape, dtype) -> np.ndarray:
    """
    Decrypts an encrypted embedding and returns it as a NumPy array.

    Args:
        encrypted (str): The encrypted embedding, encoded in base64.
        shape (tuple): The shape to reshape the decrypted data into.
        dtype (np.dtype): The data type of the resulting NumPy array.

    Returns:
        np.ndarray: The decrypted embedding as a NumPy array.
    """
    encrypted_bytes = base64.b64decode(encrypted)

    nonce = encrypted_bytes[:16]
    tag = encrypted_bytes[16:32]
    ciphertext = encrypted_bytes[32:]
    cipher = AES.new(key_setting.key_bytes, AES.MODE_EAX, nonce=nonce)
    data = cipher.decrypt_and_verify(ciphertext, tag)
    return np.frombuffer(data, dtype=dtype).reshape(shape)
