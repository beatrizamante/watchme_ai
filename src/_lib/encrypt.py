from Crypto.Cipher import AES
import numpy as np
import base64

from config import KeySetting
key_setting = KeySetting()

def encrypt_embedding(embedding: np.ndarray) -> str:
    """
    Encrypts a NumPy embedding array using AES encryption in EAX mode.

    Args:
        embedding (np.ndarray): The NumPy array representing the embedding to encrypt.

    Returns:
        str: The base64-encoded encrypted data, including the nonce, authentication tag, and ciphertext.

    Note:
        The encryption key is retrieved from `key_setting.key_bytes`. Ensure that `key_setting` and `AES` are properly configured and imported.
    """

    data = embedding.tobytes()
    cipher = AES.new(key_setting.key_bytes, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    encrypted_data = cipher.nonce + tag + ciphertext
    return base64.b64encode(encrypted_data).decode('utf-8')
