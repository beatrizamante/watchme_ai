import base64
import numpy as np
from Crypto.Cipher import AES

from config import KeySetting


class EncryptionService:
    """EncryptionService provides AES encryption for NumPy embedding arrays.

    Attributes:
        key_setting (KeySetting): Configuration object containing the AES key bytes.
    """

    def __init__(self, key_setting: KeySetting):
        self.key_setting = key_setting

    def encrypt_embedding(self, embedding: np.ndarray) -> str:
        """
        Encrypts a NumPy embedding array using AES encryption in EAX mode.

        Args:
            embedding (np.ndarray): The NumPy array representing the embedding to encrypt.

        Returns:
            str: The base64-encoded encrypted data, including the nonce, authentication tag, and ciphertext.
        """
        data = embedding.tobytes()
        cipher = AES.new(self.key_setting.key_bytes, AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(data)
        encrypted_data = cipher.nonce + tag + ciphertext
        return base64.b64encode(encrypted_data).decode('utf-8')
