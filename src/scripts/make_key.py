import base64
import secrets

key = secrets.token_bytes(32)
key_b64 = base64.b64encode(key).decode()
print(f"ENCRYPTION_KEY={key_b64}")
