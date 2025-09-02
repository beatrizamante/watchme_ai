from pydantic import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    YOLO_MODEL_PATH: str
    YOLO_EPOCHS: int = 50
    YOLO_IMAGE_SIZE: int = 640

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
