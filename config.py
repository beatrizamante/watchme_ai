from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    YOLO_MODEL_PATH: str = Field()
    YOLO_EPOCHS: int = Field(150, description="Default number of epoch to run on YOLO")
    YOLO_IMAGE_SIZE: int = Field(64, description="Default image batch size for each epoch")
    YOLO_LEARNING_RATE: float = Field(0.001, description="Default learning rate fo CNN training")
    YOLO_LOSS_FUNC: str = Field()
    YOLO_DROPOUT: float = Field()
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings() # type: ignore
