"This is the .env decrypt configuration module"

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    "This class receives a set of configuration variables and forces its types and default values"
    YOLO_MODEL_PATH: str = Field("yolo11n.pt", description="Default to partial YOLO")
    YOLO_EPOCHS: int = Field(150, description="Default number of epoch to run on YOLO")
    YOLO_BATCH_SIZE: int = Field(1, description="Default image batch size for each epoch")
    YOLO_LEARNING_RATE: float = Field(0.001, description="Default learning rate for CNN training")
    YOLO_LOSS_FUNC: str = Field("Adam", description="Default Loss function")
    YOLO_DROPOUT: float = Field(0.1, description="Default dropout value")
    YOLO_DEVICE: str = Field('cpu', description="Which device are we running") 
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings() # type: ignore
