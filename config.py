"This is the .env decrypt configuration module"

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv
from typing import Union
import base64

load_dotenv()

class YOLOSettings(BaseSettings):
    YOLO_MODEL_PATH: str = Field(default="src/infrastructure/yolo/client/best.pt", description="Default to partial YOLO")
    YOLO_EPOCHS: int = Field(default=50, description="Default number of epoch to run on YOLO")
    YOLO_BATCH_SIZE: int = Field(default=16, description="Default image batch size for each epoch")
    YOLO_LEARNING_RATE: float = Field(default=0.001, description="Default learning rate for CNN training")
    YOLO_DROPOUT: float = Field(default=0.0, description="Default dropout rate for YOLO.")
    YOLO_LOSS_FUNC: str = Field(default="AdamW", description="Default Loss function")
    YOLO_DEVICE: Union[int, str] = Field(default=0, description="Which device are we running, default to GPU")
    YOLO_DATASET_PATH: str = Field(default="src/dataset/yolo/dataset.yml", description="Path where the dataset is saved")
    YOLO_PROJECT_PATH: str = Field(default="src/dataset/yolo/runs/detect", description="The path where the weights and metrics will be saved")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

class OSNetSettings(BaseSettings):
    OSNET_EPOCHS: int = Field(default=150, description="Default number of epochs for OSNet training")
    OSNET_LEARNING_RATE: float = Field(default=0.0003, description="Default learning rate for OSNet")
    OSNET_WEIGHT_DECAY: float = Field(default=5e-4, description="Default weight decay for OSNet")
    OSNET_BATCH_SIZE: int = Field(default=64, description="Default batch size for OSNet")
    OSNET_OPTIMIZER: str = Field(default="adam", description="Default optimizer for OSNet")
    OSNET_NUM_CLASSES: int = Field(default=751, description="Number of identity classes (dataset dependent)")
    OSNET_IMG_HEIGHT: int = Field(default=256, description="Input image height for OSNet")
    OSNET_IMG_WIDTH: int = Field(default=128, description="Input image width for OSNet")
    OSNET_NUM_INSTANCES: int = Field(default=4, description="Number of instances per identity in batch")
    OSNET_MARGIN: float = Field(default=0.3, description="Margin for triplet loss")
    OSNET_STEPSIZE: int = Field(default=20, description="Step size for learning rate scheduler")
    OSNET_EVAL_FREQ: int = Field(default=30, description="Evaluation frequency during training")
    OSNET_PRINT_FREQ: int = Field(default=10, description="Print frequency during training")
    OSNET_DATASET_NAME: str = Field(default="dukemtmcreid", description="Name of the ReID dataset")
    OSNET_ROOT_DIR: str = Field(default="src/dataset/osnet", description="Directory to save OSNet results")
    OSNET_SAVE_DIR: str = Field(default="src/dataset/osnet/saved_results", description="Directory to save OSNet results")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
class KeySetting(BaseSettings):
    ENCRYPTION_KEY: str = Field(default="Insert Key", description="AES Encryption key for embedding security. Add in .env")

    @property
    def key_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.ENCRYPTION_KEY)
        except Exception:
            return self.ENCRYPTION_KEY.encode()

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
