"This is the .env decrypt configuration module"

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    "This class receives a set of configuration variables and forces its types and default values"
    # YOLO Settings
    YOLO_MODEL_PATH: str = Field("yolo11n.pt", description="Default to partial YOLO")
    YOLO_EPOCHS: int = Field(50, description="Default number of epoch to run on YOLO")
    YOLO_BATCH_SIZE: int = Field(16, description="Default image batch size for each epoch")
    YOLO_LEARNING_RATE: float = Field(0.001, description="Default learning rate for CNN training")
    YOLO_LOSS_FUNC: str = Field("AdamW", description="Default Loss function")
    YOLO_DROPOUT: float = Field(0.0, description="Default dropout value")
    YOLO_DEVICE: int | str = Field(0, description="Which device are we running") 
    
    # OSNet Settings
    OSNET_EPOCHS: int = Field(60, description="Default number of epochs for OSNet training")
    OSNET_LEARNING_RATE: float = Field(0.0003, description="Default learning rate for OSNet")
    OSNET_WEIGHT_DECAY: float = Field(5e-4, description="Default weight decay for OSNet")
    OSNET_BATCH_SIZE: int = Field(64, description="Default batch size for OSNet")
    OSNET_OPTIMIZER: str = Field("adam", description="Default optimizer for OSNet")
    OSNET_NUM_CLASSES: int = Field(751, description="Number of identity classes (dataset dependent)")
    OSNET_IMG_HEIGHT: int = Field(256, description="Input image height for OSNet")
    OSNET_IMG_WIDTH: int = Field(128, description="Input image width for OSNet")
    OSNET_NUM_INSTANCES: int = Field(4, description="Number of instances per identity in batch")
    OSNET_MARGIN: float = Field(0.3, description="Margin for triplet loss")
    OSNET_STEPSIZE: int = Field(20, description="Step size for learning rate scheduler")
    OSNET_EVAL_FREQ: int = Field(10, description="Evaluation frequency during training")
    OSNET_PRINT_FREQ: int = Field(10, description="Print frequency during training")
    OSNET_SAVE_DIR: str = Field("osnet_results", description="Directory to save OSNet results")
    OSNET_DATASET_NAME: str = Field("market1501", description="Name of the ReID dataset")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings() # type: ignore
