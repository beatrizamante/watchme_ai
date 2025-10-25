"This is the .env decrypt configuration module"

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """
    Settings configuration for YOLO model training and inference.

    Attributes:
        YOLO_MODEL_PATH (str): Path to the YOLO model weights file. Default is "src/infrastructure/yolo/client/best.pt".
        YOLO_EPOCHS (int): Number of epochs to train the YOLO model. Default is 50.
        YOLO_BATCH_SIZE (int): Batch size for training each epoch. Default is 16.
        YOLO_LEARNING_RATE (float): Learning rate for CNN training. Default is 0.001.
        YOLO_LOSS_FUNC (str): Loss function used for training. Default is "AdamW".
        YOLO_DROPOUT (float): Dropout value for regularization. Default is 0.0.
        YOLO_DEVICE (int | str): Device identifier for running the model (e.g., GPU index or "cpu"). Default is 0.
        YOLO_DATASET_PATH (str): Path to the YOLO dataset configuration file. Default is "src/dataset/yolo/dataset.yml".
        YOLO_PROJECT_PATH (str): Directory where training outputs (weights, metrics) are saved. Default is "src/dataset/yolo/runs/detect".
    """

    YOLO_MODEL_PATH: str = Field("src/infrastructure/yolo/client/best.pt", description="Default to partial YOLO")
    YOLO_EPOCHS: int = Field(50, description="Default number of epoch to run on YOLO")
    YOLO_BATCH_SIZE: int = Field(16, description="Default image batch size for each epoch")
    YOLO_LEARNING_RATE: float = Field(0.001, description="Default learning rate for CNN training")
    YOLO_LOSS_FUNC: str = Field("AdamW", description="Default Loss function")
    YOLO_DROPOUT: float = Field(0.0, description="Default dropout value")
    YOLO_DEVICE: int | str = Field(0, description="Which device are we running")
    YOLO_DATASET_PATH: str = Field("src/dataset/yolo/dataset.yml", description="Path where the dataset is saved")
    YOLO_PROJECT_PATH: str = Field("src/dataset/yolo/runs/detect", description="The path where the weights and metrics will be saved")


settings = Settings() # type: ignore

class OSnet_Settings(BaseSettings):
    """
    Configuration settings for OSNet model training.
    Attributes:
        OSNET_EPOCHS (int): Default number of epochs for OSNet training.
        OSNET_LEARNING_RATE (float): Default learning rate for OSNet.
        OSNET_WEIGHT_DECAY (float): Default weight decay for OSNet.
        OSNET_BATCH_SIZE (int): Default batch size for OSNet.
        OSNET_OPTIMIZER (str): Default optimizer for OSNet.
        OSNET_NUM_CLASSES (int): Number of identity classes (dataset dependent).
        OSNET_IMG_HEIGHT (int): Input image height for OSNet.
        OSNET_IMG_WIDTH (int): Input image width for OSNet.
        OSNET_NUM_INSTANCES (int): Number of instances per identity in batch.
        OSNET_MARGIN (float): Margin for triplet loss.
        OSNET_STEPSIZE (int): Step size for learning rate scheduler.
        OSNET_EVAL_FREQ (int): Evaluation frequency during training.
        OSNET_PRINT_FREQ (int): Print frequency during training.
        OSNET_SOURCE_DIR (str): Dataset source directory.
        OSNET_DATASET_NAME (str): Name of the ReID dataset.
        OSNET_SAVE_DIR (str): Directory to save OSNet results.
        model_config (SettingsConfigDict): Configuration for environment file and encoding.
    """

    OSNET_EPOCHS: int = Field(150, description="Default number of epochs for OSNet training")
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
    OSNET_EVAL_FREQ: int = Field(30, description="Evaluation frequency during training")
    OSNET_PRINT_FREQ: int = Field(10, description="Print frequency during training")
    OSNET_DATASET_NAME: str = Field("dukemtmcreid", description="Name of the ReID dataset")
    OSNET_ROOT_DIR: str = Field("src/dataset/osnet", description="Directory to save OSNet results")
    OSNET_SAVE_DIR: str = Field("src/dataset/osnet/saved_results", description="Directory to save OSNet results")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

osnet_settings = OSnet_Settings() # type: ignore
