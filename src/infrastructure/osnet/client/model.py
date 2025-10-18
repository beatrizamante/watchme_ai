import torchreid
import torch

from config import osnet_settings

class OsnetModel:
    """OsnetModel is a client class for managing the OSNet model and its associated data manager using torchreid.
        model (torch.nn.Module or None): The OSNet model instance.
        datamanager (torchreid.data.ImageDataManager or None): The data manager for handling image datasets.
        settings (object): Configuration settings for the OSNet model.
    Methods:
        __init__():
            Initializes the OsnetModel instance with default values for model, datamanager, and settings.
        create_osnet_model(num_classes=None):
            Builds and returns an OSNet model using torchreid with the specified number of identity classes.
            If num_classes is not provided, uses the value from settings.
            Moves the model to CUDA if available.
        create_datamanager():
            Creates and returns a torchreid ImageDataManager for training and testing.
            Uses configuration values from settings to set up the data manager.
    """
    def __init__(self):
        """
        Initializes the model client with default values.
        """
        self.model = None
        self.datamanager = None
        self.settings = osnet_settings

    def create_osnet_model(self, num_classes=None):
        """
        Create an OSNet model using torchreid

        Args:
            num_classes: Number of identity classes in your dataset

        Returns:
            model: OSNet model instance
        """
        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=num_classes or self.settings.OSNET_NUM_CLASSES,
            loss='triplet',
        )

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        return self.model

    def create_datamanager(self):
        """
        Create a data manager for training/testing

        Returns:
            datamanager: Torchreid data manager
        """
        self.datamanager = torchreid.data.VideoDataManager(
            root=self.settings.OSNET_ROOT_DIR,
            sources=self.settings.OSNET_DATASET_NAME,
            height=self.settings.OSNET_IMG_HEIGHT,
            width=self.settings.OSNET_IMG_WIDTH,
            batch_size_train=self.settings.OSNET_BATCH_SIZE,
            batch_size_test=self.settings.OSNET_BATCH_SIZE,
            transforms=['random_flip', 'random_crop', 'random_erase'],
            num_instances=self.settings.OSNET_NUM_INSTANCES,
            train_sampler='RandomIdentitySampler',
        )

        return self.datamanager
