import torchreid
import torch

from config import OSNetSettings
class OSNetModel:
    """
    A wrapper class for creating and managing an OSNet model and its data manager using torchreid.
    Attributes:
        model: The OSNet model instance.
        datamanager: The torchreid data manager instance.
        settings: Configuration settings for OSNet.
    """

    def __init__(self):
        self.model = None
        self.datamanager = None
        self.settings = OSNetSettings()

    def create_osnet_model(self, num_classes=None):
        """
        Create an OSNet model using torchreid
        Args:
            num_classes: Number of identity classes in your dataset

        Returns:
            model: OSNet model instance
        """
        self.model = torchreid.models.build_model(
            name='osnet_ibn_x1_0',
            num_classes=num_classes or self.settings.OSNET_NUM_CLASSES,
            loss='triplet',
            pretrained=False
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
            transforms=['random_flip', 'random_crop', 'random_erase', 'color_jitter'],
            num_instances=self.settings.OSNET_NUM_INSTANCES,
            train_sampler='RandomIdentitySampler',
            seq_len=8,
            sample_method='evenly'
        )

        return self.datamanager
