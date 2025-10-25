"""Module for encoding images using trained OSNet models."""

from pathlib import Path

import numpy as np
import torch

from config import osnet_settings
from src.infrastructure.osnet.scripts.shared.load_checkpoint import load_checkpoint
from src.infrastructure.osnet.client.model import OsnetModel
from src.infrastructure.osnet.scripts.transformers.transformers import \
    create_transforms, preprocess_image
class OSNetEncoder:
    """Handle OSNet encoding operations for person re-identification."""

    def __init__(self):
        self.osnet_client = OsnetModel()
        self.settings = osnet_settings
        self.model = self.osnet_client.create_osnet_model(
            num_classes=self.settings.OSNET_NUM_CLASSES
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_path = Path("src/infrastructure/osnet/client/model.pth.tar")
        self.transform = create_transforms(self.settings.OSNET_IMG_HEIGHT,
                                           self.settings.OSNET_IMG_WIDTH)
        self._load_model()

    def _load_model(self):
        """Load the OSNet model with pre-trained weights."""
        self.model = load_checkpoint(self.weights_path, self.device, self.model)
        print("OSNet model loaded successfully")

    def encode_single_image(self, image):
        """
        Encode a single image to feature vector.

        Args:
            image: Input image (PIL Image, numpy array, or tensor)

        Returns:
            numpy.ndarray: Feature vector (1D array)
        """

        image_tensor = preprocess_image(image, self.transform)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            features = self.model(image_tensor)

            if isinstance(features, tuple):
                features = features[0]

            features = torch.nn.functional.normalize(features, p=2, dim=1)
            features = features.cpu().numpy().flatten()

        return features

    def encode_batch(self, images):
        """
        Encode a batch of images to feature vectors.

        Args:
            images: List of images (PIL Images, numpy arrays, or tensors)

        Returns:
            numpy.ndarray: Feature matrix (num_images x feature_dim)
        """
        if not images:
            return np.array([])

        image_tensors = []
        for image in images:
            tensor = preprocess_image(image, self.transform)
            image_tensors.append(tensor.squeeze(0))

        batch_tensor = torch.stack(image_tensors).to(self.device)

        with torch.no_grad():
            features = self.model(batch_tensor)

            if isinstance(features, tuple):
                features = features[0]

            features = torch.nn.functional.normalize(features, p=2, dim=1)
            features = features.cpu().numpy()

        return features
