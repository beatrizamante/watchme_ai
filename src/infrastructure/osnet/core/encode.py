"""Module for encoding images using trained OSNet models."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import OSNetSettings
from src.infrastructure.osnet.client.model import OSNetModel
from src.infrastructure.osnet.scripts.load_checkpoint import load_checkpoint
from src.infrastructure.osnet.scripts.transformers.transformers import \
    create_transforms, preprocess_image

class OSNetEncoder:
    """Handle OSNet encoding operations for person re-identification."""

    def __init__(self):
        self.osnet_client = OSNetModel()
        self.settings = OSNetSettings()
        self.model = self.osnet_client.create_osnet_model(
            num_classes=self.settings.OSNET_NUM_CLASSES
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_path = Path(self.settings.OSNET_SAVE_DIR, self.settings.OSNET_MODEL_NAME)
        self.transform = create_transforms(
            self.settings.OSNET_IMG_HEIGHT,
            self.settings.OSNET_IMG_WIDTH
        )
        self._load_model()

    def _load_model(self):
        """Load the OSNet model with pre-trained weights."""
        self.model = load_checkpoint(self.weights_path, self.device, self.model)
        print("OSNet model loaded successfully")

    def encode_single_image(self, image):
        """
        Encode a single image - treat it as a single frame, not a video sequence.
        """
        image_tensor = preprocess_image(image, self.transform)
        image_tensor = image_tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            features = self.model(image_tensor)

            if isinstance(features, (tuple, list)):
                features = features[0]

            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)

            features = F.normalize(features, p=2, dim=1)
            features = features.cpu().numpy().flatten()

        return features.astype(np.float32)

    def encode_batch(self, images):
        """
        Encode a batch of images to feature vectors.
        """
        if not images:
            return []

        batch_tensors = []
        for image in images:
            image_tensor = preprocess_image(image, self.transform)
            batch_tensors.append(image_tensor.squeeze(0))

        batch_tensor = torch.stack(batch_tensors).to(self.device)

        self.model.eval()
        with torch.no_grad():
            features = self.model(batch_tensor)

            if isinstance(features, (tuple, list)):
                features = features[0]

            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)

            features = F.normalize(features, p=2, dim=1)
            features = features.cpu().numpy()

        return [feat.astype(np.float32) for feat in features]
