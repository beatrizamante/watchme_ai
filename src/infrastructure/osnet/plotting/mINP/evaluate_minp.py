"""
Script to calculate mean Inverse Negative Penalty (mINP) metrics for OSNet model.
"""

from pathlib import Path

import torch

from config import OSNetSettings
from src.infrastructure.osnet.client.model import OSNetModel
from src.infrastructure.osnet.plotting.mINP.calculate_minp import calculate_minp
from src.infrastructure.osnet.plotting.shared.extract_features import extract_features
from src.infrastructure.osnet.plotting.shared.load_checkpoint import load_checkpoint

class OSNetmINPEvaluator:
    """Evaluate OSNet model using mean Inverse Negative Penalty (mINP) metrics."""

    def __init__(self):
        self.osnet_client = OSNetModel()
        self.settings = OSNetSettings()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_path = Path("src/infrastructure/osnet/client/model.pth.tar")
        self.dataset_name = self.settings.OSNET_DATASET_NAME
        self.model = self.osnet_client.create_osnet_model(
            num_classes=self.settings.OSNET_NUM_CLASSES
        )
        self.datamanager = self.osnet_client.create_datamanager()
        self.feature_extractor = None

        self._load_model()

    def _load_model(self):
        """Load the trained OSNet model."""
        self.model = load_checkpoint(self.weights_path, self.device, self.model)
        print("OSNet model loaded successfully")


    def evaluate(self):
        """
        Run complete mINP evaluation.

        Returns:
            dict: Evaluation results
        """
        print("=" * 60)
        print("Starting mINP Evaluation for OSNet")
        print("=" * 60)

        (query_features, query_pids, query_camids,
         gallery_features, gallery_pids, gallery_camids) = extract_features(self.model, self.device, self.datamanager)

        results = calculate_minp(
            query_features, query_pids, query_camids,
            gallery_features, gallery_pids, gallery_camids
        )

        return results
