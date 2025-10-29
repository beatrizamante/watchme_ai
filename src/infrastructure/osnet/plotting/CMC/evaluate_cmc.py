"""
Script to calculate Cumulative Matching Characteristic (CMC) metrics for OSNet model.

CMC measures the probability that the correct match appears in the top-k retrieved results.
"""

import torch

from config import OSNetSettings
from src.infrastructure.osnet.client.model import OSNetModel
from src.infrastructure.osnet.plotting.CMC.calculate_cmc import calculate_cmc
from src.infrastructure.osnet.plotting.shared.extract_features import extract_features
from src.infrastructure.osnet.plotting.shared.load_checkpoint import load_checkpoint

class OSNetCMCEvaluator:
    """Evaluate OSNet model using Cumulative Matching Characteristic (CMC) metrics."""

    def __init__(self):
        self.osnet_client = OSNetModel()
        self.settings = OSNetSettings()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_path = self.settings.OSNET_SAVE_DIR,
        self.dataset_name = "dukemtmcreid"
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
        Run complete CMC evaluation.

        Returns:
            dict: Evaluation results
        """
        print("=" * 60)
        print("Starting CMC Evaluation for OSNet")
        print("=" * 60)

        (query_features, query_pids, query_camids,
          gallery_features, gallery_pids, gallery_camids) = extract_features(
             self.model,
             self.device,
             self.datamanager
          )

        results = calculate_cmc(
            query_features, query_pids, query_camids,
            gallery_features, gallery_pids, gallery_camids
          )

        return results
