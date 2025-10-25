"""
Script to calculate Cumulative Matching Characteristic (CMC) metrics for OSNet model.

CMC measures the probability that the correct match appears in the top-k retrieved results.
"""

import json
from datetime import datetime
from pathlib import Path

import torch

from config import osnet_settings
from src.infrastructure.osnet.client.model import OsnetModel
from src.infrastructure.osnet.scripts.CMC.calculate_cmc import calculate_cmc
from src.infrastructure.osnet.scripts.shared.extract_features import \
    extract_features
from src.infrastructure.osnet.scripts.shared.load_checkpoint import \
    load_checkpoint


class OSNetCMCEvaluator:
    """Evaluate OSNet model using Cumulative Matching Characteristic (CMC) metrics."""

    def __init__(self):
        self.osnet_client = OsnetModel()
        self.settings = osnet_settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_path = Path("src/infrastructure/osnet/client/model.pth.tar")
        self.dataset_name = "dukemtmcreid"
        self.model = self.osnet_client.create_osnet_model(
            num_classes=self.settings.OSNET_NUM_CLASSES
        )
        self.datamanager = self.osnet_client.create_datamanager()
        self.feature_extractor = None

        self._load_model()

    def _load_model(self):
        """Load the trained OSNet model."""
        print(f"Loading OSNet model from {self.weights_path}")
        self.model = load_checkpoint(self.weights_path, self.device, self.model)
        print("OSNet model loaded successfully")

    def evaluate(self, results_dir="src/infrastructure/osnet/scripts/results"):
        """
        Run complete CMC evaluation.

        Args:
            save_results: Whether to save results to file
            results_dir: Directory to save results

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

        self._save_results(results, results_dir)

        return results

    def _save_results(self, results, results_dir):
        """Save results to JSON file."""
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cmc_evaluation_{timestamp}.json"
        filepath = results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filepath}")
