"""
Script to calculate mean Inverse Negative Penalty (mINP) metrics for OSNet model.

mINP is a robust evaluation metric for person re-identification that better handles
the negative impact of hard false positives compared to traditional mAP.
"""

import json
from datetime import datetime
from pathlib import Path

import torch

from config import osnet_settings
from src.infrastructure.osnet.client.model import OsnetModel
from src.infrastructure.osnet.scripts.mINP.calculate_minp import calculate_minp
from src.infrastructure.osnet.scripts.shared.extract_features import \
    extract_features
from src.infrastructure.osnet.scripts.shared.load_checkpoint import \
    load_checkpoint


class OSNetmINPEvaluator:
    """Evaluate OSNet model using mean Inverse Negative Penalty (mINP) metrics."""

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


    def evaluate(self, save_results=True, results_dir="src/infrastructure/osnet/scripts/results"):
        """
        Run complete mINP evaluation.

        Args:
            save_results: Whether to save results to file
            results_dir: Directory to save results

        Returns:
            dict: Evaluation results
        """
        print("=" * 60)
        print("Starting mINP Evaluation for OSNet")
        print("=" * 60)

        (query_features, query_pids, query_camids,
         gallery_features, gallery_pids, gallery_camids) = extract_features(self.weights_path, self.device, self.datamanager)

        results = calculate_minp(
            query_features, query_pids, query_camids,
            gallery_features, gallery_pids, gallery_camids
        )

        self._print_results(results)

        if save_results:
            self._save_results(results, results_dir)

        return results

    def _print_results(self, results):
        """Print evaluation results."""
        print("\n" + "=" * 40)
        print("mINP EVALUATION RESULTS")
        print("=" * 40)
        print(f"Dataset: {results['dataset']}")
        print(f"Query samples: {results['num_query']}")
        print(f"Gallery samples: {results['num_gallery']}")
        print(f"mINP: {results['mINP']:.6f}")
        print(f"mAP (for comparison): {results['mAP']:.4f}")
        print("\nNote: Lower mINP values indicate better performance")
        print("(mINP measures negative penalty, so lower is better)")

    def _save_results(self, results, results_dir):
        """Save results to JSON file."""
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"minp_evaluation_{timestamp}.json"
        filepath = results_dir / filename

        with open(filepath, 'w', encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filepath}")
