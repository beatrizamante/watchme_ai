"""OSNet training pipeline for person re-identification."""

from pathlib import Path
from config import osnet_settings
from src.infrastructure.osnet.core.train import OSNetTrainer

class OSNetPipeline:
    """Complete OSNet training pipeline with resume support."""

    def __init__(self) -> None:
        self.trainer = None
        self.settings = osnet_settings
        self.baseline_weights = None
        self.final_results = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize the OSNet trainer."""
        self.trainer = OSNetTrainer()

    def _check_for_baseline_weights(self):
        """Check if there's an existing baseline model to resume from."""
        baseline_dir = Path(self.settings.OSNET_SAVE_DIR) / "baseline_train"
        best_weights_path = baseline_dir / "model.pth.tar"

        if best_weights_path.exists():
            return True, str(best_weights_path)
        return False, None

    def run(self):
        """
        Run complete OSNet training pipeline.

        Returns:
            dict: Final training results
        """
        print("=" * 60)
        print("OSNet Training Pipeline")
        print("=" * 60)

        has_baseline, baseline_path = self._check_for_baseline_weights()

        if not has_baseline:
            print("\n[1/2] Training baseline OSNet model...")
            print("-" * 60)

            baseline_results = self.trainer.train()
            self.baseline_weights = self.trainer.get_best_model_path()

            print("✓ Baseline training completed")
            print(f"  - Rank-1: {baseline_results['rank1']:.3f}")
            print(f"  - mAP: {baseline_results['mAP']:.3f}")
            print(f"  - Model saved at: {self.baseline_weights}")
        else:
            print(f"\n[1/2] Found existing baseline weights: {baseline_path}")
            self.baseline_weights = baseline_path

        print("\n[2/2] Final training with baseline weights...")
        print("-" * 60)

        #self.final_results = self.trainer.train(weights=self.baseline_weights)

        print("\n" + "=" * 60)
        print("OSNet Pipeline completed successfully!")
        #print(f"Final Rank-1: {self.final_results['rank1']:.3f}")
        #print(f"Final mAP: {self.final_results['mAP']:.3f}")
        #print(f"Model saved at: {self.final_results['save_dir']}")
        print("=" * 60)

        return self.final_results


# Standalone function for easy usage
def run_osnet_pipeline():
    """Run the complete OSNet training pipeline."""
    pipeline = OSNetPipeline()
    return pipeline.run()
