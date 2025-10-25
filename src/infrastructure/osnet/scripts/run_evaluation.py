"""
This script demonstrates how to run CMC and mINP evaluations on your trained OSNet model.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import the evaluation modules
sys.path.append(str(Path(__file__).parent))

from src.infrastructure.osnet.scripts.CMC.evaluate_cmc import OSNetCMCEvaluator
from src.infrastructure.osnet.scripts.mINP.evaluate_minp import OSNetmINPEvaluator


def example_cmc_evaluation():
    """Example of running CMC evaluation."""
    print("=" * 60)
    print("EXAMPLE: CMC Evaluation")
    print("=" * 60)

    try:
        # Initialize CMC evaluator
        evaluator = OSNetCMCEvaluator(
            weights_path="src/infrastructure/osnet/client/model.pth.tar-250",
            dataset_name="dukemtmcreid"  # or your dataset name
        )

        # Run evaluation
        results = evaluator.evaluate(save_results=True)

        # Print key metrics
        print(f"\nKey Results:")
        print(f"  mAP: {results['mAP']:.4f}")
        print(f"  Rank-1: {results['CMC'].get('Rank-1', 'N/A')}")
        print(f"  Rank-5: {results['CMC'].get('Rank-5', 'N/A')}")

        return results

    except Exception as e:
        print(f"Error in CMC evaluation: {e}")
        return None


def example_minp_evaluation():
    """Example of running mINP evaluation."""
    print("\n" + "=" * 60)
    print("EXAMPLE: mINP Evaluation")
    print("=" * 60)

    try:
        # Initialize mINP evaluator
        evaluator = OSNetmINPEvaluator(
            weights_path="src/infrastructure/osnet/client/model.pth.tar-250",
            dataset_name="dukemtmcreid"  # or your dataset name
        )

        # Run evaluation
        results = evaluator.evaluate(save_results=True)

        # Print key metrics
        print(f"\nKey Results:")
        print(f"  mINP: {results['mINP']:.6f} (lower is better)")
        print(f"  mAP: {results['mAP']:.4f}")

        return results

    except Exception as e:
        print(f"Error in mINP evaluation: {e}")
        return None


def run_both_evaluations():
    """Run both CMC and mINP evaluations."""
    print("=" * 70)
    print("RUNNING BOTH CMC AND mINP EVALUATIONS")
    print("=" * 70)

    # Run CMC evaluation
    cmc_results = example_cmc_evaluation()

    # Run mINP evaluation
    minp_results = example_minp_evaluation()

    # Compare results
    if cmc_results and minp_results:
        print("\n" + "=" * 50)
        print("COMPARISON SUMMARY")
        print("=" * 50)
        print(f"CMC mAP:     {cmc_results['mAP']:.4f}")
        print(f"mINP mAP:    {minp_results['mAP']:.4f}")
        print(f"mINP Score:  {minp_results['mINP']:.6f} (lower is better)")

        if cmc_results.get('CMC', {}).get('Rank-1'):
            print(f"Rank-1 Acc: {cmc_results['CMC']['Rank-1']:.4f}")


def quick_evaluation_guide():
    """Print a quick guide for running evaluations."""
    print("=" * 70)
    print("QUICK EVALUATION GUIDE")
    print("=" * 70)

    print("\n📋 Available Evaluation Scripts:")
    print("  1. evaluate_cmc.py        - Cumulative Matching Characteristic")
    print("  2. evaluate_minp.py       - mean Inverse Negative Penalty")
    print("  3. evaluate_comprehensive.py - Both CMC and mINP")

    print("\n🚀 Command Line Usage:")
    print("  # CMC evaluation only")
    print("  python src/infrastructure/osnet/scripts/evaluate_cmc.py --save-results")

    print("\n  # mINP evaluation only")
    print("  python src/infrastructure/osnet/scripts/evaluate_minp.py --save-results")

    print("\n  # Both evaluations with custom weights")
    print("  python src/infrastructure/osnet/scripts/evaluate_comprehensive.py \\")
    print("    --weights src/infrastructure/osnet/client/model.pth.tar-250 \\")
    print("    --save-results")

    print("\n  # CMC only with custom dataset")
    print("  python src/infrastructure/osnet/scripts/evaluate_comprehensive.py \\")
    print("    --cmc-only --dataset market1501 --save-results")

    print("\n📊 What Each Metric Means:")
    print("  • mAP: Mean Average Precision (0-1, higher is better)")
    print("  • Rank-1: Top-1 accuracy (0-1, higher is better)")
    print("  • Rank-5: Top-5 accuracy (0-1, higher is better)")
    print("  • mINP: Negative penalty score (0+, lower is better)")

    print("\n📁 Results Location:")
    print("  Results are saved to: src/infrastructure/osnet/scripts/results/")

    print("\n⚙️  Requirements:")
    print("  Make sure you have:")
    print("  • Trained model weights at: src/infrastructure/osnet/client/model.pth.tar-250")
    print("  • Dataset configured in config.py (OSNET_DATASET_NAME)")
    print("  • PyTorch and torchreid installed")


def check_setup():
    """Check if the setup is ready for evaluation."""
    print("=" * 60)
    print("SETUP CHECK")
    print("=" * 60)

    # Check for weights file
    weights_path = Path("src/infrastructure/osnet/client/model.pth.tar-250")
    if weights_path.exists():
        print("✅ Model weights found")
        print(f"   Path: {weights_path}")
        print(f"   Size: {weights_path.stat().st_size / (1024*1024):.1f} MB")
    else:
        print("❌ Model weights not found")
        print(f"   Expected at: {weights_path}")
        print("   Make sure you have trained the model first")

    # Check for dataset directory
    dataset_dir = Path("src/dataset/osnet")
    if dataset_dir.exists():
        print("✅ Dataset directory found")
        print(f"   Path: {dataset_dir}")
    else:
        print("❌ Dataset directory not found")
        print(f"   Expected at: {dataset_dir}")

    # Check for results directory
    results_dir = Path("src/infrastructure/osnet/scripts/results")
    if not results_dir.exists():
        print("📁 Creating results directory...")
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {results_dir}")
    else:
        print("✅ Results directory exists")

    # Try to import required packages
    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        print("❌ PyTorch not available - install with: pip install torch")

    try:
        import torchreid
        print("✅ Torchreid available")
    except ImportError:
        print("❌ Torchreid not available - install with: pip install torchreid")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Check setup first
    check_setup()

    # Show usage guide
    quick_evaluation_guide()

    # Ask user what they want to do
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Choose an option:")
    print("1. Run CMC evaluation")
    print("2. Run mINP evaluation")
    print("3. Run both evaluations")
    print("4. Exit")

    try:
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            example_cmc_evaluation()
        elif choice == "2":
            example_minp_evaluation()
        elif choice == "3":
            run_both_evaluations()
        elif choice == "4":
            print("Goodbye!")
        else:
            print("Invalid choice. Use command line arguments instead.")
            print("Example: python this_script.py")

    except KeyboardInterrupt:
        print("\nEvaluation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all dependencies are installed and weights are available.")
