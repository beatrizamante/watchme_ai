from datetime import datetime
import json
from pathlib import Path
from config import OSNetSettings
from src.infrastructure.osnet.plotting.CMC.evaluate_cmc import OSNetCMCEvaluator
from src.infrastructure.osnet.plotting.mINP.evaluate_minp import OSNetmINPEvaluator

minpEval = OSNetmINPEvaluator()
cmcEval = OSNetCMCEvaluator()
dataset_name = OSNetSettings().OSNET_DATASET_NAME

def _save_results(results, results_dir):
    """Save results to JSON file."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"minp_evaluation_{timestamp}.json"
    filepath = results_dir / filename

    with open(filepath, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")

def build_evaluation_json():
    minp_json = minpEval.evaluate()
    cmc_json = cmcEval.evaluate()

    return {
        "dataset": dataset_name,
        "CMC": {
            "Rank-1": cmc_json.get("rank1"),
            "Rank-5": cmc_json.get("rank5"),
            "Rank-10": cmc_json.get("rank10"),
            "Rank-20": cmc_json.get("rank20"),
            "all_ranks": cmc_json.get("all_ranks", [])
        },
        "mAP": cmc_json.get("mAP"),
        "mINP": minp_json.get("minp"),
        "num_query": cmc_json.get("num_query"),
        "num_gallery": cmc_json.get("num_gallery"),
        "timestamp": datetime.now().isoformat()
    }

final_json = build_evaluation_json()
_save_results(final_json, "src/infrastructure/osnet/plotting/results")
