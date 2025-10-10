import torch
from .core.train import train
from .core.tune import model_tune
import os
import json

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if __name__ == "__main__":
    print("="*60)
    print("YOLO Training & Hyperparameter Tuning Pipeline")
    print("="*60)

    print("\n[1/3] Training baseline model...")
    print("-" * 60)
    base_results = train()
    best_weights = os.path.join(str(base_results.save_dir), "weights", "best.pt") # type: ignore
    print("✓ Baseline training complete!")
    print(f"  Best weights: {best_weights}")

    print("\n[2/3] Running hyperparameter tuning with Ray Tune...")
    print("-" * 60)
    print("This may take a while...")
    
    best_hp = model_tune(best_weights)
    
    print("✓ Hyperparameter tuning complete!")
    print("  Best hyperparameters:")
    if best_hp is not None:
        for key, value in best_hp.items():
            print(f"    - {key}: {value}")
    else:
        print("    No hyperparameters found (best_hp is None).")

    print("\n[3/3] Retraining with optimized hyperparameters...")
    print("-" * 60)
    
    final_results = train(weights=best_weights, hp=best_hp)
    
    print("✓ Final training complete!")
    print(f"  Model saved to: {final_results.save_dir}")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    
    config_path = os.path.join(str(final_results.save_dir), "best_hyperparameters.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(best_hp, f, indent=2)
    print(f"\nBest hyperparameters saved to: {config_path}")