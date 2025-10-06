import torch
from .core.train import train
from .core.tune import model_tune

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if __name__ == "__main__":
    print("Initializing...")

    print("\n[1/3] Training base model...")
    base_results = train()
    best_weights = str(base_results.save_dir) + "/weights/best.pt" # type: ignore
    print(f"Best weight in {best_weights}")

    print("\n[2/3] Running hyperparameter tuning...")
    best_hp = model_tune(best_weights)
    print("Best hp in", best_hp.get_best_result().config)

    print("\n[3/3] Retraining with best parameters...")
    final_results = train(weights=best_weights, hp=best_hp.values)
    print("Saved in:", final_results.save_dir) # type: ignore
