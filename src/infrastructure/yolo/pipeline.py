from .core.train import train
from .core.tune import tune

if __name__ == "__main__":
    print("Initializing...")

    print("\n[1/3] Training base model...")
    base_results = train()
    best_weights = base_results.save_dir + "/weights/best.pt"
    print(f"Best weight in {best_weights}")

    print("\n[2/3] Running hyperparameter tuning...")
    best_hp = tune(best_weights)
    print("Best hp in", best_hp.values)

    print("\n[3/3] Retraining with best parameters...")
    final_results = train(best_weights, best_hp)
    print("Saved in:", final_results.save_dir)
