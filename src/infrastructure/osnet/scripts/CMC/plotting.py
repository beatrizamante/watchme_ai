import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_cmc_curve(json_path, save_path=None):
    """
    Plot CMC curve from evaluation results.

    Args:
        json_path: Path to JSON file with CMC results
        save_path: Optional path to save the figure
    """
    with open(json_path, 'r') as f:
        results = json.load(f)

    cmc_scores = results['CMC']['all_ranks']
    ranks = np.arange(1, len(cmc_scores) + 1)

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))

    plt.plot(ranks, np.array(cmc_scores) * 100,
             linewidth=3, color='#3b82f6', label='CMC Curve')

    important_ranks = [1, 5, 10, 20]
    for rank in important_ranks:
        if rank <= len(cmc_scores):
            plt.scatter(rank, cmc_scores[rank-1] * 100,
                       s=100, color='#ef4444', zorder=5)
            plt.text(rank, cmc_scores[rank-1] * 100 - 3,
                    f'{cmc_scores[rank-1]*100:.2f}%',
                    ha='center', fontsize=10, fontweight='bold')

    plt.xlabel('Rank', fontsize=14, fontweight='bold')
    plt.ylabel('Matching Rate (%)', fontsize=14, fontweight='bold')
    plt.title(f'CMC Curve - {results["dataset"]}\nmAP: {results["mAP"]*100:.2f}%',
              fontsize=16, fontweight='bold', pad=20)

    plt.grid(True, alpha=0.3)
    plt.xlim(0, min(50, len(ranks)))
    plt.ylim(0, 105)

    plt.legend(fontsize=12, loc='lower right')

    info_text = f"Query: {results['num_query']}\nGallery: {results['num_gallery']}\nRank-1: {cmc_scores[0]*100:.2f}%"
    plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, verticalalignment='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
