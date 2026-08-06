"""
Plot a bar chart of the mINP (mean Inverse Negative Penalty) score for the OSNet model.

The chart shows the overall mINP score and, optionally, a baseline for comparison.
Saves publication-ready figures as PDF and PNG under results/figures/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.infrastructure.osnet.plotting.mINP.evaluate_minp import OSNetmINPEvaluator

_FIGURES_DIR = Path("src/infrastructure/osnet/plotting/results/figures")

# Optional baseline scores from prior experiments — update or extend as needed.
# Set to None to omit baseline comparison.
_BASELINES: dict[str, float] | None = None


def plot_minp_chart(
    minp_score: float,
    baselines: dict[str, float] | None = _BASELINES,
    out_dir: Path = _FIGURES_DIR,
) -> None:
    """
    Plot and save the mINP bar chart.

    Args:
        minp_score: Overall mINP score in [0, 1].
        baselines: Optional dict mapping experiment name → mINP score for
                   side-by-side comparison (e.g. ``{"Baseline": 0.42}``).
        out_dir: Directory where the figures are saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: list[str] = ["OSNet (ours)"]
    values: list[float] = [minp_score]

    if baselines:
        for name, score in baselines.items():
            labels.append(name)
            values.append(score)

    x = np.arange(len(labels))
    colors = ["#1f77b4"] + ["#aec7e8"] * (len(labels) - 1)

    fig, ax = plt.subplots(figsize=(max(5, 2.5 * len(labels)), 6))

    bars = ax.bar(x, [v * 100 for v in values], width=0.5, color=colors,
                  edgecolor="black", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("mINP (%)", fontsize=15)
    ax.set_title("mINP — OSNet Re-ID", fontsize=16)
    ax.set_ylim(0, min(100, max(v * 100 for v in values) * 1.25))
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    fig.tight_layout()

    pdf_path = out_dir / "minp_chart.pdf"
    png_path = out_dir / "minp_chart.png"
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"mINP chart saved to:\n  {pdf_path}\n  {png_path}")


def main() -> None:
    evaluator = OSNetmINPEvaluator()
    results = evaluator.evaluate()

    minp_score: float = results["mINP"]
    print(f"\nOverall mINP: {minp_score * 100:.2f}%")

    plot_minp_chart(minp_score)


if __name__ == "__main__":
    main()
