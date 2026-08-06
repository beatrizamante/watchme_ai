"""
Plot the Cumulative Matching Characteristic (CMC) curve for the OSNet model.

Generates a publication-ready figure (Rank-1 to Rank-20) and saves it as
both PDF and PNG under results/figures/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.infrastructure.osnet.plotting.CMC.evaluate_cmc import OSNetCMCEvaluator

_FIGURES_DIR = Path("src/infrastructure/osnet/plotting/results/figures")
_ANNOTATED_RANKS = [1, 5, 10]
_MAX_RANK = 20


def plot_cmc_curve(all_ranks: list[float], out_dir: Path = _FIGURES_DIR) -> None:
    """
    Plot and save the CMC curve.

    Args:
        all_ranks: CMC matching rates per rank (0-indexed, values in [0, 1]).
        out_dir: Directory where the figures are saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ranks = list(range(1, _MAX_RANK + 1))
    rates = [all_ranks[r - 1] * 100 for r in ranks]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(ranks, rates, color="#1f77b4", linewidth=2.5, marker="o", markersize=5)

    for r in _ANNOTATED_RANKS:
        if r <= _MAX_RANK:
            rate = all_ranks[r - 1] * 100
            ax.annotate(
                f"Rank-{r}: {rate:.1f}%",
                xy=(r, rate),
                xytext=(r + 0.4, rate - 3.5),
                fontsize=12,
                arrowprops=dict(arrowstyle="-", color="gray", lw=1.2),
            )

    ax.set_xlabel("Rank", fontsize=15)
    ax.set_ylabel("Matching Rate (%)", fontsize=15)
    ax.set_title("CMC Curve — OSNet Re-ID", fontsize=16)
    ax.set_xlim(1, _MAX_RANK)
    ax.set_ylim(0, 105)
    ax.set_xticks(ranks)
    ax.tick_params(labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    fig.tight_layout()

    pdf_path = out_dir / "cmc_curve.pdf"
    png_path = out_dir / "cmc_curve.png"
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"CMC curve saved to:\n  {pdf_path}\n  {png_path}")


def main() -> None:
    evaluator = OSNetCMCEvaluator()
    results = evaluator.evaluate()

    all_ranks: list[float] = results["CMC"]["all_ranks"]
    plot_cmc_curve(all_ranks)


if __name__ == "__main__":
    main()
