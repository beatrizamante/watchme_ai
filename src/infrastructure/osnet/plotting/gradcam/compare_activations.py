"""Side-by-side Grad-CAM comparison: canonical pose vs. extreme pose.

Produces a 2×2 figure::

    ┌─────────────────────┬──────────────────────────┐
    │  Canonical (orig)   │  Canonical (Grad-CAM)    │
    ├─────────────────────┼──────────────────────────┤
    │  Extreme   (orig)   │  Extreme   (Grad-CAM)    │
    └─────────────────────┴──────────────────────────┘

The plot title includes the cosine distance between the two embeddings so
it is immediately clear "how much" the extreme pose breaks the embedding.

Each Grad-CAM heatmap shows which spatial regions the model relies on when
comparing that pose against the *other* pose — symmetric cross-comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.infrastructure.osnet.core.encode import OSNetEncoder
from src.infrastructure.osnet.plotting.gradcam.osnet_gradcam import OSNetGradCAM

# Default output directory (relative to this file's package root)
_DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "plotting" / "results" / "gradcam"
)


def _load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as an RGB numpy array (H×W×3, uint8)."""
    return np.array(Image.open(Path(path)).convert("RGB"))


def _overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a normalised [0,1] heatmap over an RGB image.

    Args:
        image_rgb: Original image, shape ``(H, W, 3)``, dtype uint8.
        heatmap: Grad-CAM output, shape ``(h, w)``, values in ``[0, 1]``.
        alpha: Heatmap opacity (0 = invisible, 1 = fully opaque).

    Returns:
        Blended RGB image as uint8 array of the same shape as *image_rgb*.
    """
    h, w = image_rgb.shape[:2]

    # Resize heatmap to match image resolution
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (w, h), resample=Image.BILINEAR
        )
    ) / 255.0

    # Apply jet colormap → (H, W, 4) RGBA, drop alpha channel
    colormap = cm.jet(heatmap_resized)[..., :3]  # (H, W, 3), float [0,1]

    blended = (1 - alpha) * (image_rgb / 255.0) + alpha * colormap
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def compare_activations(
    canonical_path: str | Path,
    extreme_path: str | Path,
    encoder: OSNetEncoder,
    output_dir: str | Path | None = None,
    save_name: str = "gradcam_comparison.png",
    heatmap_alpha: float = 0.5,
) -> Path:
    """Generate and save a side-by-side Grad-CAM comparison figure.

    Runs Grad-CAM symmetrically:
    - Canonical row: query=canonical, gallery=extreme
    - Extreme row:   query=extreme,   gallery=canonical

    This reveals which body regions each pose activates when the model tries
    to match it against the opposite-pose reference.

    Args:
        canonical_path: Path to the canonical (ground-truth) pose image.
        extreme_path: Path to the extreme pose image.
        encoder: Loaded :class:`OSNetEncoder` instance (model must be ready).
        output_dir: Directory to save the figure.  Defaults to
            ``src/infrastructure/osnet/plotting/results/gradcam/``.
        save_name: Output filename (including extension).
        heatmap_alpha: Opacity of the Grad-CAM overlay (0–1).

    Returns:
        :class:`~pathlib.Path` to the saved figure.
    """
    output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load images ────────────────────────────────────────────────────
    canonical_rgb = _load_rgb(canonical_path)
    extreme_rgb = _load_rgb(extreme_path)

    # ── Compute embeddings for cosine distance ─────────────────────────
    emb_canonical = encoder.encode_single_image(canonical_rgb)  # L2-normalised float32
    emb_extreme = encoder.encode_single_image(extreme_rgb)

    # Embeddings are already L2-normalised → dot product == cosine similarity
    cosine_similarity = float(np.dot(emb_canonical, emb_extreme))
    cosine_distance = 1.0 - cosine_similarity

    # ── Compute Grad-CAM heatmaps (symmetric cross-comparison) ─────────
    gradcam = OSNetGradCAM(encoder)

    heatmap_canonical = gradcam.compute(
        image_query=canonical_rgb,
        image_gallery=extreme_rgb,
    )
    heatmap_extreme = gradcam.compute(
        image_query=extreme_rgb,
        image_gallery=canonical_rgb,
    )

    # ── Build overlays ─────────────────────────────────────────────────
    overlay_canonical = _overlay_heatmap(canonical_rgb, heatmap_canonical, alpha=heatmap_alpha)
    overlay_extreme = _overlay_heatmap(extreme_rgb, heatmap_extreme, alpha=heatmap_alpha)

    # ── Plot 2×2 figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f"Grad-CAM: Canonical vs. Extreme Pose\n"
        f"Cosine Distance = {cosine_distance:.4f}  "
        f"(Similarity = {cosine_similarity:.4f})",
        fontsize=13,
        fontweight="bold",
    )

    _plot_cell(axes[0, 0], canonical_rgb, "Canonical — Original")
    _plot_cell(axes[0, 1], overlay_canonical, "Canonical — Grad-CAM")
    _plot_cell(axes[1, 0], extreme_rgb, "Extreme — Original")
    _plot_cell(axes[1, 1], overlay_extreme, "Extreme — Grad-CAM")

    plt.tight_layout()

    output_path = output_dir / save_name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def _plot_cell(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
