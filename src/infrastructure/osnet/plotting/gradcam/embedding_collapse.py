"""Embedding collapse analysis for person re-identification robustness.

Two distinct responsibilities live here:

1. :func:`embedding_collapse` — differentiable cosine-similarity scalar used
   as the Grad-CAM backprop target (kept in this module so imports remain
   stable).

2. Analysis utilities (:func:`heatmap_entropy`, :func:`analyse_pose_pairs`,
   :func:`plot_collapse_scatter`) — validate the central hypothesis of the
   paper: extreme poses produce diffuse Grad-CAM heatmaps (high entropy) that
   correlate with large cosine distances from the canonical embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
    from PIL import Image


if TYPE_CHECKING:
    from src.infrastructure.osnet.core.encode import OSNetEncoder
    from src.infrastructure.osnet.plotting.gradcam.osnet_gradcam import OSNetGradCAM

# ---------------------------------------------------------------------------
# Grad-CAM backprop target (original helper — do not remove)
# ---------------------------------------------------------------------------


def embedding_collapse(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two embeddings as a differentiable scalar.

    Used as the target score for Grad-CAM backpropagation — the gradient of this
    scalar w.r.t. the last convolutional feature map indicates which spatial regions
    contributed most to the similarity between the two embeddings.

    Args:
        embedding1: Embedding from the query image, shape ``(D,)`` or ``(1, D)``.
        embedding2: Embedding from the gallery image, shape ``(D,)`` or ``(1, D)``.

    Returns:
        Scalar tensor (shape ``(1,)``) with the cosine similarity in ``[-1, 1]``.
        Call ``.backward()`` on it to propagate gradients for Grad-CAM.
    """
    e1 = embedding1.view(1, -1)
    e2 = embedding2.view(1, -1)
    return F.cosine_similarity(e1, e2, dim=1)


# ---------------------------------------------------------------------------
# Analysis utilities (issue #22)
# ---------------------------------------------------------------------------


def heatmap_entropy(heatmap: np.ndarray, eps: float = 1e-9) -> float:
    """Compute the spatial entropy of a normalised Grad-CAM heatmap.

    A focused heatmap (model attends to a small region) has **low** entropy;
    a diffuse heatmap (model does not know where to look) has **high** entropy.

    The heatmap is treated as an unnormalised probability distribution over
    spatial positions: values are summed to 1 before computing Shannon entropy.

    Args:
        heatmap: 2-D array with values in ``[0, 1]``, shape ``(H, W)``.
        eps: Small constant to avoid ``log(0)``.

    Returns:
        Shannon entropy in nats (float).
    """
    flat = heatmap.flatten().astype(np.float64)
    flat = flat + eps
    prob = flat / flat.sum()
    return float(-np.sum(prob * np.log(prob)))


def analyse_pose_pairs(
    pairs: list[dict],
    encoder: "OSNetEncoder",
    gradcam: "OSNetGradCAM",
) -> list[dict]:
    """Compute cosine distance and heatmap entropy for a list of image pairs.

    Each entry in *pairs* must be a dict with the following keys:

    - ``"canonical"`` (*str | Path*): path to the canonical (ground-truth) pose.
    - ``"query"``     (*str | Path*): path to the query (possibly extreme) pose.
    - ``"pose_category"`` (*str*): label used to colour the scatter plot
      (e.g. ``"frontal"``, ``"lateral"``, ``"occlusion"``).

    Optional key:

    - ``"identity"`` (*str*): person ID, used only for logging / debugging.

    Args:
        pairs: List of pair descriptors (see above).
        encoder: Loaded :class:`~src.infrastructure.osnet.core.encode.OSNetEncoder`.
        gradcam: Initialised :class:`~src.infrastructure.osnet.plotting.gradcam.osnet_gradcam.OSNetGradCAM`.

    Returns:
        List of result dicts, each containing:

        - ``"cosine_distance"`` (float)
        - ``"entropy"``         (float)
        - ``"pose_category"``   (str)
        - ``"identity"``        (str | None)
    """

    results: list[dict] = []

    for pair in pairs:
        canonical_img = np.array(Image.open(Path(pair["canonical"])).convert("RGB"))
        query_img = np.array(Image.open(Path(pair["query"])).convert("RGB"))

        emb_canonical = encoder.encode_single_image(canonical_img)
        emb_query = encoder.encode_single_image(query_img)

        # Embeddings are L2-normalised → dot == cosine similarity
        cosine_distance = float(1.0 - np.dot(emb_canonical, emb_query))

        # Grad-CAM: query activation relative to canonical gallery
        heatmap = gradcam.compute(image_query=query_img, image_gallery=canonical_img)
        entropy = heatmap_entropy(heatmap)

        results.append(
            {
                "cosine_distance": cosine_distance,
                "entropy": entropy,
                "pose_category": pair["pose_category"],
                "identity": pair.get("identity"),
            }
        )

    return results


def plot_collapse_scatter(
    results: list[dict],
    output_path: str | Path,
    title: str = "Embedding Collapse: Cosine Distance × Heatmap Entropy",
) -> Path:
    """Scatter plot of cosine distance vs. heatmap entropy with Pearson correlation.

    Each point represents one image pair from :func:`analyse_pose_pairs`.
    Points are coloured by ``pose_category``; the Pearson *r* and *p*-value are
    shown in the plot title to directly support the paper's central claim.

    Args:
        results: Output of :func:`analyse_pose_pairs`.
        output_path: Full path (including filename) where the figure is saved.
        title: Main title of the figure.

    Returns:
        :class:`~pathlib.Path` to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    distances = np.array([r["cosine_distance"] for r in results])
    entropies = np.array([r["entropy"] for r in results])
    categories = [r["pose_category"] for r in results]

    # Pearson correlation
    r_value, p_value = pearsonr(distances, entropies)

    # Assign a colour per unique category
    unique_cats = sorted(set(categories))
    palette = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
    colour_map = dict(zip(unique_cats, palette))

    fig, ax = plt.subplots(figsize=(8, 6))

    for cat in unique_cats:
        mask = [c == cat for c in categories]
        ax.scatter(
            distances[mask],
            entropies[mask],
            label=cat,
            color=colour_map[cat],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            s=60,
        )

    # Trend line
    if len(distances) > 1:
        z = np.polyfit(distances, entropies, 1)
        x_line = np.linspace(distances.min(), distances.max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), color="black", linewidth=1, linestyle="--", alpha=0.6)

    ax.set_xlabel("Cosine Distance", fontsize=11)
    ax.set_ylabel("Heatmap Entropy (nats)", fontsize=11)
    ax.set_title(
        f"{title}\nPearson r = {r_value:.3f}  (p = {p_value:.2e},  n = {len(results)})",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(title="Pose Category", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
