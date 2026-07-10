"""Grad-CAM integration with OSNetEncoder for person re-identification analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from src.infrastructure.osnet.plotting.gradcam.embedding_collapse import embedding_collapse
from src.infrastructure.osnet.scripts.transformers.transformers import preprocess_image

if TYPE_CHECKING:
    from src.infrastructure.osnet.core.encode import OSNetEncoder


class OSNetGradCAM:
    """Compute Grad-CAM heatmaps reusing a loaded OSNetEncoder instance.

    Grad-CAM highlights the spatial regions of a query image that contribute
    most to its cosine similarity with a gallery image — useful for visualising
    which body parts the model relies on under pose deformations.

    The encoder's model is never re-loaded; only forward/backward hooks are
    temporarily attached to the target layer and removed immediately after
    the heatmap is produced, so normal inference is unaffected.

    Args:
        encoder: A fully initialised :class:`OSNetEncoder` instance.
        target_layer_name: Dot-separated attribute path to the convolutional
            layer to hook.  Defaults to ``"conv5"``, the last 1×1 conv before
            global average pooling in ``osnet_ibn_x1_0`` — highest-level
            spatial features before embedding aggregation.
    """

    def __init__(
        self,
        encoder: "OSNetEncoder",
        target_layer_name: str = "conv5",
    ) -> None:
        self.encoder = encoder
        self.model = encoder.model
        self.device = encoder.device
        self.transform = encoder.transform
        self._target_layer = self._resolve_layer(target_layer_name)

        self._feature_maps: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

    def _resolve_layer(self, name: str) -> torch.nn.Module:
        """Traverse dot-separated attribute names to reach the target layer."""
        layer: torch.nn.Module = self.model
        for part in name.split("."):
            layer = getattr(layer, part)
        return layer

    def _register_hooks(self) -> tuple[torch.utils.hooks.RemovableHandle, torch.utils.hooks.RemovableHandle]:
        """Attach forward and backward hooks; return handles for later removal."""

        def _save_feature_maps(_module, _input, output: torch.Tensor) -> None:
            self._feature_maps = output

        def _save_gradients(_module, _grad_in, grad_out: tuple[torch.Tensor, ...]) -> None:
            self._gradients = grad_out[0]

        fwd_handle = self._target_layer.register_forward_hook(_save_feature_maps)
        bwd_handle = self._target_layer.register_full_backward_hook(_save_gradients)
        return fwd_handle, bwd_handle

    def _build_heatmap(self) -> np.ndarray:
        """Pool gradients over spatial dimensions and produce a normalised CAM."""
        if self._gradients is None or self._feature_maps is None:
            raise RuntimeError("Hooks did not capture data — did the forward pass run?")

        # (1, C, H, W) → channel-wise importance weights via global avg pool
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)

        # weighted sum over channels → (1, 1, H, W), then ReLU
        cam = F.relu((weights * self._feature_maps).sum(dim=1, keepdim=True))

        cam_np: np.ndarray = cam.squeeze().cpu().detach().numpy()

        # min-max normalise to [0, 1]
        cam_np -= cam_np.min()
        if cam_np.max() > 0:
            cam_np /= cam_np.max()

        return cam_np

    def compute(self, image_query, image_gallery) -> np.ndarray:
        """Generate a Grad-CAM heatmap for *image_query* relative to *image_gallery*.

        The model stays in ``eval()`` throughout.  ``torch.enable_grad()`` is
        used only for the query forward+backward pass; the gallery embedding is
        computed beforehand inside ``torch.no_grad()`` so it does not interfere
        with the hooks or the gradient graph.

        Args:
            image_query: Query image (numpy array or PIL Image accepted by the
                encoder's transform pipeline).
            image_gallery: Gallery / reference image.

        Returns:
            2-D numpy array of shape ``(H, W)`` with values in ``[0, 1]``.
        """
        self.model.eval()

        # ── Step 1: gallery embedding — no hooks, no gradients ──────────
        t_gallery = preprocess_image(image_gallery, self.transform).to(self.device)
        with torch.no_grad():
            emb_gallery = self.model(t_gallery)
            if isinstance(emb_gallery, (tuple, list)):
                emb_gallery = emb_gallery[0]
            emb_gallery = F.normalize(emb_gallery.view(1, -1), p=2, dim=1).detach()

        # ── Step 2: query embedding — hooks active, gradients enabled ───
        fwd_handle, bwd_handle = self._register_hooks()
        try:
            with torch.enable_grad():
                t_query = preprocess_image(image_query, self.transform).to(self.device)
                emb_query = self.model(t_query)
                if isinstance(emb_query, (tuple, list)):
                    emb_query = emb_query[0]
                emb_query = F.normalize(emb_query.view(1, -1), p=2, dim=1)

                score = embedding_collapse(emb_query, emb_gallery)
                self.model.zero_grad()
                score.backward()
        finally:
            # always remove hooks — even if an exception is raised
            fwd_handle.remove()
            bwd_handle.remove()

        return self._build_heatmap()
