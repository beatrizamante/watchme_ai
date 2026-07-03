import torch
import torch.nn.functional as F


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
