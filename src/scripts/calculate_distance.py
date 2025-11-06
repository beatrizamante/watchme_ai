import numpy as np

def compute_euclidean_distance(embedding1, embedding2):
    """
    Compute Euclidean distance between two person embeddings.

    Args:
        embedding1: Person embedding (1D array, shape: 512)
        embedding2: Person embedding (1D array, shape: 512)

    Returns:
        float: Euclidean distance
    """
    return float(np.linalg.norm(embedding1 - embedding2))

def compute_batch_distances(target_embedding, candidate_embeddings):
    """
    Compare target person embedding against multiple candidates.

    Args:
        target_embedding: Target person embedding (1D array, shape: 512)
        candidate_embeddings: Multiple embeddings (2D array, shape: N x 512)

    Returns:
        numpy.ndarray: Distances to each candidate (shape: N)
    """
    if target_embedding.ndim != 1:
        target_embedding = target_embedding.flatten()

    if candidate_embeddings.ndim == 1:
        candidate_embeddings = candidate_embeddings.reshape(1, -1)

    distances = np.linalg.norm(candidate_embeddings - target_embedding, axis=1)
    return distances
