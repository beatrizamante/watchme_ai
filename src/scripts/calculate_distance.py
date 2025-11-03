import numpy as np

def compute_euclidean_distance(features1, features2):
    """
    Compute Euclidean distance between two feature vectors or matrices.

    Args:
        features1: Feature vector/matrix (1D or 2D numpy array)
        features2: Feature vector/matrix (1D or 2D numpy array)

    Returns:
        float or numpy.ndarray: Distance(s)
    """
    if features1.ndim == 1:
        features1 = features1.reshape(1, -1)
    if features2.ndim == 1:
        features2 = features2.reshape(1, -1)

    diff = features1[:, np.newaxis, :] - features2[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)

    if dist.shape == (1, 1):
        return dist[0, 0]

    return dist
