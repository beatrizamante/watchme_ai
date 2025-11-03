import numpy as np

def compute_distance_matrix(query_features, gallery_features):
    """
    Compute distance matrix between query and gallery features.

    Args:
        query_features: Query features (N x D)
        gallery_features: Gallery features (M x D)

    Returns:
        numpy.ndarray: Distance matrix (N x M)
    """
    if query_features.ndim == 1:
        query_features = query_features.reshape(1, -1)
    if gallery_features.ndim == 1:
        gallery_features = gallery_features.reshape(1, -1)

    if query_features.shape[0] == 0 or gallery_features.shape[0] == 0:
        raise ValueError("Empty feature arrays detected!")

    query_norm = np.linalg.norm(query_features, axis=1, keepdims=True)
    gallery_norm = np.linalg.norm(gallery_features, axis=1, keepdims=True)

    query_norm = np.maximum(query_norm, 1e-12)
    gallery_norm = np.maximum(gallery_norm, 1e-12)

    query_features = query_features / query_norm
    gallery_features = gallery_features / gallery_norm

    similarity_matrix = np.dot(query_features, gallery_features.T)

    distance_matrix = 1 - similarity_matrix

    return distance_matrix
