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
    query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    gallery_features = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)

    similarity_matrix = np.dot(query_features, gallery_features.T)
    distance_matrix = 1 - similarity_matrix

    return distance_matrix
