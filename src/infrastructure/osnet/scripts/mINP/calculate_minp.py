from datetime import datetime

import numpy as np
import torch

from src.infrastructure.osnet.scripts.mINP.compute_minp_score import \
    compute_minp_score
from src.infrastructure.osnet.scripts.shared.compute_distance_matrix import \
    compute_distance_matrix

def calculate_minp(query_features, query_pids, query_camids,
                   gallery_features, gallery_pids, gallery_camids):
    """
    Calculate mean Inverse Negative Penalty (mINP) metrics.

    Args:
        query_features: Query feature matrix
        query_pids: Query person IDs
        query_camids: Query camera IDs
        gallery_features: Gallery feature matrix
        gallery_pids: Gallery person IDs
        gallery_camids: Gallery camera IDs

    Returns:
        dict: mINP metrics and results
    """
    print("Calculating mINP metrics...")

    if isinstance(query_features, torch.Tensor):
        query_features = query_features.numpy()
    if isinstance(gallery_features, torch.Tensor):
        gallery_features = gallery_features.numpy()

    query_pids = np.array(query_pids)
    query_camids = np.array(query_camids)
    gallery_pids = np.array(gallery_pids)
    gallery_camids = np.array(gallery_camids)

    print("Computing distance matrix...")
    distmat = compute_distance_matrix(query_features, gallery_features)
    minp_score = compute_minp_score(
        distmat, query_pids, gallery_pids, query_camids, gallery_camids
    )

    results = {
        'mINP': float(minp_score),
        'num_query': len(query_pids),
        'num_gallery': len(gallery_pids),
        'timestamp': datetime.now().isoformat()
    }

    return results
