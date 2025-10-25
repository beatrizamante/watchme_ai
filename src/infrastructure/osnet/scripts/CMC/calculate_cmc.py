from datetime import datetime

import numpy as np
import torch

from src.infrastructure.osnet.scripts.shared.compute_distance_matrix import \
    compute_distance_matrix
import torchreid


def calculate_cmc(query_features, query_pids, query_camids,
                      gallery_features, gallery_pids, gallery_camids,
                      max_rank=50):
    """
    Calculate CMC metrics.

    Args:
        query_features: Query feature matrix
        query_pids: Query person IDs
        query_camids: Query camera IDs
        gallery_features: Gallery feature matrix
        gallery_pids: Gallery person IDs
        gallery_camids: Gallery camera IDs
        max_rank: Maximum rank to compute CMC for

    Returns:
        dict: CMC metrics and results
    """
    print("Calculating CMC metrics...")

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

    cmc, _ = torchreid.metrics.evaluate_rank(
        distmat,
        query_pids,
        gallery_pids,
        query_camids,
        gallery_camids,
        max_rank=max_rank,
        use_metric_cuhk03=False
    )

    results = {
        'CMC': {},
        'num_query': len(query_pids),
        'num_gallery': len(gallery_pids),
        'timestamp': datetime.now().isoformat()
    }

    for rank in [1, 5, 10, 20]:
        if rank <= len(cmc):
            results['CMC'][f'Rank-{rank}'] = float(cmc[rank-1])

    results['CMC']['all_ranks'] = [float(x) for x in cmc]

    return results
