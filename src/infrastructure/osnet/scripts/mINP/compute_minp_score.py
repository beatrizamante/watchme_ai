import numpy as np


def compute_minp_score(distmat, query_pids, gallery_pids, query_camids, gallery_camids):
    """
    Compute mean Inverse Negative Penalty (mINP) score.

    The mINP metric penalizes hard negatives more than standard mAP.
    It computes the negative penalty for each query and averages them.

    Args:
        distmat: Distance matrix (num_query x num_gallery)
        query_pids: Query person IDs
        gallery_pids: Gallery person IDs
        query_camids: Query camera IDs
        gallery_camids: Gallery camera IDs

    Returns:
        float: mINP score
    """
    num_q, _ = distmat.shape

    indices = np.argsort(distmat, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, np.newaxis])

    inp_scores = []

    for q_idx in range(num_q):
        order = indices[q_idx]

        remove = (gallery_camids[order] == query_camids[q_idx])
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]

        if not np.any(orig_cmc):
            inp_scores.append(0.0)
            continue

        pos_indices = np.where(orig_cmc)[0]

        if len(pos_indices) == 0:
            inp_scores.append(0.0)
            continue

        neg_penalty = 0.0
        num_valid_pos = 0

        for pos_idx in pos_indices:
            neg_before = pos_idx  # All samples before current position are negatives

            if neg_before > 0:
                penalty = np.sum(1.0 / np.arange(1, neg_before + 1))
                neg_penalty += penalty
                num_valid_pos += 1

        if num_valid_pos > 0:
            inp = neg_penalty / num_valid_pos
            inp_scores.append(inp)
        else:
            inp_scores.append(0.0)

    minp = np.mean(inp_scores)

    return minp
