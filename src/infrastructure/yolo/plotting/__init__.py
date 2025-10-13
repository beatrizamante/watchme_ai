"""
This module provides functionality to compute ROC AUC and Precision-Recall AUC metrics for each class in a multi-class classification setting.
"""
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

def compute_auc(per_class_scores, per_class_labels):
    """
    Computes ROC AUC and Precision-Recall AUC for each class.

    Args:
        per_class_scores (dict): Dictionary mapping class labels to lists or arrays of predicted scores.
        per_class_labels (dict): Dictionary mapping class labels to lists or arrays of true binary labels.

    Returns:
        dict: Dictionary mapping each class label to a dictionary with keys 'roc_auc' and 'pr_auc', containing the respective AUC values.
              If a class has less than two unique label values, both AUCs are set to NaN.
    """
    results = {}
    for cls, scores in per_class_scores.items():
        labels = per_class_labels[cls]
        if len(set(labels)) < 2:
            results[cls] = {'roc_auc': float('nan'), 'pr_auc': float('nan')}
            continue
        roc_auc_val = roc_auc_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc_val = auc(recall, precision)
        results[cls] = {'roc_auc': roc_auc_val, 'pr_auc': pr_auc_val}
    return results
