"""
Module for plotting ROC and Precision-Recall curves per class.
This module provides functions to visualize the performance of multi-class classification models
by plotting Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves for each class.
It uses matplotlib for visualization and scikit-learn for metric computation.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_roc_per_class(per_class_scores, per_class_labels):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for each class using provided scores and labels.

    Args:
        per_class_scores (dict): A dictionary where keys are class identifiers and values are arrays or lists of predicted scores for each sample in that class.
        per_class_labels (dict): A dictionary where keys are class identifiers and values are arrays or lists of true binary labels (0 or 1) for each sample in that class.

    Notes:
        - Only classes with at least two unique label values (both 0 and 1) are plotted.
        - The function displays the ROC curve for each class, including the AUC value in the legend.
        - A diagonal line representing random chance is also plotted for reference.
        - The plot is shown using matplotlib.
    """
    plt.figure(figsize=(8,6))
    for cls, scores in per_class_scores.items():
        labels = per_class_labels[cls]
        if len(set(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {cls} (AUC={roc_auc_val:.3f})')
    plt.plot([0,1], [0,1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC per class')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def plot_pr_per_class(per_class_scores, per_class_labels):
    """
    Plots the Precision-Recall (PR) curve for each class using provided scores and labels.

    Args:
        per_class_scores (dict): A dictionary where keys are class identifiers and values are arrays or lists of prediction scores for each sample in the class.
        per_class_labels (dict): A dictionary where keys are class identifiers and values are arrays or lists of ground truth binary labels (0 or 1) for each sample in the class.

    Notes:
        - Classes with less than two unique label values are skipped (cannot compute PR curve).
        - The PR curve for each class is plotted with its corresponding PR AUC value in the legend.
        - Displays the plot with labeled axes, title, legend, and grid.
    """
    plt.figure(figsize=(8,6))
    for cls, scores in per_class_scores.items():
        labels = per_class_labels[cls]
        if len(set(labels)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc_val = auc(recall, precision)
        plt.plot(recall, precision, label=f'Class {cls} (PR AUC={pr_auc_val:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve per Class')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()
