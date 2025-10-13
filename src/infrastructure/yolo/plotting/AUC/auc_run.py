import glob
import os
from PIL import Image

from src.infrastructure.yolo.plotting import compute_auc
from src.infrastructure.yolo.plotting.AUC.build_score_labels import build_scores_labels
from src.infrastructure.yolo.plotting.AUC.loaders import load_ground_truth, load_predictions
from src.infrastructure.yolo.plotting.AUC.plot_auc import plot_pr_per_class, plot_roc_per_class
#NOTE - THIS ONLY WORKS IF YOU USE PREDICT WITH save_txt=True AND source=YOUR_VAL_DATASET

def run():
    """
    Executes the AUC evaluation pipeline for YOLO predictions.
    This function performs the following steps:
    1. Loads image sizes from the validation image directory.
    2. Loads predicted bounding boxes from the specified predictions directory.
    3. Loads ground-truth bounding boxes from the specified ground-truth directory.
    4. Builds per-class score and label lists for evaluation using a specified IoU threshold.
    5. Computes ROC AUC and PR AUC metrics for each class.
    6. Prints the AUC results for each class.
    Directories and parameters are hardcoded within the function.
    """
    pred_dir = "src/dataset/yolo/runs/detect/predict/labels"
    gt_dir = "src/dataset/yolo/dataset/labels"

    img_sizes = {}
    for img_file in glob.glob("/content/yolo/images/val/*.jpg"):
        im_id = os.path.basename(img_file).replace(".jpg","")
        im = Image.open(img_file)
        img_sizes[im_id] = im.size

    print("Carregando predições...")
    all_preds = load_predictions(pred_dir, img_sizes)
    print("Carregando ground-truths...")
    all_gts = load_ground_truth(gt_dir, img_sizes)
    print("Construindo listas y_true / y_scores...")
    per_class_scores, per_class_labels = build_scores_labels(all_preds, all_gts, iou_threshold=0.5)

    print("Calculando AUC...")
    results = compute_auc(per_class_scores, per_class_labels)
    for cls, vals in results.items():
        print(f"Classe {cls}: ROC AUC={vals['roc_auc']:.4f}, PR AUC={vals['pr_auc']:.4f}")

    plot_roc_per_class(per_class_scores, per_class_labels)
    plot_pr_per_class(per_class_scores, per_class_labels)
