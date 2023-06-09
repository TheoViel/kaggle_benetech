import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from scipy.optimize import linear_sum_assignment
from rapidfuzz.distance.Levenshtein import distance as levenshtein


def accuracy(labels, predictions, beta=1):
    """
    Accuracy metric.

    Args:
        labels (np array [n]): Labels.
        predictions (np array [n] or [n x num_classes]): Predictions.

    Returns:
        float: Accuracy value.
    """
    labels = np.array(labels).squeeze()
    predictions = np.array(predictions).squeeze()

    if len(predictions.shape) > 1:
        predictions = predictions.argmax(-1)

    acc = (predictions == labels).mean()
    return acc


def iou_score(bbox1, bbox2):
    """
    IoU metric between boxes in the pascal_voc format.

    Args:
        bbox1 (np array or list [4]): Box 1.
        bbox2 (np array or list [4]): Box 2.

    Returns:
        float: IoU.
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def precision_calc(gt_boxes, pred_boxes, threshold=0.1, return_assignment=False):
    """
    Counts TPs, FPs and FNs for a given IoU threshold between boxes in the pascal_voc format.
    If return_assignment is True, it returns the assigments between predictions and GTs.

    Args:
        gt_boxes (np array or list [n x 4]): Ground truth boxes.
        pred_boxes (np array or list [m x 4]): Prediction boxes.
        threshold (float, optional): _description_. Defaults to 0.25.
        return_assignment (bool, optional): Whether to returns GT/Pred assigment. Defaults to False.

    Returns:
        ints [3]: TPs, FPs, FNs
    """
    cost_matrix = np.ones((len(gt_boxes), len(pred_boxes)))
    for i, box1 in enumerate(gt_boxes):
        for j, box2 in enumerate(pred_boxes):
            iou = iou_score(box1, box2)

            if iou < threshold:
                continue

            else:
                cost_matrix[i, j] = 0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    if return_assignment:
        return cost_matrix, row_ind, col_ind

    fn = len(gt_boxes) - row_ind.shape[0]
    fp = len(pred_boxes) - col_ind.shape[0]
    tp = 0
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] == 0:
            tp += 1
        else:
            fp += 1
            fn += 1

    return tp, fp, fn


def compute_metrics(preds, truths):
    """
    Computes metrics for boxes.
    Output contains TP, FP, FN, precision, recall & f1 values.

    Args:
        preds (List of Boxes): Predictions.
        truths (List of Boxes): Truths.

    Returns:
        dict: Metrics
    """
    ftp, ffp, ffn = [], [], []
    
    if isinstance(preds, list):
        for pred, truth in zip(preds, truths):
            tp, fp, fn = precision_calc(truth['pascal_voc'].copy(), pred['pascal_voc'].copy())
            ftp.append(tp)
            ffp.append(fp)
            ffn.append(fn)
            
            assert ftp + ffn == len(truth)
            
        tp = np.sum(ftp)
        fp = np.sum(ffp)
        fn = np.sum(ffn)
    else:
        tp, fp, fn = precision_calc(truths.copy(), preds.copy())
        assert tp + fn == len(truths), (tp, fp, fn, len(truths), len(preds))
        assert len(truths)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) # if tp + fn else 1

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def sigmoid(x):
    return 2 - 2 / (1 + np.exp(-x))


def normalized_rmse(y_true, y_pred):
    # The argument to the sigmoid transform is equal to 
    # rmse(y_true, y_pred) / rmse(y_true, np.mean(y_true))
    return sigmoid((1 - r2_score(y_true, y_pred)) ** 0.5)


def normalized_levenshtein_score(y_true, y_pred):
    total_distance = np.sum([levenshtein(yt, yp) for yt, yp in zip(y_true, y_pred)])
    length_sum = np.sum([len(yt) for yt in y_true])
    return sigmoid(total_distance / length_sum)


def score_series(y_true, y_pred):
    if len(y_true) != len(y_pred):
        return 0.0
    if isinstance(y_true[0], str):
        return normalized_levenshtein_score(y_true, y_pred)
    else:
        return normalized_rmse(y_true, y_pred)


def benetech_score(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """
    Evaluate predictions using the metric from the Benetech - Making Graphs Accessible.
    
    Parameters
    ----------
    ground_truth: pd.DataFrame
        Has columns `[data_series, chart_type]` and an index `id`. Values in `data_series` 
        should be either arrays of floats or arrays of strings.
    
    predictions: pd.DataFrame
    """
    if not ground_truth.index.equals(predictions.index):
        raise ValueError("Must have exactly one prediction for each ground-truth instance.")
    if not ground_truth.columns.equals(predictions.columns):
        raise ValueError(f"Predictions must have columns: {ground_truth.columns}.")
    pairs = zip(ground_truth.itertuples(index=False), predictions.itertuples(index=False))
    scores = []
    for (gt_series, gt_type), (pred_series, pred_type) in pairs:
        if gt_type != pred_type:  # Check chart_type condition
            scores.append(0.0)
        else:  # Score with RMSE or Levenshtein as appropriate
            scores.append(score_series(gt_series, pred_series))
    return np.mean(scores)
