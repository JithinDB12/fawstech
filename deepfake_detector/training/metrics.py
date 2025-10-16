from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def compute_metrics(y_true: Iterable[int], y_prob: Iterable[float]) -> Dict[str, float]:
    y_true_arr = np.asarray(list(y_true))
    y_prob_arr = np.asarray(list(y_prob))

    if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
        # softmax probs for 2 classes -> use positive class prob
        y_pos = y_prob_arr[:, 1]
    else:
        y_pos = y_prob_arr

    y_pred = (y_pos >= 0.5).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_arr, y_pred, average="binary", zero_division=0)

    try:
        auc = float(roc_auc_score(y_true_arr, y_pos))
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": auc,
    }
