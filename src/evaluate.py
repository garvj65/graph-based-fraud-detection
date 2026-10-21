"""Explicit illicit-class metrics and validation-only threshold selection.

Average precision and trapezoidal PR-AUC are reported under different names.
No accuracy metric is computed. Unknown labels are excluded, never made negative.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def known_binary_scores(
    labels: ArrayLike, scores: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Validate aligned probability scores and remove unknown-labeled rows."""
    y, score = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or score.shape != y.shape:
        raise ValueError("Labels and scores must be aligned one-dimensional arrays.")
    if not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("Labels must be -1 (unknown), 0 (licit), or 1 (illicit).")
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("Scores must be finite probabilities in [0, 1].")
    known = y != -1
    return y[known].astype(np.int64), score[known]


def select_validation_threshold(
    labels: ArrayLike, scores: ArrayLike, *, split: str = "validation"
) -> dict[str, Any]:
    """Maximize illicit F1 on validation; choose the higher threshold on a tie.

    Predictions use score >= threshold, including tied scores as a whole group.
    Requiring both classes prevents a degenerate validation set selecting policy.
    """
    if split != "validation":
        raise ValueError("Decision thresholds may be selected on validation only.")
    y, score = known_binary_scores(labels, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both validation classes.")
    precision, recall, thresholds = precision_recall_curve(y, score, pos_label=1)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return {
        "value": float(thresholds[best]),
        "objective": "max_illicit_f1",
        "selection_split": "validation",
        "tie_break": "highest_threshold",
        "comparison": "score >= threshold",
        "validation_f1_at_selection": float(f1[best]),
        "validation_labeled_support": len(y),
    }


def binary_metrics(labels: ArrayLike, scores: ArrayLike, *, threshold: float) -> dict[str, Any]:
    """Compute illicit metrics at a supplied frozen threshold, with defined nulls.

    No labeled data: all rates are null. No illicit support: AP/PR-AUC/recall/F1
    are null. No predicted alerts with positive support: precision and F1 are 0.
    ROC-AUC is null whenever either class is missing. Scores are not calibrated
    fraud probabilities merely because they lie in [0, 1].
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and lie in [0, 1].")
    y, score = known_binary_scores(labels, scores)
    pred = (score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n, positives = len(y), int(y.sum())
    precision = float(tp / (tp + fp)) if tp + fp else (0.0 if n else None)
    recall = float(tp / positives) if positives else None
    f1 = float(2 * tp / (2 * tp + fp + fn)) if positives else None
    ap, pr_auc = None, None
    if positives:
        ap = float(average_precision_score(y, score))
        curve_precision, curve_recall, _ = precision_recall_curve(y, score)
        pr_auc = float(auc(curve_recall, curve_precision))
    return {
        "average_precision": ap,
        "pr_auc_trapezoid": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": float(roc_auc_score(y, score)) if 0 < positives < n else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "support_labeled": n,
        "support_illicit": positives,
        "support_licit": n - positives,
        "observed_illicit_rate_labeled": positives / n if n else None,
        "alert_count_labeled": int(pred.sum()),
        "alert_rate_labeled": float(pred.mean()) if n else None,
        "mean_score_labeled": float(score.mean()) if n else None,
        "threshold": float(threshold),
    }
