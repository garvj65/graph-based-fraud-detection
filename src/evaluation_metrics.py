"""Alert-budget and recall-policy diagnostics with explicit sparse-label denominators."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.evaluate import known_binary_scores


def ranked_indices(scores: ArrayLike, tx_ids: ArrayLike) -> np.ndarray:
    """Descending score, then lexical transaction ID; never use labels to break ties."""
    score, ids = np.asarray(scores, dtype=np.float64), np.asarray(tx_ids, dtype=str)
    if score.ndim != 1 or score.shape != ids.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("Ranking needs aligned scores and unique transaction IDs.")
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("Ranking scores must lie in [0, 1].")
    return np.lexsort((ids, -score))


def budget_metrics(
    labels: ArrayLike,
    scores: ArrayLike,
    tx_ids: ArrayLike,
    *,
    k: int,
    view: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Rank a labeled benchmark or an operational queue without inventing unknown truth.

    `precision_at_k_labeled` is defined only for the retrospective labeled-only
    cohort. An all-transaction queue reports label coverage and precision among
    its known-label subset, never overall precision or unknown false positives.
    Both policies select up to k alerts, regardless of the frozen F1 cutoff.
    """
    y, score, ids = np.asarray(labels), np.asarray(scores, dtype=np.float64), np.asarray(tx_ids)
    known_binary_scores(y, score)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("Alert budget k must be a positive integer.")
    if view not in {"labeled_only", "all_transactions"}:
        raise ValueError("Choose labeled_only or all_transactions.")
    order = ranked_indices(score, ids)
    if view == "labeled_only":
        order = order[y[order] != -1]
    chosen = order[:k]
    illicit = int((y[chosen] == 1).sum())
    licit = int((y[chosen] == 0).sum())
    unknown = int((y[chosen] == -1).sum())
    known = illicit + licit
    positives = int((y == 1).sum())
    result = {
        "view": view,
        "requested_k": k,
        "eligible_nodes": len(order),
        "alerts_selected": len(chosen),
        "budget_filled": len(chosen) == k,
        "known_illicit_alerts": illicit,
        "known_licit_alerts": licit,
        "unknown_alerts": unknown,
        "known_alerts": known,
        "known_label_coverage": known / len(chosen) if len(chosen) else None,
        "precision_at_k_labeled": illicit / len(chosen)
        if view == "labeled_only" and len(chosen)
        else None,
        "precision_among_known_alerts": illicit / known if known else None,
        "recall_known_illicit": illicit / positives if positives else None,
        "support_known_illicit": positives,
        "lowest_selected_score": float(score[chosen[-1]]) if len(chosen) else None,
    }
    return result, chosen


def select_recall_threshold(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    target_recall: float,
    split: str = "validation",
) -> dict[str, Any]:
    """Highest validation threshold meeting target recall, including tied scores together."""
    if split != "validation":
        raise ValueError("Recall-policy thresholds may use validation only.")
    if not np.isfinite(target_recall) or not 0 < target_recall <= 1:
        raise ValueError("Target recall must lie in (0, 1].")
    y, score = known_binary_scores(labels, scores)
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Recall policies require both known validation classes.")
    order = np.argsort(-score, kind="stable")
    sorted_score, sorted_y = score[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(score) - 1]
    recall = np.cumsum(sorted_y)[ends] / y.sum()
    index = np.flatnonzero(recall >= target_recall)[0]
    return {
        "target_recall": float(target_recall),
        "threshold": float(sorted_score[ends[index]]),
        "selection_split": "validation",
        "comparison": "score >= threshold",
        "achieved_validation_recall": float(recall[index]),
        "validation_alerts_labeled": int(ends[index] + 1),
    }


def alert_counts(labels: ArrayLike, scores: ArrayLike, threshold: float) -> dict[str, Any]:
    """Count all-node workload while leaving unknown alert correctness unassigned."""
    y, score = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    known_binary_scores(y, score)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0, 1].")
    alerts = score >= threshold
    return {
        "alert_count_all": int(alerts.sum()),
        "alert_rate_all": float(alerts.mean()) if len(alerts) else None,
        "known_illicit_alerts": int((alerts & (y == 1)).sum()),
        "known_licit_alerts": int((alerts & (y == 0)).sum()),
        "unknown_alerts": int((alerts & (y == -1)).sum()),
        "total_nodes": len(y),
    }
