"""Hand-checked metric values, threshold ties and sparse-label edge cases."""

import numpy as np
import pytest

from src.evaluate import binary_metrics, select_validation_threshold


def test_metrics_match_hand_calculation() -> None:
    metrics = binary_metrics([0, 1, 1, 0], [0.1, 0.4, 0.35, 0.8], threshold=0.4)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["average_precision"] == pytest.approx(7 / 12)
    assert metrics["pr_auc_trapezoid"] == pytest.approx(5 / 12)
    assert metrics["roc_auc"] == pytest.approx(0.5)
    for name in ("precision", "recall", "f1", "alert_rate_labeled"):
        assert metrics[name] == pytest.approx(0.5)
    assert "accuracy" not in metrics


def test_unknown_cannot_affect_metrics_or_threshold() -> None:
    y, score = [0, 1, 1, 0], [0.1, 0.4, 0.35, 0.8]
    first = binary_metrics(y, score, threshold=0.4)
    second = binary_metrics([*y, -1], [*score, 0.999], threshold=0.4)
    assert first == second
    assert select_validation_threshold(y, score) == select_validation_threshold(
        [*y, -1], [*score, 0.999]
    )


def test_equal_f1_prefers_higher_threshold() -> None:
    selected = select_validation_threshold([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.6])
    assert selected["value"] == 0.9
    assert selected["validation_f1_at_selection"] == pytest.approx(2 / 3)


def test_equal_scores_cannot_be_split_into_arbitrary_alerts() -> None:
    selected = select_validation_threshold([0, 1], [0.5, 0.5])
    assert selected["value"] == 0.5
    metrics = binary_metrics([0, 1], [0.5, 0.5], threshold=selected["value"])
    assert metrics["alert_count_labeled"] == 2


def test_threshold_selection_rejects_test_split() -> None:
    with pytest.raises(ValueError, match="validation only"):
        select_validation_threshold([0, 1], [0.1, 0.9], split="test")


@pytest.mark.parametrize("labels", [[0, 0], [1, 1], [-1, -1]])
def test_threshold_selection_requires_both_known_classes(labels: list[int]) -> None:
    with pytest.raises(ValueError, match="both validation classes"):
        select_validation_threshold(labels, [0.2, 0.8])


def test_no_illicit_support_has_null_undefined_metrics() -> None:
    metrics = binary_metrics([0, 0, -1], [0.1, 0.7, 0.9], threshold=0.5)
    for key in ("average_precision", "pr_auc_trapezoid", "recall", "f1", "roc_auc"):
        assert metrics[key] is None
    assert metrics["precision"] == 0
    assert metrics["support_labeled"] == 2


def test_empty_known_support_has_no_fake_zero_rates() -> None:
    metrics = binary_metrics([-1], [0.9], threshold=0.5)
    for key in ("precision", "recall", "f1", "average_precision", "alert_rate_labeled"):
        assert metrics[key] is None
    assert metrics["confusion_matrix"] == [[0, 0], [0, 0]]


def test_no_alerts_with_positive_support_is_zero_precision_f1() -> None:
    metrics = binary_metrics([0, 1], [0.1, 0.2], threshold=0.9)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0


@pytest.mark.parametrize("scores", [[0.1, np.nan], [0.1, np.inf], [0.1, 1.1], [-0.1, 0.9]])
def test_invalid_probability_scores_rejected(scores: list[float]) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        binary_metrics([0, 1], scores, threshold=0.5)


@pytest.mark.parametrize("threshold", [np.nan, -0.1, 1.1])
def test_invalid_frozen_threshold_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="Threshold"):
        binary_metrics([0, 1], [0.1, 0.9], threshold=threshold)
