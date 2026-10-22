"""Monitoring denominators, train-only drift references and causal time boundaries."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.inference import Observations
from src.monitor import (
    distribution_drift,
    fit_histogram,
    rolling_performance,
    run_monitoring,
    selected_structure_inputs,
    step_metrics,
    validate_monitor_config,
)
from src.monitor_reporting import warning_actions


def test_psi_matches_hand_calculation_and_bins_do_not_adapt():
    train, current = [0, 1, 2, 3], [0, 0, 0, 2]
    histogram = fit_histogram(train, bins=2)
    before = copy.deepcopy(histogram)
    drift = distribution_drift(train, current, histogram, probability_floor=1e-6)
    expected = 0.25 * np.log(1.5) - 0.25 * np.log(0.5)
    assert drift["psi"] == pytest.approx(expected)
    assert drift["ks_statistic"] == 0.5
    assert histogram == before
    repeated = distribution_drift(train, train * 5, histogram, probability_floor=1e-6)
    assert repeated["psi"] == 0 and repeated["ks_statistic"] == 0


@pytest.mark.parametrize("change", [-1.0, 1.0])
def test_constant_reference_detects_changes_both_directions(change):
    train = np.zeros(10)
    histogram = fit_histogram(train, bins=10)
    assert histogram["constant_reference"] and histogram["actual_bins"] == 3
    drift = distribution_drift(train, np.full(10, change), histogram, probability_floor=1e-6)
    assert np.isfinite(drift["psi"]) and drift["psi"] > 0.2
    assert drift["ks_statistic"] == 1 and drift["outside_training_range_rate"] == 1


def test_drift_refuses_heldout_fitting_invalid_values_and_changed_reference():
    with pytest.raises(ValueError, match="training data only"):
        fit_histogram([1, 2], bins=2, split="validation")
    with pytest.raises(ValueError, match="finite"):
        fit_histogram([0, np.nan], bins=2)
    histogram = fit_histogram([0, 1], bins=2)
    with pytest.raises(ValueError, match="does not match"):
        distribution_drift([0, 0], [1, 1], histogram, probability_floor=1e-6)
    drift = distribution_drift([0, 1], [], histogram, probability_floor=1e-6)
    assert drift["psi"] is None and drift["ks_statistic"] is None


def test_monitoring_unknowns_change_workload_but_not_classification_truth():
    result = step_metrics([1, 0, -1], [0.9, 0.1, 0.99], threshold=0.5)
    assert result["f1"] == 1 and result["average_precision"] == 1
    assert result["fp"] == 0 and result["unknown_alerts"] == 1
    assert result["alert_rate_all"] == 2 / 3 and result["alert_rate_labeled"] == 0.5
    assert result["label_coverage"] == 2 / 3
    assert result["observed_illicit_rate_labeled"] == 0.5
    assert result["known_illicit_share_all_nodes"] == 1 / 3
    unlabeled = step_metrics([-1, -1], [0.99, 0.01], threshold=0.5)
    assert unlabeled["average_precision"] is None and unlabeled["f1"] is None
    negative = step_metrics([0, -1], [0.9, 0.8], threshold=0.5)
    assert negative["f1"] is None and negative["recall"] is None


def test_rolling_metrics_pool_counts_and_exclude_future_and_partial_windows():
    policy = {
        "rolling_steps": 3,
        "minimum_known_labels": 6,
        "minimum_illicit_labels": 3,
        "f1_floor_fraction_of_validation": 0.8,
    }
    y, times = np.tile([1, 0], 4), np.repeat([35, 36, 37, 38], 2)
    score = np.array([0.9, 0.8, 0.1, 0.2, 0.8, 0.1, 0.1, 0.9])
    result = rolling_performance(
        y, score, times, step=37, first_step=35, threshold=0.5, validation_f1=1.0, policy=policy
    )
    assert result["f1"] == 2 / 3 and result["performance_warning"]
    assert result["support_labeled"] == 6 and result["support_illicit"] == 3
    score[times == 38] = 1 - score[times == 38]
    second = rolling_performance(
        y, score, times, step=37, first_step=35, threshold=0.5, validation_f1=1.0, policy=policy
    )
    assert second == result
    partial = rolling_performance(
        y, score, times, step=36, first_step=35, threshold=0.5, validation_f1=1.0, policy=policy
    )
    assert not partial["complete_window"] and not partial["performance_warning"]


@pytest.mark.parametrize(
    "field", ["split", "population", "fit", "threshold", "reference", "window"]
)
def test_monitor_rejects_unreviewed_protocol_changes(field):
    config = yaml.safe_load(Path("configs/stage6.yaml").read_text())
    if field == "split":
        config["split"]["test"] = [34, 49]
    elif field == "population":
        config["drift"]["population"] = "known_labels_only"
    elif field == "fit":
        config["experiment"]["fit_models_or_preprocessors"] = True
    elif field == "threshold":
        config["experiment"]["reselect_decision_thresholds"] = True
    elif field == "reference":
        config["drift"]["reference_steps"] = [1, 34]
    else:
        config["performance"]["rolling_steps"] = 0
    with pytest.raises(ValueError):
        validate_monitor_config(config)


def test_monitor_training_structure_never_calls_score_helper_or_uses_future(data):
    config = yaml.safe_load(Path("configs/stage5.yaml").read_text())
    from src.features import STRUCTURE_NAMES

    _, names = data.features("all", include_time=True)
    nodes, original = selected_structure_inputs(
        Observations.from_data(data), (1, 24), config["graph"], [*names, *STRUCTURE_NAMES]
    )
    x = data.x.copy()
    x[data.time_step > 24] = -10000
    altered = replace(data, x=x, y=np.full(len(data.y), -1))
    _, changed = selected_structure_inputs(
        Observations.from_data(altered), (1, 24), config["graph"], [*names, *STRUCTURE_NAMES]
    )
    assert data.time_step[nodes].max() == 24
    np.testing.assert_array_equal(original, changed)


def test_drift_warning_needs_consecutive_steps_and_never_auto_promotes():
    reference = {
        "validation_selected_variant": "candidate",
        "config": {"drift": {"psi_warning": 0.2, "consecutive_steps": 2}},
    }
    score = pd.DataFrame(
        {
            "variant": ["candidate"] * 4,
            "seed": [0] * 4,
            "time_step": [35, 36, 37, 38],
            "split": ["test"] * 4,
            "psi": [0.3, 0.1, 0.3, 0.4],
            "ks_statistic": [0.1] * 4,
        }
    )
    rolling = pd.DataFrame(
        {
            "variant": ["candidate"] * 4,
            "seed": [0] * 4,
            "time_step": [35, 36, 37, 38],
            "performance_warning": [False, False, False, True],
        }
    )
    result = warning_actions(score, rolling, reference)
    assert result.persistent_score_drift.tolist() == [False, False, False, True]
    assert result.retraining_review.tolist() == [False, False, False, True]
    assert not result.automatic_retraining_or_promotion.any()


def test_end_to_end_monitor_reuses_frozen_scores_without_fitting(raw_csvs, tmp_path, monkeypatch):
    from src.final_evaluate import run_final_evaluation
    from src.graph_train import run_graph_experiment
    from src.train import run_experiment

    base = yaml.safe_load(Path("configs/stage2.yaml").read_text())
    base["data"].update(raw_dir=str(raw_csvs), require_full_dataset=False)
    base.update(output_dir=str(tmp_path / "s2"), artifact_dir=str(tmp_path / "a2"))
    base["experiment"].update(feature_groups=["all"], include_time_options=[True])
    base["runtime"]["thread_limit"] = 1
    base["models"]["xgboost"]["candidates"] = [{"max_depth": 2}]
    base["models"]["xgboost"]["common"].update(n_estimators=3, n_jobs=1)
    run_experiment(base, models=["xgboost"])
    graph = yaml.safe_load(Path("configs/stage3.yaml").read_text())
    graph["data"].update(raw_dir=str(raw_csvs), require_full_dataset=False)
    graph.update(output_dir=str(tmp_path / "s3"), artifact_dir=str(tmp_path / "a3"))
    graph["runtime"]["thread_limit"] = 1
    graph["experiment"].update(
        selection_record=str(tmp_path / "s2/selection.json"),
        baseline_run_manifest=str(tmp_path / "s2/latest_run.json"),
        fixed_candidate_index=0,
        variants=["all_control", "all_structure"],
    )
    run_graph_experiment(graph)
    final = yaml.safe_load(Path("configs/stage5.yaml").read_text())
    final["data"].update(raw_dir=str(raw_csvs), require_full_dataset=False)
    final.update(output_dir=str(tmp_path / "s5"), artifact_dir=str(tmp_path / "a5"))
    final["runtime"]["thread_limit"] = 1
    final["references"] = {"stage3": str(tmp_path / "s3/latest_run.json"), "score_helper_log": None}
    final["experiment"]["primary_variants"] = ["s3_xgboost_all_structure"]
    run_final_evaluation(final)
    config = yaml.safe_load(Path("configs/stage6.yaml").read_text())
    config["data"].update(raw_dir=str(raw_csvs), require_full_dataset=False)
    config.update(output_dir=str(tmp_path / "s6"), artifact_dir=str(tmp_path / "a6"))
    config["runtime"]["thread_limit"] = 1
    config["references"]["stage5_manifest"] = str(tmp_path / "s5/latest_run.json")

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Monitoring attempted to fit a model or scaler.")

    monkeypatch.setattr(XGBClassifier, "fit", forbidden_fit)
    monkeypatch.setattr(StandardScaler, "fit", forbidden_fit)
    result = run_monitoring(config)
    assert result["status"] == "completed" and not result["decision_thresholds_changed"]
    output = Path(config["output_dir"])
    protocol = json.loads((output / "frozen_reference.json").read_text())
    assert protocol["reference_nodes"] == 72 and protocol["reference_known_labels"] == 48
    assert len(result["runs"]) == 5
    metrics = pd.read_csv(output / "per_step_metrics.csv")
    assert len(metrics) == 75 and metrics.support_labeled.eq(2).all()
    assert metrics.unknown_nodes.eq(1).all()
    assert (output / "per_step_metrics.json").is_file()
    assert len(list((output / "figures").glob("*.png"))) == 3
