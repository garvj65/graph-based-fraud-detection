"""Hand-checked budget metrics and real synthetic fits for fit-free test evaluation."""

import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import load_csv, sha256_file
from src.evaluation_metrics import budget_metrics, ranked_indices, select_recall_threshold
from src.final_evaluate import run_final_evaluation, validate_final_config
from src.inference import Observations, SnapshotFeatureCache, predict_frozen
from src.train import run_experiment


def test_budget_unknowns_are_not_false_positives_and_views_differ():
    y, score, ids = [-1, 0, 1, 1], [0.99, 0.9, 0.8, 0.7], ["a", "b", "c", "d"]
    operational, chosen = budget_metrics(y, score, ids, k=2, view="all_transactions")
    assert chosen.tolist() == [0, 1]
    assert operational["unknown_alerts"] == 1
    assert operational["known_licit_alerts"] == 1
    assert operational["known_label_coverage"] == 0.5
    assert operational["precision_at_k_labeled"] is None
    assert "false_positives" not in operational
    labeled, chosen = budget_metrics(y, score, ids, k=2, view="labeled_only")
    assert chosen.tolist() == [1, 2]
    assert labeled["precision_at_k_labeled"] == 0.5
    assert labeled["recall_known_illicit"] == 0.5


def test_budget_ties_are_label_blind_and_order_invariant():
    first = ranked_indices([0.8, 0.8, 0.9], ["b", "a", "c"])
    assert first.tolist() == [2, 1, 0]
    for labels in ([1, 0, -1], [-1, 1, 0]):
        _, chosen = budget_metrics(
            labels, [0.8, 0.8, 0.9], ["b", "a", "c"], k=2, view="all_transactions"
        )
        assert chosen.tolist() == [2, 1]


def test_budget_caps_small_cohorts_and_undefined_truth_stays_null():
    metrics, _ = budget_metrics([-1, -1], [0.4, 0.3], ["a", "b"], k=100, view="all_transactions")
    assert metrics["alerts_selected"] == 2 and metrics["budget_filled"] is False
    assert metrics["precision_among_known_alerts"] is None
    assert metrics["recall_known_illicit"] is None
    labeled, _ = budget_metrics([-1], [0.8], ["a"], k=100, view="labeled_only")
    assert labeled["alerts_selected"] == 0 and labeled["precision_at_k_labeled"] is None


def test_recall_policy_is_validation_only_and_preserves_ties():
    policy = select_recall_threshold([1, 0, 1, -1], [0.9, 0.8, 0.8, 0.99], target_recall=0.8)
    assert policy["threshold"] == 0.8
    assert policy["validation_alerts_labeled"] == 3
    assert policy["achieved_validation_recall"] == 1
    with pytest.raises(ValueError, match="validation only"):
        select_recall_threshold([0, 1], [0.1, 0.9], target_recall=0.5, split="test")


@pytest.mark.parametrize("mutation", ["split", "fit", "budget", "threshold"])
def test_final_config_rejects_unreviewed_policies(mutation):
    config = yaml.safe_load(Path("configs/stage5.yaml").read_text())
    if mutation == "split":
        config["split"]["test"] = [34, 49]
    elif mutation == "fit":
        config["experiment"]["fit_models_or_preprocessors"] = True
    elif mutation == "budget":
        config["alert_budget"]["per_time_step"] = None
    else:
        config["experiment"]["threshold_policy"] = "optimize_test_f1"
    with pytest.raises(ValueError):
        validate_final_config(config)


@pytest.fixture
def frozen_baseline(raw_csvs, tmp_path):
    baseline = yaml.safe_load(Path("configs/stage2.yaml").read_text())
    baseline["data"].update(raw_dir=str(raw_csvs), require_full_dataset=False)
    baseline["output_dir"] = str(tmp_path / "stage2")
    baseline["artifact_dir"] = str(tmp_path / "baseline_artifacts")
    baseline["experiment"].update(feature_groups=["all"], include_time_options=[True])
    baseline["models"]["xgboost"]["candidates"] = [{"max_depth": 2}]
    baseline["models"]["xgboost"]["common"].update(n_estimators=5, n_jobs=1)
    batch = run_experiment(baseline, models=["xgboost"])
    record_path = Path(baseline["output_dir"]) / "runs" / f"{batch['selected_runs'][0]}.json"
    return baseline, json.loads(record_path.read_text())


def test_label_free_predictions_and_snapshot_future_invariance(raw_csvs, frozen_baseline):
    data = load_csv(raw_csvs)
    _, record = frozen_baseline
    config = yaml.safe_load(Path("configs/stage5.yaml").read_text())
    observed = Observations.from_data(data)
    assert not hasattr(observed, "y")
    cache = SnapshotFeatureCache(observed, config["graph"], None, thread_limit=1)
    nodes, original, _ = predict_frozen(observed, (35, 49), record, cache, thread_limit=1)
    x = data.x.copy()
    x[data.time_step == 49] *= -1000
    changed = Observations.from_data(replace(data, x=x, y=np.full(len(data.y), -1)))
    cache_changed = SnapshotFeatureCache(changed, config["graph"], None, thread_limit=1)
    _, prediction, _ = predict_frozen(changed, (35, 49), record, cache_changed, thread_limit=1)
    np.testing.assert_array_equal(
        original[data.time_step[nodes] == 35], prediction[data.time_step[nodes] == 35]
    )


def test_fit_free_end_to_end_preserves_cutoffs_and_freezes_before_test(
    raw_csvs,
    frozen_baseline,
    tmp_path,
    monkeypatch,
):
    baseline, _ = frozen_baseline
    config = yaml.safe_load(Path("configs/stage5.yaml").read_text())
    config["data"].update(raw_dir=str(raw_csvs), require_full_dataset=False)
    config["references"] = {
        "stage2": str(Path(baseline["output_dir"]) / "latest_run.json"),
        "score_helper_log": None,
    }
    config["output_dir"] = str(tmp_path / "stage5")
    config["artifact_dir"] = str(tmp_path / "evaluation_artifacts")
    config["experiment"]["primary_variants"] = ["s2_xgboost_all_time"]
    config["runtime"]["thread_limit"] = 1
    source_batch = json.loads(Path(config["references"]["stage2"]).read_text())
    record_paths = [
        Path(baseline["output_dir"]) / "runs" / f"{run_id}.json"
        for run_id in source_batch["selected_runs"]
    ]
    hashes = {str(path): sha256_file(path) for path in record_paths}

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Stage 5 attempted to fit a model/preprocessor.")

    monkeypatch.setattr(XGBClassifier, "fit", forbidden_fit)
    monkeypatch.setattr(StandardScaler, "fit", forbidden_fit)
    result = run_final_evaluation(config)
    assert result["status"] == "completed" and result["test_evaluated"] is True
    output = Path(config["output_dir"])
    plan = json.loads(Path(result["protocol"]["path"]).read_text())
    assert plan["status"] == "frozen_before_test" and plan["test_scored_before_freeze"] is False
    assert len(result["evaluation_runs"]) == 5
    assert plan["validation_selected_variant"] == "s2_xgboost_all_time"
    for path in record_paths:
        assert sha256_file(path) == hashes[str(path)]
    for run_id in result["evaluation_runs"]:
        run = json.loads((output / "runs" / f"{run_id}.json").read_text())
        source = next(
            entry for entry in plan["references"] if entry["source_run_id"] == run["source_run_id"]
        )
        assert run["frozen_threshold"] == source["frozen_threshold"]
        assert run["models_or_preprocessors_fitted"] is False
        assert run["threshold_reselected_on_test"] is False
        assert run["test_metrics"]["support_labeled"] == 30
        with np.load(run["test_predictions"]["path"]) as prediction:
            assert prediction["time_step"].min() == 35 and prediction["time_step"].max() == 49
            assert (prediction["label"] == -1).sum() == 15
    assert pd.read_csv(output / "test_comparison.csv").iloc[0].n_seeds == 5
    assert (output / "figures/test_comparison.png").is_file()


def test_snapshot_cache_uses_frozen_helper_without_any_label_access(data):
    from src.data import make_temporal_masks

    config = yaml.safe_load(Path("configs/stage5.yaml").read_text())

    # A deliberately constant synthetic score helper, never used for benchmark results.
    class FixedHelper:
        def predict_proba(self, x):
            return np.tile([0.3, 0.7], (len(x), 1))

    observed = Observations.from_data(data)
    cache = SnapshotFeatureCache(observed, config["graph"], FixedHelper(), thread_limit=1)
    nodes = np.flatnonzero(make_temporal_masks(data).test_nodes)
    structure, neighbor = cache.aligned(nodes)
    altered = replace(data, y=np.full(len(data.y), -1))
    other = SnapshotFeatureCache(
        Observations.from_data(altered), config["graph"], FixedHelper(), thread_limit=1
    )
    second = other.aligned(nodes)
    np.testing.assert_array_equal(structure, second[0])
    np.testing.assert_array_equal(neighbor, second[1])
    assert set(cache.audit) == set(range(35, 50))


def test_changed_artifact_is_rejected_before_loading(tmp_path):
    from src.inference import verify_artifact

    path = tmp_path / "weights.joblib"
    joblib.dump({"synthetic": True}, path)
    artifact = {"path": str(path), "sha256": sha256_file(path)}
    verify_artifact(artifact)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        verify_artifact(artifact)


@pytest.mark.parametrize("architecture", ["gcn", "graphsage"])
def test_frozen_gnn_inference_preserves_snapshot_isolation(data, tmp_path, architecture):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.gnn import prepare_gnn_data
    from src.gnn_train import fit_gnn_run

    training = yaml.safe_load(Path("configs/stage4.yaml").read_text())
    training["output_dir"] = str(tmp_path / "training_results")
    training["artifact_dir"] = str(tmp_path / "training_artifacts")
    training["runtime"]["thread_limit"] = 1
    training["training"].update(max_epochs=2, patience=1)
    view = prepare_gnn_data(data, "all", training["graph"])
    record = fit_gnn_run(
        view,
        model_name=architecture,
        group="all",
        seed=0,
        candidate_index=0,
        parameters={"hidden_channels": 8, "dropout": 0.0, "learning_rate": 0.01},
        phase="synthetic_fixture",
        config=training,
        batch_id="synthetic_gnn",
        provenance={"synthetic_fixture": True},
    )
    config = yaml.safe_load(Path("configs/stage5.yaml").read_text())
    observed = Observations.from_data(data)
    cache = SnapshotFeatureCache(observed, config["graph"], None, thread_limit=1)
    _, validation, _ = predict_frozen(observed, (25, 34), record, cache, thread_limit=1)
    with np.load(record["artifacts"]["validation_predictions"]["path"]) as saved:
        np.testing.assert_array_equal(validation, saved["score"])
    nodes, original, _ = predict_frozen(observed, (35, 49), record, cache, thread_limit=1)
    x = data.x.copy()
    x[data.time_step == 49] *= -10000
    changed = Observations.from_data(replace(data, x=x, y=np.full(len(data.y), -1)))
    other = SnapshotFeatureCache(changed, config["graph"], None, thread_limit=1)
    _, scores, _ = predict_frozen(changed, (35, 49), record, other, thread_limit=1)
    np.testing.assert_array_equal(
        original[data.time_step[nodes] == 35], scores[data.time_step[nodes] == 35]
    )


def test_atomic_json_rejects_invalid_payload_without_replacing_evidence(tmp_path):
    from src.atomic_artifacts import write_json

    path = tmp_path / "run.json"
    write_json(path, {"status": "completed"})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {"invalid": float("nan")})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_npz_exception_cannot_publish_partial_archive(tmp_path, monkeypatch):
    from src.atomic_artifacts import write_npz

    path = tmp_path / "scores.npz"
    write_npz(path, score=np.array([0.1, 0.9]))
    before = path.read_bytes()

    def interrupted_writer(stream, **arrays):
        stream.write(b"partial archive")
        raise OSError("synthetic write interruption")

    monkeypatch.setattr(np, "savez_compressed", interrupted_writer)
    with pytest.raises(OSError, match="interruption"):
        write_npz(path, score=np.array([0.2]))
    assert path.read_bytes() == before
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["score"], [0.1, 0.9])
    assert list(tmp_path.iterdir()) == [path]
