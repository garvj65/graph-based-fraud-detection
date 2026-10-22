"""Recompute Stage 4 evidence from saved arrays/checkpoints without retraining.

Run from the repository root after `make stage4`:
    python scripts/validate_stage4.py
Test data is never scored. NumPy independently reconstructs AP and max-F1 policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_csv, sha256_file, write_json  # noqa: E402
from src.gnn import SmallGNN, prepare_gnn_data, seed_everything  # noqa: E402


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, dict]:
    """Independently group tied scores to compute AP and validation-max-F1 cutoff."""
    known = labels != -1
    y, score = labels[known], scores[known]
    order = np.argsort(-score, kind="stable")
    sorted_score, sorted_y = score[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(y) - 1]
    tp = np.cumsum(sorted_y)[ends]
    alerts = ends + 1
    precision, recall = tp / alerts, tp / y.sum()
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    f1 = 2 * tp / (alerts + y.sum())
    # Descending thresholds: the first numerically tied optimum is the higher cutoff.
    index = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[0]
    threshold = float(sorted_score[ends[index]])
    positive = score >= threshold
    true_positive = int(((y == 1) & positive).sum())
    false_positive = int(((y == 0) & positive).sum())
    false_negative = int(((y == 1) & ~positive).sum())
    true_negative = int(((y == 0) & ~positive).sum())
    metrics = {
        "precision": true_positive / (true_positive + false_positive),
        "recall": true_positive / (true_positive + false_negative),
        "f1": 2 * true_positive / (2 * true_positive + false_positive + false_negative),
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
    }
    return ap, threshold, metrics


def validate(output: Path) -> dict:
    """Fail loudly on changed inputs, wrong selection, non-reproducible scores or tables."""
    batch = json.loads((output / "latest_run.json").read_text())
    assert batch["status"] == "completed" and batch["test_evaluated"] is False
    config = batch["config"]
    for filename, digest in batch["provenance"]["input_files_sha256"].items():
        assert sha256_file(Path(config["data"]["raw_dir"]) / filename) == digest
    for filename, digest in batch["provenance"]["source_files_sha256"].items():
        assert sha256_file(Path("src") / filename) == digest, filename
    for key, digest in batch["provenance"]["reference_files_sha256"].items():
        assert sha256_file(Path(config["experiment"][key])) == digest
    data = load_csv(Path(config["data"]["raw_dir"]), require_full_dataset=True)
    seed_everything(0, thread_limit=config["runtime"]["thread_limit"], deterministic=True)
    prepared = {
        group: prepare_gnn_data(data, group, config["graph"])
        for group in config["experiment"]["feature_groups"]
    }
    prepared["all_no_messages"] = prepare_gnn_data(data, "all", config["graph"], no_messages=True)
    all_ids = sorted(set(batch["trial_runs"] + batch["selected_runs"]))
    records = []
    for run_id in all_ids:
        run = json.loads((output / "runs" / f"{run_id}.json").read_text())
        assert run["status"] == "completed" and run["test_evaluated"] is False
        assert run["config"] == config
        view = prepared[run["feature_group"]]
        assert run["data_audit"] == view.audit
        assert view.train.times.max() == 24 and view.validation.times.max() == 34
        assert view.train.times.max() < view.validation.times.min()
        assert int(view.train.known.sum()) == 23606
        assert int(view.validation.known.sum()) == 6288
        assert run["class_weights"]["scale_pos_weight"] == 21387 / 2219
        for artifact in run["artifacts"].values():
            assert sha256_file(Path(artifact["path"])) == artifact["sha256"]
        scaler = joblib.load(run["artifacts"]["scaler"]["path"])
        np.testing.assert_array_equal(scaler.mean_, view.scaler.mean_)
        np.testing.assert_array_equal(scaler.scale_, view.scaler.scale_)
        assert scaler.n_samples_seen_ == 23606
        checkpoint = torch.load(
            run["artifacts"]["checkpoint"]["path"], map_location="cpu", weights_only=True
        )
        assert checkpoint["best_epoch"] == run["best_epoch"]
        parameters = run["parameters"]
        model = SmallGNN(
            run["model"],
            checkpoint["in_channels"],
            parameters["hidden_channels"],
            parameters["dropout"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            actual = model(view.validation.x, view.validation.edge_index).sigmoid()
            actual = actual.numpy().astype(np.float64)
        with np.load(run["artifacts"]["validation_predictions"]["path"]) as prediction:
            np.testing.assert_array_equal(prediction["tx_id"], view.validation.tx_ids)
            np.testing.assert_array_equal(prediction["time_step"], view.validation.times)
            np.testing.assert_array_equal(prediction["label"], view.validation.y.numpy())
            np.testing.assert_array_equal(prediction["score"], actual)
            ap, threshold, metrics = ranking_metrics(prediction["label"], actual)
        assert np.isclose(ap, run["validation_metrics"]["average_precision"], atol=1e-14, rtol=0)
        assert threshold == run["threshold"]["value"]
        for key, value in metrics.items():
            assert value == run["validation_metrics"][key]
        history = pd.read_csv(run["artifacts"]["history"]["path"], float_precision="round_trip")
        assert history.epoch.tolist() == list(range(1, run["epochs_run"] + 1))
        assert np.isfinite(history.select_dtypes("number").to_numpy()).all()
        best = history.loc[history.validation_ap.idxmax()]
        assert run["best_epoch"] == int(best.epoch)
        assert run["best_validation_ap"] == best.validation_ap
        if run["stop_reason"] == "patience":
            assert len(history) - run["best_epoch"] == config["training"]["patience"]
        else:
            assert len(history) == config["training"]["max_epochs"]
        records.append(run)
    trial_ids = set(batch["trial_runs"])
    for key, choice in batch["choices"].items():
        trials = [
            run
            for run in records
            if run["run_id"] in trial_ids and f"{run['model']}_{run['feature_group']}" == key
        ]
        assert len(trials) == len(config["training"]["candidates"])
        best = max(trials, key=lambda run: run["validation_metrics"]["average_precision"])
        assert best["candidate_index"] == choice["candidate_index"]
    selected = [run for run in records if run["run_id"] in set(batch["selected_runs"])]
    summary = pd.read_csv(output / "validation_summary.csv", float_precision="round_trip")
    for row in summary.to_dict("records"):
        runs = [
            run
            for run in selected
            if run["model"] == row["model"] and run["feature_group"] == row["feature_group"]
        ]
        assert sorted(run["seed"] for run in runs) == config["experiment"]["seeds"]
        assert len({run["candidate_index"] for run in runs}) == 1
        choice_group = "all" if row["feature_group"] == "all_no_messages" else row["feature_group"]
        choice = batch["choices"][f"{row['model']}_{choice_group}"]
        assert all(run["candidate_index"] == choice["candidate_index"] for run in runs)
        for metric in (
            "average_precision",
            "pr_auc_trapezoid",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "alert_rate_labeled",
            "mean_score_labeled",
            "threshold",
        ):
            values = np.array([run["validation_metrics"][metric] for run in runs])
            assert np.isclose(values.mean(), row[f"{metric}_mean"], atol=1e-14, rtol=0)
            assert np.isclose(values.std(ddof=1), row[f"{metric}_std"], atol=1e-14, rtol=0)
    paired = pd.read_csv(output / "message_passing_lift.csv", float_precision="round_trip")
    for row in paired.to_dict("records"):
        runs = {
            run["feature_group"]: run
            for run in selected
            if run["seed"] == row["seed"] and run["model"] == row["model"]
        }
        for metric, column in (
            ("average_precision", "ap_delta_message_minus_no_message"),
            ("f1", "f1_delta_message_minus_no_message"),
        ):
            expected = (
                runs["all"]["validation_metrics"][metric]
                - runs["all_no_messages"]["validation_metrics"][metric]
            )
            assert row[column] == expected
    # Verify the committed representative CSV round-trips scores and thresholds exactly.
    representative = pd.read_csv(
        output / "representative_validation_scores.csv.gz",
        dtype={"tx_id": str},
        float_precision="round_trip",
    )
    for run in selected:
        if run["seed"] != config["experiment"]["representative_seed"]:
            continue
        frame = representative[
            (representative.model == run["model"])
            & (representative.feature_group == run["feature_group"])
        ]
        with np.load(run["artifacts"]["validation_predictions"]["path"]) as prediction:
            known = prediction["label"] != -1
            np.testing.assert_array_equal(frame.score.to_numpy(), prediction["score"][known])
            np.testing.assert_array_equal(frame.tx_id.to_numpy(), prediction["tx_id"][known])
        assert (frame.threshold == run["threshold"]["value"]).all()
    result = {
        "batch_id": batch["batch_id"],
        "status": "passed",
        "unique_checkpoints_reloaded": len(records),
        "selected_runs_checked": len(selected),
        "input_and_source_hashes_verified": True,
        "train_only_scalers_and_weights_verified": True,
        "unknown_exclusion_and_temporal_membership_verified": True,
        "checkpoint_validation_scores_bitwise_identical": True,
        "ap_threshold_and_confusion_independently_recomputed": True,
        "best_epochs_and_candidate_choices_verified": True,
        "summary_means_sample_std_and_message_deltas_verified": True,
        "representative_csv_round_trip_verified": True,
        "test_evaluated": False,
        "test_graph_features_computed": False,
    }
    interrupted = list((output / "interrupted_runs").glob("*.json"))
    for path in interrupted:
        record = json.loads(path.read_text())
        assert record["status"] == "interrupted"
        assert record["excluded_from_selection_and_metrics"] is True
        assert record["resumed_run_id"] in all_ids
        assert "validation_metrics" not in record
    result["interrupted_attempts_excluded"] = len(interrupted)
    write_json(output / "independent_validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage4"))
    print(json.dumps(validate(parser.parse_args().output_dir), indent=2))
