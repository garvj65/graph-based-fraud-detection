"""Independently audit frozen policies, all model scores and final test summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_csv, sha256_file, write_json  # noqa: E402
from src.final_evaluate import make_cache  # noqa: E402
from src.inference import predict_frozen, verify_artifact  # noqa: E402


def independent_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    """NumPy tied-score AP and direct known-label confusion counts."""
    known = y != -1
    y, score = y[known], score[known]
    order = np.argsort(-score, kind="stable")
    sorted_y, sorted_score = y[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(y) - 1]
    tp_at_rank = np.cumsum(sorted_y)[ends]
    recall = tp_at_rank / y.sum()
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * (tp_at_rank / (ends + 1))))
    pred = score >= threshold
    tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tp = int(((y == 1) & pred).sum())
    return {
        "average_precision": ap,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def validate(output: Path) -> dict[str, Any]:
    """Reload every frozen model and independently reconstruct all test policies."""
    batch = json.loads((output / "latest_run.json").read_text())
    assert batch["status"] == "completed" and batch["models_or_preprocessors_fitted"] is False
    protocol_path = verify_artifact(batch["protocol"])
    plan = json.loads(protocol_path.read_text())
    assert plan["status"] == "frozen_before_test" and plan["test_scored_before_freeze"] is False
    config = plan["config"]
    for filename, digest in plan["critical_source_sha256"].items():
        if sha256_file(Path("src") / filename) != digest:
            from scripts.recover_stage5_artifacts import unchanged_evaluation_logic

            recovery = json.loads(verify_artifact(batch["artifact_recovery"]).read_text())
            amendment = recovery["source_amendment"]
            assert filename == amendment["file"] == "final_evaluate.py"
            assert digest == amendment["before_sha256"]
            assert sha256_file(Path("src") / filename) == amendment["after_sha256"]
            snapshot = verify_artifact(amendment["before_source_snapshot"])
            assert unchanged_evaluation_logic(
                snapshot.read_text(), (Path("src") / filename).read_text()
            )
            assert recovery["all_original_test_metrics_reproduced"] is True
            assert recovery["thresholds_or_selection_changed"] is False
    for name, digest in plan["input_files_sha256"].items():
        assert sha256_file(Path(config["data"]["raw_dir"]) / name) == digest
    for artifact in plan["reference_manifest_sha256"].values():
        verify_artifact(artifact)
    data = load_csv(Path(config["data"]["raw_dir"]), require_full_dataset=True)
    cache = make_cache(data, config)
    expected_nodes = np.flatnonzero((data.time_step >= 35) & (data.time_step <= 49))
    entries = {entry["source_run_id"]: entry for entry in plan["references"]}
    runs = []
    for run_id in batch["evaluation_runs"]:
        run = json.loads((output / "runs" / f"{run_id}.json").read_text())
        assert run["status"] == "completed" and not run["models_or_preprocessors_fitted"]
        assert not run["threshold_reselected_on_test"]
        entry = entries[run["source_run_id"]]
        source = json.loads(verify_artifact(entry["source_record"]).read_text())
        assert source["threshold"]["value"] == run["frozen_threshold"] == entry["frozen_threshold"]
        nodes, actual, _ = predict_frozen(
            cache.observed, (35, 49), source, cache, thread_limit=config["runtime"]["thread_limit"]
        )
        np.testing.assert_array_equal(nodes, expected_nodes)
        with np.load(verify_artifact(run["test_predictions"])) as saved:
            np.testing.assert_array_equal(saved["tx_id"], data.tx_ids[nodes])
            np.testing.assert_array_equal(saved["time_step"], data.time_step[nodes])
            np.testing.assert_array_equal(saved["label"], data.y[nodes])
            np.testing.assert_array_equal(saved["score"], actual)
            y, score, ids, times = (
                saved["label"],
                saved["score"],
                saved["tx_id"],
                saved["time_step"],
            )
        computed = independent_metrics(y, score, run["frozen_threshold"])
        for key, value in computed.items():
            if key == "average_precision":
                assert np.isclose(value, run["test_metrics"][key], rtol=0, atol=1e-14)
            else:
                assert value == run["test_metrics"][key]
        alerts = score >= run["frozen_threshold"]
        workload = run["frozen_threshold_workload"]
        assert workload["alert_count_all"] == int(alerts.sum())
        assert workload["unknown_alerts"] == int((alerts & (y == -1)).sum())
        assert workload["known_illicit_alerts"] == int((alerts & (y == 1)).sum())
        assert workload["known_licit_alerts"] == int((alerts & (y == 0)).sum())
        assert sum(map(sum, computed["confusion_matrix"])) == int((data.y[nodes] != -1).sum())
        for row in run["per_step_budget_metrics"]:
            current = np.flatnonzero(times == row["time_step"])
            eligible = [int(i) for i in current if row["view"] != "labeled_only" or y[i] != -1]
            ordered = sorted(eligible, key=lambda i: (-float(score[i]), str(ids[i])))
            chosen = ordered[: config["alert_budget"]["per_time_step"]]
            illicit = sum(y[i] == 1 for i in chosen)
            licit = sum(y[i] == 0 for i in chosen)
            unknown = sum(y[i] == -1 for i in chosen)
            assert row["known_illicit_alerts"] == illicit
            assert row["known_licit_alerts"] == licit and row["unknown_alerts"] == unknown
            assert row["alerts_selected"] == len(chosen)
            assert row["known_alerts"] == illicit + licit
            if row["view"] == "all_transactions":
                assert row["precision_at_k_labeled"] is None
            elif chosen:
                assert row["precision_at_k_labeled"] == illicit / len(chosen)
        with np.load(verify_artifact(entry["validation_predictions"])) as validation:
            vy, vs = validation["label"], validation["score"]
            known = vy != -1
            order = np.argsort(-vs[known], kind="stable")
            sorted_score, sorted_y = vs[known][order], vy[known][order]
            ends = np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(sorted_y) - 1]
            recall = np.cumsum(sorted_y)[ends] / sorted_y.sum()
            for policy in run["recall_policies"]:
                index = np.flatnonzero(recall >= policy["target_recall"])[0]
                assert policy["threshold"] == float(sorted_score[ends[index]])
                values = independent_metrics(y, score, policy["threshold"])
                for metric in ("precision", "recall", "f1", "confusion_matrix"):
                    assert values[metric] == policy["test_metrics"][metric]
        runs.append(run)
    summary = pd.read_csv(output / "test_summary_all_variants.csv", float_precision="round_trip")
    for row in summary.to_dict("records"):
        group = [run for run in runs if run["variant"] == row["variant"]]
        assert sorted(run["seed"] for run in group) == sorted(config["experiment"]["seeds"])
        for metric in (
            "average_precision",
            "pr_auc_trapezoid",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ):
            values = np.array([run["test_metrics"][metric] for run in group])
            assert np.isclose(values.mean(), row[f"{metric}_mean"], rtol=0, atol=1e-14)
            assert np.isclose(values.std(ddof=1), row[f"{metric}_std"], rtol=0, atol=1e-14)
    validation_means = plan["primary_validation_ap_means"]
    expected_winner = max(config["experiment"]["primary_variants"], key=validation_means.get)
    assert expected_winner == plan["validation_selected_variant"]
    assert (
        json.loads((output / "evaluation_summary.json").read_text())["validation_selected_variant"]
        == expected_winner
    )
    representative = pd.read_csv(
        output / "representative_test_scores_labeled.csv.gz",
        dtype={"tx_id": str},
        float_precision="round_trip",
    )
    queue_path = output / "representative_operational_queues.csv.gz"
    queues = (
        pd.read_csv(queue_path, dtype={"tx_id": str}, float_precision="round_trip")
        if queue_path.exists()
        else None
    )
    for run in runs:
        if not run["primary"] or run["seed"] != config["experiment"]["representative_seed"]:
            continue
        frame = representative[representative.variant == run["variant"]]
        with np.load(run["test_predictions"]["path"]) as saved:
            known = saved["label"] != -1
            saved_scores, saved_ids, saved_times = (
                saved["score"],
                saved["tx_id"],
                saved["time_step"],
            )
            np.testing.assert_array_equal(frame.score.to_numpy(), saved_scores[known])
            np.testing.assert_array_equal(frame.tx_id.to_numpy(), saved_ids[known])
            if queues is not None:
                for step in range(35, 50):
                    indices = np.flatnonzero(saved_times == step)
                    chosen = sorted(
                        indices, key=lambda i: (-float(saved_scores[i]), str(saved_ids[i]))
                    )
                    chosen = chosen[: config["alert_budget"]["per_time_step"]]
                    queue = queues[(queues.variant == run["variant"]) & (queues.time_step == step)]
                    np.testing.assert_array_equal(queue.tx_id.to_numpy(), saved_ids[chosen])
                    np.testing.assert_array_equal(queue.score.to_numpy(), saved_scores[chosen])
    budget_micro = pd.read_csv(output / "budget_per_seed.csv", float_precision="round_trip")
    for run in runs:
        for view in config["alert_budget"]["views"]:
            values = [row for row in run["per_step_budget_metrics"] if row["view"] == view]
            row = budget_micro[
                (budget_micro.variant == run["variant"])
                & (budget_micro.seed == run["seed"])
                & (budget_micro.view == view)
            ].iloc[0]
            for metric in (
                "alerts_selected",
                "known_illicit_alerts",
                "known_licit_alerts",
                "unknown_alerts",
                "known_alerts",
            ):
                assert row[metric] == sum(value[metric] for value in values)
            if view == "labeled_only":
                assert row.precision_at_k_labeled == row.known_illicit_alerts / row.alerts_selected
    budget_summary = pd.read_csv(output / "budget_summary.csv", float_precision="round_trip")
    for row in budget_summary.to_dict("records"):
        group = budget_micro[
            (budget_micro.variant == row["variant"]) & (budget_micro.view == row["view"])
        ]
        for metric in (
            "alerts_selected",
            "known_illicit_alerts",
            "known_licit_alerts",
            "unknown_alerts",
            "known_label_coverage",
        ):
            assert np.isclose(group[metric].mean(), row[f"{metric}_mean"], atol=1e-14, rtol=0)
            assert np.isclose(group[metric].std(ddof=1), row[f"{metric}_std"], atol=1e-14, rtol=0)
    for run in runs:
        if run["variant"] == "s3_xgboost_all_control":
            baseline = next(
                r
                for r in runs
                if r["variant"] == "s2_xgboost_all_time" and r["seed"] == run["seed"]
            )
            assert run["test_metrics"] == baseline["test_metrics"]
            with (
                np.load(run["test_predictions"]["path"]) as a,
                np.load(baseline["test_predictions"]["path"]) as b,
            ):
                np.testing.assert_array_equal(a["score"], b["score"])
    result = {
        "batch_id": batch["batch_id"],
        "status": "passed",
        "frozen_models_reloaded": len(runs),
        "test_scores_bitwise_reproduced": True,
        "protocol_and_input_hashes_verified": True,
        "original_validation_thresholds_unchanged": True,
        "test_ap_and_confusion_independently_recomputed": True,
        "both_budget_denominators_and_unknown_counts_verified": True,
        "budget_totals_and_summary_statistics_verified": True,
        "representative_queue_rankings_verified": True,
        "stage3_control_reproduces_stage2_on_test": True,
        "recall_policies_rederived_from_validation_only": True,
        "five_seed_means_and_sample_std_verified": True,
        "model_selection_uses_validation_only": True,
        "representative_scores_round_trip_verified": True,
        "models_or_preprocessors_fitted": False,
        "monitoring_completed": False,
    }
    write_json(output / "independent_validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage5"))
    print(json.dumps(validate(parser.parse_args().output_dir), indent=2))
