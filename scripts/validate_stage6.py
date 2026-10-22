"""Independently reconstruct monitoring quality, drift, seed summaries and warnings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.atomic_artifacts import write_json  # noqa: E402
from src.data import load_csv, sha256_file  # noqa: E402
from src.inference import Observations, verify_artifact  # noqa: E402
from src.monitor import selected_structure_inputs  # noqa: E402
from src.train import array_digest  # noqa: E402


def independent_quality(y: np.ndarray, scores: np.ndarray, cutoff: float) -> dict[str, Any]:
    """Compute direct confusion counts and tied-score AP without production metrics."""
    known, prediction = y != -1, scores >= cutoff
    yy, ss, pp = y[known], scores[known], prediction[known]
    tp, fp = int(((yy == 1) & pp).sum()), int(((yy == 0) & pp).sum())
    fn, tn = int(((yy == 1) & ~pp).sum()), int(((yy == 0) & ~pp).sum())
    positives, n = int((yy == 1).sum()), len(yy)
    ap = None
    if positives:
        order = np.argsort(-ss, kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(ss[order]) != 0), len(ss) - 1]
        at_rank = np.cumsum(yy[order])[ends]
        ap = float(np.sum(np.diff(np.r_[0, at_rank / positives]) * at_rank / (ends + 1)))
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "average_precision": ap,
        "precision": tp / (tp + fp) if tp + fp else 0.0 if n else None,
        "recall": tp / positives if positives else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if positives else None,
        "support_labeled": n,
        "support_illicit": positives,
        "unknown_nodes": int((y == -1).sum()),
        "label_coverage": n / len(y),
        "observed_illicit_rate_labeled": positives / n if n else None,
        "mean_score_all": float(scores.mean()),
        "alert_rate_all": float(prediction.mean()),
        "unknown_alerts": int((prediction & ~known).sum()),
    }


def check_values(row: Any, expected: dict[str, Any]) -> None:
    """Compare scalar evidence with strict numerical tolerance and preserved nulls."""
    for key, value in expected.items():
        actual = row[key]
        if value is None or pd.isna(value):
            assert pd.isna(actual), (key, actual, value)
        elif isinstance(value, (float, np.floating)):
            assert np.isclose(actual, value, rtol=0, atol=1e-12), (key, actual, value)
        else:
            assert actual == value, (key, actual, value)


def independent_drift(
    train: np.ndarray, current: np.ndarray, histogram: dict, floor: float
) -> dict[str, float]:
    """Use bin lookup/counting and empirical CDFs, independently of production PSI/KS."""
    cuts = np.asarray(histogram["interior_edges"])
    if train.min() != train.max():
        quantiles = np.linspace(0, 1, histogram["requested_bins"] + 1)[1:-1]
        np.testing.assert_array_equal(cuts, np.unique(np.quantile(train, quantiles)))
    assert histogram["fit_split"] == "train" and histogram["n"] == len(train)
    ref = np.bincount(np.searchsorted(cuts, train, side="right"), minlength=len(cuts) + 1)
    observed = np.bincount(np.searchsorted(cuts, current, side="right"), minlength=len(cuts) + 1)
    np.testing.assert_array_equal(ref, histogram["counts"])
    p, q = np.clip(ref / len(train), floor, None), np.clip(observed / len(current), floor, None)
    p, q = p / p.sum(), q / q.sum()
    positions = np.unique(np.r_[train, current])
    cdf_train = np.searchsorted(np.sort(train), positions, side="right") / len(train)
    cdf_current = np.searchsorted(np.sort(current), positions, side="right") / len(current)
    return {
        "psi": float(np.sum((q - p) * np.log(q / p))),
        "ks_statistic": float(np.max(np.abs(cdf_train - cdf_current))),
        "current_n": len(current),
        "outside_training_range_rate": float(
            ((current < train.min()) | (current > train.max())).mean()
        ),
    }


def validate(output: Path) -> dict[str, Any]:
    """Audit saved evidence without modifying models, thresholds or input references."""
    batch = json.loads((output / "latest_run.json").read_text())
    assert batch["status"] == "completed" and not batch["models_or_preprocessors_fitted"]
    assert not batch["decision_thresholds_changed"]
    reference = json.loads(verify_artifact(batch["protocol"]).read_text())
    config = reference["config"]
    assert reference["status"] == "frozen_before_monitoring_test_analysis"
    assert (
        reference["test_already_opened_in_stage5"]
        and not reference["prospective_trigger_validation"]
    )
    for name, digest in reference["critical_source_sha256"].items():
        assert sha256_file(Path("src") / name) == digest
    for name, digest in reference["input_files_sha256"].items():
        assert sha256_file(Path(config["data"]["raw_dir"]) / name) == digest
    stage5 = json.loads(verify_artifact(reference["stage5_manifest"]).read_text())
    plan = json.loads(verify_artifact(reference["stage5_protocol"]).read_text())
    primary = plan["config"]["experiment"]["primary_variants"]
    assert reference["validation_selected_variant"] == max(
        primary, key=plan["primary_validation_ap_means"].get
    )
    source_runs = [
        json.loads(verify_artifact(record).read_text()) for record in reference["source_records"]
    ]
    assert len(source_runs) == len(primary) * len(config["experiment"]["seeds"])
    for record in batch["runs"]:
        run = json.loads(verify_artifact(record).read_text())
        assert not run["models_or_preprocessors_fitted"] and not run["decision_threshold_changed"]
    data = load_csv(
        Path(config["data"]["raw_dir"]), require_full_dataset=config["data"]["require_full_dataset"]
    )

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(output / f"{name}.csv", float_precision="round_trip")

    steps, rolling, periods, warnings = [
        read(name)
        for name in ("per_step_metrics", "rolling_metrics", "regime_per_seed", "warnings")
    ]
    assert (
        len(steps) == len(source_runs) * 15
        and not steps.duplicated(["variant", "seed", "time_step"]).any()
    )
    predictions, selected_records = {}, []
    entries = {e["source_run_id"]: e for e in plan["references"]}
    candidate = reference["validation_selected_variant"]
    for run in source_runs:
        with np.load(verify_artifact(run["test_predictions"])) as archive:
            values = {key: archive[key] for key in archive.files}
        current_nodes = (data.time_step >= 35) & (data.time_step <= 49)
        np.testing.assert_array_equal(values["tx_id"], data.tx_ids[current_nodes])
        np.testing.assert_array_equal(values["label"], data.y[current_nodes])
        np.testing.assert_array_equal(values["time_step"], data.time_step[current_nodes])
        y, ss, tt = values["label"], values["score"], values["time_step"]
        cutoff = run["frozen_threshold"]
        source = json.loads(verify_artifact(run["source_record"]).read_text())
        assert source["threshold"]["value"] == cutoff
        group = steps[(steps.variant == run["variant"]) & (steps.seed == run["seed"])]
        assert group.time_step.tolist() == list(range(35, 50))
        for row in group.to_dict("records"):
            mask = tt == row["time_step"]
            check_values(row, independent_quality(y[mask], ss[mask], cutoff))
        roll = rolling[(rolling.variant == run["variant"]) & (rolling.seed == run["seed"])]
        policy = config["performance"]
        for row in roll.to_dict("records"):
            start = max(35, row["time_step"] - policy["rolling_steps"] + 1)
            mask = (tt >= start) & (tt <= row["time_step"])
            metrics = independent_quality(y[mask], ss[mask], cutoff)
            check_values(row, metrics)
            complete = row["time_step"] - start + 1 == policy["rolling_steps"]
            supported = (
                complete
                and metrics["support_labeled"] >= policy["minimum_known_labels"]
                and metrics["support_illicit"] >= policy["minimum_illicit_labels"]
            )
            floor = source["validation_metrics"]["f1"] * policy["f1_floor_fraction_of_validation"]
            check_values(
                row,
                {
                    "window_start": start,
                    "complete_window": complete,
                    "support_sufficient": supported,
                    "f1_floor": floor,
                    "performance_warning": bool(
                        supported and metrics["f1"] is not None and metrics["f1"] < floor
                    ),
                },
            )
        for row in periods[
            (periods.variant == run["variant"]) & (periods.seed == run["seed"])
        ].to_dict("records"):
            bounds = config["investigation"]["periods"][row["period"]]
            mask = (tt >= bounds[0]) & (tt <= bounds[1])
            check_values(row, independent_quality(y[mask], ss[mask], cutoff))
        if run["variant"] == candidate:
            selected_records.append((run, source))
            predictions[run["seed"]] = values
    selected_records.sort(key=lambda pair: pair[0]["seed"])
    observation = Observations.from_data(data)
    names = selected_records[0][1]["feature_names"]
    nodes, xx = selected_structure_inputs(observation, (1, 49), plan["config"]["graph"], names)
    train = data.time_step[nodes] <= 24
    known_train = train & (data.y[nodes] != -1)
    fingerprint = array_digest(
        xx[known_train],
        data.y[nodes][known_train],
        data.tx_ids[nodes][known_train],
        data.time_step[nodes][known_train],
    )
    assert fingerprint == reference["training_input_fingerprint"]
    with np.load(verify_artifact(reference["training_reference"])) as archive:
        saved_reference = {key: archive[key] for key in archive.files}
    np.testing.assert_array_equal(saved_reference["tx_id"], data.tx_ids[nodes][train])
    assert reference["reference_nodes"] == int(train.sum())
    assert reference["reference_known_labels"] == int(known_train.sum())
    score_drift, feature_drift = read("score_drift"), read("feature_drift")
    floor = config["drift"]["probability_floor"]
    for run, source in selected_records:
        assert fingerprint == source["input_fingerprints"]["train"]
        model = XGBClassifier(n_jobs=config["runtime"]["thread_limit"])
        model.load_model(verify_artifact(source["artifacts"]["model"]))
        with threadpool_limits(limits=config["runtime"]["thread_limit"]):
            predicted = model.predict_proba(xx[train])[:, 1].astype(float)
        np.testing.assert_array_equal(predicted, saved_reference[f"score_s{run['seed']}"])
        histogram = reference["score_histograms"][str(run["seed"])]
        entry = entries[run["source_run_id"]]
        with np.load(verify_artifact(entry["validation_predictions"])) as val:
            val_score, val_times = val["score"], val["time_step"]
        group = score_drift[score_drift.seed == run["seed"]]
        assert len(group) == 49
        for row in group.to_dict("records"):
            st = row["time_step"]
            current = (
                predicted[data.time_step[nodes][train] == st]
                if st <= 24
                else val_score[val_times == st]
                if st <= 34
                else predictions[run["seed"]]["score"][predictions[run["seed"]]["time_step"] == st]
            )
            check_values(row, independent_drift(predicted, current, histogram, floor))
        if run["seed"] == config["experiment"]["representative_seed"]:
            importance = model.feature_importances_.astype(float)
            ranking = sorted(range(len(names)), key=lambda i: (-importance[i], names[i]))
            top = ranking[: config["drift"]["top_features"]]
            assert [names[i] for i in top] == reference["top_feature_names"]
            np.testing.assert_array_equal(saved_reference["feature_values"], xx[train][:, top])
    for row in feature_drift.to_dict("records"):
        index = names.index(row["feature"])
        ref = xx[train, index]
        values = xx[data.time_step[nodes] == row["time_step"], index]
        check_values(
            row,
            independent_drift(ref, values, reference["feature_histograms"][row["feature"]], floor),
        )
    assert len(feature_drift) == 49 * config["drift"]["top_features"]
    for _, group in warnings.groupby("seed"):
        consecutive = 0
        for row in group.sort_values("time_step").to_dict("records"):
            exceeded = row["psi"] > config["drift"]["psi_warning"]
            consecutive = consecutive + 1 if exceeded else 0
            expected = rolling[
                (rolling.variant == candidate)
                & (rolling.seed == row["seed"])
                & (rolling.time_step == row["time_step"])
            ].iloc[0]
            check_values(
                row,
                {
                    "score_drift_warning": exceeded,
                    "persistent_score_drift": consecutive >= config["drift"]["consecutive_steps"],
                    "performance_warning": bool(expected.performance_warning),
                    "retraining_review": bool(expected.performance_warning),
                    "automatic_retraining_or_promotion": False,
                },
            )
    for raw, filename, keys in (
        (steps, "per_step_summary", ["variant", "time_step"]),
        (periods, "regime_summary", ["variant", "period"]),
    ):
        aggregate = read(filename)
        for row in aggregate.to_dict("records"):
            mask = np.ones(len(raw), dtype=bool)
            for key in keys:
                mask &= raw[key].to_numpy() == row[key]
            group = raw.loc[mask]
            assert sorted(group.seed) == config["experiment"]["seeds"]
            for metric in [col[:-5] for col in aggregate if col.endswith("_mean")]:
                check_values(
                    row,
                    {
                        f"{metric}_mean": group[metric].mean(),
                        f"{metric}_std": group[metric].std(ddof=1),
                    },
                )
    result = {
        "batch_id": batch["batch_id"],
        "status": "passed",
        "candidate_models_reloaded": len(selected_records),
        "frozen_evaluation_score_archives_verified": len(source_runs),
        "per_step_quality_rows_independently_checked": len(steps),
        "score_and_feature_drift_rows_independently_checked": len(score_drift) + len(feature_drift),
        "train_only_reference_and_training_gain_ranking_verified": True,
        "training_inputs_match_original_fit_fingerprints": True,
        "unknown_denominators_and_undefined_metrics_verified": True,
        "rolling_counts_support_floors_and_warnings_verified": True,
        "event_periods_and_seed_means_sample_std_verified": True,
        "original_stage5_thresholds_and_evidence_unchanged": True,
        "models_retrained": False,
        "prospective_detection_or_causality_claimed": False,
        "stage5_batch_id": stage5["batch_id"],
    }
    write_json(output / "independent_validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage6"))
    print(json.dumps(validate(parser.parse_args().output_dir), indent=2))
