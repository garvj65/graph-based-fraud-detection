"""Retrospective performance and train-referenced drift for frozen fraud models.

No model, scaler, decision threshold, or feature ranking is fitted on held-out
data. Unknown nodes enter distribution monitoring, never classification truth.
"""

from __future__ import annotations

import argparse
import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import ArrayLike
from scipy.stats import ks_2samp
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from src.atomic_artifacts import write_json, write_npz
from src.data import RAW_FILES, TemporalSplit, environment_versions, load_csv, sha256_file
from src.evaluate import binary_metrics, known_binary_scores
from src.evaluation_metrics import alert_counts
from src.features import STRUCTURE_NAMES, snapshot_structure
from src.inference import Observations, observed_view, raw_inputs, verify_artifact
from src.train import array_digest


def finite_vector(values: ArrayLike, *, allow_empty: bool = False) -> np.ndarray:
    """Reject malformed or missing observations rather than silently dropping rows."""
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all() or (not len(result) and not allow_empty):
        raise ValueError("Distribution inputs must be nonempty finite one-dimensional arrays.")
    return result


def fit_histogram(values: ArrayLike, *, bins: int, split: str = "train") -> dict[str, Any]:
    """Freeze quantile cuts on training observations, retaining open outer bins.

    Repeated quantiles collapse to fewer bins. A constant training variable uses
    a three-bin partition so both upward and downward changes remain visible.
    Only finite interior cuts are serialized; no nonstandard JSON infinities.
    """
    if split != "train":
        raise ValueError("Drift reference bins may be fitted on training data only.")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("At least two requested histogram bins are required.")
    x = finite_vector(values)
    constant = bool(x.min() == x.max())
    if constant:
        value = x[0]
        upper = np.nextafter(value, np.inf)
        cuts = (
            np.array([value, upper])
            if np.isfinite(upper)
            else np.array([np.nextafter(value, -np.inf), value])
        )
    else:
        cuts = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    counts = np.histogram(x, bins=np.r_[-np.inf, cuts, np.inf])[0]
    return {
        "fit_split": "train",
        "requested_bins": bins,
        "interior_edges": cuts.tolist(),
        "counts": counts.tolist(),
        "n": len(x),
        "minimum": float(x.min()),
        "maximum": float(x.max()),
        "constant_reference": constant,
        "actual_bins": len(counts),
    }


def distribution_drift(
    reference: ArrayLike,
    current: ArrayLike,
    histogram: dict[str, Any],
    *,
    probability_floor: float,
) -> dict[str, Any]:
    """PSI with fixed train bins and clipped/renormalized proportions, plus KS D.

    KS p-values are diagnostic only: graph dependence, discrete/tied values,
    fitted-model reference scores and repeated comparisons violate a simple
    independent continuous-sample interpretation. No p-value triggers actions.
    """
    train, x = finite_vector(reference), finite_vector(current, allow_empty=True)
    if not 0 < probability_floor < 1 / histogram["actual_bins"]:
        raise ValueError("PSI probability floor must be positive and below 1 / bin count.")
    edges = np.r_[-np.inf, histogram["interior_edges"], np.inf]
    expected_counts = np.histogram(train, bins=edges)[0]
    if (
        histogram["fit_split"] != "train"
        or not np.array_equal(expected_counts, histogram["counts"])
        or len(train) != histogram["n"]
    ):
        raise ValueError("Histogram does not match its frozen training reference.")
    if not len(x):
        return {
            "psi": None,
            "ks_statistic": None,
            "ks_pvalue_diagnostic": None,
            "current_n": 0,
            "outside_training_range_rate": None,
        }
    expected = np.maximum(expected_counts / len(train), probability_floor)
    actual_counts = np.histogram(x, bins=edges)[0]
    actual = np.maximum(actual_counts / len(x), probability_floor)
    expected, actual = expected / expected.sum(), actual / actual.sum()
    psi = float(np.sum((actual - expected) * np.log(actual / expected)))
    ks = ks_2samp(train, x, alternative="two-sided", method="asymp")
    return {
        "psi": max(0.0, psi),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue_diagnostic": float(ks.pvalue),
        "current_n": len(x),
        "outside_training_range_rate": float(
            ((x < histogram["minimum"]) | (x > histogram["maximum"])).mean()
        ),
    }


def step_metrics(labels: ArrayLike, scores: ArrayLike, *, threshold: float) -> dict[str, Any]:
    """Known-label quality alongside explicitly all-node score/workload denominators."""
    y, score = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    known_binary_scores(y, score)
    result = binary_metrics(y, score, threshold=threshold)
    result.update(alert_counts(y, score, threshold))
    result.update(
        unknown_nodes=int((y == -1).sum()),
        label_coverage=float((y != -1).mean()) if len(y) else None,
        mean_score_all=float(score.mean()) if len(score) else None,
        known_illicit_share_all_nodes=float((y == 1).mean()) if len(y) else None,
    )
    tn, fp = result.pop("confusion_matrix")[0]
    fn = result["support_illicit"] - result["known_illicit_alerts"]
    result.update(tn=tn, fp=fp, fn=fn, tp=result["known_illicit_alerts"])
    return result


def rolling_performance(
    labels: ArrayLike,
    scores: ArrayLike,
    times: ArrayLike,
    *,
    step: int,
    first_step: int,
    threshold: float,
    validation_f1: float,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Pool consecutive snapshots through the current step, never averaging step F1."""
    y, score, time = np.asarray(labels), np.asarray(scores), np.asarray(times)
    if time.shape != y.shape:
        raise ValueError("Rolling timestamps must align with labels.")
    start = max(first_step, step - policy["rolling_steps"] + 1)
    selected = (time >= start) & (time <= step)
    values = step_metrics(y[selected], score[selected], threshold=threshold)
    complete = step - start + 1 == policy["rolling_steps"]
    supported = (
        complete
        and values["support_labeled"] >= policy["minimum_known_labels"]
        and values["support_illicit"] >= policy["minimum_illicit_labels"]
    )
    floor = validation_f1 * policy["f1_floor_fraction_of_validation"]
    return {
        "window_start": start,
        "window_end": step,
        "complete_window": complete,
        "support_sufficient": supported,
        "validation_f1": validation_f1,
        "f1_floor": floor,
        "performance_warning": bool(
            supported and values["f1"] is not None and values["f1"] < floor
        ),
        **values,
    }


def selected_structure_inputs(
    observed: Observations,
    bounds: tuple[int, int],
    graph: dict[str, Any],
    expected_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the selected 172-column schema without any neighbor-label score helper.

    This avoids applying the prefix-24 helper to training rows. Each snapshot
    supplies its own topology; global train/validation/test graph fits are absent.
    """
    nodes = observed_view(observed, bounds).node_indices
    raw, names = raw_inputs(observed, nodes, "all", include_time=True)
    if [*names, *STRUCTURE_NAMES] != expected_names:
        raise ValueError("Monitoring supports the frozen all-plus-structure schema only.")
    structure = np.empty((len(nodes), len(STRUCTURE_NAMES)), dtype=np.float32)
    for step in np.unique(observed.time_step[nodes]):
        view = observed_view(observed, (int(step), int(step)))
        values, _ = snapshot_structure(view, graph)
        rows = np.flatnonzero(observed.time_step[nodes] == step)
        if not np.array_equal(nodes[rows], view.node_indices):
            raise ValueError("Snapshot feature alignment changed.")
        structure[rows] = values
    return nodes, np.column_stack([raw, structure]).astype(np.float32)


def validate_monitor_config(config: dict[str, Any]) -> None:
    """Reject silent changes to monitoring populations, selection, splits or policies."""
    if config.get("stage") != 6:
        raise ValueError("This entry point implements Stage 6 only.")
    TemporalSplit(**{name: tuple(bounds) for name, bounds in config["split"].items()})
    experiment, drift, performance = config["experiment"], config["drift"], config["performance"]
    if (
        experiment["fit_models_or_preprocessors"] is not False
        or experiment["reselect_decision_thresholds"] is not False
        or experiment["candidate_selection"] != "reuse_stage5_validation_selected_variant"
        or experiment["primary_metric"] != "average_precision"
    ):
        raise ValueError("Monitoring must preserve frozen models, AP and decision policies.")
    seeds = experiment["seeds"]
    if (
        len(seeds) < 5
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or experiment["representative_seed"] not in seeds
    ):
        raise ValueError("Declare at least five distinct training seeds and a representative.")
    if (
        drift["population"] != "all_nodes_including_unknown"
        or drift["reference_steps"] != [1, 24]
        or drift["feature_selection"] != "representative_model_training_gain"
    ):
        raise ValueError("Use all-node training references and training-only feature selection.")
    for value, minimum in (
        (drift["quantile_bins"], 2),
        (drift["top_features"], 1),
        (drift["consecutive_steps"], 1),
        (performance["rolling_steps"], 1),
        (performance["minimum_known_labels"], 1),
        (performance["minimum_illicit_labels"], 1),
        (config["runtime"]["thread_limit"], 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError("Monitoring counts and windows must be positive integers.")
    if (
        not 0 < drift["probability_floor"] < 1 / (drift["quantile_bins"] + 2)
        or not np.isfinite(drift["psi_warning"])
        or drift["psi_warning"] <= 0
        or not 0 < performance["f1_floor_fraction_of_validation"] <= 1
        or performance["incomplete_window_warning"] is not False
    ):
        raise ValueError("Declare finite drift/F1 thresholds and require complete windows.")
    if config["investigation"] != {
        "literature_event_step": 43,
        "periods": {"before_event": [35, 42], "event_and_after": [43, 49]},
    }:
        raise ValueError("Keep the literature-defined event comparison separate from tuning.")


def load_monitor_inputs(config: dict[str, Any]) -> tuple[dict, dict, list[dict]]:
    """Verify Stage 5 evidence, exact seed coverage and the previously selected candidate."""
    path = Path(config["references"]["stage5_manifest"])
    batch = json.loads(path.read_text())
    if batch["status"] != "completed" or not batch["test_evaluated"]:
        raise ValueError("A completed frozen-model evaluation is required before monitoring.")
    plan = json.loads(verify_artifact(batch["protocol"]).read_text())
    if plan["config"]["split"] != config["split"]:
        raise ValueError("Stage 5 and monitoring split policies differ.")
    primary = plan["config"]["experiment"]["primary_variants"]
    runs = []
    for run_id in batch["evaluation_runs"]:
        record_path = path.parent / "runs" / f"{run_id}.json"
        record = json.loads(record_path.read_text())
        if record["variant"] in primary:
            if record["status"] != "completed" or record["models_or_preprocessors_fitted"]:
                raise ValueError("A referenced evaluation record is incomplete or refitted.")
            source = json.loads(verify_artifact(record["source_record"]).read_text())
            if record["frozen_threshold"] != source["threshold"]["value"]:
                raise ValueError("A frozen threshold changed before monitoring.")
            record = {
                **record,
                "monitor_source_record": str(record_path),
                "monitor_source_hash": sha256_file(record_path),
                "source": source,
            }
            verify_artifact(record["test_predictions"])
            runs.append(record)
    for variant in primary:
        if sorted(r["seed"] for r in runs if r["variant"] == variant) != sorted(
            config["experiment"]["seeds"]
        ):
            raise ValueError("Every primary variant must have the exact declared seeds.")
    winner = max(primary, key=plan["primary_validation_ap_means"].get)
    if winner != plan["validation_selected_variant"] or any(
        r["source"]["feature_group"] != "all_structure" for r in runs if r["variant"] == winner
    ):
        raise ValueError("Preserve the validation-selected structural XGBoost candidate.")
    return batch, plan, runs


def prediction_archive(artifact: dict[str, str], data: Any, bounds: tuple[int, int]) -> dict:
    """Require exact IDs, labels and times before reusing scores from a prior stage."""
    nodes = np.flatnonzero((data.time_step >= bounds[0]) & (data.time_step <= bounds[1]))
    with np.load(verify_artifact(artifact)) as saved:
        arrays = {key: saved[key] for key in ("tx_id", "time_step", "label", "score")}
    for key, expected in (
        ("tx_id", data.tx_ids[nodes]),
        ("time_step", data.time_step[nodes]),
        ("label", data.y[nodes]),
    ):
        if not np.array_equal(arrays[key], expected):
            raise ValueError(f"Saved monitoring {key} does not align with observed transactions.")
    known_binary_scores(arrays["label"], arrays["score"])
    return arrays


def run_monitoring(config: dict[str, Any]) -> dict[str, Any]:
    """Freeze train references/validation floors, then describe the already-open test period."""
    config = copy.deepcopy(config)
    validate_monitor_config(config)
    output = Path(config["output_dir"])
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    batch_path = output / "batches" / f"{batch_id}.json"
    batch = {
        "batch_id": batch_id,
        "stage": 6,
        "status": "running",
        "config": config,
        "models_or_preprocessors_fitted": False,
        "decision_thresholds_changed": False,
    }
    write_json(batch_path, batch)
    try:
        previous, plan, runs = load_monitor_inputs(config)
        data = load_csv(
            Path(config["data"]["raw_dir"]),
            require_full_dataset=config["data"]["require_full_dataset"],
        )
        hashes = {name: sha256_file(Path(config["data"]["raw_dir"]) / name) for name in RAW_FILES}
        if hashes != plan["input_files_sha256"]:
            raise ValueError("Monitoring data differs from the frozen evaluation dataset.")
        candidate = plan["validation_selected_variant"]
        selected = sorted([r for r in runs if r["variant"] == candidate], key=lambda r: r["seed"])
        representative = next(
            r for r in selected if r["seed"] == config["experiment"]["representative_seed"]
        )
        observations = Observations.from_data(data)
        nodes, train_x = selected_structure_inputs(
            observations,
            (1, 24),
            plan["config"]["graph"],
            representative["source"]["feature_names"],
        )
        train_known = data.y[nodes] != -1
        fingerprint = array_digest(
            train_x[train_known],
            data.y[nodes][train_known],
            data.tx_ids[nodes][train_known],
            data.time_step[nodes][train_known],
        )
        train_scores, score_histograms, importance = {}, {}, None
        for run in selected:
            source = run["source"]
            if fingerprint != source["input_fingerprints"]["train"]:
                raise ValueError("Reconstructed monitoring training features changed.")
            model = XGBClassifier(n_jobs=config["runtime"]["thread_limit"])
            model.load_model(verify_artifact(source["artifacts"]["model"]))
            with threadpool_limits(limits=config["runtime"]["thread_limit"]):
                score = model.predict_proba(train_x)[:, 1].astype(np.float64)
            train_scores[f"score_s{run['seed']}"] = score
            score_histograms[str(run["seed"])] = fit_histogram(
                score, bins=config["drift"]["quantile_bins"]
            )
            if run["seed"] == config["experiment"]["representative_seed"]:
                importance = model.feature_importances_.astype(np.float64)
        names = representative["source"]["feature_names"]
        ranking = np.lexsort((np.asarray(names), -importance))
        top_indices = ranking[: config["drift"]["top_features"]]
        top_names = [names[i] for i in top_indices]
        feature_histograms = {
            name: fit_histogram(train_x[:, index], bins=config["drift"]["quantile_bins"])
            for name, index in zip(top_names, top_indices)
        }
        artifact = Path(config["artifact_dir"]) / batch_id / "training_reference.npz"
        write_npz(
            artifact,
            tx_id=data.tx_ids[nodes],
            time_step=data.time_step[nodes],
            label=data.y[nodes],
            feature_values=train_x[:, top_indices],
            feature_names=np.asarray(top_names),
            **train_scores,
        )
        floor_records = {
            f"{r['variant']}_s{r['seed']}": {
                "validation_f1": r["source"]["validation_metrics"]["f1"],
                "f1_floor": r["source"]["validation_metrics"]["f1"]
                * config["performance"]["f1_floor_fraction_of_validation"],
                "decision_threshold": r["frozen_threshold"],
            }
            for r in runs
        }
        reference = {
            "batch_id": batch_id,
            "stage": 6,
            "status": "frozen_before_monitoring_test_analysis",
            "test_already_opened_in_stage5": True,
            "prospective_trigger_validation": False,
            "config": config,
            "versions": environment_versions(),
            "input_files_sha256": hashes,
            "stage5_manifest": {
                "path": config["references"]["stage5_manifest"],
                "sha256": sha256_file(Path(config["references"]["stage5_manifest"])),
            },
            "stage5_protocol": previous["protocol"],
            "validation_selected_variant": candidate,
            "training_input_fingerprint": fingerprint,
            "training_reference": {"path": str(artifact), "sha256": sha256_file(artifact)},
            "reference_population": "all training nodes including unknown",
            "reference_nodes": len(nodes),
            "reference_known_labels": int(train_known.sum()),
            "score_reference_in_sample": True,
            "top_feature_names": top_names,
            "top_feature_indices": top_indices.tolist(),
            "training_gain_importance": importance.tolist(),
            "score_histograms": score_histograms,
            "feature_histograms": feature_histograms,
            "validation_floors": floor_records,
            "source_records": [
                {"path": r["monitor_source_record"], "sha256": r["monitor_source_hash"]}
                for r in runs
            ],
            "critical_source_sha256": {
                name: sha256_file(Path("src") / name)
                for name in (
                    "monitor.py",
                    "monitor_reporting.py",
                    "atomic_artifacts.py",
                    "evaluate.py",
                    "inference.py",
                    "features.py",
                    "data.py",
                )
            },
            "frozen_at": datetime.now(timezone.utc).isoformat(),
        }
        reference_path = output / "protocols" / f"{batch_id}.json"
        write_json(reference_path, reference)
        write_json(output / "frozen_reference.json", reference)
        reference_hash = sha256_file(reference_path)
        print(
            json.dumps(
                {
                    "monitoring_reference_frozen": str(reference_path),
                    "candidate": candidate,
                    "top_features": top_names,
                }
            ),
            flush=True,
        )
        del train_x
        step_rows, rolling_rows, period_rows, score_rows, run_logs = [], [], [], [], []
        references = {e["source_run_id"]: e for e in plan["references"]}
        for run in runs:
            arrays = prediction_archive(run["test_predictions"], data, (35, 49))
            y, scores, times = arrays["label"], arrays["score"], arrays["time_step"]
            common = {
                "variant": run["variant"],
                "seed": run["seed"],
                "threshold": run["frozen_threshold"],
            }
            for step in range(35, 50):
                current = times == step
                step_rows.append(
                    {
                        **common,
                        "time_step": step,
                        **step_metrics(
                            y[current], scores[current], threshold=run["frozen_threshold"]
                        ),
                    }
                )
                rolling_rows.append(
                    {
                        **common,
                        "time_step": step,
                        **rolling_performance(
                            y,
                            scores,
                            times,
                            step=step,
                            first_step=35,
                            threshold=run["frozen_threshold"],
                            validation_f1=run["source"]["validation_metrics"]["f1"],
                            policy=config["performance"],
                        ),
                    }
                )
            for period, bounds in config["investigation"]["periods"].items():
                current = (times >= bounds[0]) & (times <= bounds[1])
                period_rows.append(
                    {
                        **common,
                        "period": period,
                        "first_step": bounds[0],
                        "last_step": bounds[1],
                        **step_metrics(
                            y[current], scores[current], threshold=run["frozen_threshold"]
                        ),
                    }
                )
            if run["variant"] == candidate:
                validation = prediction_archive(
                    references[run["source_run_id"]]["validation_predictions"], data, (25, 34)
                )
                train_score = train_scores[f"score_s{run['seed']}"]
                for split, st, ss in (
                    ("train", data.time_step[nodes], train_score),
                    ("validation", validation["time_step"], validation["score"]),
                    ("test", times, scores),
                ):
                    for step in np.unique(st):
                        score_rows.append(
                            {
                                **common,
                                "split": split,
                                "time_step": int(step),
                                **distribution_drift(
                                    train_score,
                                    ss[st == step],
                                    score_histograms[str(run["seed"])],
                                    probability_floor=config["drift"]["probability_floor"],
                                ),
                            }
                        )
            log = {
                "batch_id": batch_id,
                "stage": 6,
                "status": "completed",
                "config": config,
                **common,
                "source_evaluation_record": {
                    "path": run["monitor_source_record"],
                    "sha256": run["monitor_source_hash"],
                },
                "protocol": {"path": str(reference_path), "sha256": reference_hash},
                "test_metrics": run["test_metrics"],
                "test_scores_reused": True,
                "models_or_preprocessors_fitted": False,
                "decision_threshold_changed": False,
            }
            log_path = output / "runs" / f"{batch_id}_{run['variant']}_s{run['seed']}.json"
            write_json(log_path, log)
            run_logs.append({"path": str(log_path), "sha256": sha256_file(log_path)})
        # Only after the reference is persisted do held-out feature distributions enter monitoring.
        all_nodes, all_x = selected_structure_inputs(
            observations, (1, 49), plan["config"]["graph"], names
        )
        feature_rows = []
        for name, index in zip(top_names, top_indices):
            train_values = all_x[data.time_step[all_nodes] <= 24, index]
            for step in range(1, 50):
                current = data.time_step[all_nodes] == step
                feature_rows.append(
                    {
                        "feature": name,
                        "time_step": step,
                        "split": "train" if step <= 24 else "validation" if step <= 34 else "test",
                        **distribution_drift(
                            train_values,
                            all_x[current, index],
                            feature_histograms[name],
                            probability_floor=config["drift"]["probability_floor"],
                        ),
                    }
                )
        from src.monitor_reporting import build_monitor_report

        report = build_monitor_report(
            reference, step_rows, rolling_rows, period_rows, score_rows, feature_rows, output
        )
        if sha256_file(reference_path) != reference_hash:
            raise ValueError("Monitoring reference changed during analysis.")
        for source in reference["source_records"]:
            verify_artifact(source)
        for name, digest in reference["critical_source_sha256"].items():
            if sha256_file(Path("src") / name) != digest:
                raise ValueError(f"Monitoring source changed during execution: {name}")
        batch.update(
            status="completed",
            protocol={"path": str(reference_path), "sha256": reference_hash},
            runs=run_logs,
            report=report,
        )
        write_json(batch_path, batch)
        write_json(output / "latest_run.json", batch)
        return batch
    except Exception as error:
        batch.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        write_json(batch_path, batch)
        raise


def main() -> None:
    """CLI: python -m src.monitor --config configs/stage6.yaml."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage6.yaml"))
    run_monitoring(yaml.safe_load(parser.parse_args().config.read_text()))


if __name__ == "__main__":
    main()
