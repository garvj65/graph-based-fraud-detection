"""Reproducible Stage 2 tabular fitting, selection and validation-only reporting.

Example: python -m src.train --model xgboost --seed 0
The default command runs both model families, all feature variants and five seeds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from src.data import (
    RAW_FILES,
    TemporalSplit,
    download_csvs,
    environment_versions,
    load_csv,
    load_via_pyg,
    sha256_file,
    write_json,
)
from src.evaluate import binary_metrics, select_validation_threshold
from src.models import DevelopmentData, development_data, fit_tabular


def array_digest(*arrays: np.ndarray) -> str:
    """Hash array shape, dtype and content without Python's randomized hash()."""
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str((contiguous.shape, contiguous.dtype.str)).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    """Reject accidental test evaluation or unimplemented decision policies."""
    if config.get("stage") != 2:
        raise ValueError("This entry point implements stage 2 only.")
    TemporalSplit(**{name: tuple(bounds) for name, bounds in config["split"].items()})
    experiment = config["experiment"]
    seeds = experiment["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Seeds must be nonnegative integers.")
    if experiment["evaluate_test"] is not False:
        raise ValueError("Stage 2 does not evaluate test data; reserve that for stage 5.")
    if experiment["primary_metric"] not in {"average_precision", "pr_auc_trapezoid"}:
        raise ValueError("Choose an explicit supported PR metric definition.")
    if experiment["threshold_objective"] != "max_f1":
        raise ValueError("Only the explicitly defined validation-max-F1 policy is implemented.")
    if experiment["threshold_tie_break"] != "highest_threshold":
        raise ValueError("Threshold tie policy must be highest_threshold.")
    if not isinstance(experiment["primary_include_time"], bool):
        raise ValueError("Primary time-feature policy must be boolean.")
    options = experiment["include_time_options"]
    if not options or any(not isinstance(value, bool) for value in options):
        raise ValueError("Time-feature variants must be explicit booleans.")
    if len(set(options)) != len(options) or experiment["primary_include_time"] not in options:
        raise ValueError("Time variants must be unique and include the primary policy.")
    if not experiment["feature_groups"] or not set(experiment["feature_groups"]) <= {
        "local",
        "all",
    }:
        raise ValueError("Feature groups must be local and/or all.")
    if experiment["tuning_seed"] not in seeds or experiment["representative_seed"] not in seeds:
        raise ValueError("Configured tuning/representative seeds must belong to the seed list.")
    for name in ("logreg", "xgboost"):
        if not config["models"][name]["candidates"]:
            raise ValueError("Each model family needs at least one predeclared candidate.")
    if not isinstance(config["data"]["require_full_dataset"], bool):
        raise ValueError("require_full_dataset must be a YAML boolean.")
    if config["runtime"]["thread_limit"] < 1:
        raise ValueError("thread_limit must be positive.")


def fit_run(
    *,
    model_name: str,
    group: str,
    include_time: bool,
    seed: int,
    candidate_index: int,
    phase: str,
    parameters: dict[str, Any],
    data: DevelopmentData,
    config: dict[str, Any],
    batch_id: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Fit one immutable trial, save its artifacts and log success or failure."""
    variant = f"{model_name}_{group}_{'with_time' if include_time else 'no_time'}"
    run_id = f"{batch_id}_{variant}_c{candidate_index}_s{seed}"
    output = Path(config["output_dir"])
    artifact = Path(config["artifact_dir"]) / batch_id / run_id
    artifact.mkdir(parents=True, exist_ok=False)
    log_path = output / "runs" / f"{run_id}.json"
    record: dict[str, Any] = {
        "run_id": run_id,
        "batch_id": batch_id,
        "stage": config["stage"],
        "status": "running",
        "model": model_name,
        "feature_group": group,
        "include_time": include_time,
        "n_features": len(data.feature_names),
        "feature_names": data.feature_names,
        "seed": seed,
        "candidate_index": candidate_index,
        "phase": phase,
        "parameters": parameters,
        "config": config,
        "provenance": provenance,
        "fit_scope": "known labels in time steps 1–24 only",
        "selection_scope": "known labels in time steps 25–34 only",
        "test_evaluated": False,
        "input_fingerprints": {
            "train": array_digest(data.x_train, data.y_train, data.train_ids, data.train_times),
            "validation": array_digest(
                data.x_validation, data.y_validation, data.validation_ids, data.validation_times
            ),
        },
    }
    write_json(log_path, record)
    started = time.perf_counter()
    try:
        model, weights, captured_warnings = fit_tabular(
            model_name,
            data,
            seed=seed,
            parameters=parameters,
            thread_limit=config["runtime"]["thread_limit"],
        )
        fit_seconds = time.perf_counter() - started
        with threadpool_limits(limits=config["runtime"]["thread_limit"]):
            scores = model.predict_proba(data.x_validation)[:, 1].astype(np.float64)
        threshold = select_validation_threshold(data.y_validation, scores)
        metrics = binary_metrics(data.y_validation, scores, threshold=threshold["value"])
        if model_name == "xgboost":
            model_path = artifact / "model.ubj"
            model.save_model(model_path)
            preprocess = {"scaler": None, "histogram_fit_scope": "training only"}
        else:
            model_path = artifact / "pipeline.joblib"
            joblib.dump(model, model_path)
            scaler = model.named_steps["scaler"]
            preprocess = {
                "scaler": "StandardScaler",
                "fit_scope": "labeled training rows only",
                "n_samples_seen": int(scaler.n_samples_seen_),
                "mean_sha256": array_digest(scaler.mean_),
                "scale_sha256": array_digest(scaler.scale_),
            }
        prediction_path = artifact / "validation_predictions.npz"
        np.savez_compressed(
            prediction_path,
            tx_id=data.validation_ids,
            time_step=data.validation_times,
            label=data.y_validation,
            score=scores,
        )
        threshold["validation_scores_sha256"] = array_digest(
            data.validation_ids, data.y_validation, scores
        )
        record.update(
            status="completed",
            class_weights=weights,
            preprocessing=preprocess,
            warnings=captured_warnings,
            fit_seconds=fit_seconds,
            threshold=threshold,
            validation_metrics=metrics,
            artifacts={
                "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
                "validation_predictions": {
                    "path": str(prediction_path),
                    "sha256": sha256_file(prediction_path),
                },
            },
        )
        write_json(log_path, record)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "phase": phase,
                    "average_precision": metrics["average_precision"],
                    "f1": metrics["f1"],
                    "fit_seconds": round(fit_seconds, 3),
                }
            ),
            flush=True,
        )
        return record
    except Exception as error:
        record.update(
            status="failed",
            elapsed_seconds=time.perf_counter() - started,
            error={"type": type(error).__name__, "message": str(error)},
        )
        write_json(log_path, record)
        raise


def run_experiment(
    config: dict[str, Any],
    *,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    download: bool = False,
) -> dict[str, Any]:
    """Tune on a fixed seed, freeze candidates, then replicate across requested seeds.

    Tuning uses validation AP (or explicitly configured trapezoidal PR-AUC), not
    validation F1. The winning seed-0 fit is reused, not counted as another fit.
    No test predictions, test metrics or train+validation refit are performed.
    """
    config = copy.deepcopy(config)
    output = Path(config["output_dir"])
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    batch_path = output / "batches" / f"{batch_id}.json"
    batch: dict[str, Any] = {
        "batch_id": batch_id,
        "stage": 2,
        "status": "running",
        "config": config,
        "test_evaluated": False,
        "selected_runs": [],
        "trial_runs": [],
    }
    write_json(batch_path, batch)
    try:
        validate_config(config)
        chosen_models = models if models is not None else ["logreg", "xgboost"]
        requested_seeds = seeds if seeds is not None else config["experiment"]["seeds"]
        if not chosen_models or not set(chosen_models) <= {"logreg", "xgboost"}:
            raise ValueError("Choose logreg and/or xgboost.")
        if not requested_seeds or not set(requested_seeds) <= set(config["experiment"]["seeds"]):
            raise ValueError("Requested seeds must belong to the configured seed list.")
        batch.update(requested_models=chosen_models, requested_seeds=requested_seeds)
        raw_dir = Path(config["data"]["raw_dir"])
        if config["data"]["source"] == "pyg":
            root = Path(config["data"]["root"])
            data = load_via_pyg(root, require_full_dataset=config["data"]["require_full_dataset"])
            raw_dir = root / "raw"
        elif config["data"]["source"] == "csv":
            if download:
                download_csvs(raw_dir)
            data = load_csv(raw_dir, require_full_dataset=config["data"]["require_full_dataset"])
        else:
            raise ValueError("Unsupported data source.")
        provenance = {
            "versions": environment_versions(),
            "source_files_sha256": {
                path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))
            },
            "input_files_sha256": {name: sha256_file(raw_dir / name) for name in RAW_FILES},
        }
        batch["provenance"] = provenance
        selected, tuning = [], []
        metric_name = config["experiment"]["primary_metric"]
        tuning_seed = config["experiment"]["tuning_seed"]
        for group in config["experiment"]["feature_groups"]:
            for include_time in config["experiment"]["include_time_options"]:
                development = development_data(data, group, include_time=include_time)
                for model_name in chosen_models:
                    specification = config["models"][model_name]
                    candidates = []
                    for index, candidate in enumerate(specification["candidates"]):
                        parameters = {**specification["common"], **candidate}
                        trial = fit_run(
                            model_name=model_name,
                            group=group,
                            include_time=include_time,
                            seed=tuning_seed,
                            candidate_index=index,
                            phase="tuning",
                            parameters=parameters,
                            data=development,
                            config=config,
                            batch_id=batch_id,
                            provenance=provenance,
                        )
                        candidates.append(trial)
                        tuning.append(trial)
                        batch["trial_runs"].append(trial["run_id"])
                        write_json(batch_path, batch)
                    # Stable candidate ordering is the tie break; never choose a best random seed.
                    winner = max(candidates, key=lambda run: run["validation_metrics"][metric_name])
                    for seed in requested_seeds:
                        trial = (
                            winner
                            if seed == tuning_seed
                            else fit_run(
                                model_name=model_name,
                                group=group,
                                include_time=include_time,
                                seed=seed,
                                candidate_index=winner["candidate_index"],
                                phase="replication",
                                parameters=winner["parameters"],
                                data=development,
                                config=config,
                                batch_id=batch_id,
                                provenance=provenance,
                            )
                        )
                        selected.append(trial)
                        batch["selected_runs"].append(trial["run_id"])
                        if seed != tuning_seed:
                            batch["trial_runs"].append(trial["run_id"])
                        write_json(batch_path, batch)
        from src.reporting import build_stage2_report

        report = build_stage2_report(selected, tuning, config, output)
        batch.update(status="completed", report=report)
        write_json(batch_path, batch)
        write_json(output / "latest_run.json", batch)
        return batch
    except Exception as error:
        batch.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        write_json(batch_path, batch)
        raise


def main() -> None:
    """CLI for one model/seed or the complete predeclared Stage 2 experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2.yaml"))
    parser.add_argument("--model", choices=("all", "logreg", "xgboost"), default="all")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.raw_dir:
        config["data"]["raw_dir"] = str(args.raw_dir)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    batch = run_experiment(
        config,
        models=None if args.model == "all" else [args.model],
        seeds=None if args.seed is None else [args.seed],
        download=args.download,
    )
    print(
        json.dumps(
            {"batch_id": batch["batch_id"], "status": batch["status"], "report": batch["report"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
