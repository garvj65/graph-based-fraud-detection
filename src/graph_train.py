"""Stage 3: fixed-parameter XGBoost comparisons with leakage-guarded graph features."""

from __future__ import annotations

import argparse
import copy
import json
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data import (
    RAW_FILES,
    TemporalSplit,
    download_csvs,
    environment_versions,
    load_csv,
    make_temporal_masks,
    sha256_file,
    write_json,
)
from src.features import NEIGHBOR_NAMES, STRUCTURE_NAMES, GraphFeatures, build_graph_features
from src.models import DevelopmentData, development_data, validate_development_data
from src.train import array_digest, fit_run

VARIANTS = {"all_control", "all_structure", "all_graph", "structure_only"}


def validate_graph_config(config: dict[str, Any]) -> None:
    """Require the agreed temporal/metric protocol and development-only processing."""
    if config.get("stage") != 3:
        raise ValueError("This entry point implements Stage 3 only.")
    TemporalSplit(**{name: tuple(bounds) for name, bounds in config["split"].items()})
    experiment = config["experiment"]
    if experiment["evaluate_test"] is not False or config["graph"]["max_time_step"] != 34:
        raise ValueError("Stage 3 processes steps 1–34 and does not evaluate test performance.")
    if experiment["graph_availability"] != "completed_snapshot":
        raise ValueError("These graph features require the completed-snapshot protocol.")
    if (
        experiment["primary_metric"],
        experiment["threshold_objective"],
        experiment["threshold_tie_break"],
    ) != ("average_precision", "max_f1", "highest_threshold"):
        raise ValueError("Stage 3 must preserve the Stage 2 metric/threshold definitions.")
    variants = experiment["variants"]
    if not variants or len(set(variants)) != len(variants) or not set(variants) <= VARIANTS:
        raise ValueError("Choose unique documented graph variants.")
    if "all_control" not in variants:
        raise ValueError("A paired all-feature control is required.")
    seeds = experiment["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Seeds must be nonnegative integers.")
    if experiment["representative_seed"] not in seeds:
        raise ValueError("Representative seed must be run, not selected after seeing scores.")
    if not isinstance(config["data"]["require_full_dataset"], bool):
        raise ValueError("require_full_dataset must be a YAML boolean.")
    if config["runtime"]["thread_limit"] < 1:
        raise ValueError("Thread limit must be positive.")


def graph_development_view(
    base: DevelopmentData,
    graph: GraphFeatures,
    *,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    variant: str,
) -> DevelopmentData:
    """Attach only the intended columns; test feature rows never enter this view."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown graph variant: {variant}")
    if variant == "all_control":
        return base
    names = list(STRUCTURE_NAMES)
    values = graph.structure
    if variant == "all_graph":
        values = np.column_stack([values, graph.neighbor_scores])
        names += NEIGHBOR_NAMES
    train, validation = values[train_mask], values[validation_mask]
    if variant != "structure_only":
        train = np.column_stack([base.x_train, train])
        validation = np.column_stack([base.x_validation, validation])
        names = [*base.feature_names, *names]
    view = replace(base, x_train=train, x_validation=validation, feature_names=names)
    validate_development_data(view)
    return view


def load_stage2_reference(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Load the already-selected baseline, never choose a new candidate using graph results."""
    experiment = config["experiment"]
    selection_path = Path(experiment["selection_record"])
    manifest_path = Path(experiment["baseline_run_manifest"])
    selection = json.loads(selection_path.read_text())
    batch = json.loads(manifest_path.read_text())
    best = selection["best_primary_tabular_variant"]
    if (best["model"], best["feature_group"], best["include_time"]) != ("xgboost", "all", True):
        raise ValueError(
            "This comparison requires the reviewed Stage 2 all-feature XGBoost baseline."
        )
    if best["candidate_index"] != experiment["fixed_candidate_index"]:
        raise ValueError("Stage 2 candidate changed; review the fixed Stage 3 contract first.")
    records = {}
    for run_id in batch["selected_runs"]:
        record = json.loads((manifest_path.parent / "runs" / f"{run_id}.json").read_text())
        if (
            record["model"] == "xgboost"
            and record["feature_group"] == "all"
            and record["include_time"]
        ):
            records[record["seed"]] = record
    if not set(experiment["seeds"]) <= set(records):
        raise ValueError("Stage 2 must have matching baseline records for every requested seed.")
    parameters = records[experiment["seeds"][0]]["parameters"]
    if any(r["parameters"] != parameters for r in records.values()):
        raise ValueError("Stage 2 candidate parameters are inconsistent across seeds.")
    return parameters, records


def run_graph_experiment(config: dict[str, Any], *, download: bool = False) -> dict[str, Any]:
    """Run four paired ablations with frozen baseline hyperparameters and five seeds."""
    config = copy.deepcopy(config)
    output = Path(config["output_dir"])
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    batch_path = output / "batches" / f"{batch_id}.json"
    batch: dict[str, Any] = {
        "batch_id": batch_id,
        "stage": 3,
        "status": "running",
        "config": config,
        "test_evaluated": False,
        "selected_runs": [],
    }
    write_json(batch_path, batch)
    try:
        validate_graph_config(config)
        parameters, baseline = load_stage2_reference(config)
        raw_dir = Path(config["data"]["raw_dir"])
        if download:
            download_csvs(raw_dir)
        data = load_csv(raw_dir, require_full_dataset=config["data"]["require_full_dataset"])
        provenance = {
            "versions": environment_versions(),
            "source_files_sha256": {
                p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "input_files_sha256": {name: sha256_file(raw_dir / name) for name in RAW_FILES},
            "stage2_reference_files": {
                key: {
                    "path": config["experiment"][key],
                    "sha256": sha256_file(Path(config["experiment"][key])),
                }
                for key in ("selection_record", "baseline_run_manifest")
            },
        }
        batch["provenance"] = provenance
        feature_artifacts = Path(config["artifact_dir"]) / batch_id / "feature_build"
        graph = build_graph_features(data, config, artifact_dir=feature_artifacts)
        masks = make_temporal_masks(data)
        if not np.isnan(graph.structure[masks.test_nodes]).all():
            raise ValueError("Test graph statistics were computed before final evaluation.")
        feature_path = feature_artifacts / "graph_features.npz"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            feature_path,
            structure=graph.structure,
            neighbor_scores=graph.neighbor_scores,
            node_scores=graph.node_scores,
        )
        feature_evidence = {
            "batch_id": batch_id,
            "graph_config": config["graph"],
            "risk_model_config": config["risk_model"],
            "structure_feature_names": STRUCTURE_NAMES,
            "neighbor_feature_names": NEIGHBOR_NAMES,
            "graph_scope": "separate completed snapshots 1–34; test rows remain NaN",
            "raw_label_neighbor_features_used": False,
            "feature_artifact": {"path": str(feature_path), "sha256": sha256_file(feature_path)},
            "feature_values_sha256": array_digest(graph.structure, graph.neighbor_scores),
            "snapshot_audit": graph.snapshot_audit,
            "score_generation_audit": graph.score_audit,
            "test_features_computed": False,
        }
        write_json(output / "graph_feature_manifest.json", feature_evidence)
        unique_helper_fits = {row["fit_end_step"]: row["fit"] for row in graph.score_audit}
        for cutoff, info in unique_helper_fits.items():
            write_json(
                output / "helper_fit_logs" / f"{batch_id}_prefix_{cutoff:02d}.json",
                {
                    **info,
                    "batch_id": batch_id,
                    "seed": config["risk_model"]["seed"],
                    "metrics": None,
                    "metrics_note": "No held-out labels used to train/tune score helpers.",
                    "provenance": provenance,
                },
            )
        provenance["graph_feature_values_sha256"] = feature_evidence["feature_values_sha256"]
        base = development_data(data, "all", include_time=True)
        selected = []
        for variant in config["experiment"]["variants"]:
            view = graph_development_view(
                base,
                graph,
                train_mask=masks.train_labeled,
                validation_mask=masks.validation_labeled,
                variant=variant,
            )
            for seed in config["experiment"]["seeds"]:
                run = fit_run(
                    model_name="xgboost",
                    group=variant,
                    include_time=variant != "structure_only",
                    seed=seed,
                    candidate_index=config["experiment"]["fixed_candidate_index"],
                    phase="fixed_stage2_parameters",
                    parameters=parameters,
                    data=view,
                    config=config,
                    batch_id=batch_id,
                    provenance=provenance,
                )
                selected.append(run)
                batch["selected_runs"].append(run["run_id"])
                write_json(batch_path, batch)
        from src.graph_reporting import build_graph_report

        report = build_graph_report(selected, baseline, feature_evidence, config, output)
        batch.update(status="completed", report=report)
        write_json(batch_path, batch)
        write_json(output / "latest_run.json", batch)
        return batch
    except Exception as error:
        batch.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        write_json(batch_path, batch)
        raise


def main() -> None:
    """CLI: python -m src.graph_train --config configs/stage3.yaml --download."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage3.yaml"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    batch = run_graph_experiment(yaml.safe_load(args.config.read_text()), download=args.download)
    print(
        json.dumps(
            {"batch_id": batch["batch_id"], "status": batch["status"], "report": batch["report"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
