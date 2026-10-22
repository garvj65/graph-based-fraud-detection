"""Stage 4: temporal PyG training, validation-AP early stopping and frozen thresholds."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score

from src.data import (
    RAW_FILES,
    TemporalSplit,
    download_csvs,
    environment_versions,
    load_csv,
    sha256_file,
    write_json,
)
from src.evaluate import binary_metrics, select_validation_threshold
from src.gnn import (
    GNNDevelopmentData,
    SmallGNN,
    prepare_gnn_data,
    seed_everything,
    supervised_loss,
    validate_gnn_data,
)
from src.models import training_class_weights
from src.train import array_digest


@dataclass
class EarlyStopping:
    """Track strict validation-AP improvement; exact ties keep the earliest epoch."""

    patience: int
    min_delta: float = 0.0
    best_ap: float = -1.0
    best_epoch: int = 0
    stale_epochs: int = 0

    def update(self, epoch: int, ap: float) -> bool:
        """Return whether to checkpoint; reject invalid selection metrics."""
        if not np.isfinite(ap) or not 0 <= ap <= 1:
            raise ValueError("Early stopping requires finite validation AP in [0, 1].")
        if ap > self.best_ap + self.min_delta:
            self.best_ap, self.best_epoch, self.stale_epochs = ap, epoch, 0
            return True
        self.stale_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        """Stop after the configured count of epochs without improvement."""
        return self.stale_epochs >= self.patience


def validate_gnn_config(config: dict[str, Any]) -> None:
    """Keep the reviewed split, AP definition and threshold policy unchanged."""
    if config.get("stage") != 4:
        raise ValueError("This entry point implements Stage 4 only.")
    TemporalSplit(**{key: tuple(value) for key, value in config["split"].items()})
    experiment, training, runtime = (config["experiment"], config["training"], config["runtime"])
    if experiment["evaluate_test"] is not False or config["graph"]["max_time_step"] != 34:
        raise ValueError("Stage 4 does not evaluate test or prepare test graph statistics.")
    if (
        experiment["primary_metric"],
        experiment["threshold_objective"],
        experiment["threshold_tie_break"],
        experiment["graph_availability"],
    ) != ("average_precision", "max_f1", "highest_threshold", "completed_snapshot"):
        raise ValueError("Preserve the reviewed AP, threshold and snapshot protocol.")
    if experiment["include_time"] is not True or experiment["primary_feature_group"] != "all":
        raise ValueError("The primary 94/166-feature convention must remain explicit.")
    seeds = experiment["seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("Seeds must be unique nonnegative integers.")
    if experiment["tuning_seed"] not in seeds:
        raise ValueError("The fixed tuning seed must be included.")
    for key, allowed in (
        ("models", {"gcn", "graphsage"}),
        ("feature_groups", {"local", "all", "structure_only"}),
    ):
        values = experiment[key]
        if not values or len(set(values)) != len(values) or not set(values) <= allowed:
            raise ValueError(f"Choose unique supported {key}.")
    if "all" not in experiment["feature_groups"]:
        raise ValueError("The all-feature primary comparison is required.")
    if not isinstance(experiment["run_no_message_control"], bool):
        raise ValueError("No-message control must be explicitly enabled or disabled.")
    if config["graph"]["direction"] != "bidirectional_unique":
        raise ValueError("Use the declared bidirectional message graph.")
    if training["max_epochs"] < 1 or training["patience"] < 1:
        raise ValueError("Epoch count and patience must be positive.")
    if training["min_delta"] != 0 or training["weight_decay"] < 0:
        raise ValueError("Strict AP improvement and nonnegative weight decay are required.")
    if training["gradient_clip_norm"] <= 0 or not training["candidates"]:
        raise ValueError("A positive gradient limit and predeclared candidates are required.")
    for candidate in training["candidates"]:
        if set(candidate) != {"hidden_channels", "dropout", "learning_rate"}:
            raise ValueError("Candidates require explicit width, dropout and learning rate.")
        if candidate["hidden_channels"] < 1 or not 0 <= candidate["dropout"] < 1:
            raise ValueError("Invalid hidden width or dropout.")
        if candidate["learning_rate"] <= 0:
            raise ValueError("Learning rate must be positive.")
    if runtime["device"] not in {"cpu", "cuda"} or runtime["thread_limit"] < 1:
        raise ValueError("Choose cpu/cuda and a positive thread limit.")
    if runtime["deterministic_algorithms"] is not True:
        raise ValueError("Deterministic algorithms must be requested explicitly.")
    if not isinstance(config["data"]["require_full_dataset"], bool):
        raise ValueError("require_full_dataset must be a boolean.")


def fit_gnn_run(
    data: GNNDevelopmentData,
    *,
    model_name: str,
    group: str,
    seed: int,
    candidate_index: int,
    parameters: dict[str, Any],
    phase: str,
    config: dict[str, Any],
    batch_id: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Fit on the training graph only; validation is used without gradients for AP.

    Checkpoint selection uses AP only. F1 threshold selection happens once,
    after restoring the best checkpoint. Unknown labels never reach BCE loss.
    """
    validate_gnn_data(data)
    run_id = f"{batch_id}_{model_name}_{group}_c{candidate_index}_s{seed}"
    output = Path(config["output_dir"])
    artifact = Path(config["artifact_dir"]) / batch_id / run_id
    artifact.mkdir(parents=True, exist_ok=False)
    log_path = output / "runs" / f"{run_id}.json"
    record: dict[str, Any] = {
        "run_id": run_id,
        "batch_id": batch_id,
        "stage": 4,
        "status": "running",
        "model": model_name,
        "feature_group": group,
        "include_time": group != "structure_only",
        "feature_names": data.feature_names,
        "n_features": len(data.feature_names),
        "seed": seed,
        "candidate_index": candidate_index,
        "phase": phase,
        "parameters": parameters,
        "config": config,
        "provenance": provenance,
        "data_audit": data.audit,
        "fit_scope": "graph nodes 1–24; known train labels only in loss",
        "selection_scope": "known validation labels 25–34; AP early stopping",
        "test_evaluated": False,
    }
    write_json(log_path, record)
    started = time.perf_counter()
    try:
        runtime, training = config["runtime"], config["training"]
        device = runtime["device"]
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA explicitly requested but unavailable; no silent fallback.")
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        seed_everything(seed, thread_limit=runtime["thread_limit"], deterministic=True)
        train, validation = data.train.to(device), data.validation.to(device)
        weights = training_class_weights(train.y[train.known].cpu().numpy())
        model = SmallGNN(
            model_name,
            len(data.feature_names),
            parameters["hidden_channels"],
            parameters["dropout"],
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=parameters["learning_rate"],
            weight_decay=training["weight_decay"],
        )
        stopper = EarlyStopping(training["patience"], training["min_delta"])
        known_val = validation.known
        val_y = validation.y[known_val].cpu().numpy()
        history, best_state = [], None
        for epoch in range(1, training["max_epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(train.x, train.edge_index)
            loss = supervised_loss(logits, train, positive_weight=weights["scale_pos_weight"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; run is not reportable.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training["gradient_clip_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_logits = model(validation.x, validation.edge_index)
                scores = val_logits[known_val].sigmoid().cpu().numpy().astype(np.float64)
                val_loss = supervised_loss(
                    val_logits,
                    validation,
                    positive_weight=weights["scale_pos_weight"],
                )
            ap = float(average_precision_score(val_y, scores))
            improved = stopper.update(epoch, ap)
            history.append(
                {
                    "epoch": epoch,
                    "train_weighted_bce": float(loss.detach()),
                    "validation_weighted_bce": float(val_loss),
                    "validation_ap": ap,
                    "gradient_norm_before_clip": float(norm),
                    "checkpoint_improved": improved,
                }
            )
            if improved:
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            if epoch % 50 == 0:
                print(
                    json.dumps(
                        {"run_id": run_id, "epoch": epoch, "best_validation_ap": stopper.best_ap}
                    ),
                    flush=True,
                )
            if stopper.should_stop:
                break
        if best_state is None:
            raise RuntimeError("Training produced no valid checkpoint.")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            all_scores = model(validation.x, validation.edge_index).sigmoid()
            scores = all_scores.cpu().numpy().astype(np.float64)
        labels = validation.y.cpu().numpy()
        threshold = select_validation_threshold(labels, scores)
        metrics = binary_metrics(labels, scores, threshold=threshold["value"])
        if metrics["average_precision"] != stopper.best_ap:
            raise RuntimeError("Restored checkpoint does not reproduce best validation AP.")
        checkpoint_path = artifact / "checkpoint.pt"
        torch.save(
            {
                "state_dict": best_state,
                "model": model_name,
                "parameters": parameters,
                "in_channels": len(data.feature_names),
                "best_epoch": stopper.best_epoch,
            },
            checkpoint_path,
        )
        scaler_path = artifact / "scaler.joblib"
        joblib.dump(data.scaler, scaler_path)
        score_path = artifact / "validation_predictions.npz"
        np.savez_compressed(
            score_path,
            tx_id=validation.tx_ids,
            time_step=validation.times,
            label=labels,
            score=scores,
        )
        history_path = output / "histories" / f"{run_id}.csv"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(history_path, index=False)
        threshold["validation_scores_sha256"] = array_digest(validation.tx_ids, labels, scores)
        record.update(
            status="completed",
            fit_seconds=time.perf_counter() - started,
            class_weights=weights,
            best_epoch=stopper.best_epoch,
            epochs_run=len(history),
            stop_reason="patience" if stopper.should_stop else "max_epochs",
            best_validation_ap=stopper.best_ap,
            threshold=threshold,
            validation_metrics=metrics,
            parameter_count=sum(value.numel() for value in model.parameters()),
            device=device,
            artifacts={
                key: {"path": str(path), "sha256": sha256_file(path)}
                for key, path in (
                    ("checkpoint", checkpoint_path),
                    ("scaler", scaler_path),
                    ("validation_predictions", score_path),
                    ("history", history_path),
                )
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
                    "best_epoch": stopper.best_epoch,
                    "epochs_run": len(history),
                    "fit_seconds": round(record["fit_seconds"], 3),
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


def run_gnn_experiment(config: dict[str, Any], *, download: bool = False) -> dict[str, Any]:
    """Tune on seed 0, replicate each frozen winner, then run no-message controls."""
    config = copy.deepcopy(config)
    output = Path(config["output_dir"])
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    batch_path = output / "batches" / f"{batch_id}.json"
    batch: dict[str, Any] = {
        "batch_id": batch_id,
        "stage": 4,
        "status": "running",
        "config": config,
        "test_evaluated": False,
        "trial_runs": [],
        "selected_runs": [],
        "choices": {},
    }
    write_json(batch_path, batch)
    try:
        validate_gnn_config(config)
        raw_dir = Path(config["data"]["raw_dir"])
        if download:
            download_csvs(raw_dir)
        data = load_csv(raw_dir, require_full_dataset=config["data"]["require_full_dataset"])
        provenance = {
            "versions": environment_versions(),
            "input_files_sha256": {name: sha256_file(raw_dir / name) for name in RAW_FILES},
            "source_files_sha256": {
                path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))
            },
            "reference_files_sha256": {
                key: sha256_file(Path(config["experiment"][key]))
                for key in ("baseline_summary", "graph_summary")
            },
        }
        batch["provenance"] = provenance
        experiment = config["experiment"]
        selected, tuning, prepared = [], [], {}
        for group in experiment["feature_groups"]:
            prepared[group] = prepare_gnn_data(data, group, config["graph"])
        write_json(
            output / "data_manifest.json",
            {
                "batch_id": batch_id,
                "provenance": provenance,
                "variants": {key: value.audit for key, value in prepared.items()},
                "test_evaluated": False,
            },
        )
        for model_name in experiment["models"]:
            for group in experiment["feature_groups"]:
                candidates = []
                for index, parameters in enumerate(config["training"]["candidates"]):
                    run = fit_gnn_run(
                        prepared[group],
                        model_name=model_name,
                        group=group,
                        seed=experiment["tuning_seed"],
                        candidate_index=index,
                        parameters=parameters,
                        phase="tuning",
                        config=config,
                        batch_id=batch_id,
                        provenance=provenance,
                    )
                    candidates.append(run)
                    tuning.append(run)
                    batch["trial_runs"].append(run["run_id"])
                    write_json(batch_path, batch)
                # Stable candidate order breaks an exact AP tie; never pick the best seed.
                winner = max(
                    candidates, key=lambda run: run["validation_metrics"]["average_precision"]
                )
                key = f"{model_name}_{group}"
                batch["choices"][key] = {
                    "candidate_index": winner["candidate_index"],
                    "parameters": winner["parameters"],
                    "selection_seed": experiment["tuning_seed"],
                    "selection_metric": "validation_average_precision",
                }
                for seed in experiment["seeds"]:
                    run = (
                        winner
                        if seed == experiment["tuning_seed"]
                        else fit_gnn_run(
                            prepared[group],
                            model_name=model_name,
                            group=group,
                            seed=seed,
                            candidate_index=winner["candidate_index"],
                            parameters=winner["parameters"],
                            phase="replication",
                            config=config,
                            batch_id=batch_id,
                            provenance=provenance,
                        )
                    )
                    selected.append(run)
                    batch["selected_runs"].append(run["run_id"])
                    write_json(batch_path, batch)
            if experiment["run_no_message_control"]:
                control = prepare_gnn_data(data, "all", config["graph"], no_messages=True)
                choice = batch["choices"][f"{model_name}_all"]
                for seed in experiment["seeds"]:
                    run = fit_gnn_run(
                        control,
                        model_name=model_name,
                        group="all_no_messages",
                        seed=seed,
                        candidate_index=choice["candidate_index"],
                        parameters=choice["parameters"],
                        phase="no_message_control_fixed_all_parameters",
                        config=config,
                        batch_id=batch_id,
                        provenance=provenance,
                    )
                    selected.append(run)
                    batch["selected_runs"].append(run["run_id"])
                    write_json(batch_path, batch)
        from src.gnn_reporting import build_gnn_report

        report = build_gnn_report(selected, tuning, batch["choices"], config, output)
        batch.update(status="completed", report=report)
        write_json(batch_path, batch)
        write_json(output / "latest_run.json", batch)
        return batch
    except Exception as error:
        batch.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        write_json(batch_path, batch)
        raise


def main() -> None:
    """Run the full experiment or explicitly smaller model/seed subsets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage4.yaml"))
    parser.add_argument("--model", choices=["gcn", "graphsage"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.model:
        config["experiment"]["models"] = [args.model]
    if args.seed is not None:
        config["experiment"].update(
            seeds=[args.seed], tuning_seed=args.seed, representative_seed=args.seed
        )
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.artifact_dir:
        config["artifact_dir"] = str(args.artifact_dir)
    if args.device:
        config["runtime"]["device"] = args.device
    run_gnn_experiment(config, download=args.download)


if __name__ == "__main__":
    main()
